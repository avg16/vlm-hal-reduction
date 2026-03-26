"""
VQA Model with Counterfactual SCM
===================================
Architecture mirrors the paper's setup:
  - Visual encoder  fv(v, u^v)  : bottom-up features → embedding
  - Language encoder fq(q, u^q) : LSTM / GloVe → embedding
  - Fusion h(e_v, e_q)          : element-wise product + MLP
  - Classifier                  : MLP → answer logits

For the SCM, the embedding functions are replaced by their
counterfactual versions ẽ_v = f̃_v(v, u^v) during training.

Supports:
  - UpDn-style (UpDn+C from Table 3)
  - LSTM / TreeLSTM language encoder (Table 1)
  - Direct pixel / GloVe interpolation (Section 4)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Visual Encoder  fv: image features → e_v
# ---------------------------------------------------------------------------

class BottomUpVisualEncoder(nn.Module):
    """
    Mirrors UpDn's visual branch.
    Input: pre-extracted bottom-up features (B, num_boxes, feat_dim)
           e.g. num_boxes=36, feat_dim=2048
    Output: attended visual embedding (B, embed_dim)
    """

    def __init__(self, feat_dim: int = 2048, embed_dim: int = 512,
                 num_boxes: int = 36):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Self-attention over boxes
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.embed_dim = embed_dim

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: (B, num_boxes, feat_dim)  — bottom-up features
        returns: (B, embed_dim)
        """
        h = self.feat_proj(v)                    # (B, K, D)
        attn_w = self.attn(h).squeeze(-1)        # (B, K)
        attn_w = F.softmax(attn_w, dim=-1)       # (B, K)
        e_v = (attn_w.unsqueeze(-1) * h).sum(1)  # (B, D)
        return e_v


class SimpleVisualEncoder(nn.Module):
    """
    Lightweight CNN visual encoder for pixel inputs.
    Uses ResNet backbone + projection.
    """

    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        resnet = tvm.resnet50(
            weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool
        self.proj = nn.Linear(2048, embed_dim)
        self.embed_dim = embed_dim
        import torchvision.transforms as T
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: (B,3,H,W) in [0,1] → (B, embed_dim)"""
        v_n = torch.stack([self.norm(vi) for vi in v])
        feat = self.backbone(v_n).flatten(1)   # (B, 2048)
        return self.proj(feat)


# ---------------------------------------------------------------------------
# Language Encoder  fq: question tokens → e_q
# ---------------------------------------------------------------------------

class LSTMQuestionEncoder(nn.Module):
    """
    Standard LSTM question encoder (Table 1 baseline).
    Optionally initialized with GloVe embeddings.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 hidden_dim: int = 512, num_layers: int = 1,
                 glove_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if glove_weights is not None:
            self.embedding.weight.data.copy_(glove_weights)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=False,
        )
        self.embed_dim = hidden_dim

    def forward(self, q: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        q:       (B, T) token ids
        lengths: (B,)   actual sequence lengths (optional)
        returns: (B, hidden_dim)
        """
        emb = self.embedding(q)           # (B, T, 300)
        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            _, (h, _) = self.lstm(packed)
        else:
            _, (h, _) = self.lstm(emb)
        return h[-1]                      # (B, hidden_dim)


class GloVeEmbeddingEncoder(nn.Module):
    """
    Simple mean-pooled GloVe encoder.
    Faster than LSTM, used in ablations.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 glove_weights: Optional[torch.Tensor] = None,
                 output_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if glove_weights is not None:
            self.embedding.weight.data.copy_(glove_weights)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.embed_dim = output_dim

    def forward(self, q: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(q)                         # (B, T, 300)
        if lengths is not None:
            mask = (torch.arange(q.shape[1], device=q.device)
                    .unsqueeze(0) < lengths.unsqueeze(1)).float()
            emb = (emb * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            emb = emb.mean(1)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Fusion  h(e_v, e_q)  →  joint representation z
# ---------------------------------------------------------------------------

class MultiModalFusion(nn.Module):
    """
    Element-wise product fusion (UpDn style) + MLP.
    z = MLP( LayerNorm(e_v) ⊙ LayerNorm(e_q) )
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 1024,
                 dropout: float = 0.3):
        super().__init__()
        self.ln_v = nn.LayerNorm(embed_dim)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, e_v: torch.Tensor, e_q: torch.Tensor) -> torch.Tensor:
        """
        e_v, e_q: (B, embed_dim)
        returns z: (B, hidden_dim)
        """
        z = self.ln_v(e_v) * self.ln_q(e_q)  # element-wise product
        return self.mlp(z)


class ConcatFusion(nn.Module):
    """Concatenation fusion — simpler alternative."""

    def __init__(self, v_dim: int, q_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(v_dim + q_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, e_v: torch.Tensor, e_q: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([e_v, e_q], dim=-1))


# ---------------------------------------------------------------------------
# Full VQA Model
# ---------------------------------------------------------------------------

class VQAModel(nn.Module):
    """
    Full VQA model:
      fv → e_v
      fq → e_q
      h(e_v, e_q) → z
      classifier(z) → answer logits

    Exposes fuse_and_classify(e_v, e_q) separately so the SCM
    can inject counterfactual embeddings directly.
    """

    def __init__(
        self,
        visual_encoder:   nn.Module,
        question_encoder: nn.Module,
        fusion:           nn.Module,
        num_answers:      int = 3129,   # VQA-CP / VQA v2 default
        dropout:          float = 0.3,
    ):
        super().__init__()
        self.visual_encoder   = visual_encoder
        self.question_encoder = question_encoder
        self.fusion           = fusion

        self.classifier = nn.Sequential(
            nn.Linear(fusion.output_dim, fusion.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion.output_dim // 2, num_answers),
        )

    def encode_visual(self, v) -> torch.Tensor:
        return self.visual_encoder(v)

    def encode_question(self, q, lengths=None) -> torch.Tensor:
        return self.question_encoder(q, lengths)

    def fuse_and_classify(
        self, e_v: torch.Tensor, e_q: torch.Tensor
    ) -> torch.Tensor:
        """Used by SCM to inject counterfactual embeddings."""
        z = self.fusion(e_v, e_q)
        return self.classifier(z)

    def forward(
        self,
        v,                    # images or bottom-up features
        q: torch.Tensor,      # question token ids  (B, T)
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard forward pass. Returns answer logits (B, num_answers)."""
        e_v = self.encode_visual(v)
        e_q = self.encode_question(q, lengths)
        return self.fuse_and_classify(e_v, e_q)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def build_updn_model(
    vocab_size:    int,
    num_answers:   int,
    feat_dim:      int = 2048,
    embed_dim:     int = 512,
    glove_weights: Optional[torch.Tensor] = None,
) -> VQAModel:
    """Build UpDn-style model (Table 3: UpDn+C baseline)."""
    v_enc = BottomUpVisualEncoder(feat_dim=feat_dim, embed_dim=embed_dim)
    q_enc = LSTMQuestionEncoder(vocab_size=vocab_size, embed_dim=300,
                                hidden_dim=embed_dim, glove_weights=glove_weights)
    fusion = MultiModalFusion(embed_dim=embed_dim, hidden_dim=embed_dim * 2)
    return VQAModel(v_enc, q_enc, fusion, num_answers=num_answers)


def build_simple_model(
    vocab_size:  int,
    num_answers: int,
    embed_dim:   int = 512,
) -> VQAModel:
    """Build lightweight model for CIFAR-10/SST-style experiments."""
    v_enc  = SimpleVisualEncoder(embed_dim=embed_dim)
    q_enc  = LSTMQuestionEncoder(vocab_size=vocab_size, hidden_dim=embed_dim)
    fusion = MultiModalFusion(embed_dim=embed_dim, hidden_dim=embed_dim * 2)
    return VQAModel(v_enc, q_enc, fusion, num_answers=num_answers)


def build_image_classifier(
    num_classes: int,
    embed_dim:   int = 512,
) -> nn.Module:
    """
    Unimodal image classifier (Section 5.1 CIFAR experiments).
    Wraps SimpleVisualEncoder → classifier, no language branch.
    """
    class ImageOnlyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SimpleVisualEncoder(embed_dim=embed_dim)
            self.head    = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(embed_dim // 2, num_classes),
            )
            self.embed_dim = embed_dim

        def encode_visual(self, v):
            return self.encoder(v)

        def fuse_and_classify(self, e_v, e_q=None):
            return self.head(e_v)

        def forward(self, v, q=None, lengths=None):
            return self.head(self.encoder(v))

    return ImageOnlyModel()
