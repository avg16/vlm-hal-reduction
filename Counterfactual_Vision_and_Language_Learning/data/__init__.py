"""
Data Loaders
============
Supports:
  - VQA-CP v2  (Section 5.2, Table 3)
  - VQA v2     (Section 5.2, Table 4)
  - CIFAR-10 / CIFAR-100  (Section 5.1, Table 2)
  - SST (Stanford Sentiment Treebank)  (Section 5.1, Table 1)

Each dataset returns batches with a `distractors` field:
  randomly sampled OTHER samples from the dataset used to
  form the counterfactual pairs (q', v') in Eq. (2).
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as tvd
import torchvision.transforms as T
import numpy as np
import os
import json
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Batch dataclass (common interface)
# ---------------------------------------------------------------------------

@dataclass
class VQABatch:
    images:           torch.Tensor        # (B, 3, H, W) or (B, K, feat_dim)
    questions:        torch.Tensor        # (B, T)
    q_lengths:        torch.Tensor        # (B,)
    answers:          torch.Tensor        # (B,) hard label
    soft_targets:     Optional[torch.Tensor]  # (B, num_answers) soft
    # Distractors — randomly sampled OTHER samples for Eq. (2)
    images_prime:     torch.Tensor        # (B, ...)
    questions_prime:  torch.Tensor        # (B, T)
    q_lengths_prime:  torch.Tensor        # (B,)


# ---------------------------------------------------------------------------
# 1. CIFAR-10 / CIFAR-100  (unimodal vision)
# ---------------------------------------------------------------------------

class CIFARWithDistractors(Dataset):
    """
    CIFAR-10 or CIFAR-100 with random distractor images.
    Each item returns (image, label, distractor_image).
    """

    def __init__(self, root: str, num_classes: int = 10,
                 train: bool = True, img_size: int = 224, seed: int = 42):
        transform = T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip() if train else T.Lambda(lambda x: x),
            T.ToTensor(),
        ])
        cls = tvd.CIFAR10 if num_classes == 10 else tvd.CIFAR100
        self.dataset  = cls(root=root, train=train, transform=transform, download=True)
        self.rng      = np.random.RandomState(seed)
        self.n        = len(self.dataset)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Random distractor (different index)
        dist_idx  = self.rng.randint(0, self.n)
        dist_img, _ = self.dataset[dist_idx]
        return img, label, dist_img

    @staticmethod
    def collate_fn(batch):
        imgs, labels, dists = zip(*batch)
        return (
            torch.stack(imgs),
            torch.tensor(labels),
            torch.stack(dists),
        )


def get_cifar_loader(root: str = "./data", num_classes: int = 10,
                     train: bool = True, batch_size: int = 128,
                     img_size: int = 224, seed: int = 42,
                     n_samples: Optional[int] = None) -> DataLoader:
    ds = CIFARWithDistractors(root, num_classes, train, img_size, seed)
    if n_samples:
        ds = Subset(ds, list(range(min(n_samples, len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=2, collate_fn=CIFARWithDistractors.collate_fn,
                      pin_memory=True)


# ---------------------------------------------------------------------------
# 2. SST — Stanford Sentiment Treebank (unimodal language)
# ---------------------------------------------------------------------------

class SSTDataset(Dataset):
    """
    Simple SST binary sentiment dataset.
    Expects a TSV file: sentence \\t label (0/1).
    If not available, falls back to a tiny synthetic demo set.
    """

    MAX_LEN = 60

    def __init__(self, path: Optional[str], vocab: Optional[Dict[str, int]] = None,
                 split: str = "train", seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.samples: List[Tuple[List[int], int]] = []

        if path and os.path.exists(path):
            self._load_tsv(path, vocab)
        else:
            print(f"  [SST] File not found: {path}. Using synthetic demo data.")
            self._load_synthetic()

    def _tokenize(self, sentence: str, vocab: Dict[str, int]) -> List[int]:
        unk = vocab.get("<unk>", 1)
        return [vocab.get(w.lower(), unk) for w in sentence.split()][:self.MAX_LEN]

    def _load_tsv(self, path: str, vocab: Dict[str, int]):
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2: continue
                tokens = self._tokenize(parts[0], vocab)
                label  = int(parts[1])
                if tokens:
                    self.samples.append((tokens, label))

    def _load_synthetic(self):
        # 200 random samples for smoke-testing
        for i in range(200):
            length = np.random.randint(5, 20)
            tokens = list(np.random.randint(2, 1000, length))
            label  = np.random.randint(0, 2)
            self.samples.append((tokens, label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        tokens, label = self.samples[idx]
        dist_idx = self.rng.randint(0, len(self.samples))
        dist_tokens, _ = self.samples[dist_idx]
        return tokens, label, dist_tokens

    @staticmethod
    def collate_fn(batch):
        seqs, labels, dists = zip(*batch)

        def pad(sequences):
            max_len = max(len(s) for s in sequences)
            padded  = [s + [0] * (max_len - len(s)) for s in sequences]
            lengths = torch.tensor([len(s) for s in sequences])
            return torch.tensor(padded, dtype=torch.long), lengths

        q_pad, q_len   = pad(seqs)
        qp_pad, qp_len = pad(dists)
        return q_pad, torch.tensor(labels), q_len, qp_pad, qp_len


def get_sst_loader(path: Optional[str] = None, vocab: Optional[Dict] = None,
                   batch_size: int = 32, seed: int = 42) -> DataLoader:
    ds = SSTDataset(path, vocab, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=0, collate_fn=SSTDataset.collate_fn)


# ---------------------------------------------------------------------------
# 3. VQA-CP / VQA v2  (multimodal)
# ---------------------------------------------------------------------------

class VQADataset(Dataset):
    """
    Generic VQA dataset loader.

    Expects:
        questions_json : COCO-format questions file
        annotations_json: COCO-format annotations file
        image_dir      : directory of images (or pre-extracted features .npy)
        vocab          : answer vocabulary {answer_str: idx}

    If files don't exist, falls back to synthetic demo data.
    """

    MAX_Q_LEN = 20

    def __init__(
        self,
        questions_json:   Optional[str],
        annotations_json: Optional[str],
        image_dir:        Optional[str],
        answer_vocab:     Optional[Dict[str, int]] = None,
        word_vocab:       Optional[Dict[str, int]] = None,
        img_size:         int = 224,
        num_answers:      int = 3129,
        seed:             int = 42,
    ):
        self.rng        = np.random.RandomState(seed)
        self.img_size   = img_size
        self.num_answers = num_answers
        self.answer_vocab = answer_vocab or {}
        self.word_vocab   = word_vocab or {}
        self.image_dir  = image_dir
        self.transform  = T.Compose([T.Resize(256), T.CenterCrop(img_size), T.ToTensor()])
        self.samples: List[Dict] = []

        if questions_json and os.path.exists(questions_json):
            self._load_real(questions_json, annotations_json)
        else:
            print(f"  [VQA] Data not found. Using synthetic demo data.")
            self._load_synthetic()

    def _load_real(self, q_file: str, a_file: Optional[str]):
        with open(q_file) as f:
            q_data = json.load(f)
        a_data = {}
        if a_file and os.path.exists(a_file):
            with open(a_file) as f:
                raw = json.load(f)
            for ann in raw.get("annotations", []):
                a_data[ann["question_id"]] = ann

        for q in q_data.get("questions", []):
            qid    = q["question_id"]
            tokens = self._tok(q["question"])
            img_id = q["image_id"]
            ann    = a_data.get(qid, {})
            answers = ann.get("answers", [])
            # Build soft target
            soft = torch.zeros(self.num_answers)
            for a in answers:
                idx = self.answer_vocab.get(a.get("answer", ""), -1)
                if idx >= 0:
                    soft[idx] += min(1.0, len([x for x in answers
                                               if x.get("answer") == a.get("answer")]) / 3.0)
            hard = soft.argmax().item() if soft.sum() > 0 else 0
            self.samples.append({
                "img_id": img_id, "tokens": tokens,
                "hard": hard, "soft": soft,
            })

    def _load_synthetic(self):
        for i in range(500):
            tokens = list(np.random.randint(2, 1000, np.random.randint(3, 15)))
            self.samples.append({
                "img_id": i % 50,
                "tokens": tokens,
                "hard":   np.random.randint(0, min(self.num_answers, 10)),
                "soft":   None,
            })

    def _tok(self, q: str) -> List[int]:
        unk = self.word_vocab.get("<unk>", 1)
        return [self.word_vocab.get(w.lower(), unk)
                for w in q.rstrip("?").split()][:self.MAX_Q_LEN]

    def _load_image(self, img_id: int) -> torch.Tensor:
        if self.image_dir:
            # Try common COCO filename formats
            for fmt in [f"COCO_train2014_{img_id:012d}.jpg",
                        f"COCO_val2014_{img_id:012d}.jpg",
                        f"{img_id}.jpg"]:
                path = os.path.join(self.image_dir, fmt)
                if os.path.exists(path):
                    from PIL import Image
                    return self.transform(Image.open(path).convert("RGB"))
        # Fallback: random image tensor
        return torch.rand(3, self.img_size, self.img_size)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img      = self._load_image(s["img_id"])
        tokens   = s["tokens"]
        hard     = s["hard"]
        soft     = s["soft"] if s["soft"] is not None else torch.zeros(self.num_answers)

        # Random distractor
        dist_idx = self.rng.randint(0, len(self.samples))
        ds = self.samples[dist_idx]
        img_p    = self._load_image(ds["img_id"])
        tokens_p = ds["tokens"]
        return img, tokens, hard, soft, img_p, tokens_p

    @staticmethod
    def collate_fn(batch):
        imgs, qs, hards, softs, imgs_p, qs_p = zip(*batch)

        def pad_seqs(seqs):
            ml = max(len(s) for s in seqs)
            padded  = [s + [0] * (ml - len(s)) for s in seqs]
            lengths = torch.tensor([len(s) for s in seqs])
            return torch.tensor(padded, dtype=torch.long), lengths

        q_tok, q_len   = pad_seqs(qs)
        qp_tok, qp_len = pad_seqs(qs_p)

        return VQABatch(
            images          = torch.stack(imgs),
            questions       = q_tok,
            q_lengths       = q_len,
            answers         = torch.tensor(hards, dtype=torch.long),
            soft_targets    = torch.stack(softs),
            images_prime    = torch.stack(imgs_p),
            questions_prime = qp_tok,
            q_lengths_prime = qp_len,
        )


def get_vqa_loader(
    questions_json:   Optional[str] = None,
    annotations_json: Optional[str] = None,
    image_dir:        Optional[str] = None,
    answer_vocab:     Optional[Dict] = None,
    word_vocab:       Optional[Dict] = None,
    num_answers:      int = 3129,
    batch_size:       int = 64,
    shuffle:          bool = True,
    seed:             int = 42,
) -> DataLoader:
    ds = VQADataset(
        questions_json, annotations_json, image_dir,
        answer_vocab, word_vocab, num_answers=num_answers, seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=VQADataset.collate_fn,
                      pin_memory=False)
