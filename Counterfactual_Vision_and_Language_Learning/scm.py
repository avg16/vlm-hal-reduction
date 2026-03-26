"""
Structural Causal Model (SCM) for Counterfactual Vision and Language
=====================================================================
Implements:
  - Definition 1: SCM with exogenous variables u_v, u_q  (Section 2.2)
  - Definition 3: Counterfactual distribution p(u|D)      (Section 4)
  - Eq. (2):  Generative process for v~, q~               (Section 4)
  - Eq. (3):  Counterfactual optimization (gradient ascent on u) (Section 4.1)
  - Beta posterior update: α, β parameters                (Section 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from typing import Tuple, Optional


class ExogenousVariable(nn.Module):
    """
    Represents one exogenous variable u ∈ [0, 1) with a Beta prior.

    Paper: u^v ~ Beta(α₀, β₀), u^q ~ Beta(α₀, β₀)
    
    The posterior is also Beta (conjugate):
        α = α₀ + Σ 𝟙[aᵢ = argmax p(aᵢ|qᵢ,vᵢ)]   (correct predictions)
        β = β₀ + Σ 𝟙[aᵢ ≠ argmax p(aᵢ|qᵢ,vᵢ)]   (incorrect predictions)

    During forward pass, u is a learnable scalar that is projected to [0,1).
    For the counterfactual optimizer (Eq.3), u becomes a leaf tensor.
    """

    def __init__(self, alpha0: float = 0.1, beta0: float = 0.1):
        super().__init__()
        self.alpha0 = alpha0
        self.beta0  = beta0
        # Learnable unconstrained parameter; sigmoid → (0,1)
        self.u_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def u(self) -> torch.Tensor:
        """u ∈ (0, 1)"""
        return torch.sigmoid(self.u_raw)

    def sample_posterior(self, alpha: float, beta: float) -> torch.Tensor:
        """Sample from Beta posterior Beta(α, β)."""
        return Beta(torch.tensor(alpha), torch.tensor(beta)).sample()

    def update_posterior(
        self,
        correct_count: int,
        incorrect_count: int,
    ) -> Tuple[float, float]:
        """
        Conjugate Beta update.
        Returns updated (α, β).
        """
        alpha = self.alpha0 + correct_count
        beta  = self.beta0  + incorrect_count
        return alpha, beta


class CounterfactualGenerator(nn.Module):
    """
    Implements the generative process from Eq. (2):

        v  ~  p(v | u^v):
            if u^v ≥ 1-ε  →  v  (identity)
            else           →  u^v · v  ⊕  (1 - u^v) · v'    (interpolation)

    And similarly for q.

    The interpolation ⊕ is linear (MixUp-style) in embedding space.
    v' / q' are randomly sampled from the dataset (distractor samples).

    Parameters
    ----------
    epsilon : float
        Threshold below which interpolation happens.  Default 0.5.
        ε→0: more interpolated samples.  ε→1: independent of prior.
    """

    def __init__(self, epsilon: float = 0.5):
        super().__init__()
        self.epsilon = epsilon

    def generate(
        self,
        e: torch.Tensor,          # original embedding  (B, D)
        e_prime: torch.Tensor,    # distractor embedding (B, D)
        u: torch.Tensor,          # exogenous variable   (B,) or scalar
    ) -> torch.Tensor:
        """
        Generate counterfactual embedding ẽ following Eq. (2).

        Returns ẽ: (B, D)
        """
        if u.dim() == 0:
            u = u.expand(e.shape[0])

        # Mask: where u ≥ 1-ε, keep original; otherwise interpolate
        keep_mask = (u >= (1.0 - self.epsilon)).float().unsqueeze(1)  # (B,1)

        # Linear interpolation: u·e + (1-u)·e'
        e_interp = u.unsqueeze(1) * e + (1.0 - u.unsqueeze(1)) * e_prime

        # Combine: keep original where u ≥ 1-ε, else interpolated
        e_cf = keep_mask * e + (1.0 - keep_mask) * e_interp
        return e_cf

    def forward(
        self,
        e_v: torch.Tensor,        # visual embedding      (B, D_v)
        e_q: torch.Tensor,        # question embedding    (B, D_q)
        e_v_prime: torch.Tensor,  # distractor visual     (B, D_v)
        e_q_prime: torch.Tensor,  # distractor question   (B, D_q)
        u_v: torch.Tensor,        # visual exogenous var  scalar or (B,)
        u_q: torch.Tensor,        # question exogenous var
        intervene_v: bool = True,
        intervene_q: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (ẽ_v, ẽ_q): counterfactual embeddings.
        If intervene_v=False, visual embedding is unchanged (and vice versa).
        """
        e_v_cf = self.generate(e_v, e_v_prime, u_v) if intervene_v else e_v
        e_q_cf = self.generate(e_q, e_q_prime, u_q) if intervene_q else e_q
        return e_v_cf, e_q_cf


class CounterfactualOptimizer:
    """
    Implements Eq. (3): finds u* that maximizes ||u||² - λ·log p(a|q̃,ṽ)
    i.e., maximum-norm u that makes the model predict the WRONG answer.

    This is the "intervention" step — finding the hardest counterfactual
    (minimum edit = highest u) that still changes the prediction.

    Algorithm:
        1. Initialize u as free variable (projected gradient to [0,1))
        2. Gradient ASCENT on: ||u||² - λ·log p(correct_answer | counterfactual)
        3. The constraint a~ ≠ a is enforced via the loss
    """

    def __init__(
        self,
        cf_generator: CounterfactualGenerator,
        vqa_model: nn.Module,       # must accept (e_v, e_q) → logits
        lambda_cf: float = 1.0,
        lr: float = 0.01,
        n_steps: int = 10,
        epsilon: float = 0.5,
    ):
        self.cf_gen    = cf_generator
        self.model     = vqa_model
        self.lambda_cf = lambda_cf
        self.lr        = lr
        self.n_steps   = n_steps
        self.epsilon   = epsilon

    def optimize(
        self,
        e_v:       torch.Tensor,   # (B, D_v)
        e_q:       torch.Tensor,   # (B, D_q)
        e_v_prime: torch.Tensor,   # (B, D_v)  distractor
        e_q_prime: torch.Tensor,   # (B, D_q)  distractor
        answers:   torch.Tensor,   # (B,)      true answer indices
        intervene_v: bool = True,
        intervene_q: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            u_v_star: (B,) optimized visual exogenous variable
            u_q_star: (B,) optimized question exogenous variable
            e_v_cf:   (B, D_v) counterfactual visual embedding
            e_q_cf:   (B, D_q) counterfactual question embedding
        """
        B = e_v.shape[0]
        device = e_v.device

        # Initialize u in unconstrained space; sigmoid → (0,1)
        u_v_raw = torch.zeros(B, device=device, requires_grad=True)
        u_q_raw = torch.zeros(B, device=device, requires_grad=True)

        opt_params = []
        if intervene_v: opt_params.append(u_v_raw)
        if intervene_q: opt_params.append(u_q_raw)

        if not opt_params:
            # Nothing to optimize
            return (torch.ones(B, device=device),
                    torch.ones(B, device=device),
                    e_v, e_q)

        optimizer = torch.optim.Adam(opt_params, lr=self.lr)

        for _ in range(self.n_steps):
            optimizer.zero_grad()

            u_v = torch.sigmoid(u_v_raw)   # project to (0,1)
            u_q = torch.sigmoid(u_q_raw)

            # Generate counterfactual embeddings
            e_v_cf, e_q_cf = self.cf_gen(
                e_v, e_q, e_v_prime, e_q_prime,
                u_v, u_q, intervene_v, intervene_q,
            )

            # Forward through VQA fusion + classifier
            logits = self.model.fuse_and_classify(e_v_cf, e_q_cf)  # (B, num_answers)

            # Eq. (3): max ||u||² - λ·log p(correct_answer | counterfactual)
            # = min  -||u||² + λ·log p(a | cf)
            norm_term  = -(u_v.pow(2).sum() + u_q.pow(2).sum()) / B
            log_p_correct = -F.cross_entropy(logits, answers)   # log p(a|cf)
            loss = norm_term + self.lambda_cf * log_p_correct

            loss.backward(retain_graph=False)
            optimizer.step()

        with torch.no_grad():
            u_v_star = torch.sigmoid(u_v_raw).detach()
            u_q_star = torch.sigmoid(u_q_raw).detach()
            e_v_cf, e_q_cf = self.cf_gen(
                e_v, e_q, e_v_prime, e_q_prime,
                u_v_star, u_q_star, intervene_v, intervene_q,
            )

        return u_v_star, u_q_star, e_v_cf, e_q_cf
