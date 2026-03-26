"""
Counterfactual Risk Minimization (CRM) Losses
==============================================
Implements:
  - Standard ERM loss (observational)
  - Eq. (4):  R̂_M(θ) — clipped counterfactual risk
  - ωᵢ(θ):   importance weight = min{M, p(a|q,v) / p(a|q̃,ṽ)}
  - Theorem 4: variance-penalized bound (used for monitoring)
  - Full training objective: alternating ERM + CRM

The combined loss is:
    L = L_ERM(factual) + L_CF(counterfactual, weighted by ω)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


# ---------------------------------------------------------------------------
# 1. Importance Weight  ωᵢ(θ) — Eq. (4)
# ---------------------------------------------------------------------------

def importance_weight(
    logits_factual:        torch.Tensor,   # (B, num_answers) on original data
    logits_counterfactual: torch.Tensor,   # (B, num_answers) on counterfactual
    answers:               torch.Tensor,   # (B,) true answer indices
    M:                     float = 5.0,    # clipping constant
) -> torch.Tensor:
    """
    ωᵢ(θ) = min{ M,  p(aᵢ|qᵢ,vᵢ,θ) / p_do(I)(aᵢ|q̃ᵢ,ṽᵢ,θ) }

    The ratio is the confidence on factual vs. confidence on counterfactual
    for the TRUE answer.  Clips at M to bound variance (Theorem 4).

    Returns ω: (B,)
    """
    # Gather probability of true answer under each distribution
    p_factual = F.softmax(logits_factual, dim=-1)           # (B, A)
    p_cf      = F.softmax(logits_counterfactual, dim=-1)    # (B, A)

    p_a_factual = p_factual.gather(1, answers.unsqueeze(1)).squeeze(1)   # (B,)
    p_a_cf      = p_cf.gather(1, answers.unsqueeze(1)).squeeze(1)        # (B,)

    # Avoid division by zero
    ratio = p_a_factual / (p_a_cf + 1e-8)
    omega = torch.clamp(ratio, max=M)
    return omega.detach()   # ω is treated as a constant (not differentiated through)


# ---------------------------------------------------------------------------
# 2. Counterfactual Risk  R̂_M(θ) — Eq. (4)
# ---------------------------------------------------------------------------

class CounterfactualRisk(nn.Module):
    """
    R̂_M(θ) = (1/n) Σᵢ  ℓ(fθ(qᵢ,vᵢ), aᵢ)  ×  ωᵢ(θ)

    where ωᵢ(θ) = min{M, p(a|q,v)/p_do(a|q̃,ṽ)}

    The loss is applied to the FACTUAL samples but weighted by ω,
    so the model is penalised more when it's over-confident on
    factuals that produce easy counterfactuals.
    """

    def __init__(self, M: float = 5.0, reduction: str = "mean"):
        super().__init__()
        self.M = M
        self.reduction = reduction

    def forward(
        self,
        logits_factual:        torch.Tensor,   # (B, A)
        logits_counterfactual: torch.Tensor,   # (B, A)
        answers:               torch.Tensor,   # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (loss, omega) where omega can be logged for diagnostics.
        """
        omega = importance_weight(
            logits_factual, logits_counterfactual, answers, self.M
        )

        # ℓ(fθ(qᵢ,vᵢ), aᵢ): cross-entropy on FACTUAL data
        per_sample_loss = F.cross_entropy(logits_factual, answers, reduction="none")

        # Weighted loss
        weighted = per_sample_loss * omega

        if self.reduction == "mean":
            loss = weighted.mean()
        elif self.reduction == "sum":
            loss = weighted.sum()
        else:
            loss = weighted

        return loss, omega


# ---------------------------------------------------------------------------
# 3. VQA Soft Loss (multi-label BCE, standard for VQA benchmarks)
# ---------------------------------------------------------------------------

class VQASoftLoss(nn.Module):
    """
    VQA evaluation uses soft labels (multiple human answers).
    Standard loss: BCE with soft targets.
    Used in VQA-CP and VQA v2 experiments (Section 5.2).
    """

    def forward(
        self,
        logits:       torch.Tensor,   # (B, num_answers)
        soft_targets: torch.Tensor,   # (B, num_answers) in [0, 1]
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, soft_targets, reduction="mean"
        )


# ---------------------------------------------------------------------------
# 4. Combined Training Objective
# ---------------------------------------------------------------------------

class CombinedCVLLoss(nn.Module):
    """
    Full training objective from Section 4.2:

    L = L_ERM + L_CF

    Where:
        L_ERM = cross-entropy on factual samples (standard ERM)
        L_CF  = counterfactual risk R̂_M (Eq. 4)

    In practice, the paper alternates between ERM steps and CF steps.
    This module supports both alternating (via the `cf_step` flag)
    and combined (summed) modes.

    Parameters
    ----------
    M:      clipping constant for ω (paper uses M=5 implicitly via clipping)
    cf_weight: weight on the counterfactual loss term (1.0 = equal weight)
    use_soft_labels: if True, uses VQA soft BCE loss; else cross-entropy
    """

    def __init__(
        self,
        M:               float = 5.0,
        cf_weight:       float = 1.0,
        use_soft_labels: bool  = False,
    ):
        super().__init__()
        self.cf_loss = CounterfactualRisk(M=M)
        self.cf_weight = cf_weight
        self.use_soft_labels = use_soft_labels
        self.soft_loss = VQASoftLoss() if use_soft_labels else None

    def erm_loss(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Standard ERM. targets: hard (B,) or soft (B, A)."""
        if self.use_soft_labels and targets.dim() == 2:
            return self.soft_loss(logits, targets)
        else:
            if targets.dim() == 2:
                targets = targets.argmax(1)
            return F.cross_entropy(logits, targets)

    def forward(
        self,
        logits_factual:        torch.Tensor,   # (B, A)
        logits_counterfactual: torch.Tensor,   # (B, A)
        answers:               torch.Tensor,   # (B,) hard labels
        soft_targets:          Optional[torch.Tensor] = None,  # (B, A) soft
        cf_step:               bool = False,   # if True, only CF loss (alternating mode)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys: total, erm, cf, omega_mean, omega_var
        """
        # ERM loss on factual data
        targets = soft_targets if (self.use_soft_labels and soft_targets is not None) else answers
        L_erm = self.erm_loss(logits_factual, targets)

        # Counterfactual risk (Eq. 4)
        L_cf, omega = self.cf_loss(logits_factual, logits_counterfactual, answers)

        if cf_step:
            # Alternating mode: only CF loss this step
            total = self.cf_weight * L_cf
        else:
            total = L_erm + self.cf_weight * L_cf

        return {
            "total":      total,
            "erm":        L_erm,
            "cf":         L_cf,
            "omega_mean": omega.mean(),
            "omega_var":  omega.var(),   # Theorem 4: monitor variance
        }


# ---------------------------------------------------------------------------
# 5. Theorem 4 — Variance Bound Monitor
# ---------------------------------------------------------------------------

class VarianceBoundMonitor:
    """
    Theorem 4:  R(θ) ≤ R̂_M(θ) + sqrt(18·V̂(u)·Qγ/n) + 15·M·Qγ/(n-1)

    where Qγ = log(10·ε/γ), V̂(u) = sample variance of uᵢ(θ).

    This is used for monitoring/logging — not for optimization.
    It tells us how tight the generalization bound is.
    """

    def __init__(self, M: float = 5.0, gamma: float = 0.05, epsilon: float = 1.0):
        self.M       = M
        self.gamma   = gamma
        self.epsilon = epsilon

    def compute_bound(
        self,
        empirical_risk:  float,
        u_values:        torch.Tensor,   # per-sample uᵢ(θ) = ℓ × ω
        n:               int,
    ) -> Dict[str, float]:
        """
        Returns the generalization bound components.
        u_values: (n,) tensor of per-sample weighted losses
        """
        u_bar  = u_values.mean().item()
        v_hat  = u_values.var().item()   # V̂(u)
        Q_gamma = math.log(10 * self.epsilon / self.gamma)

        if n <= 1:
            return {"bound": float("inf"), "v_hat": v_hat, "gap": float("inf")}

        term1 = math.sqrt(18 * v_hat * Q_gamma / n)
        term2 = 15 * self.M * Q_gamma / (n - 1)
        gap   = term1 + term2
        bound = empirical_risk + gap

        return {
            "empirical_risk": empirical_risk,
            "v_hat":          v_hat,
            "gap":            gap,
            "bound":          bound,
            "Q_gamma":        Q_gamma,
        }
