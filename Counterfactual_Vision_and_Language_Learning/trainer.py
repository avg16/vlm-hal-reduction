"""
Trainer
=======
Implements the full training loop from Section 4.2 and Section 5.

Three experiment modes:
  A) Unimodal Vision     — CIFAR-10/100  (Table 2)
  B) Unimodal Language   — SST           (Table 1)
  C) Multimodal VQA      — VQA-CP/v2     (Table 3/4)

Core training procedure (Fig. 1):
  1. Infer posterior on exogenous variables from observations  (ERM step)
  2. Perform intervention I via CounterfactualOptimizer  →  q̃, ṽ
  3. Compute counterfactual risk R̂_M and update θ  (CF step)

Alternates between conventional ERM (ω=1) and CRM (Eq. 4).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from typing import Optional, Dict, List
import os
import time

from models.scm import (
    ExogenousVariable,
    CounterfactualGenerator,
    CounterfactualOptimizer,
)
from losses import CombinedCVLLoss, VarianceBoundMonitor


# ---------------------------------------------------------------------------
# Base Trainer
# ---------------------------------------------------------------------------

class BaseTrainer:
    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "cf_loss":    [], "erm_loss":  [],
            "omega_mean": [], "omega_var": [],
        }
        self.bound_monitor = VarianceBoundMonitor(
            M=cfg.get("M", 5.0), gamma=0.05
        )

    def save(self, model: nn.Module, path: str, extra: dict = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {"model": model.state_dict(), "history": self.history}
        if extra: state.update(extra)
        torch.save(state, path)
        print(f"  [ckpt] Saved → {path}")

    def _acc(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        return (logits.argmax(1) == labels).float().mean().item()

    def _log(self, ep: int, total_ep: int, metrics: dict):
        parts = [f"Ep {ep+1:03d}/{total_ep}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
        print("  " + " | ".join(parts))


# ---------------------------------------------------------------------------
# A) Unimodal Vision Trainer (CIFAR)
# ---------------------------------------------------------------------------

class CIFARTrainer(BaseTrainer):
    """
    Implements Section 5.1 CIFAR experiments (Table 2).

    Modes:
        baseline   — standard ERM only
        +P         — posterior sampling (Ours+P)
        +C         — counterfactual optimization (Ours+C)  ← main contribution

    Exogenous variable u^v is optimized via Eq. (3).
    Interpolation is in image (pixel) space directly.
    """

    def __init__(self, model: nn.Module, cfg: dict, device: str = "cpu"):
        super().__init__(cfg, device)
        self.model = model.to(device)
        self.u_v   = ExogenousVariable(alpha0=cfg.get("alpha0", 0.1),
                                       beta0=cfg.get("beta0", 0.1))
        self.cf_gen = CounterfactualGenerator(epsilon=cfg.get("epsilon", 0.5))
        self.loss_fn = CombinedCVLLoss(
            M=cfg.get("M", 5.0),
            cf_weight=cfg.get("cf_weight", 1.0),
        )
        # Track posterior parameters
        self.alpha, self.beta = cfg.get("alpha0", 0.1), cfg.get("beta0", 0.1)

    def _make_cf_images(
        self,
        imgs: torch.Tensor,       # (B,3,H,W) factual
        imgs_prime: torch.Tensor, # (B,3,H,W) distractor
        labels: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """
        Generate counterfactual images following Eq. (2).
        mode: "posterior" samples u from Beta posterior
              "optimized" optimizes u via Eq. (3)
        """
        B = imgs.shape[0]

        if mode == "posterior":
            # Sample u from current Beta posterior
            u_scalar = self.u_v.sample_posterior(self.alpha, self.beta)
            u = u_scalar.expand(B).to(self.device)
        elif mode == "optimized":
            # Optimize u via gradient ascent (Eq. 3) in image space
            u = self._optimize_u_image(imgs, imgs_prime, labels)
        else:
            u = torch.ones(B, device=self.device)  # identity (baseline)

        # Generate counterfactual: u·img + (1-u)·img'  (interpolation in pixel space)
        u_exp = u.view(B, 1, 1, 1)
        imgs_cf = (u_exp * imgs + (1.0 - u_exp) * imgs_prime).clamp(0, 1)
        return imgs_cf, u

    def _optimize_u_image(
        self,
        imgs:       torch.Tensor,
        imgs_prime: torch.Tensor,
        labels:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Eq. (3) in image space:
        max_u  ||u||² - λ·log p(correct_answer | u·img + (1-u)·img')
        """
        B = imgs.shape[0]
        u_raw = torch.zeros(B, device=self.device, requires_grad=True)
        opt   = torch.optim.Adam([u_raw], lr=self.cfg.get("cf_lr", 0.01))

        for _ in range(self.cfg.get("cf_steps", 10)):
            opt.zero_grad()
            u    = torch.sigmoid(u_raw)
            u_e  = u.view(B, 1, 1, 1)
            imgs_cf = (u_e * imgs + (1.0 - u_e) * imgs_prime).clamp(0, 1)

            logits   = self.model(imgs_cf)
            norm_sq  = -(u.pow(2).mean())
            log_p    = -nn.functional.cross_entropy(logits, labels)
            loss     = norm_sq + self.cfg.get("lambda_cf", 1.0) * log_p
            loss.backward()
            opt.step()

        return torch.sigmoid(u_raw).detach()

    def _update_posterior(self, logits: torch.Tensor, labels: torch.Tensor):
        """Update Beta posterior based on current model's correctness."""
        preds   = logits.argmax(1)
        correct = (preds == labels).sum().item()
        wrong   = labels.shape[0] - correct
        self.alpha, self.beta = self.u_v.update_posterior(int(correct), int(wrong))

    def train_epoch(
        self,
        loader,
        optimizer: torch.optim.Optimizer,
        mode: str = "+C",         # "baseline", "+P", "+C"
        cf_alternate: bool = True, # alternate ERM and CF steps (paper default)
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = erm_t = cf_t = acc_t = omega_t = var_t = 0.0
        n_batches  = 0

        for batch in loader:
            imgs, labels, imgs_prime = batch
            imgs       = imgs.to(self.device)
            labels     = labels.to(self.device)
            imgs_prime = imgs_prime.to(self.device)

            optimizer.zero_grad()

            # --- Factual forward pass ---
            logits_factual = self.model(imgs)

            if mode == "baseline":
                loss = nn.functional.cross_entropy(logits_factual, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item(); n_batches += 1
                acc_t += self._acc(logits_factual, labels)
                continue

            # --- Update posterior ---
            with torch.no_grad():
                self._update_posterior(logits_factual, labels)

            # --- Generate counterfactuals ---
            cf_mode_map = {"+P": "posterior", "+C": "optimized"}
            imgs_cf, u  = self._make_cf_images(imgs, imgs_prime, labels,
                                                cf_mode_map.get(mode, "posterior"))

            # --- Counterfactual forward pass ---
            logits_cf = self.model(imgs_cf)

            # --- Combined loss ---
            losses = self.loss_fn(
                logits_factual=logits_factual,
                logits_counterfactual=logits_cf,
                answers=labels,
                cf_step=(cf_alternate and n_batches % 2 == 1),
            )

            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += losses["total"].item()
            erm_t      += losses["erm"].item()
            cf_t       += losses["cf"].item()
            omega_t    += losses["omega_mean"].item()
            var_t      += losses["omega_var"].item()
            acc_t      += self._acc(logits_factual, labels)
            n_batches  += 1

        nb = max(n_batches, 1)
        return {
            "loss": total_loss/nb, "erm": erm_t/nb,
            "cf":   cf_t/nb,       "acc": acc_t/nb,
            "omega_mean": omega_t/nb, "omega_var": var_t/nb,
        }

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        correct = total = 0
        for batch in loader:
            imgs, labels, _ = batch
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits  = self.model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.shape[0]
        return {"acc": correct / max(total, 1)}

    def train(self, train_loader, val_loader, epochs: int = 100,
              mode: str = "+C", output_dir: str = "./results/cifar"):
        os.makedirs(output_dir, exist_ok=True)

        optimizer = optim.SGD(self.model.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

        best_acc = 0.0
        print(f"\n  CIFAR Training  |  mode={mode}  |  epochs={epochs}")

        for ep in range(epochs):
            tr = self.train_epoch(train_loader, optimizer, mode=mode)
            vl = self.evaluate(val_loader)

            scheduler.step()

            self.history["train_loss"].append(tr["loss"])
            self.history["train_acc"].append(tr["acc"])
            self.history["val_acc"].append(vl["acc"])
            self.history["cf_loss"].append(tr.get("cf", 0))
            self.history["omega_mean"].append(tr.get("omega_mean", 0))
            self.history["omega_var"].append(tr.get("omega_var", 0))

            if vl["acc"] > best_acc:
                best_acc = vl["acc"]
                self.save(self.model, os.path.join(output_dir, f"best_{mode}.pt"),
                          {"epoch": ep, "best_acc": best_acc})

            if ep % 10 == 0 or ep == epochs - 1:
                self._log(ep, epochs, {**tr, "val_acc": vl["acc"]})

        print(f"\n  Best Val Acc: {best_acc:.4f}")
        return best_acc


# ---------------------------------------------------------------------------
# B) Unimodal Language Trainer (SST)
# ---------------------------------------------------------------------------

class SSTTrainer(BaseTrainer):
    """
    Table 1: LSTM / TreeLSTM + counterfactuals on SST.
    Language-only — only u^q is inferred (no image branch).
    """

    def __init__(self, model: nn.Module, cfg: dict, device: str = "cpu"):
        super().__init__(cfg, device)
        self.model  = model.to(device)
        self.u_q    = ExogenousVariable(alpha0=cfg.get("alpha0", 0.1),
                                        beta0=cfg.get("beta0", 0.1))
        self.cf_gen = CounterfactualGenerator(epsilon=cfg.get("epsilon", 0.5))
        self.loss_fn = CombinedCVLLoss(M=cfg.get("M", 5.0))
        self.alpha, self.beta = 0.1, 0.1

    def _make_cf_text(self, q, q_prime, lengths, lengths_prime, labels, mode):
        """Interpolate in embedding space (not token space)."""
        B = q.shape[0]
        # Get embeddings
        e_q       = self.model.encoder(q, lengths)          # (B, D)
        e_q_prime = self.model.encoder(q_prime, lengths_prime)

        if mode == "posterior":
            u_scalar = self.u_q.sample_posterior(self.alpha, self.beta)
            u = u_scalar.expand(B).to(self.device)
        else:
            # Optimize u in embedding space
            u = self._optimize_u_text(e_q, e_q_prime, labels)

        e_q_cf = self.cf_gen.generate(e_q, e_q_prime, u)
        return e_q_cf, u

    def _optimize_u_text(self, e_q, e_q_prime, labels):
        B = e_q.shape[0]
        u_raw = torch.zeros(B, device=self.device, requires_grad=True)
        opt   = torch.optim.Adam([u_raw], lr=self.cfg.get("cf_lr", 0.01))
        for _ in range(self.cfg.get("cf_steps", 10)):
            opt.zero_grad()
            u     = torch.sigmoid(u_raw)
            e_cf  = self.cf_gen.generate(e_q.detach(), e_q_prime.detach(), u)
            logits = self.model.classify(e_cf)
            loss  = -(u.pow(2).mean()) + self.cfg.get("lambda_cf", 1.0) * (
                -nn.functional.cross_entropy(logits, labels)
            )
            loss.backward(); opt.step()
        return torch.sigmoid(u_raw).detach()

    def train_epoch(self, loader, optimizer, mode="+C") -> Dict[str, float]:
        self.model.train()
        total = erm_t = cf_t = acc_t = 0.0
        nb = 0

        for q, labels, lengths, q_prime, lengths_prime in loader:
            q, labels = q.to(self.device), labels.to(self.device)
            lengths   = lengths.to(self.device)
            q_prime, lengths_prime = q_prime.to(self.device), lengths_prime.to(self.device)

            optimizer.zero_grad()
            logits_factual = self.model(q, lengths)

            if mode == "baseline":
                loss = nn.functional.cross_entropy(logits_factual, labels)
                loss.backward(); optimizer.step()
                total += loss.item(); acc_t += self._acc(logits_factual, labels)
                nb += 1; continue

            with torch.no_grad():
                preds   = logits_factual.argmax(1)
                correct = (preds == labels).sum().item()
                self.alpha, self.beta = self.u_q.update_posterior(
                    correct, labels.shape[0] - correct)

            e_q_cf, u = self._make_cf_text(
                q, q_prime, lengths, lengths_prime, labels,
                "posterior" if mode == "+P" else "optimized"
            )
            logits_cf = self.model.classify(e_q_cf)

            losses = self.loss_fn(logits_factual, logits_cf, labels,
                                  cf_step=(nb % 2 == 1))
            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total += losses["total"].item()
            erm_t += losses["erm"].item()
            cf_t  += losses["cf"].item()
            acc_t += self._acc(logits_factual, labels)
            nb += 1

        nb = max(nb, 1)
        return {"loss": total/nb, "erm": erm_t/nb, "cf": cf_t/nb, "acc": acc_t/nb}

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        correct = total = 0
        for q, labels, lengths, *_ in loader:
            q, labels, lengths = (q.to(self.device), labels.to(self.device),
                                  lengths.to(self.device))
            preds    = self.model(q, lengths).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.shape[0]
        return {"acc": correct / max(total, 1)}

    def train(self, train_loader, val_loader, epochs=10,
              mode="+C", output_dir="./results/sst"):
        os.makedirs(output_dir, exist_ok=True)
        opt  = optim.Adam(self.model.parameters(), lr=2e-4)
        best = 0.0
        print(f"\n  SST Training  |  mode={mode}  |  epochs={epochs}")
        for ep in range(epochs):
            tr = self.train_epoch(train_loader, opt, mode=mode)
            vl = self.evaluate(val_loader)
            if vl["acc"] > best:
                best = vl["acc"]
                self.save(self.model, os.path.join(output_dir, f"best_{mode}.pt"))
            if ep % 2 == 0 or ep == epochs - 1:
                self._log(ep, epochs, {**tr, "val_acc": vl["acc"]})
        print(f"  Best SST Acc: {best:.4f}")
        return best


# ---------------------------------------------------------------------------
# C) Multimodal VQA Trainer (VQA-CP / VQA v2)
# ---------------------------------------------------------------------------

class VQATrainer(BaseTrainer):
    """
    Table 3/4: UpDn+C — multimodal counterfactual training.

    Both u^v and u^q are inferred simultaneously.
    Counterfactual optimizer works on embedding space (not pixels).
    The paper interpolates word/visual embeddings (Section 5.2).
    """

    def __init__(self, model: nn.Module, cfg: dict, device: str = "cpu"):
        super().__init__(cfg, device)
        self.model   = model.to(device)
        self.u_v     = ExogenousVariable(alpha0=cfg.get("alpha0", 0.1),
                                         beta0=cfg.get("beta0", 0.1))
        self.u_q     = ExogenousVariable(alpha0=cfg.get("alpha0", 0.1),
                                         beta0=cfg.get("beta0", 0.1))
        self.cf_gen  = CounterfactualGenerator(epsilon=cfg.get("epsilon", 0.5))
        self.cf_opt  = CounterfactualOptimizer(
            cf_generator=self.cf_gen,
            vqa_model=model,
            lambda_cf=cfg.get("lambda_cf", 1.0),
            lr=cfg.get("cf_lr", 0.01),
            n_steps=cfg.get("cf_steps", 10),
        )
        self.loss_fn = CombinedCVLLoss(
            M=cfg.get("M", 5.0),
            cf_weight=cfg.get("cf_weight", 1.0),
            use_soft_labels=cfg.get("use_soft_labels", True),
        )
        self.alpha_v = self.alpha_q = cfg.get("alpha0", 0.1)
        self.beta_v  = self.beta_q  = cfg.get("beta0", 0.1)

    def _make_cf_embeddings(self, batch, mode, intervene_v, intervene_q):
        imgs, qs, q_len, _, _, imgs_p, qs_p, qp_len = (
            batch.images, batch.questions, batch.q_lengths,
            batch.answers, batch.soft_targets,
            batch.images_prime, batch.questions_prime, batch.q_lengths_prime,
        )

        e_v  = self.model.encode_visual(imgs)
        e_q  = self.model.encode_question(qs, q_len)
        e_vp = self.model.encode_visual(imgs_p)
        e_qp = self.model.encode_question(qs_p, qp_len)

        if mode == "posterior":
            u_v_s = self.u_v.sample_posterior(self.alpha_v, self.beta_v)
            u_q_s = self.u_q.sample_posterior(self.alpha_q, self.beta_q)
            u_v   = u_v_s.expand(e_v.shape[0]).to(self.device)
            u_q   = u_q_s.expand(e_q.shape[0]).to(self.device)
            e_v_cf, e_q_cf = self.cf_gen(
                e_v, e_q, e_vp, e_qp, u_v, u_q, intervene_v, intervene_q
            )
        else:  # optimized
            _, _, e_v_cf, e_q_cf = self.cf_opt.optimize(
                e_v, e_q, e_vp, e_qp, batch.answers.to(self.device),
                intervene_v, intervene_q,
            )

        return e_v_cf, e_q_cf

    def train_epoch(self, loader, optimizer,
                    mode="+C", intervene_v=True, intervene_q=True) -> Dict[str, float]:
        self.model.train()
        total = erm_t = cf_t = acc_t = 0.0
        nb = 0

        for batch in loader:
            # Move to device
            def _d(x): return x.to(self.device) if x is not None else None
            batch = type(batch)(
                images=_d(batch.images), questions=_d(batch.questions),
                q_lengths=_d(batch.q_lengths), answers=_d(batch.answers),
                soft_targets=_d(batch.soft_targets),
                images_prime=_d(batch.images_prime),
                questions_prime=_d(batch.questions_prime),
                q_lengths_prime=_d(batch.q_lengths_prime),
            )

            optimizer.zero_grad()

            e_v = self.model.encode_visual(batch.images)
            e_q = self.model.encode_question(batch.questions, batch.q_lengths)
            logits_factual = self.model.fuse_and_classify(e_v, e_q)

            if mode == "baseline":
                loss = nn.functional.cross_entropy(logits_factual, batch.answers)
                loss.backward(); optimizer.step()
                total += loss.item(); acc_t += self._acc(logits_factual, batch.answers)
                nb += 1; continue

            # Update posterior
            with torch.no_grad():
                preds   = logits_factual.argmax(1)
                B       = batch.answers.shape[0]
                correct = (preds == batch.answers).sum().item()
                wrong   = B - correct
                self.alpha_v, self.beta_v = self.u_v.update_posterior(correct, wrong)
                self.alpha_q, self.beta_q = self.u_q.update_posterior(correct, wrong)

            # Generate counterfactuals
            e_v_cf, e_q_cf = self._make_cf_embeddings(
                batch, "posterior" if mode == "+P" else "optimized",
                intervene_v, intervene_q
            )
            logits_cf = self.model.fuse_and_classify(e_v_cf, e_q_cf)

            losses = self.loss_fn(
                logits_factual=logits_factual,
                logits_counterfactual=logits_cf,
                answers=batch.answers,
                soft_targets=batch.soft_targets,
                cf_step=(nb % 2 == 1),
            )
            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total += losses["total"].item()
            erm_t += losses["erm"].item()
            cf_t  += losses["cf"].item()
            acc_t += self._acc(logits_factual, batch.answers)
            nb += 1

        nb = max(nb, 1)
        return {"loss": total/nb, "erm": erm_t/nb, "cf": cf_t/nb, "acc": acc_t/nb}

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        correct = total = 0
        for batch in loader:
            v  = batch.images.to(self.device)
            q  = batch.questions.to(self.device)
            ql = batch.q_lengths.to(self.device)
            a  = batch.answers.to(self.device)
            logits   = self.model(v, q, ql)
            correct += (logits.argmax(1) == a).sum().item()
            total   += a.shape[0]
        return {"acc": correct / max(total, 1)}

    def train(self, train_loader, val_loader, epochs=30,
              mode="+C", intervene_v=True, intervene_q=True,
              output_dir="./results/vqa"):
        os.makedirs(output_dir, exist_ok=True)
        lr  = self.cfg.get("lr", 1e-4)
        opt = optim.Adamax(self.model.parameters(), lr=lr, weight_decay=0)
        sched = CosineAnnealingLR(opt, T_max=epochs)
        best = 0.0
        iv = "I+Q" if (intervene_v and intervene_q) else ("I" if intervene_v else "Q")
        print(f"\n  VQA Training  |  mode={mode}  |  intervene={iv}  |  epochs={epochs}")

        for ep in range(epochs):
            tr = self.train_epoch(train_loader, opt, mode, intervene_v, intervene_q)
            vl = self.evaluate(val_loader)
            sched.step()

            for k in ["train_loss","cf_loss","erm_loss","val_acc"]:
                self.history[k].append(tr.get("loss",0) if "loss" in k
                                        else tr.get("cf",0) if "cf" in k
                                        else tr.get("erm",0) if "erm" in k
                                        else vl.get("acc",0))

            if vl["acc"] > best:
                best = vl["acc"]
                self.save(self.model, os.path.join(output_dir, f"best_{mode}_{iv}.pt"))

            if ep % 5 == 0 or ep == epochs - 1:
                self._log(ep, epochs, {**tr, "val_acc": vl["acc"]})

        print(f"  Best VQA Acc: {best:.4f}")
        return best
