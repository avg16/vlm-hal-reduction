"""
Main Runner — Counterfactual Vision and Language (CVL)
======================================================
Reproduces all three experiment tracks from the paper:

    python run.py --exp cifar10   --mode +C
    python run.py --exp cifar100  --mode +C
    python run.py --exp sst       --mode +C
    python run.py --exp vqa_cp    --mode +C --intervene_v --intervene_q
    python run.py --exp vqa_v2    --mode +C

    --mode: baseline | +P | +C
"""

import argparse
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from models.vqa_model import (
    build_updn_model, build_simple_model, build_image_classifier
)
from data import get_cifar_loader, get_sst_loader, get_vqa_loader
from trainer import CIFARTrainer, SSTTrainer, VQATrainer
import json


# ---------------------------------------------------------------------------
# Simple language-only model for SST
# ---------------------------------------------------------------------------

class SSTModel(torch.nn.Module):
    """LSTM classifier for SST. Exposes .encoder() and .classify() for trainer."""

    def __init__(self, vocab_size=10000, embed_dim=300, hidden_dim=512, num_classes=2):
        super().__init__()
        import torch.nn as nn
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.head      = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.hidden_dim = hidden_dim

    def encoder(self, q, lengths=None):
        emb = self.embedding(q)
        _, (h, _) = self.lstm(emb)
        return h[-1]

    def classify(self, e):
        return self.head(e)

    def forward(self, q, lengths=None):
        return self.head(self.encoder(q, lengths))


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_cifar(args, cfg, device):
    print("\n" + "="*60)
    print(f"  CIFAR-{args.num_classes} | mode={args.mode}")
    print("="*60)

    # Build model
    from models.vqa_model import build_image_classifier
    model = build_image_classifier(num_classes=args.num_classes, embed_dim=512)

    # Data
    train_loader = get_cifar_loader(
        root=args.data_path, num_classes=args.num_classes,
        train=True, batch_size=cfg["batch_size"],
        img_size=args.img_size, seed=args.seed,
    )
    val_loader = get_cifar_loader(
        root=args.data_path, num_classes=args.num_classes,
        train=False, batch_size=cfg["batch_size"],
        img_size=args.img_size, seed=args.seed,
    )

    trainer = CIFARTrainer(model, cfg, device)
    best_acc = trainer.train(
        train_loader, val_loader,
        epochs=cfg["epochs"], mode=args.mode,
        output_dir=os.path.join(args.output_dir, f"cifar{args.num_classes}"),
    )

    # Save history for variance plot (Fig. 4)
    out = os.path.join(args.output_dir, f"cifar{args.num_classes}")
    os.makedirs(out, exist_ok=True)
    torch.save(trainer.history, os.path.join(out, f"history_{args.mode}.pt"))

    print(f"\n  Final best acc: {best_acc:.4f}")
    return best_acc


def run_sst(args, cfg, device):
    print("\n" + "="*60)
    print(f"  SST | mode={args.mode}")
    print("="*60)

    model = SSTModel(vocab_size=cfg.get("vocab_size", 10000),
                     embed_dim=300, hidden_dim=512, num_classes=2)

    train_loader = get_sst_loader(
        path=args.sst_train, batch_size=cfg["batch_size"], seed=args.seed,
    )
    val_loader = get_sst_loader(
        path=args.sst_val, batch_size=cfg["batch_size"], seed=args.seed,
    )

    trainer = SSTTrainer(model, cfg, device)
    best_acc = trainer.train(
        train_loader, val_loader,
        epochs=cfg["epochs"], mode=args.mode,
        output_dir=os.path.join(args.output_dir, "sst"),
    )
    return best_acc


def run_vqa(args, cfg, device):
    print("\n" + "="*60)
    print(f"  VQA | dataset={args.exp}  mode={args.mode}  "
          f"intervene_v={args.intervene_v}  intervene_q={args.intervene_q}")
    print("="*60)

    # Load vocabularies if provided
    answer_vocab = word_vocab = None
    if args.answer_vocab and os.path.exists(args.answer_vocab):
        with open(args.answer_vocab) as f:
            answer_vocab = json.load(f)
    if args.word_vocab and os.path.exists(args.word_vocab):
        with open(args.word_vocab) as f:
            word_vocab = json.load(f)

    num_answers = len(answer_vocab) if answer_vocab else cfg.get("num_answers", 3129)
    vocab_size  = len(word_vocab)   if word_vocab  else cfg.get("vocab_size",  10000)

    model = build_updn_model(
        vocab_size=vocab_size,
        num_answers=num_answers,
        embed_dim=cfg.get("embed_dim", 512),
    )

    # Data
    q_train = args.vqa_questions_train
    a_train = args.vqa_annotations_train
    q_val   = args.vqa_questions_val
    a_val   = args.vqa_annotations_val

    train_loader = get_vqa_loader(
        questions_json=q_train, annotations_json=a_train,
        image_dir=args.vqa_image_dir, answer_vocab=answer_vocab,
        word_vocab=word_vocab, num_answers=num_answers,
        batch_size=cfg["batch_size"], shuffle=True, seed=args.seed,
    )
    val_loader = get_vqa_loader(
        questions_json=q_val, annotations_json=a_val,
        image_dir=args.vqa_image_dir, answer_vocab=answer_vocab,
        word_vocab=word_vocab, num_answers=num_answers,
        batch_size=cfg["batch_size"], shuffle=False, seed=args.seed,
    )

    trainer = VQATrainer(model, cfg, device)
    best_acc = trainer.train(
        train_loader, val_loader,
        epochs=cfg["epochs"], mode=args.mode,
        intervene_v=args.intervene_v,
        intervene_q=args.intervene_q,
        output_dir=os.path.join(args.output_dir, args.exp),
    )
    return best_acc


def run_ablation_sweep(args, cfg, device):
    """
    Reproduce ablation in Table 3: run all 3 modes and both intervention configs.
    """
    print("\n  ABLATION SWEEP: comparing baseline, +P, +C")
    results = {}

    for mode in ["baseline", "+P", "+C"]:
        args.mode = mode
        if args.exp.startswith("cifar"):
            acc = run_cifar(args, cfg, device)
        elif args.exp == "sst":
            acc = run_sst(args, cfg, device)
        else:
            acc = run_vqa(args, cfg, device)
        results[mode] = acc

    if args.exp.startswith("vqa"):
        for iv, iq in [(True, False), (False, True), (True, True)]:
            args.mode = "+C"
            args.intervene_v = iv
            args.intervene_q = iq
            tag = "I" + ("v" if iv else "") + ("q" if iq else "")
            acc = run_vqa(args, cfg, device)
            results[f"+C_{tag}"] = acc

    print("\n  ABLATION RESULTS:")
    print("  " + "-"*30)
    for mode, acc in results.items():
        print(f"  {mode:15s} : {acc:.4f}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Counterfactual Vision and Language (CVL)")

    p.add_argument("--exp", default="cifar10",
                   choices=["cifar10","cifar100","sst","vqa_cp","vqa_v2"])
    p.add_argument("--mode", default="+C", choices=["baseline", "+P", "+C"])

    # CIFAR
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--img_size",    type=int, default=224)

    # SST paths
    p.add_argument("--sst_train", default=None)
    p.add_argument("--sst_val",   default=None)

    # VQA paths
    p.add_argument("--vqa_questions_train",   default=None)
    p.add_argument("--vqa_annotations_train", default=None)
    p.add_argument("--vqa_questions_val",     default=None)
    p.add_argument("--vqa_annotations_val",   default=None)
    p.add_argument("--vqa_image_dir",         default=None)
    p.add_argument("--answer_vocab",          default=None)
    p.add_argument("--word_vocab",            default=None)

    # Intervention
    p.add_argument("--intervene_v", action="store_true", default=True)
    p.add_argument("--intervene_q", action="store_true", default=False)
    p.add_argument("--no_intervene_v", dest="intervene_v", action="store_false")
    p.add_argument("--no_intervene_q", dest="intervene_q", action="store_false")

    # Training
    p.add_argument("--data_path",  default="./data")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",       type=int,   default=42)

    # SCM hyperparameters
    p.add_argument("--alpha0",     type=float, default=0.1)
    p.add_argument("--beta0",      type=float, default=0.1)
    p.add_argument("--epsilon",    type=float, default=0.5,  help="Eq.2 threshold")
    p.add_argument("--lambda_cf",  type=float, default=1.0,  help="Eq.3 weight")
    p.add_argument("--cf_lr",      type=float, default=0.01, help="Exogenous var lr")
    p.add_argument("--cf_steps",   type=int,   default=10,   help="Eq.3 gradient steps")
    p.add_argument("--M",          type=float, default=5.0,  help="Clipping constant Eq.4")
    p.add_argument("--cf_weight",  type=float, default=1.0,  help="CF loss weight")

    # Sweep
    p.add_argument("--sweep", action="store_true", help="Run ablation sweep over all modes")

    return p.parse_args()


def main():
    args  = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # If CIFAR, override num_classes from exp name
    if args.exp == "cifar10":  args.num_classes = 10
    if args.exp == "cifar100": args.num_classes = 100

    cfg = {
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "alpha0":      args.alpha0,
        "beta0":       args.beta0,
        "epsilon":     args.epsilon,
        "lambda_cf":   args.lambda_cf,
        "cf_lr":       args.cf_lr,
        "cf_steps":    args.cf_steps,
        "M":           args.M,
        "cf_weight":   args.cf_weight,
        "use_soft_labels": args.exp.startswith("vqa"),
    }

    print(f"\n{'='*60}")
    print("  Counterfactual Vision and Language (CVL)")
    print(f"  Abbasnejad et al., CVPR 2020")
    print(f"{'='*60}")
    print(f"  Exp:    {args.exp}   Mode: {args.mode}")
    print(f"  Device: {device}     Seed: {args.seed}")
    print(f"  SCM:  ε={args.epsilon}  λ={args.lambda_cf}  M={args.M}")
    print(f"        α₀={args.alpha0}  β₀={args.beta0}")

    if args.sweep:
        run_ablation_sweep(args, cfg, device)
    elif args.exp.startswith("cifar"):
        run_cifar(args, cfg, device)
    elif args.exp == "sst":
        run_sst(args, cfg, device)
    elif args.exp.startswith("vqa"):
        run_vqa(args, cfg, device)

    print(f"\n  Done. Outputs → {args.output_dir}")


if __name__ == "__main__":
    main()
