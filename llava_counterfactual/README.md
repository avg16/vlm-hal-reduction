# Counterfactual Vision and Language Learning (CVL) for LLaVA

Implementation of **"Counterfactual Vision and Language Learning"** (Abbasnejad et al., CVPR 2020) applied to **LLaVA** on Visual Question Answering (VQA).

## Overview

This codebase adapts the CVL framework to LLaVA by introducing:
- **Counterfactual generation** via exogenous variable interpolation in the embedding space
- **Beta-distributed exogenous variables** over vision and language embeddings
- **Counterfactual Risk Minimization (CRM)** as an alternative to standard ERM
- Full training loop alternating between observational and counterfactual updates

## Architecture

```
cvl_llava/
├── configs/
│   └── cvl_llava_vqa.yaml        # All hyperparameters
├── data/
│   ├── vqa_dataset.py            # VQA v2 / VQA-CP dataset loaders
│   └── collator.py               # Batch collation for LLaVA
├── models/
│   ├── llava_cvl.py              # LLaVA + CVL wrapper
│   ├── exogenous_module.py       # Exogenous variable module (Beta posterior)
│   └── counterfactual_generator.py  # Counterfactual sample generation
├── training/
│   ├── trainer.py                # CVL training loop
│   ├── crm_loss.py               # Counterfactual Risk Minimization loss
│   └── erm_loss.py               # Standard ERM loss
├── evaluation/
│   └── evaluator.py              # VQA evaluation metrics
├── utils/
│   ├── logging_utils.py
│   └── checkpoint_utils.py
└── scripts/
    ├── train.py                  # Main training script
    └── evaluate.py               # Evaluation script
```

## Setup

```bash
pip install -r requirements.txt
```

## Training on VQA-CP v2 (recommended for bias evaluation)

```bash
python scripts/train.py --config configs/cvl_llava_vqa.yaml \
    --dataset vqa_cp \
    --data_dir /path/to/vqa_cp \
    --output_dir ./outputs/cvl_llava_vqacp
```

## Training on VQA v2

```bash
python scripts/train.py --config configs/cvl_llava_vqa.yaml \
    --dataset vqa_v2 \
    --data_dir /path/to/vqa_v2 \
    --output_dir ./outputs/cvl_llava_vqa2
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint ./outputs/cvl_llava_vqacp/best_model \
    --dataset vqa_cp \
    --data_dir /path/to/vqa_cp
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_cf` | 0.5 | Counterfactual loss weight λ |
| `epsilon` | 0.1 | Interpolation threshold ε |
| `clip_M` | 10.0 | Importance weight clipping M |
| `beta_alpha0` | 0.1 | Beta prior α₀ |
| `beta_beta0` | 0.1 | Beta prior β₀ |
| `cf_steps` | 5 | Gradient ascent steps for u optimization |
| `cf_lr` | 0.01 | Learning rate for u optimization |
| `alternate_every` | 1 | Alternate ERM/CRM every N steps |

## Citation

```bibtex
@inproceedings{abbasnejad2020counterfactual,
  title={Counterfactual Vision and Language Learning},
  author={Abbasnejad, Ehsan and Teney, Damien and Parvaneh, Amin and Shi, Javen and van den Hengel, Anton},
  booktitle={CVPR},
  year={2020}
}
```