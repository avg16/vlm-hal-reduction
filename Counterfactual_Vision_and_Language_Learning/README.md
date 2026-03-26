# Counterfactual Vision and Language (CVL)
Implementation of **Abbasnejad et al., CVPR 2020**

---

## Paper → Code Map

| Paper | File | What It Does |
|-------|------|-------------|
| Def 1 — SCM | `models/scm.py :: ExogenousVariable` | u^v, u^q with Beta prior |
| Eq. (2) — generative process | `models/scm.py :: CounterfactualGenerator` | ṽ = u·v ⊕ (1-u)·v' |
| Eq. (3) — counterfactual opt | `models/scm.py :: CounterfactualOptimizer` | max \|\|u\|\|² - λ·log p(a\|ṽ) |
| Beta posterior update | `models/scm.py :: ExogenousVariable.update_posterior` | α = α₀ + Σ𝟙[correct] |
| Eq. (4) — counterfactual risk | `losses/__init__.py :: CounterfactualRisk` | R̂_M(θ) = Σ ℓ × ω |
| ω importance weight | `losses/__init__.py :: importance_weight` | min{M, p(a\|factual)/p(a\|cf)} |
| Theorem 4 — variance bound | `losses/__init__.py :: VarianceBoundMonitor` | R(θ) ≤ R̂_M + gap |
| Fig 1 — training loop | `trainer.py :: {CIFAR,SST,VQA}Trainer` | alternate ERM + CF steps |
| Table 1 — SST | `run.py :: run_sst` | LSTM + u^q only |
| Table 2 — CIFAR | `run.py :: run_cifar` | ResNet + u^v only |
| Table 3 — VQA-CP | `run.py :: run_vqa` | UpDn + u^v + u^q |
| Table 4 — VQA v2 | `run.py :: run_vqa --exp vqa_v2` | same model |
| Ablation: Q/I/Q+I | `run.py --sweep` | intervene_v / intervene_q flags |

---

## Install

```bash
pip install torch torchvision
```

---

## Usage

### CIFAR-10/100 (Table 2)

```bash
# Baseline
python run.py --exp cifar10 --mode baseline --epochs 100

# Ours+P (posterior sampling)
python run.py --exp cifar10 --mode +P --epochs 100

# Ours+C (counterfactual optimization) — main method
python run.py --exp cifar10 --mode +C --epochs 100

# Full sweep (reproduces Table 2 for one architecture)
python run.py --exp cifar10 --sweep --epochs 100
```

### SST (Table 1)

```bash
python run.py --exp sst --mode +C \
    --sst_train ./data/sst/train.tsv \
    --sst_val   ./data/sst/dev.tsv \
    --epochs 10
```

SST TSV format: `sentence\tlabel` (label: 0=neg, 1=pos, one per line).

### VQA-CP / VQA v2 (Tables 3/4)

```bash
# Images only (UpDn+C Images in Table 3)
python run.py --exp vqa_cp --mode +C --intervene_v --no_intervene_q \
    --vqa_questions_train   data/vqa_cp/train_questions.json \
    --vqa_annotations_train data/vqa_cp/train_annotations.json \
    --vqa_questions_val     data/vqa_cp/test_questions.json \
    --vqa_annotations_val   data/vqa_cp/test_annotations.json \
    --vqa_image_dir         data/coco/images/ \
    --answer_vocab          data/answer_vocab.json \
    --word_vocab            data/word_vocab.json \
    --epochs 30

# Questions only (UpDn+C Questions)
python run.py --exp vqa_cp --mode +C --no_intervene_v --intervene_q ...

# Both (UpDn+C Q+I — best result, Table 3: 42.12%)
python run.py --exp vqa_cp --mode +C --intervene_v --intervene_q ...
```

---

## Key SCM Hyperparameters

| Arg | Default | Paper Eq. | Meaning |
|-----|---------|-----------|---------|
| `--alpha0` | 0.1 | Beta prior | u ~ Beta(α₀, β₀) |
| `--beta0` | 0.1 | Beta prior | |
| `--epsilon` | 0.5 | Eq. (2) | Interpolation threshold |
| `--lambda_cf` | 1.0 | Eq. (3) | Weight on log p(a\|cf) |
| `--cf_lr` | 0.01 | Eq. (3) | Gradient ascent lr for u |
| `--cf_steps` | 10 | Eq. (3) | Gradient ascent iterations |
| `--M` | 5.0 | Eq. (4) | Importance weight clip |
| `--cf_weight` | 1.0 | Eq. (4) | Weight on CF loss term |

---

## File Structure

```
cvl/
├── models/
│   ├── scm.py          # SCM: ExogenousVariable, CounterfactualGenerator, CounterfactualOptimizer
│   └── vqa_model.py    # Visual/language encoders, fusion, full VQA model
├── losses/
│   └── __init__.py     # ImportanceWeight, CounterfactualRisk, CombinedCVLLoss, VarianceBoundMonitor
├── data/
│   └── __init__.py     # CIFAR, SST, VQA loaders (all with distractors for Eq.2)
├── trainer.py          # CIFARTrainer, SSTTrainer, VQATrainer
└── run.py              # CLI — reproduces all paper tables
```

---

## Training Loop (Fig. 1)

```
for each batch (q, v, a):
    1. Forward factual:  logits = model(q, v)
    2. ERM update:       L_erm = CE(logits, a)

    3. Update Beta posterior:
       α = α₀ + #{correct predictions}
       β = β₀ + #{wrong predictions}

    4. Generate counterfactuals via Eq. (2):
       u = optimize_u(v, v') via Eq. (3)     # max ||u||² - λ·log p(a|cf)
       ṽ = u·v + (1-u)·v'                    # interpolate

    5. Counterfactual forward: logits_cf = model(q̃, ṽ)
    6. ω = min{M, p(a|q,v) / p(a|q̃,ṽ)}     # importance weight
    7. L_cf = CE(logits_factual, a) × ω      # Eq. (4)

    8. L_total = L_erm + L_cf  (or alternate each step)
       θ ← θ - ∇L_total
```
