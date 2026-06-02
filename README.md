# ISIC 2024 — Skin-Lesion Malignancy Detection under Extreme Class Imbalance

Predicting whether a dermoscopic skin lesion is malignant, from **images + clinical metadata**, in a
dataset where malignant cases are vanishingly rare: **343 / 400,959 ≈ 0.0855 % (~1 : 1168)**. This
repository is the cleaned, documented record of an end-to-end deep-learning project for the
[ISIC 2024 challenge](https://www.kaggle.com/competitions/isic-2024-challenge) — from a near-random
baseline to two competitive endgame architectures, including the validation discipline, leakage
audits, and failure postmortems that the result depended on.

> **Metric note.** The competition's official metric is **partial AUC (pAUC) above 80 % TPR** —
> ranking quality in the high-sensitivity regime, not plain AUC. Local AUC (0.93–0.98) is used as a
> *development proxy* throughout; it is **not** the same scale as the leaderboard pAUC. Every number
> below is labeled *verified* (recomputed from on-disk predictions) or *self-reported* (recorded in a
> log, no independent receipt). See [`docs/domain-rules.md`](docs/domain-rules.md).

---

## Results

| Approach | Metric | Value | Provenance |
|---|---|---|---|
| **Dual-backbone end-to-end** (EVA02 + EdgeNeXt + metadata) | 5-fold CV AUC (regular) | **0.9503 ± 0.0092** | **Verified** — recomputed from `results/dual_hybrid_v2/oof_fold*.csv` |
| same, EMA weights | 5-fold CV AUC | 0.9484 ± 0.0128 | **Verified** — from `oof_ema_fold*.csv` (EMA ≈ regular here, slightly higher variance) |
| **Two-stage stacking** (vision → XGBoost+MLP) | Public LB (best single fold) | 0.98997 | *Self-reported* (submission log); metric-scale caveat below |
| same, 5-fold ensemble | Public LB | 0.98245 | *Self-reported* (submission log) |
| Baseline (scratch CNN, images only) | local AUC | ~0.51 | self-reported (≈ random under imbalance) |
| + metadata fusion (Focal Loss) | local AUC | ~0.936 | self-reported — the single biggest jump |

> **Caveat on the stacking LB numbers.** They are recorded verbatim in the submission log as Public
> LB scores, but a value near 0.99 is unusually high for a pAUC@80 %TPR (the project's *pAUC*-scale
> figures elsewhere sit around 0.16–0.18). They are most likely an AUC-scaled course leaderboard. Treat
> them as the strongest *recorded* result, not as an independently reproduced pAUC. See
> [`docs/train-inference-mismatch.md`](docs/train-inference-mismatch.md) for the pAUC-scale discussion.

---

## Why this problem is hard

1. **Extreme imbalance (~1:1168).** Naïve training collapses to predicting "benign" everywhere
   (~0.51 AUC). Addressed with **1:1 balanced sampling** (which beat Focal-Loss-alone) and
   patient-aware evaluation. → [`docs/model-evolution.md`](docs/model-evolution.md)
2. **The metric rewards the high-sensitivity tail**, so calibration and ranking in the top-TPR band
   matter more than global accuracy.
3. **Label leakage is everywhere in the metadata.** Post-biopsy columns (`mel_thick_mm`,
   `mel_mitotic_index`, the `iddx_*` diagnosis codes) are ~100 % malignant when present. They must be
   excluded in *both* training and inference. → [`docs/metadata-audit.md`](docs/metadata-audit.md)
4. **Patient leakage.** One patient contributes many lesions (28–9,184). Splits must group by
   `patient_id` or validation scores are fiction.
5. **Train→test distribution shift.** The vision models' prediction distributions shift hard from
   train to test (**EVA02 +2.31σ, EdgeNeXt +2.49σ**, K-S p≪1e-26) — which makes rank-normalization at
   inference actively dangerous. → [`docs/distribution-shift.md`](docs/distribution-shift.md)

## The two endgame architectures

**A) Dual-backbone, end-to-end** — [`dual_backbone/`](dual_backbone/)
One model fuses **EVA02-Small** (`eva02_small_patch14_336.mim_in22k_ft_in1k`, @336) **+ EdgeNeXt-Base**
(`edgenext_base.in21k_ft_in1k`, @384) **+ a metadata encoder** through a fusion MLP. Uses EMA, AMP,
1:1 balanced sampling, and patient-relative ("Ugly Duckling") + LOF features. Trained end-to-end over
5 patient-grouped folds.

**B) Two-stage stacking** — [`stacking/`](stacking/)
Stage 1: an EVA02 + EdgeNeXt vision ensemble (real data only) emitting **logits + top-50 PCA
embeddings**. Stage 2: an **XGBoost** model on [metadata + vision logits + embeddings] **and** an
**MLP** on [metadata + DAE latents + logits + embeddings], averaged. Includes the **"Golden Split"**
finding (train folds 0–3, exclude the noisy fold 4) and the corrected, parity-safe inference path.

## Repository layout

```
dual_backbone/   Endgame A — train + submit (isic_model.py is the shared model/dataset core)
stacking/        Endgame B — vision → DAE → stacking, with feature_list.txt (38 vetted features)
generative/      Synthetic-malignant augmentation: fine-tuned SD-1.5 → classifier-filtered positives
tools/           Reusable utilities (HDF5 preprocessing, scaler recovery, OOF/submission checks)
docs/            Deep dives: evolution, architecture, metadata audit, postmortems, skills map
archive/         The full 1→17 evolution, read-only, grouped by era (see archive/INDEX.md)
references/      The 1st-place solution writeup + competition pages (external, clearly attributed)
reports/         The Slidev presentation (source) and built decks
results/sample/  Tiny illustrative artifacts (a sample submission + OOF; the safe feature list)
```

## What this project demonstrates (engineering rigor)

Beyond the models, the project is a record of **catching silent failures** — the kind that quietly
cost leaderboard points:

- **Train/inference preprocessing mismatch** (z-score in training vs rank-normalization at inference,
  plus a too-short exclude list) broke the stacking submission — ~0.48 instead of the expected
  ~0.16–0.18 pAUC. Root-caused over a 6-phase audit and fixed.
  → [`docs/train-inference-mismatch.md`](docs/train-inference-mismatch.md)
- **A `GradScaler` saved in place of a `StandardScaler`** silently dropped a fold's test AUC ~95 %→85 %;
  reconstructed and repaired by re-deriving the exact fold splits.
  → [`docs/scaler-recovery-postmortem.md`](docs/scaler-recovery-postmortem.md)
- **Numbers were re-verified against the raw OOF files** during this cleanup — which corrected a
  long-quoted "0.9612 CV AUC" down to the file-backed **0.9503 ± 0.0092**. Reproducibility beats
  memory. See [`docs/model-evolution.md`](docs/model-evolution.md).

Full competency inventory: [`docs/skills-map.md`](docs/skills-map.md).

## Quickstart

```bash
# 1. Install (Python ≥3.11). uv recommended:
uv venv && uv pip install -e .          # or: pip install -e .

# 2. Get the data — see DATA.md (ISIC 2024; not redistributed here)
#    Place CSVs + HDF5 image stores under ./data/, then build fast stores:
python tools/preprocess_hdf5.py         # 224px
python tools/preprocess_hdf5_v2.py      # 384px

# 3a. Endgame A — dual-backbone, end-to-end
python dual_backbone/01_train.py --fold 0
python dual_backbone/03_generate_submissions.py

# 3b. Endgame B — two-stage stacking
python stacking/src/create_folds.py
python stacking/src/train_vision.py
python stacking/src/train_stacking.py
python stacking/src/inference_stacking_corrected.py
```

GPU is expected; scripts accept smaller `--batch-size` and CPU fallback. See each tier's README.

## Data & licensing

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Data:** ISIC 2024 imagery/metadata is **not** included and is governed by the competition terms —
  see [`DATA.md`](DATA.md).

## Acknowledgments

Patient-relative ("Ugly Duckling") and LOF features were adopted from the publicly shared
**1st-place ISIC 2024 solution** (reproduced for reference, clearly attributed, in
[`references/1st-place-solution.md`](references/1st-place-solution.md)). This repository is independent
coursework and is not affiliated with ISIC or Kaggle.
