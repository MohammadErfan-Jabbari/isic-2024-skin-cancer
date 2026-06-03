# Finding 343 Cancers in 400,959 Photos
#### Skin-lesion malignancy detection (ISIC 2024) from dermoscopic images + clinical metadata, under extreme class imbalance

A classifier that labels every lesion benign scores **99.91 % accuracy** — and never catches a single
cancer. With malignant cases at **343 in 400,959 (≈ 1 : 1168)**, accuracy is meaningless; the entire
problem is *ranking the few true positives above the many look-alikes.* This repository takes that
problem from a coin-flip baseline (0.51 AUC) to two working architectures — and it carries the
verification to show the reported numbers hold up against the raw predictions.

That last part is not rhetorical. While cleaning this repository up, I recomputed the headline
cross-validation score directly from the saved out-of-fold predictions. It came back **0.9503, not the
0.9612 I had been quoting** from a training log, and I corrected it everywhere. Building the model is
half the work; knowing which of your own numbers to trust is the other half — and that discipline is
what this repository is meant to demonstrate as much as the architectures are.

> **On the metric.** ISIC 2024 is scored by **partial AUC above 80 % TPR (pAUC@80%TPR)** — performance
> in the high-sensitivity regime a screening tool actually operates in, not global accuracy or plain
> AUC. Standard AUC is reported as a development proxy alongside the official pAUC. Every figure below
> is labeled **verified** (recomputed from on-disk predictions by
> [`tools/compute_pauc.py`](tools/compute_pauc.py)) or **self-reported** (from a log, no independent
> receipt).

## Results

Computed from the 5-fold out-of-fold (OOF) predictions in `results/dual_hybrid_v2/`. The official ISIC
pAUC ranges from **0.02 (random) to 0.12 (perfect)**, so a value near 0.10 is strong.

| Model | pAUC@80%TPR (official) | Standard AUC | Provenance |
|---|---|---|---|
| **Dual-backbone, end-to-end** (regular weights) | **0.099 ± 0.004** | **0.9503 ± 0.0092** | verified — 5-fold OOF |
| Dual-backbone (EMA weights) | 0.098 ± 0.006 | 0.9483 ± 0.0128 | verified — EMA ≈ regular here |
| Two-stage stacking (XGBoost + MLP) | 0.087 | 0.928 | verified — pooled OOF |
| Baseline → + metadata fusion | — | 0.51 → ~0.936 | self-reported — the single largest jump |

> The two-stage stacking run also logged a course-leaderboard score of **0.990**, sometimes cited as
> "pAUC." It is not: the verified pAUC@80%TPR is ≈ 0.09, so that 0.990 is an **AUC-scaled** leaderboard
> number, not a competition pAUC. Read it as a self-reported AUC. (This mislabeling is exactly the
> failure mode the [postmortems](docs/train-inference-mismatch.md) exist to catch.)

## Why this problem is hard

**The imbalance is the whole game.** At 1 malignant in 1,168, naïve training collapses to "benign for
everyone." What worked was *balanced 1 : 1 sampling* (which beat focal-loss-alone) paired with a metric
that only rewards the high-sensitivity tail. The largest single lever was not a fancier backbone but
**metadata fusion** — adding clinical features to an image-only model moved local AUC from ≈ 0.51 to
≈ 0.936. → [`docs/model-evolution.md`](docs/model-evolution.md)

**Validity is fragile, in two directions.** The metadata hides post-biopsy columns (`mel_thick_mm`,
`mel_mitotic_index`, the `iddx_*` diagnosis codes) that are ~100 % malignant whenever present — leak
them and the model "wins" on train and fails on test. And because one patient contributes 28–9,184
lesions, any split that lets a patient straddle train and validation reports a fantasy. Every fold here
is grouped by `patient_id`; every leakage column is excluded in both training and inference.
→ [`docs/metadata-audit.md`](docs/metadata-audit.md), [`docs/domain-rules.md`](docs/domain-rules.md)

**The test distribution moves.** The vision models' prediction distributions shift hard from train to
test — EVA02 by +2.31σ, EdgeNeXt by +2.49σ (Kolmogorov–Smirnov p ≪ 1e-26). That single fact is why
rank-normalizing predictions at inference is dangerous, and why one stacking submission scored ≈ 0.48
instead of working. → [`docs/distribution-shift.md`](docs/distribution-shift.md)

## The two architectures

**A) Dual-backbone, end-to-end** — [`dual_backbone/`](dual_backbone/)
One model fuses **EVA02-Small** (`eva02_small_patch14_336.mim_in22k_ft_in1k`, @336) **+ EdgeNeXt-Base**
(`edgenext_base.in21k_ft_in1k`, @384) **+ a metadata encoder** through a fusion MLP. EMA, AMP, 1 : 1
balanced sampling, patient-relative ("Ugly Duckling") and LOF features, trained over 5 patient-grouped
folds. The two backbones are complementary by design — EVA02 attends to global context, EdgeNeXt to
local lesion texture (shown in [`docs/explainability.md`](docs/explainability.md)).

**B) Two-stage stacking** — [`stacking/`](stacking/)
Stage 1: an EVA02 + EdgeNeXt vision ensemble (real data only) emitting **logits + top-50 PCA
embeddings**. Stage 2: an **XGBoost** model on [metadata + vision logits + embeddings] and an **MLP** on
[metadata + DAE latents + logits + embeddings], averaged. Includes the **"Golden Split"** finding
(train folds 0–3, exclude the noisy fold 4) and a corrected, parity-safe inference path.

## Repository layout

```
dual_backbone/   Architecture A — train + submit (isic_model.py is the shared model/dataset core)
stacking/        Architecture B — vision → DAE → stacking, with feature_list.txt (38 vetted features)
generative/      Synthetic-malignant augmentation: fine-tuned SD-1.5 → classifier-filtered positives
tools/           Reusable utilities — pAUC scorer, HDF5 preprocessing, scaler recovery, OOF checks
docs/            Deep dives: evolution, architecture, metadata audit, postmortems, explainability, skills
archive/         The full 1→17 evolution, read-only, grouped by era (see archive/INDEX.md)
references/      The 1st-place solution writeup + competition pages (external, clearly attributed)
reports/         The Slidev presentation (source) and ERRATA.md
results/sample/  Tiny illustrative artifacts (a sample submission + OOF; the safe feature list)
```

## What this project demonstrates

Beyond the two models, the repository is a record of **catching silent failures** — the kind that
quietly cost generalization:

- **Train/inference preprocessing mismatch** (z-score in training vs rank-normalization at inference,
  plus a too-short exclude list) broke a stacking submission — ≈ 0.48 instead of working. Root-caused
  over a six-phase audit and fixed. → [`docs/train-inference-mismatch.md`](docs/train-inference-mismatch.md)
- **A `GradScaler` pickled in place of a `StandardScaler`** silently dropped a fold's test AUC
  ≈ 95 % → 85 %; repaired by re-deriving the exact seeded fold splits.
  → [`docs/scaler-recovery-postmortem.md`](docs/scaler-recovery-postmortem.md)
- **The headline number was re-verified against raw predictions**, which corrected a long-quoted
  0.9612 CV AUC to the file-backed 0.9503 ± 0.0092. → [`docs/model-evolution.md`](docs/model-evolution.md)
- **The model is looking in the right place** — Grad-CAM and SHAP confirm attention on the lesion and
  clinically sensible feature attributions. → [`docs/explainability.md`](docs/explainability.md)

The full, evidence-anchored competency inventory is in [`docs/skills-map.md`](docs/skills-map.md).

## Open questions & next steps

What this project does *not* yet establish (and would, given more time):

- **No synthetic-data ablation.** The dual-backbone was trained *with* synthetic positives, but there
  is no with/without comparison, so their contribution is unquantified. An earlier synthetic experiment
  on a weaker backbone actively hurt the leaderboard — reason for caution, not confidence.
- **Phase-0 stacking choices rest on simulated embeddings.** "A linear head suffices," "DAE helps the
  MLP but not the trees," and "logits + PCA-50 beats logits alone" came from experiments on *simulated*
  vision features; they should be re-run on real embeddings before being treated as settled.
- **The official pAUC was only ever computed locally (OOF), never on a held-out test set** — the course
  leaderboard reported an AUC-scaled score. A true held-out pAUC@80%TPR would close the loop.
- **Explainability is qualitative.** Deletion/insertion curves for Grad-CAM and SHAP-interaction
  analysis on the metadata would turn the sanity checks into quantitative evidence.
- **Calibration and test-time augmentation are untested levers**, both plausibly worth points in the
  high-sensitivity regime.

## Quickstart

```bash
# 1. Install (Python ≥3.11). uv recommended:
uv venv && uv pip install -e .          # or: pip install -e .

# 2. Get the data — see DATA.md (ISIC 2024; not redistributed here)
#    Place CSVs + HDF5 image stores under ./data/, then build fast stores:
python tools/preprocess_hdf5.py         # 224px
python tools/preprocess_hdf5_v2.py      # 384px

# 3a. Architecture A — dual-backbone, end-to-end
python dual_backbone/01_train.py --fold 0
python dual_backbone/03_generate_submissions.py

# 3b. Architecture B — two-stage stacking
python stacking/src/create_folds.py
python stacking/src/train_vision.py
python stacking/src/train_stacking.py
python stacking/src/inference_stacking_corrected.py

# Reproduce the headline metric from OOF predictions:
python tools/compute_pauc.py --oof-glob "results/dual_hybrid_v2/oof_fold*.csv"
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
