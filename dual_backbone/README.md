# Dual-Backbone Hybrid — Runbook

End-to-end joint training of EVA02 + EdgeNeXt + metadata MLP on ISIC 2024 images and clinical features.
See `docs/architecture-dual-backbone.md` for the full architecture description.

---

## Files

| File | Role |
|---|---|
| `isic_model.py` | Shared core: model class, HDF5 dataset, `BalancedBatchSampler`, feature engineering, preprocessing helpers. Co-located so scripts resolve imports with a plain `import isic_model`. |
| `00_prepare_synthetic_metadata.py` | Assigns realistic metadata to the ~6,000 synthetic malignant samples from `generative/` before training. Run once. |
| `01_train.py` | 5-fold patient-aware training. Saves per-fold checkpoints, scalers, encoders, and OOF predictions. |
| `02_analyze_training.py` | Post-training diagnostics: fold AUC curves, OOF score aggregation, prediction distribution sanity checks. |
| `03_generate_submissions.py` | Runs inference with saved checkpoints; writes per-fold and ensemble submission CSVs. |

---

## Inputs

```
data/
  train-image.hdf5          # HDF5 image store keyed by isic_id (224px or 384px)
  train-metadata.csv        # Clinical metadata + labels
  test-image.hdf5
  test-metadata.csv

generative/data/
  synthetic_malignant_384.hdf5          # Optional: synthetic positives at 384px
  synthetic_malignant_metadata_enriched.csv  # Required if using synthetic data
```

Synthetic data is optional. If absent, training proceeds on real data only.

---

## Run Order

```bash
cd /path/to/isic-2024-skin-cancer

# Step 0: prepare synthetic metadata (once, if using synthetic data)
uv run python dual_backbone/00_prepare_synthetic_metadata.py

# Step 1: train all 5 folds (or a single fold for debugging)
uv run python dual_backbone/01_train.py [--fold 0] [--gpu 0] [--batch-size 32]

# Step 2: inspect training results
uv run python dual_backbone/02_analyze_training.py

# Step 3: generate submissions
uv run python dual_backbone/03_generate_submissions.py [--use-ema]
```

---

## Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--fold N` | all folds | Train a single fold (0–4) for quick debugging |
| `--gpu N` | 0 | CUDA device index |
| `--batch-size N` | 48 | Reduce to 16 on OOM; effective batch doubles with grad accum ×2 |
| `--use-ema` | off | `03_generate_submissions.py`: use EMA checkpoints instead of regular |

---

## Outputs

All artifacts land in `results/dual_hybrid_v2/` (gitignored):

```
results/dual_hybrid_v2/
  best_model_fold{1..5}.pth
  best_model_ema_fold{1..5}.pth
  scaler_fold{1..5}.pkl
  encoders_fold{1..5}.pkl
  patient_statistics_fold{1..5}.pkl
  precomputed_features_fold{1..5}.pkl
  feature_info_fold{1..5}.json
  oof_fold{1..5}.csv
  oof_ema_fold{1..5}.csv
  fold_{1..5}_individual_{regular,ema}_submission.csv
  ensemble_regular_average_submission.csv
  ensemble_ema_average_submission.csv   ← primary submission
```

Submit `ensemble_ema_average_submission.csv`. Verified 5-fold CV AUC: **0.9503 ± 0.0092** (regular), **0.9484 ± 0.0128** (EMA). Local AUC is a proxy; the leaderboard metric is pAUC@80%TPR.
