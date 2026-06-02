# Dual-Backbone Hybrid Architecture

**Task:** ISIC 2024 — binary malignancy classification from dermoscopic images + clinical metadata.
**Constraint:** Extreme class imbalance (~1:1168 malignant prevalence). Competition metric: pAUC above 80% TPR.

---

## Model Overview

A single end-to-end model trained jointly across all components via backpropagation. No two-stage stacking; all gradients flow through the full pipeline.

```
┌───────────────────────────────────────────────────────────────┐
│                         INPUT                                  │
│  Image (336×336)     Image (384×384)       Metadata (~110 raw) │
└──────┬──────────────────────┬──────────────────────┬──────────┘
       │                      │                      │
       ▼                      ▼                      ▼
 EVA02-Small            EdgeNeXt-Base          Metadata MLP
 eva02_small_patch14    edgenext_base           Linear → ReLU
 _336.mim_in22k         .in21k_ft_in1k          → [64-dim]
 _ft_in1k               @384px
 @336px
       │                      │
  [384-dim]             [584-dim]
       └──────────┬───────────┘
                  │
             [968-dim]
                  │
             cat([968, 64])
                  │
             [1032-dim]
                  │
            Fusion MLP
                  │
             [1-dim logit]
```

**Embedding dimensions (from plan):**
- EVA02 branch → 384-dim feature vector
- EdgeNeXt branch → 584-dim feature vector
- Concatenated vision → 968-dim
- Metadata encoder: raw features (~110-dim) → two-layer MLP → 64-dim
- Fusion input: 968 + 64 = **1032-dim** → Fusion MLP → scalar logit

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Batch size | 48 (effective 96 with grad accum ×2) |
| Optimizer | Adam |
| LR scheduler | ReduceLROnPlateau |
| Learning rate | 5e-4 |
| Weight decay | 1e-5 |
| Epochs | 40 (early stopping patience=15) |
| Loss | FocalLoss (α=0.25, γ=2.0) |
| AMP | Enabled (float16, ~1.5–2× speedup) |
| EMA decay | 0.9999 |

**Balanced sampling (1:1 ratio):** `BalancedBatchSampler` draws equal positives and negatives per batch with wraparound for the minority class. This is the primary imbalance remedy and outperforms Focal Loss alone.

---

## Cross-Validation

5-fold `StratifiedGroupKFold` with `group=patient_id` (seed=42). Patient-aware splitting is mandatory — many patients have multiple lesions, so naive splits leak identities across folds.

**Verified CV AUC (from OOF files):**
- Regular model: **0.9503 ± 0.0092**
- EMA model: **0.9484 ± 0.0128**

EMA and regular checkpoints perform comparably (EMA shows slightly higher fold variance). Use the EMA ensemble submission as the primary candidate.

> Note: Local AUC is a development proxy. The leaderboard scores pAUC above 80% TPR, which is a stricter and lower-magnitude metric.

---

## Patient-Relative ("Ugly Duckling") Features

Inspired by the 1st-place solution. For each lesion, computes deviation from the patient's own lesion population:

- **Z-scores, ratios, and absolute differences** vs. patient mean/std for: `tbp_lv_areaMM2`, `tbp_lv_perimeterMM`, `tbp_lv_deltaB`, `tbp_lv_color_std_mean`, `tbp_lv_radial_color_std_max`, `tbp_lv_norm_color`, `color_variance`.
- **LOF score** (Local Outlier Factor) per patient: flags lesions that are outliers among a patient's own lesions. Improved CV pAUC from 0.18149 → 0.18185.

These features capture the "ugly duckling" sign — a lesion that looks atypical relative to a patient's baseline is a stronger malignancy signal than absolute appearance alone.

---

## Key Artifacts Per Fold

Saved at training start (crash-safe) and required for inference:

| File | Purpose |
|---|---|
| `best_model_fold{N}.pth` | Best regular checkpoint |
| `best_model_ema_fold{N}.pth` | Best EMA checkpoint |
| `scaler_fold{N}.pkl` | `StandardScaler` fitted on fold's train split |
| `encoders_fold{N}.pkl` | Categorical encoders |
| `patient_statistics_fold{N}.pkl` | Per-patient stats for relative features |
| `precomputed_features_fold{N}.pkl` | Precomputed patient-relative + LOF features |
| `feature_info_fold{N}.json` | Feature names and metadata |
| `oof_fold{N}.csv` / `oof_ema_fold{N}.csv` | Out-of-fold predictions |

---

## Submission

Submit `ensemble_ema_average_submission.csv` (average of 5 EMA fold predictions). The regular ensemble is the fallback. Verify prediction distribution is not degenerate before uploading.
