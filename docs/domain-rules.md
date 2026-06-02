# Domain Rules — ISIC 2024

Seven non-negotiable rules derived from failures in this project. Each caused a silent, severe
performance collapse when violated. They are ordered by severity of the failure they produced.

Cross-references: [metadata-audit.md](metadata-audit.md) | [train-inference-mismatch.md](train-inference-mismatch.md) | [distribution-shift.md](distribution-shift.md)

---

## Rule 1 — Exclude post-biopsy leakage columns everywhere (train AND inference)

**Drop:** `mel_thick_mm`, `mel_mitotic_index`, and all diagnosis codes `iddx_1`–`iddx_5`,
`iddx_full`.

**Evidence:**
- `mel_thick_mm`: 99.99% missing (400,908/400,959 rows); of the 51 non-missing rows, **51/51 are
  malignant** (target=1). GBDT importance spike to 413.4 confirmed it was being used as a cheat
  feature. Source: [metadata-audit.md](metadata-audit.md), Phase 1.3 + Phase 2.
- `mel_mitotic_index`: 99.99% missing; **43/43 non-missing are malignant**. Same post-biopsy
  mechanism. Source: [metadata-audit.md](metadata-audit.md), Phase 1.2 / postmortem verification.
- `iddx_*` columns: explicit diagnosis codes — directly encode the label.

The vetted safe feature list is `last_run/feature_list.txt` (38 features). Note:
`tbp_lv_nevi_confidence` is retained; `tbp_lv_dnn_lesion_confidence` is dropped (overlap
analysis showed it is the riskier of the two).

---

## Rule 2 — Splits must be patient-aware

Use `StratifiedGroupKFold(n_splits=5, group=patient_id)`. Never let the same patient appear in
both train and validation.

**Evidence:** Patients in this dataset have 28–9,184 lesions each (mean 1,191). A random split
without grouping would put multiple lesions from the same patient in both train and val, directly
leaking patient-level appearance/risk information. Fold splits must use a fixed seed to be
reproducible; `recover_scalers.py` depends on reconstructing them exactly. Source:
[metadata-audit.md](metadata-audit.md), Phase 1 dataset characteristics.

---

## Rule 3 — Train/inference preprocessing must match exactly

Use the **same normalization** (z-score with **saved** training mean/std from
`standardization_stats.pkl`), the **same exclude list**, and output **raw probabilities** (no
rank-normalization of outputs).

**Evidence:** This caused the worst single failure in the project. The broken `16_5` stacking
submission scored **~0.48 pAUC** (expected ~0.16–0.18) because: (1) training used z-score
normalization but inference applied rank normalization — 6 code locations; (2) `exclude_cols` was
not applied at inference; (3) the final output was rank-normalized again. Fix: load
`standardization_stats.pkl` and apply training mean/std at inference. Reference:
`post_feature_analysis/audit/16_5_submission_stacking_corrected.py`. Source:
[train-inference-mismatch.md](train-inference-mismatch.md).

---

## Rule 4 — Vision predictions shift hard from train to test; do not rank-normalize them

Treat vision logits/probabilities as **raw features** when feeding them into the GBDT stage.

**Evidence:** EVA02 test-set mean is **+2.31σ** above its train-set mean (0.0053 → 0.1184;
K-S test p=8.66e-64). EdgeNeXt: **+2.49σ** (0.0082 → 0.1585; K-S test p=3.92e-26). Rank
normalization maps these shifted predictions into extreme percentiles, and then double-ranking
compounds the distortion. Raw metadata features are much milder (<1σ for all: `tbp_lv_H` 0.89σ,
`perimeterMM` 0.59σ, `deltaB` 0.54σ, `B` 0.48σ). Source: [distribution-shift.md](distribution-shift.md),
Phase 4.

---

## Rule 5 — Use the "Golden Split": train folds {0,1,2,3}, exclude fold 4

Fold 4 contains noisy/out-of-distribution samples. Including it in training degrades all other folds.

**Evidence (Public LB pAUC, self-reported):**
- Model trained on folds {0,1,2,3}, validated on fold 4: **0.990** (best)
- Any model trained on a set that *includes* fold 4: drops to 0.947–0.972

Including fold 4 hurt EdgeNeXt most (fold-2 score dropped to 0.774 when fold 4 was in training).
The 0.990 XGB/MLP model (fold-4 validation, no feature engineering) is the SOTA ceiling for the
stacking track. Source: `last_run/research_log_lb.md`.

---

## Rule 6 — More features does not imply better performance

Use the 38-feature safe list; do not add heavy ABCDE/manual composites to the Golden Split model.

**Evidence:** Extensive ABCDE feature engineering helped the noisy folds (0–2) but
**hurt** the clean Golden Split (folds {0,1,2,3}). Similarly, CatBoost (CV ~0.945–0.960)
underperformed XGBoost on the Golden Split and pulled down ensemble scores when mixed in.
The mechanism: on a clean, well-calibrated fold, additional features introduce noise rather than
signal; the model is already near its ceiling on the informative features. Source:
[model-evolution.md](model-evolution.md), Stage 5 / Stage 8.

---

## Rule 7 — `tbp_lv_y` is a body-location proxy; `tbp_lv_*_confidence` features are safe

**`tbp_lv_y`:** its very high GBDT importance (789.8 in some runs) does not reflect a novel
medical signal — it correlates with `anatom_site_general` (ANOVA F=341,202, p<0.001). It
encodes vertical position in the TBP imaging frame, which is a proxy for body-site. Use it, but
interpret SHAP/importance plots accordingly. Source: [metadata-audit.md](metadata-audit.md), Phase 3.

**TBP `*_confidence` features:** these are outputs of the TBP imaging system computed at scan
time, **before** any pathology result. They are not post-diagnosis. The overlap analysis
confirmed no perfect class separation. `tbp_lv_nevi_confidence` is kept; `tbp_lv_dnn_lesion_confidence`
is dropped as a precaution. Source: [metadata-audit.md](metadata-audit.md), Phase 1.3.
