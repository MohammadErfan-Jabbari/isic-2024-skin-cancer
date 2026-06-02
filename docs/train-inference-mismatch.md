# Train/Inference Preprocessing Mismatch — War Story

> This is the worst failure in the project. A stacking submission scored ~0.48 pAUC@80%TPR
> when the expected range was ~0.16–0.18. Four simultaneous bugs in the inference script
> explain the entire gap.

---

## Background: What pAUC@80%TPR Scores Mean

The competition metric is **partial AUC above 80% TPR** — it measures ranking quality only in the high-sensitivity regime, not overall AUC. The project's local AUC proxy (on full ROC) sits at 0.93–0.98; that is a different scale. On the pAUC@80%TPR scale, **0.16–0.18 is normal and competitive**. A score of **0.48 is broken**: it implies the model is systematically inverting rank order (higher predicted probability → more likely benign), consistent with a preprocessing transformation that corrupts the ordinal signal.

---

## The Four Simultaneous Mismatches

Training was done in `16_3_train_stacking.py`. Inference was done in `16_5_submission_stacking.py`. They diverged in four ways at once.

| # | Aspect | 16_3 Training | 16_5 Inference | Severity |
|---|---|---|---|---|
| 1 | Vision prediction normalization | Z-score (`(x − μ) / σ` using saved training stats) | Reference-based **rank normalization** | CRITICAL |
| 2 | `exclude_cols` list | 15 columns (includes all leakage columns) | 6 columns — missing `mel_thick_mm`, `mel_mitotic_index`, `iddx_1`–`iddx_5`, `iddx_full` | HIGH |
| 3 | Final output | Raw GBDT probability | **Rank-normalized** again | HIGH |
| 4 | Standardization stats | Saved to `standardization_stats.pkl` | **Not loaded** — stats recomputed from test data | CRITICAL |

### Mismatch 1 — Vision normalization

Both EVA02 and EdgeNeXt predictions were z-score standardized during training using per-column mean/std derived from training OOF predictions. At inference, the script applied rank normalization instead (mapping each prediction to its percentile within the test batch). Because test predictions are shifted +2.3–2.5σ above training predictions (see `distribution-shift.md`), rank normalization maps them to extreme high percentiles — then feeds those extreme values into a model that was trained on z-scored features near zero. The feature space mismatch is total.

### Mismatch 2 — Incomplete exclude_cols

The 9 missing exclusions (`mel_thick_mm`, `mel_mitotic_index`, `iddx_1`–`iddx_5`, `iddx_full`) are the hard-leakage columns identified in the metadata audit. While their near-100% missingness means most test rows see no value, the inconsistency means inference and training operate on feature matrices of different shapes and semantics.

### Mismatch 3 — Double rank-normalization of output

The final `submission["target"]` was rank-normalized before writing. This step converts raw probabilities to uniform-distribution percentile ranks, which destroys calibration and, combined with the inverted feature space from mismatch 1, produces scores near 0.5 (random) or worse.

### Mismatch 4 — Standardization stats not loaded

Training saves per-feature means and standard deviations to `standardization_stats.pkl`. The inference script recomputed these statistics from the test set. Because the test distribution differs from training (shifts up to 0.89σ on raw metadata features), this produces inconsistent standardization — the model sees features in a different range than it was trained on.

---

## Symptom

Kaggle leaderboard: **~0.48 pAUC@80%TPR** (higher is better on this metric; random = ~0.5, good = ~0.16–0.18).

The score is near 0.5, consistent with a model that has lost all discriminative signal due to corrupted feature space.

---

## Root Cause in One Sentence

The inference script was written independently of the training script rather than reusing the saved preprocessing artifacts, and the normalization method differed in a direction that compounded the existing train→test distribution shift.

---

## The Fix

Implemented in `stacking/src/inference_stacking_corrected.py` (source: `post_feature_analysis/audit/16_5_submission_stacking_corrected.py`):

1. **Load `standardization_stats.pkl`** — do not recompute from test data.
2. **Apply z-score normalization** — `(pred − train_mean) / train_std` for all vision predictions.
3. **Full `exclude_cols`** — all 15 columns from training, including all leakage columns.
4. **Output raw GBDT probabilities** — no rank normalization of the final submission column.
5. **Reuse saved `LabelEncoder` objects** — do not re-fit on test data.

Expected post-fix score: **0.16–0.18 pAUC@80%TPR**, consistent with OOF validation and the dual-backbone hybrid model.

---

## Checklist: Train/Inference Parity (apply to any new inference script)

- [ ] Load scaler/stats from training artifacts — never recompute from test.
- [ ] Normalization method matches training (z-score, not rank).
- [ ] `exclude_cols` list is identical to training.
- [ ] Final output is raw probability — no post-hoc rank transform.
- [ ] Categorical encoders loaded from training pickles.
- [ ] Feature matrix shape matches training (same columns, same order).
