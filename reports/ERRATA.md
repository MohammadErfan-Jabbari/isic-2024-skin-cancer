# Errata — post-submission corrections to the presentation

The Slidev deck in `presentation/` was prepared during the project. A later verification pass
(recomputing metrics from the raw on-disk prediction files) found two figures that need correction.
The slides are preserved as-is for honesty about what was originally presented; this note records the
corrections. The repository's `docs/` always reflect the corrected, verified numbers.

## 1. Dual-backbone CV AUC: 0.9612 → **0.9503 ± 0.0092**
Any slide citing a dual-backbone cross-validation AUC of **0.9612 ± 0.0038** is superseded. Recomputing
AUC from `results/dual_hybrid_v2/oof_fold{1..5}.csv` (all 5 folds, full imbalanced held-out data)
gives **0.9503 ± 0.0092** (regular weights) and **0.9484 ± 0.0128** (EMA). See
[`../docs/model-evolution.md`](../docs/model-evolution.md).

## 2. Stacking "pAUC" axis labels are metric-scale-ambiguous
Figures whose axes read **"Public LB Score (pAUC)"** with values near **0.98–0.99** (e.g. the stacking
performance and Golden-Split charts) mislabel the scale. The competition's pAUC@80%TPR for this project
is on the order of **0.16–0.18**; values near 0.99 are almost certainly a full-AUC or rescaled
leaderboard variant, not pAUC@80%TPR. Read those values as **self-reported leaderboard scores of
unconfirmed metric scale**, not as verified pAUC. See
[`../docs/train-inference-mismatch.md`](../docs/train-inference-mismatch.md).
