# Archive — the evolution, kept honest

This directory is the **read-only history** of the project: the chronological path from a ~0.51-AUC
scratch CNN to the two endgame architectures (`../dual_backbone/`, `../stacking/`). It is kept on
purpose — the learning arc, including the dead ends, is part of the story. **None of this is the
recommended way to run the project today**; see the top-level `README.md`.

Numbers below are *self-reported* from the original scripts/notebooks (development-time local AUC,
or public-LB pAUC where noted). They are development proxies — see `../docs/domain-rules.md` on why
local AUC ≠ the competition pAUC@80%TPR.

| Era (folder) | Stages | What it explored | Best result (self-reported) |
|---|---|---|---|
| `01_baselines/` | 1–2 | Scratch SimpleCNN on images only; + random augmentation | ~0.51 AUC (≈ random under imbalance) |
| `02_hybrid/` | 3 | First image **+ metadata** fusion; Focal Loss (α=0.25, γ=2.0) | ~0.936 AUC — metadata was the single biggest jump |
| `03_transfer/` | 4–5 | Transfer learning: EfficientNetV2-S (~0.951) vs ResNet34 (~0.938) | ~0.951 AUC (V2-S) |
| `04_ensemble_kfold/` | 6–9 | Weighted ensembles, TTA, 5-fold CV, EfficientNetV2-M | diversity/CV gains |
| `05_feature_eng/` | 10–11 | Engineered metadata (age/size/color/shape composites); ConvNeXt, ViT; k-fold + EMA/SWA | CV ~0.9468 ± 0.0117 |
| `06_gbdt_stacking_v1/` | 12, 14–17 | First GBDT stacking on OOF preds; the stage-16/17 stacking precursor (incl. the **broken** `16_5` and its diagnosis) | the tabular endgame takes shape |
| `07_retrospective/` | 13 | "Rising from ashes" — a pure debrief notebook that scanned all prior runs and decided to switch backbones | (analysis only) |
| `08_eva_edgenext_intro/` | 14 | First use of **EVA02-Small + EdgeNeXt-Base** — the backbones that carried to both endgames | vision-only LB pAUC ~0.93 |
| `09_legacy_modules/` | — | Superseded shared code (`modules/`, `18_model_utils.py`) kept for provenance | n/a |
| `misc_analysis/` | — | One-off debugging scripts from the stacking-failure investigation | n/a |
| `metadata_audit_scripts/` | — | The 6-phase metadata audit scripts (findings distilled in `../docs/metadata-audit.md`) | n/a |
| `postmortem_audit_scripts/` | — | The train/inference-mismatch postmortem scripts (distilled in `../docs/train-inference-mismatch.md`) | n/a |

## Notable dead ends (deliberately preserved — negative results are results)
- **Synthetic + EfficientNetV2-S** (stage 15): injecting SD-generated images into V2S training *hurt*
  the leaderboard (≈0.932 → 0.644). The generative idea was later salvaged with classifier filtering
  on the stronger EVA02/EdgeNeXt backbones — see `../generative/` and `../docs/generative-augmentation.md`.
- **Rank-normalization at inference** (stage 14/16): caused the worst failure in the project
  (stacking scored ~0.48 instead of the expected ~0.16–0.18 pAUC). Root-caused and fixed — see
  `../docs/train-inference-mismatch.md`.
- **Heavy feature engineering**: helped the noisy folds but *hurt* the clean "Golden Split" — the final
  models use simpler features (`../docs/domain-rules.md`).
