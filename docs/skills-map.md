# Skills Map — ISIC 2024 Skin Lesion Malignancy Prediction

**Project:** Kaggle ISIC 2024 — predicting malignancy from dermoscopic images + clinical metadata  
**Task difficulty:** Extreme class imbalance (~1:1168), partial AUC above 80% TPR as metric  
**Outcome:** Dual-backbone hybrid — 0.9503 ± 0.0092 CV AUC (5-fold, verified from OOF); stacking LB 0.98997 (scale-ambiguous, self-reported)

---

## Headline Capabilities

- **Imbalance at scale:** 0.9503 ± 0.0092 CV AUC on ~1:1168 malignant:benign ratio using 1:1 balanced sampling + Focal Loss + per-patient LOF; independently converged on the 1st-place solution's core strategy.
- **Multimodal fusion:** Dual-backbone end-to-end (EVA02-ViT + EdgeNeXt-ConvNet + metadata MLP → fusion head) trained on 300 K+ images with AMP + EMA; all gradients flow jointly.
- **Data quality obsession:** 6-phase metadata audit + leakage exclusion of post-biopsy columns; root-cause investigation of preprocessing mismatch that shifted pAUC from ~0.16 to 0.48 (broken).
- **Patient-aware validation:** StratifiedGroupKFold(group=patient_id) throughout; per-patient LOF captures "Ugly Duckling Sign" from domain literature.
- **Reproducibility under pressure:** Deterministic fold reconstruction via saved seed + `recover_scalers.py`; detected +2.31σ EVA02 train→test distribution shift (K-S p ≪ 1e-26).

---

## Deep Learning / Computer Vision

- **Vision Transformer fine-tuning** — EVA02-Small (`eva02_small_patch14_336`, timm) end-to-end with AdamW + ReduceLROnPlateau (`dual_backbone/01_train.py`)
- **ConvNet backbone integration** — EdgeNeXt-Base (`edgenext_base.in21k_ft_in1k` @ 384px) concatenated with ViT embeddings for complementary feature diversity (`dual_backbone/01_train.py`, `dual_backbone/models.py`)
- **Dual-resolution preprocessing** — Separate 336px (EVA02) and 384px (EdgeNeXt) streams from a single 384px HDF5 source (`dual_backbone/models.py`, `tools/isic_model.py`)
- **End-to-end multimodal fusion** — EVA02 [384-dim] + EdgeNeXt [584-dim] + metadata [64-dim] → fusion MLP (512→128→1); jointly optimised (`dual_backbone/01_train.py`)
- **AMP + GradScaler** — `torch.autocast` + `GradScaler` for 2× speedup; NaN guard via `torch.where(isnan, zeros)` in MetadataEncoder (`dual_backbone/01_train.py`)
- **EMA (decay=0.9999)** — Separate EMA checkpoint alongside regular weights; EMA ensemble submitted to Kaggle (`dual_backbone/01_train.py`, `dual_backbone/03_generate_submissions.py`)
- **Focal Loss (α=0.25, γ=2.0)** — Custom PyTorch implementation on top of BCE; combined with 1:1 balanced sampling (`dual_backbone/01_train.py`)
- **LayerNorm in metadata encoder** — Replaces BatchNorm for stability with variable/small effective batch sizes during fusion (`dual_backbone/models.py`)

---

## Tabular / Classical ML & Ensembling

- **XGBoost meta-learner** — Trained on vision logits + top-50 PCA embeddings + raw metadata; 0.990 AUC on Golden Split (`stacking/17_1_train_stacking_gbdt.py`)
- **MLP meta-learner** — Trained on metadata + DAE latents + vision embeddings; averaged with XGBoost for final stacking prediction (`stacking/`)
- **PCA dimensionality reduction** — 50-component PCA on vision embeddings before stacking; empirically beats logits-only input (`stacking/`, `dual_backbone/models.py`)
- **CatBoost evaluated and excluded** — ~0.945–0.960 AUC, underperformed XGBoost on Golden Split; dropped to avoid diluting the strong XGB/MLP ensemble (`stacking/`)
- **StandardScaler fit/transform discipline** — Scaler fit on training fold, serialised as pickle, applied identically at inference; failure to do this was the root cause of the 0.48 pAUC broken submission (`tools/recover_scalers.py`, `archive/postmortem_audit_scripts/`)
- **Ensemble averaging** — Fold-wise averaging (5 folds) + EMA vs. regular; EMA ensemble is the final submission (`dual_backbone/03_generate_submissions.py`)

---

## Imbalanced Learning

- **1:1 balanced batch sampler** — Custom `BalancedBatchSampler` ensures equal real:malignant per batch; outperforms Focal Loss alone on this dataset (`dual_backbone/01_train.py`)
- **Stratified group k-fold** — `StratifiedGroupKFold(n_splits=5, group=patient_id)`; same patient never straddles train/val (`dual_backbone/01_train.py`, all training scripts)
- **Synthetic malignant oversampling** — SD-1.5 fine-tuned on 343 real malignant crops → 10,001 generated → top-6,000 by EfficientNetV2-S classifier score kept (`generative/scripts/`)
- **Patient-relative features (Ugly Duckling Sign)** — Z-scores and ratios of each lesion against its patient's distribution; adopted from 1st-place solution (`dual_backbone/01_train.py`)
- **Local Outlier Factor per patient** — `sklearn.neighbors.LocalOutlierFactor` (n_neighbors=10) on 7-feature subset; identifies outlier lesions within a patient (`dual_backbone/01_train.py`)
- **Golden Split (fold exclusion)** — Fold 4 consistently degraded other folds; excluding it from training lifted the ceiling to 0.990 AUC (`stacking/`, `docs/model-evolution.md`)

---

## Validation, Leakage & Experimental Rigor

- **Post-biopsy leakage exclusion** — `mel_thick_mm`, `mel_mitotic_index`, `iddx_*` columns excluded everywhere; ~100% of non-missing values are malignant (`tools/feature_list.txt`)
- **Train/inference preprocessing parity** — Identical `engineer_features()`, identical scaler/encoder artifacts, raw output probabilities (no rank normalisation at inference); 6 instances of mismatch caused the worst failure in the project (`archive/postmortem_audit_scripts/16_5_submission_stacking_corrected.py`)
- **Distribution shift detection** — EVA02 test mean +2.31σ above train (K-S p ≪ 1e-26); EdgeNeXt +2.49σ; explains why rank normalisation of vision predictions is dangerous (`archive/postmortem_audit_scripts/`)
- **GradScaler artifact detection** — `dual_hybrid_v1` saved PyTorch `GradScaler` objects in place of `StandardScaler` pickles; `recover_scalers.py` reconstructs and overwrites them (`tools/recover_scalers.py`)
- **OOF integrity audits** — `check_oof_integrity.py`, `verify_scaler.py`, `compare_oofs.py`; ad-hoc but systematic validation scripts (`tools/`)
- **pAUC vs AUC distinction** — Competition metric is partial AUC above 80% TPR; local AUC (0.93–0.99) used as proxy only; never conflated in decision-making

---

## Generative Modeling

- **SD-1.5 full fine-tuning** — All layers trained on 128×128 malignant crops, 50 epochs, batch 8, fp16, float32 UNet load for numerical stability (`generative/scripts/05_train_sd_finetune.py`)
- **Top-k classifier filter** — 10,001 generated images scored by EfficientNetV2-S hybrid; top 6,000 by score kept (rank-based, not a fixed probability threshold) (`generative/scripts/08_filter_images.py`)
- **Train/inference parity for synthetic filter** — Reuses `preprocessors.pkl` (StandardScaler + encoders) and exact `engineer_features()` from the classifier's training; dummy metadata = average real-malignant profile so image content drives the score (`generative/scripts/08_filter_images.py`, `generative/FILTERING_COMPATIBILITY_ANALYSIS.md`)
- **Metadata enrichment for synthetic samples** — Assigns realistic clinical metadata to 6,000 survivors (`dual_backbone/00_prepare_synthetic_metadata.py`)
- **Negative result documented** — Unfiltered synthetic + EfficientNetV2-S (archive stage 15) dropped LB pAUC from ~0.932 to ~0.644; motivated the classifier-filter approach (`archive/`)

---

## MLOps / Engineering

- **HDF5 image store** — LZF-compressed, keyed by `isic_id`, O(1) random access across 300 K+ samples; separate 224px and 384px variants (`tools/preprocess_hdf5*.py`, `tools/isic_model.py`)
- **UV dependency management** — Reproducible PyTorch/timm/scikit-learn stack; `uv run` resolves to root `pyproject.toml` (`pyproject.toml`)
- **Per-fold artifact organisation** — Checkpoints, scalers, encoders, OOF predictions, feature metadata saved and loaded consistently per fold (`dual_backbone/`, `tools/recover_scalers.py`)
- **Gradient accumulation** — `--accumulation-steps` flag for effective batch scaling without GPU OOM (`dual_backbone/01_train.py`)
- **Deterministic splits** — Fixed seeds + deterministic fold reconstruction; reproducibility verified by `recover_scalers.py`

---

## Scientific Communication

- **6-phase metadata audit** — Systematic missingness, leakage, distribution-shift, and categorical-consistency investigation (`metadata_investigation/`)
- **Root-cause postmortem** — `archive/postmortem_audit_scripts/` documents the stacking failure end-to-end: broken submission → 6 root causes identified → corrected implementation with side-by-side comparisons
- **Distribution shift quantification** — K-S statistics and σ-magnitudes for vision predictions and raw features; actionable guidance on when to distrust normalisation tricks (`archive/postmortem_audit_scripts/`)
- **Model evolution log** — 6-stage timeline from random-level baselines to dual-backbone with per-stage AUC and lessons (`docs/model-evolution.md`)
- **Live leaderboard research log** — `stacking/research_log_lb.md` documents real-time strategy, fold exclusion decisions, and final submission selection

---

## Scope & Limitations

- **No synthetic ablation:** CV AUC 0.9503 ± 0.0092 was measured with synthetic data included; no without-synthetic baseline exists for the dual-backbone architecture, so the contribution of generative augmentation is unquantified.
- **LB metric ambiguity:** Stacking LB figures (0.98997 / 0.98245) are self-reported and the metric scale is ambiguous (pAUC vs. full AUC). Do not compare directly against the CV AUC figures.
- **Single competition dataset:** All design choices are optimised for ISIC 2024 (dermoscopy + tabular metadata); generalisation to other imaging modalities or distributions is untested.
