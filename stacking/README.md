# Two-Stage Stacking — Runbook

A two-stage pipeline: vision backbones produce OOF logits + embeddings (Stage 1), which feed a per-fold XGBoost + MLP ensemble alongside raw metadata and DAE latents (Stage 2). Outputs are averaged for the final prediction.

---

## Pipeline Overview

```
Stage 1 — Vision (EVA02-Small + EdgeNeXt-Base, 5-fold CV)
  └── OOF logits + penultimate embeddings

Stage 2 — Stacking (per-fold)
  ├── XGBoost:  [metadata (38 features) + vision logits + PCA-50 embeddings]
  └── MLP:      [metadata + DAE latents + vision logits + PCA-50 embeddings]
                └── Average → final probability
```

---

## Run Order

```bash
cd /path/to/isic-2024-skin-cancer

# 1. Create reproducible patient-aware folds
uv run python stacking/src/create_folds.py

# 2. Train vision stage — generates OOF logits + embeddings
uv run python stacking/src/train_vision.py [--fold N] [--gpu 0] [--batch-size 32]

# 3. Train metadata DAE (unsupervised, train+test metadata)
uv run python stacking/src/train_dae.py

# 4. Extract and compress embeddings (PCA-50)
uv run python stacking/src/extract_embeddings.py

# 5. Train stacking models (XGBoost + MLP, per fold)
uv run python stacking/src/train_stacking.py

# 6. Inference — use the corrected script (preprocessing-safe)
uv run python stacking/src/inference_stacking_corrected.py
```

---

## Key Design Decisions

### Golden Split
Train stacking on folds 0–3 only; **exclude fold 4**. Fold 4 is empirically noisy ("toxic"): including it in training consistently degrades the other folds' performance. The XGBoost + MLP model trained on this 4-fold split is the SOTA ceiling for this track.

### 38-Feature Safe List (`feature_list.txt`)
`feature_list.txt` (in this directory) enumerates the 38 vetted tabular features. Three categories of columns are unconditionally excluded everywhere — train and inference:
- **Post-biopsy leakage:** `mel_thick_mm`, `mel_mitotic_index` (~100% of non-missing values are malignant).
- **Diagnosis codes:** `iddx_1` through `iddx_5`, `iddx_full`.
- `tbp_lv_dnn_lesion_confidence` (drop; `tbp_lv_nevi_confidence` is kept).

Using any excluded column will silently inflate training metrics and collapse test performance.

### Vision-to-GBDT Interface
The Phase-0 science (see `experiments/`) found that passing logits + top-50 PCA components of the penultimate embeddings outperforms logits alone. XGBoost receives this hybrid feature set; the MLP additionally receives DAE latents. DAE latents improve the MLP but do **not** help tree models.

### Preprocessing Parity (Critical)
The worst failure in this project was a train/inference preprocessing mismatch — z-score normalization during training vs. rank normalization at inference — which produced a 0.48 LB score (broken) vs. the expected ~0.16–0.18 pAUC range. `inference_stacking_corrected.py` enforces the fix:
- Load `standardization_stats.pkl` saved during training.
- Apply z-score with training mean/std, not rank normalization.
- Use identical `feature_list.txt` exclude list.
- Output raw probabilities (no rank transform on output).

---

## `experiments/` — Phase-0 Science

`experiments/01_vision_head.py`, `02_metadata_dae.py`, `03_gbdt_interface.py` are small-scale ablations that were run on **simulated/synthetic embeddings** (not real trained-vision outputs) on a 10% data subset. Their conclusions — e.g., that DAE latents help the MLP but not trees, or that PCA-50 of embeddings beats logits alone — are directionally informative but **not conclusively proven** on the full pipeline. Treat them as hypotheses that informed the architecture choices, not validated benchmarks.

---

## Reported Scores

Self-reported Public LB: 0.98997 / 0.98245. The metric scale for these figures is ambiguous (LB may report a rescaled pAUC variant); treat them as relative indicators, not ground-truth pAUC@80%TPR values.
