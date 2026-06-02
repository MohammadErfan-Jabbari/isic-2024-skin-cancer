# Generative Augmentation — ISIC 2024

Synthetic malignant images were generated to address extreme class imbalance (~1:1168).
The pipeline is fully reproducible via `generative/scripts/01–09`.

---

## Motivation

343 real malignant images in the training set (out of ~400 K). Standard oversampling
(random duplication, SMOTE) is ill-suited to high-dimensional image data. SD-1.5
fine-tuning provides diversity while staying in-distribution with real dermoscopic images.

---

## Nine-Step Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_analyze_metadata.py` | Characterise class imbalance, feature distributions |
| 2 | `02_extract_positive_samples.py` | Decode 343 malignant images from HDF5 → 128×128 JPEG (LANCZOS) |
| 3 | `03_validate_extracted_data.py` | Sanity-check extracted crops and caption JSONL |
| 4 | `04_prepare_training_dataset.py` | Build Dreambooth-style dataset with captions |
| 5 | `05_train_sd_finetune.py` | Full SD-1.5 fine-tune (see below) |
| 6 | `07_generate_images.py` | Sample 10,001 images from the fine-tuned model |
| 7 | `08_filter_images.py` | Classifier-based quality filter — keep top 6,000 |
| 8 | `09_pack_synthetic_hdf5.py` | Upscale survivors 128→384 px (LANCZOS); pack to HDF5 |
| — | `00_prepare_synthetic_metadata.py` (dual_backbone/) | Assign enriched metadata to the 6,000 survivors |

`06_visualize_sd_training.py` is a diagnostic script moved to `generative/tools/`; it
is not part of the production pipeline.

---

## SD-1.5 Fine-Tuning Details

- **Base checkpoint:** "ThisIsReal" (SD 1.5 community fine-tune on photorealistic images)
- **Strategy:** Full fine-tuning (all layers, no frozen blocks)
- **Resolution:** 128×128
- **Batch size / Epochs:** 8 / 50
- **Precision:** fp16 training; UNet weights loaded as float32 first for numerical stability,
  then cast — avoids NaN initialisation with some fp16 checkpoints
- **Input:** 343 real malignant crops + text captions generated from clinical metadata

---

## Filter Mechanism — Correct Description

> **The "P(malignant) > 0.7" claim in earlier project docs is incorrect.**

The actual filter (`08_filter_images.py`):

1. Score all 10,001 generated images with a trained **EfficientNetV2-S hybrid classifier**
   (stage 11 checkpoint: `results/v2s_features_20251110_155122/best_model.pth`).
2. Load the classifier's saved `preprocessors.pkl` (StandardScaler + categorical encoders)
   and apply the **exact same `engineer_features()` function** used during that model's
   training — preserving train/inference preprocessing parity.
3. Supply each image with **"average malignant metadata"** as dummy tabular input: the
   mean of numerical features and mode of categorical features across all real malignant
   training rows. This ensures the image content drives the score, not metadata.
4. Sort all 10,001 images by classifier score descending; **keep the top 6,000**.
   A soft lower bound (`--threshold 0.15`, the average real-malignant probability) is
   applied first, but in practice enough images exceeded it — the effective selection
   criterion is rank, not a hard probability cut. The minimum probability among the 6,000
   selected images was approximately 0.29.

This is a **top-k selection** strategy, not a fixed-probability gate.

---

## Outputs

| Artifact | Path | Notes |
|----------|------|-------|
| Filtered 384px HDF5 | `generative/data/synthetic_malignant_384.hdf5` | ~6,000 images, LZF-compressed, git-ignored |
| Enriched metadata CSV | `generative/synthetic_malignant_metadata_enriched.csv` | 6,000 rows, git-ignored (large) |
| Selection log | `generative/data/synthetic_malignant_filtered/selection_log.csv` | Per-image scores + selection status |

---

## Downstream Impact — Unverified

The dual-backbone model (CV AUC 0.9503 ± 0.0092) was trained **with** synthetic data
included, but **no without-synthetic baseline was run** for the same architecture.
Attributing any CV/LB gain to synthetic augmentation is therefore speculative.

### Negative Result (Stage 15 — archived)

An earlier attempt injected synthetic images directly into EfficientNetV2-S fine-tuning
without the classifier filter. That submission saw LB pAUC drop from ~0.932 to ~0.644 —
a severe degradation, likely from low-quality or off-distribution synthetic images
contaminating the training signal. This failure motivated building the classifier-based
filter before the dual-backbone stage.

---

## Key Design Choices and Their Rationale

| Choice | Why |
|--------|-----|
| 128px generation → 384px upscale | Low-res SD training is faster and avoids tile artefacts; LANCZOS upscale is acceptable for classification (not segmentation) |
| Full SD fine-tune (not LoRA) | Maximises stylistic alignment with dermoscopic appearance given small positive class |
| Average-malignant dummy metadata | Image content should drive filter score, not synthetic metadata quality |
| Top-6,000 rank cut | Empirically set; calibrated to match roughly 6 × real-malignant count (343 × ~17.5) |
| Reuse exact preprocessors.pkl | Prevents the preprocessing-mismatch failure pattern documented in `docs/architecture-dual-backbone.md` |
