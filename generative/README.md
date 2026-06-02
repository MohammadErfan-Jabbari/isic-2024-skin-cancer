# Generative Augmentation — Runbook

Synthetic malignant images for ISIC 2024 training. See `docs/generative-augmentation.md`
for design rationale and a discussion of limitations.

---

## Prerequisites

- Real training HDF5: `data/train-image.hdf5` (keys = `isic_id`)
- Real training metadata: `data/train-metadata.csv`
- Trained EfficientNetV2-S classifier checkpoint for filtering:
  `results/v2s_features_20251110_155122/best_model.pth` + `preprocessors.pkl`
- SD-1.5 base weights (downloaded by `05_train_sd_finetune.py` on first run)

---

## Script Run Order

Run from the repo root with `uv run python generative/scripts/<script>`.

```
01_analyze_metadata.py          # inspect class distribution, feature stats
02_extract_positive_samples.py  # export 343 malignant images → 128px JPEGs + captions
03_validate_extracted_data.py   # verify extraction counts and caption JSONL
04_prepare_training_dataset.py  # build Dreambooth-compatible dataset folder
05_train_sd_finetune.py         # full SD-1.5 fine-tune (50 epochs, fp16, ~8–12h on A100)
07_generate_images.py           # sample 10,001 images from fine-tuned model
08_filter_images.py             # score with EfficientNetV2-S; keep top 6,000
09_pack_synthetic_hdf5.py       # upscale 128→384px (LANCZOS); pack to HDF5
```

After packing, run `dual_backbone/00_prepare_synthetic_metadata.py` to assign enriched
clinical metadata to the 6,000 survivors.

### Filter command reference

```bash
uv run python generative/scripts/08_filter_images.py \
  --model-dir results/v2s_features_20251110_155122/ \
  --topk 6000 \
  --gpu 0
```

The `--threshold` flag (default 0.15) is a soft pre-filter; selection is rank-based —
the top 6,000 by classifier score are kept regardless of the threshold value.

---

## Outputs (git-ignored, large files)

| File | Description |
|------|-------------|
| `generative/data/synthetic_malignant_384.hdf5` | 6,000 upscaled synthetic images, LZF-compressed |
| `generative/synthetic_malignant_metadata_enriched.csv` | Enriched metadata for 6,000 survivors |
| `generative/data/synthetic_malignant_filtered/selection_log.csv` | Per-image classifier scores |

> **Honest caveat.** No ablation (training *with* vs *without* synthetic data) was run on the
> dual-backbone architecture. The downstream effect of this augmentation on CV/LB is therefore
> **unquantified** — see `docs/generative-augmentation.md`. An earlier synthetic experiment on an
> EfficientNetV2-S backbone (archive stage 15) actively *hurt* the leaderboard.

---

## Diagnostic Tool

`generative/tools/06_visualize_sd_training.py` — plots training loss curves from the
SD fine-tuning logs. Not part of the production pipeline; run manually if needed.
