# Data

This repository ships **code, documentation, and small artifacts only**. The imagery and clinical
metadata are **not redistributed** — they belong to the ISIC 2024 challenge and are governed by the
competition's terms of use.

## What you need

The project was developed against an **ISIC 2024** ("Skin Cancer Detection with 3D-TBP") dataset
variant used in a university Deep Learning course (the "M-Health" Kaggle competition). The public
ISIC 2024 data is available from Kaggle:

- ISIC 2024 challenge: https://www.kaggle.com/competitions/isic-2024-challenge

You will need:

| File | Description |
|---|---|
| `train-metadata.csv` | Clinical/tabular metadata per lesion (incl. `patient_id`, `target`) |
| `test-metadata.csv`  | Same schema, no target |
| `train-image.hdf5`   | Training images keyed by `isic_id` (JPEG bytes) |
| `test-image.hdf5`    | Test images keyed by `isic_id` |

## Expected layout

Place the downloaded files under a git-ignored `data/` directory:

```
data/
├── train-metadata.csv
├── test-metadata.csv
├── train-image.hdf5
└── test-image.hdf5
```

Then build the fast preprocessed image stores used by the pipelines:

```bash
python tools/preprocess_hdf5.py        # → 224px store
python tools/preprocess_hdf5_v2.py     # → 384px store (used by EVA02 + EdgeNeXt)
```

## Leakage warning (read before using metadata)

Several metadata columns are **post-biopsy** and leak the label. They are excluded everywhere in this
project and must stay excluded: `mel_thick_mm`, `mel_mitotic_index`, and all diagnosis codes
`iddx_1..5`, `iddx_full`. The vetted 38-feature safe list is `stacking/feature_list.txt`.
See `docs/metadata-audit.md` and `docs/domain-rules.md`.
