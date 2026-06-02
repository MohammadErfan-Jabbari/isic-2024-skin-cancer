# Metadata Audit — ISIC 2024

> Six-phase forensic audit of the 55-column clinical metadata accompanying ~401K dermoscopic images.
> Outcome: a 38-feature safe list (`stacking/feature_list.txt`) free of leakage, with validated engineered features.

---

## Context

**Task**: binary malignancy prediction. Prevalence ≈ 343/400,959 ≈ 0.085% (1:1168 imbalance).  
**Metric**: pAUC above 80% TPR. Metadata alone accounts for the single largest AUC jump in the project (custom CNN baseline ~0.51 → hybrid with metadata ~0.936).

---

## Phase 1 — Raw Data Profiling

- 55 total columns: 4 identifiers, 1 target, 4 categorical, 46 numeric.
- Most numeric features are TBP (Total Body Photography) system outputs covering color (L\*a\*b\*, HSV), size/shape, symmetry, and border regularity.
- Key missingness: `mel_thick_mm` 99.99% missing (400,908/400,959), `mel_mitotic_index` similarly ~99.99%, all `iddx_*` codes 99.75–100% missing.
- Train/test column overlap is complete; no test-only columns observed.

---

## Phase 2 — Per-Column Leakage Analysis

### Confirmed leakage columns — **DROP from all pipelines**

| Column | Non-missing count | Malignant rate among non-missing | Verdict |
|---|---|---|---|
| `mel_thick_mm` | 51 | **51/51 = 100%** | Post-biopsy measurement — hard leakage |
| `mel_mitotic_index` | 43 | **43/43 = 100%** | Post-biopsy measurement — hard leakage |
| `iddx_1` … `iddx_5`, `iddx_full` | ~0–0.25% | Near-100% | Histopathological diagnosis codes — hard leakage |

All non-missing values of these columns originate from confirmed malignant cases. Including them is equivalent to label injection.

### TBP confidence features — **safe**

`tbp_lv_dnn_lesion_confidence` and `tbp_lv_nevi_confidence` are pre-computed TBP system outputs generated before any diagnosis. No post-diagnosis leakage; both retained, with one exception:  
`tbp_lv_dnn_lesion_confidence` was later dropped from the final safe list because it hurt the clean folds; `tbp_lv_nevi_confidence` is kept.

### Position proxy — `tbp_lv_y`

`tbp_lv_y` encodes vertical body position (ANOVA F ≈ 341,202). It is highly informative because body location is a genuine melanoma risk factor, not a leakage artifact. It is retained but interpreted accordingly — high importance reflects site-level risk, not a spurious correlation.

---

## Phase 3 — Engineered Features

### Patient-relative z-scores

For each patient, lesion-level features (color, size, shape) are expressed as z-scores within that patient's lesion population. Rationale: the "Ugly Duckling" clinical heuristic — a lesion unusual relative to a patient's own baseline carries elevated risk. Patient-relative features added meaningful signal on top of absolute TBP measurements.

### LOF (Local Outlier Factor)

Applied per-patient on TBP color features to score each lesion's local density anomaly. LOF nudged pAUC from 0.18149 → 0.18185. Small but consistent.

### Lesion count per patient

Patients with more lesions have a diluted per-lesion malignancy probability. Count was included as a feature and used in patient-relative normalization.

---

## Phase 4 — Importance Validation

Permutation importance and GBDT gain importance were compared against Spearman correlations with target. High-importance features that showed low linear correlation were inspected for proxy-leakage. Key finding: `tbp_lv_y` (body location) and patient-relative features are legitimately important, not artifacts.

CatBoost importance rankings differed from XGBoost on the Golden Split (folds {0,1,2,3}); CatBoost underperformed and was excluded from the final ensemble.

---

## Phase 5 — Data Quality and Distribution

- No exact duplicate `isic_id` entries.
- Outlier inspection: extreme TBP values (very large `areaMM2`, extreme L\*a\*b\* readings) are genuine clinical cases, not corrupted rows.
- Train→test distribution shift: **raw metadata features shift < 1σ** (largest: `tbp_lv_H` ~0.89σ, `perimeterMM` ~0.59σ, `deltaB` ~0.54σ, `B` ~0.48σ). Manageable with z-score standardization using training statistics.
- Vision prediction shift is far more severe (2.3–2.5σ) — covered separately in `distribution-shift.md`.

---

## Phase 6 — Synthesis: 38-Feature Safe List

Full list in `stacking/feature_list.txt`. Reproduced here for reference:

```
age_approx, anatom_site_general, clin_size_long_diam_mm, sex,
tbp_lv_A, tbp_lv_Aext, tbp_lv_B, tbp_lv_Bext, tbp_lv_C, tbp_lv_Cext,
tbp_lv_H, tbp_lv_Hext, tbp_lv_L, tbp_lv_Lext,
tbp_lv_areaMM2, tbp_lv_area_perim_ratio, tbp_lv_color_std_mean,
tbp_lv_deltaA, tbp_lv_deltaB, tbp_lv_deltaL, tbp_lv_deltaLB, tbp_lv_deltaLBnorm,
tbp_lv_eccentricity, tbp_lv_location, tbp_lv_location_simple,
tbp_lv_minorAxisMM, tbp_lv_nevi_confidence, tbp_lv_norm_border, tbp_lv_norm_color,
tbp_lv_perimeterMM, tbp_lv_radial_color_std_max,
tbp_lv_stdL, tbp_lv_stdLExt, tbp_lv_symm_2axis, tbp_lv_symm_2axis_angle,
tbp_lv_x, tbp_lv_y, tbp_lv_z
```

**Dropped from initial candidate set:**
- `mel_thick_mm`, `mel_mitotic_index` — hard post-biopsy leakage.
- `iddx_1`–`iddx_5`, `iddx_full` — diagnosis codes.
- `tbp_lv_dnn_lesion_confidence` — hurts clean folds despite no leakage.
- All identifier and target columns.

**Patient-aware splits are mandatory** with this feature set: `StratifiedGroupKFold(n_splits=5, group=patient_id)`. Many patients contribute multiple lesions; cross-patient contamination inflates OOF AUC silently and substantially.
