# ISIC 2024 Kaggle Presentation (Slidev)

## Presentation Overview

**Duration**: ~10 minutes  
**Topic**: ISIC 2024 Skin Cancer Detection with 3D-TBP  
**Final Submission**: Models from `DeepLearning/Kaggle/last_run/`

## Required Topics (20% Total)

| Topic | Weight | Status |
|-------|--------|--------|
| Data Imbalance Handling | 5% | 🔲 TODO |
| Preprocessing & Data Augmentation | 5% | 🔲 TODO |
| Best Model Justification | 5% | 🔲 TODO |
| Explainability | 5% | 🔲 TODO |

## Folder Structure

```
presentation/
├── slides.md              # Main Slidev deck
├── public/
│   ├── figures/           # Generated plots (PNG/SVG)
│   └── images/            # Static images/screenshots
├── scripts/               # Figure generation scripts
├── data/                  # Presentation-specific data extracts
├── notes/                 # Speaker notes, outline
└── notebooks/             # Exploration notebooks
```

## Usage

```bash
# Navigate to presentation directory
cd ./presentation

# Install Slidev (if not installed)
npm init slidev

# Start presentation dev server
npm run dev

# Build for production
npm run build
```

## Content Sources

| Topic | Source Files |
|-------|--------------|
| Imbalance | `last_run/src/train_vision.py`, `18_1_train_dual_backbone_hybrid.py` |
| Augmentation | `last_run/src/train_vision.py:26-49` (Albumentations) |
| Model Selection | `last_run/research_log.md`, `PROJECT_SUMMARY.md` |
| Explainability | `16_4_visualize_stacking.py`, **TODO: Grad-CAM/SHAP** |

## Slide Timing Guide (~10 min)

1. **Title + Problem** (1 min)
2. **Data Imbalance** (2 min)
3. **Preprocessing & Augmentation** (2 min)
4. **Best Model** (3 min)
5. **Explainability** (2 min)
