# Presentation Notes: Stacked Melanoma Detection

**Team:** Los Backpropagators  
**Competition:** ISIC 2024 - Skin Cancer Detection with 3D-TBP  
**Best Score:** 0.990 pAUC (Public LB)

---

## Slide 1: Title Slide

### Goal
Introduce the project and team.

### Key Information
- **ISIC**: International Skin Imaging Collaboration
- **3D-TBP**: 3D Total Body Photography — a medical imaging system that captures full-body dermoscopic images and automatically segments individual lesions with pre-computed metadata (size, color, position).
- **pAUC**: Partial Area Under the Curve — the competition metric, focusing on the high-sensitivity region (low false negative rate) crucial for cancer screening.

### Narration
> "Our project tackles melanoma detection using the ISIC 2024 dataset from Kaggle. We combine dermoscopic images with clinical metadata using a two-stage stacking architecture. Our team achieved a 0.990 pAUC score, and today I'll walk you through the key challenges we faced and the solutions we developed."

---

## Slide 2: Problem Overview

### Goal
Frame the classification task and outline the presentation structure.

### Key Information
- **Dermoscopic images**: High-magnification skin images taken with polarized light to reduce surface reflection and reveal subsurface structures.
- **Clinical metadata**: Patient info (age, sex, anatomical site) + lesion measurements (area, perimeter, color channels in LAB space).
- **Binary classification**: Malignant (target=1) vs Benign (target=0).

### Narration
> "The goal is simple in principle: classify each skin lesion as malignant or benign. But the devil is in the details. We have two input modalities — images and tabular metadata — and we need to fuse them intelligently. we will structure this presentation around three pillars: first, how we handled the extreme data imbalance; second, our feature engineering and preprocessing pipeline; and third, our two-stage stacking architecture with explainability analysis. At the end we also see how the differnet parts work together to achieve our final performance."

---

## Slide 3: Section Header — Data Imbalance Handling

### Goal
Transition to the imbalance discussion.

### Narration
> "Let's start with the elephant in the room: class imbalance."

---

## Slide 4: The Problem — Extreme Imbalance

### Goal
Quantify the severity of the class imbalance and explain why it matters.

### Key Information
- **343 malignant samples** out of 400,959 total = **0.085%** positive rate
- **Ratio 1:1168** — for every malignant case, there are 1,168 benign ones
- **Accuracy paradox**: A model predicting all-negative achieves 99.9% accuracy but 0.50 AUC (random)
- **AUC (Area Under ROC Curve)**: Measures discrimination ability independent of threshold; 0.5 = random, 1.0 = perfect

### Narration
> "The ISIC dataset has one of the most extreme imbalance ratios I've encountered: 343 malignant cases out of 400,000. That's a 1:1168 ratio. To put it in perspective, if you predict 'benign' for every single sample, you get 99.9% accuracy — but an AUC of 0.50, meaning you've learned nothing useful. Our baseline CNN fell into exactly this trap."

---

## Slide 5: Our Solution — Weighted Sampling

### Goal
Explain the primary solution for handling imbalance at the data loading level.

### Key Information
- **WeightedRandomSampler**: PyTorch utility that assigns sampling probability inversely proportional to class frequency
- **Effect**: Each epoch, positive samples are drawn ~1168× more often, creating balanced mini-batches
- **Advantage over loss modification**: No hyperparameter tuning for α, γ; just let sampling handle it
- The 1st place solution explicitly stated: *"Balanced sampling works better than Focal Loss alone"*

### Narration
> "Our primary solution was weighted sampling using PyTorch's WeightedRandomSampler. The idea is simple: assign each sample a weight inversely proportional to its class frequency. During training, positives get sampled roughly 1,168 times more often than negatives, so the model sees a balanced class distribution every epoch. This was more effective than modifying the loss function. Interestingly, the 1st place solution made the same observation."

---

## Slide 6: Alternative — Focal Loss

### Goal
Explain Focal Loss as an alternative (we experimented with it but didn't use it in the final submission).

### Key Information
- **Focal Loss formula**: $\mathcal{L}_{\text{focal}} = -\alpha (1-p_t)^\gamma \log(p_t)$
  - $p_t$: predicted probability for the true class
  - $\alpha$: class weighting factor (typically 0.25)
  - $\gamma$: focusing parameter (typically 2.0) — down-weights easy examples
- **Intuition**: When $\gamma=2$, an easy example with $p_t=0.9$ has its loss reduced by $(1-0.9)^2 = 0.01$ — a 100× reduction
- **BCE**: Binary Cross-Entropy = standard log loss

### Narration
> "Focal Loss is the other popular approach for imbalance. It modifies the loss function to down-weight easy negatives. The gamma parameter controls how aggressively we focus on hard examples — with gamma=2, easy samples have their loss reduced by up to 100×. We experimented with Focal Loss in earlier iterations, but our final submission uses standard BCE with weighted sampling. The combination of both didn't improve over sampling alone."

---

## Slide 7: Summary — Imbalance Handling

### Goal
Summarize what we used vs. what we didn't.

### Key Information
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic samples by interpolating between existing minority class examples in feature space — risky for images, mixed results in literature.
- **Key insight**: For extreme imbalance (>1:1000), sampling-based solutions outperformed loss modifications in our experiments.

### Narration
> "To summarize: we use WeightedRandomSampler for all vision training combined with standard BCE loss. We tested Focal Loss but found no improvement over sampling alone. We did not use SMOTE because interpolation in image space creates artifacts, and the literature is mixed on its effectiveness for such extreme ratios."

---

## Slide 8: Synthetic Data Generation with Stable Diffusion

### Goal
Explain our generative data augmentation approach.

### Key Information
- **Stable Diffusion 1.5 (SD 1.5)**: Latent diffusion model — generates images by iteratively denoising in a compressed latent space (64× smaller than pixel space)
- **Fine-tuning**: We trained SD 1.5 on our 343 malignant samples at 128×128 resolution
- **Filtration strategy**: Generated 10,000 candidates, filtered down to ~6,000 using classifier confidence threshold ($P(\text{malignant}) > 0.5$)
- **Critical limitation**: Synthetic images have NO metadata — we couldn't include them in the final stacking model

### Narration
> "Standard oversampling just duplicates the same 343 images, leading to overfitting. Instead, we fine-tuned Stable Diffusion 1.5 on our malignant subset to generate novel synthetic lesions. We trained at 128×128 for speed, then upscaled the best samples. For quality control, we used our strongest classifier at the time to filter — if the classifier predicted high malignancy probability, we kept it. The challenge? Synthetic images have no associated metadata, so they couldn't be used in our final stacking architecture. They helped vision-only models but didn't make it into the submitted solution."

---

## Slide 9: Section Header — Preprocessing & Data Augmentation

### Goal
Transition to data pipeline discussion.

### Narration
> "Now let's look at how we process the raw data before it reaches the model."

---

## Slide 10: Data Pipeline Overview

### Goal
Show the complete preprocessing flow for both modalities.

### Key Information
- **HDF5**: Hierarchical Data Format — binary container optimized for fast random access of large arrays
- **LZF compression**: Fast compression codec; ~40% size reduction with minimal CPU overhead
- **LANCZOS resampling**: Sinc-based interpolation that preserves edge sharpness (better than bilinear for upscaling)
- **ImageNet normalization**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] — required for pretrained backbones
- **StandardScaler**: Z-score normalization for metadata: $z = (x - \mu) / \sigma$

### Narration
> "Our data pipeline handles two streams. For images: the original ISIC images are surprisingly small — 67 to 139 pixels — so we upscale to 384×384 and store in HDF5 with LZF compression. This gives us ~10× faster loading than individual JPEGs. At training time, we resize to model-specific sizes and apply ImageNet normalization. For metadata: we apply feature engineering, then StandardScaler normalization. Both streams feed into the model."

---

## Slide 11: Image Augmentation

### Goal
Explain our augmentation strategy and design decisions.

### Key Information
- **Albumentations**: High-performance augmentation library with GPU support
- **Transpose/Flip/Rotate90**: Geometric transforms — skin lesions have no canonical orientation
- **ColorJitter**: Brightness, contrast, saturation, hue perturbations — handles scanner variability
- **OneOf [MotionBlur, MedianBlur, GaussianBlur, GaussNoise]**: Camera quality simulation
- **p=0.7**: Probability of applying the transform — aggressive augmentation
- **TTA (Test-Time Augmentation)**: Not used in our final submission

### Narration
> "We use aggressive augmentation during training. Since skin lesions have no inherent orientation, we apply all geometric flips and rotations. Color jitter handles variation across different imaging devices. Blur and noise simulate camera quality differences. The key design decision was using p=0.7 — 70% of samples get color and blur transforms. With only 343 positives, aggressive augmentation is essential to prevent overfitting. Importantly, at test time we only apply resize and normalize — no TTA."

---

## Slide 12: Augmentation Examples

### Goal
Visual demonstration of augmentation effects.

### Key Information
- Same malignant lesion shown with different random transforms applied
- Each training epoch sees different combinations
- Transforms stack: a single sample might get flipped, color-jittered, AND blurred

### Narration
> "Here's a real malignant lesion with various augmentations applied. Notice how each transform creates a plausible variation while preserving the diagnostic features — the irregular border, the color asymmetry. During training, each epoch sees different random combinations, effectively multiplying our effective dataset size."

---

## Slide 13: Metadata Analysis — Top Correlations

### Goal
Justify the need for feature engineering by showing weak raw correlations.

### Key Information
- **ABCDE "D"**: Diameter criterion from clinical ABCDE melanoma screening rule
- **tbp_lv_dnn_lesion_confidence**: Pre-computed confidence score from the TBP system's built-in classifier — lower confidence correlates with malignancy (unusual lesions are harder to classify)
- **All correlations < |0.05|**: No single feature is strongly predictive — ensemble/stacking is necessary
- **Pearson correlation (r)**: Linear relationship strength; ranges from -1 to +1

### Narration
> "Before engineering features, we analyzed raw metadata correlations with the target. The strongest signal comes from size features — larger lesions are more likely malignant, aligning with the 'D' in the clinical ABCDE rule. Interestingly, the TBP system's built-in DNN confidence is negatively correlated — when the system is less confident about a lesion, it's more likely malignant. Critically, all correlations are below 0.05 — no single feature strongly predicts malignancy. This justifies our complex modeling approach."

---

## Slide 14: Feature Engineering — ABCDE Rule

### Goal
Explain our domain-driven feature engineering approach.

### Key Information
- **ABCDE Rule**: Clinical mnemonic for melanoma screening:
  - **A = Asymmetry**: One half doesn't match the other
  - **B = Border**: Irregular, ragged, or blurred edges
  - **C = Color**: Uneven color distribution (brown, black, tan, red, white, blue)
  - **D = Diameter**: Larger than 6mm (pencil eraser size)
  - **E = Evolution**: Changes in size, shape, or color over time (we proxy with age)
- **28 engineered features** total across all criteria
- **Formula examples**:
  - `asymmetry_score = (norm_color + radial_color_std + 1/shape_regularity) / 3`
  - `shape_regularity = area / perimeter²` (circle maximizes this)
  - `compactness = 4πA/P²` (isoperimetric quotient; 1 = perfect circle)

### Narration
> "Our feature engineering is grounded in dermatology's ABCDE rule. For Asymmetry, we combine color norm with radial color variation and shape irregularity. For Border, we compute shape regularity — a circle maximizes area-to-perimeter-squared, so irregular borders score lower. For Color, we compute variance metrics across multiple channels. For Diameter, we create a binary flag for lesions larger than 6mm. For Evolution, we can't measure change from a single image, so we use age as a proxy — melanoma risk increases with age. In total, 28 engineered features."

---

## Slide 15: Feature Engineering — Patient-Relative Features

### Goal
Explain the "Ugly Duckling" principle and patient-relative features.

### Key Information
- **Ugly Duckling Sign**: Clinical observation that malignant lesions often look different from a patient's other moles
- **Patient Z-Score**: $z = (x - \mu_{\text{patient}}) / \sigma_{\text{patient}}$ — how unusual is this lesion for THIS patient?
- **LOF (Local Outlier Factor)**: Density-based anomaly detection; scores how isolated a point is relative to its neighbors — we apply this per-patient
- **Inspiration**: The 1st place solution specifically highlighted patient-relative features as a key insight

### Narration
> "The 'Ugly Duckling' sign is a clinical heuristic: if one mole looks different from a patient's other moles, it's suspicious. We encode this computationally. For each numerical feature, we compute a z-score relative to that patient's distribution. We also run Local Outlier Factor per patient — if a lesion is an anomaly among the patient's other lesions, LOF will flag it. This was explicitly called out in the 1st place solution as a key differentiator."

---

## Slide 16: Train/Test Distribution Shift

### Goal
Highlight the distribution mismatch between train and test sets.

### Key Information
- **Distribution shift (covariate shift)**: When train and test data come from different underlying distributions
- **Observed shifts**: 0.2–0.5 standard deviations on key features (tbp_lv_H, tbp_lv_perimeterMM, etc.)
- **Implication**: Models may not generalize as well as CV suggests; robust normalization and augmentation help
- **Adversarial validation**: Train a classifier to distinguish train from test — high accuracy = significant shift

### Narration
> "We discovered significant distribution shift between train and test sets. Test lesions are generally larger, and color distributions differ by 0.2–0.5 standard deviations. This explains some gap between local CV and leaderboard performance. We addressed this through robust feature normalization and aggressive augmentation. The shift also motivated our decision to use patient-relative features — ratios and z-scores are more invariant to absolute scale shifts."

---

## Slide 17: Data Quality & Leakage Detection

### Goal
Explain how we identified and handled data leakage.

### Key Information
- **Data leakage**: When features contain information that wouldn't be available at prediction time — leads to overly optimistic training metrics but fails in production
- **mel_thick_mm (Breslow thickness)**: Measured in mm from tissue sample after excision biopsy — only exists for confirmed melanomas
- **mel_mitotic_index**: Cell division rate, also measured post-biopsy
- Both features are 99.99% missing — but when present, 100% are malignant
- **Action**: Remove these columns from all training/inference pipelines

### Narration
> "Data leakage is a silent killer in medical ML. We found two columns — mel_thick_mm and mel_mitotic_index — that are post-biopsy measurements. They're 99.99% missing, but when present, they're 100% malignant. That's a dead giveaway: these values only exist after a diagnosis is made. Using them would give artificially perfect predictions during training but complete failure in real-world use. We dropped them entirely."

---

## Slide 18: Section Header — Best Model Architecture

### Goal
Transition to model architecture discussion.

### Narration
> "Now for the core of our solution: the two-stage stacking architecture."

---

## Slide 19: Two-Stage Stacking Architecture

### Goal
Present the complete architecture with data flow.

### Key Information
- **Stage 1 — Vision Models**:
  - **EVA02-Small (22M params)**: Vision Transformer with Masked Image Modeling pretraining
  - **EdgeNeXt-Base (18M params)**: CNN-ViT hybrid — CNN stages for local texture, Transformer for global context
  - Output: logits + 384/576-dimensional embeddings → PCA reduced to 50 dimensions
- **Stage 2 — Stacking Models**:
  - **XGBoost**: Input = raw metadata + vision logits + vision embeddings (PCA'd)
  - **MLP**: Input = raw metadata + DAE latent (64d) + vision logits + vision embeddings
  - **DAE (Denoising Autoencoder)**: Trained unsupervised on all metadata (train+test) to learn robust 64-dim latent
- **Ensemble**: Simple average of XGBoost and MLP probabilities
- **Critical note**: Synthetic data was NOT used in this final architecture (no metadata for synthetic images)

### Narration
> "Our final architecture is a two-stage stacking ensemble. Stage 1 runs two vision models — EVA02 and EdgeNeXt — independently. We extract both the final logits and the penultimate embeddings, reduce embeddings to 50 dimensions via PCA to prevent overfitting. Stage 2 is the 'brain': XGBoost and an MLP both receive the metadata plus vision outputs. The MLP additionally gets a 64-dimensional latent from a denoising autoencoder trained on all metadata. Final prediction is a simple average of both stackers. Note that synthetic data was not used here — we couldn't include it without metadata."

---

## Slide 20: Why EVA02 + EdgeNeXt?

### Goal
Justify the choice of vision backbones.

### Key Information
- **EVA02**: Vision Transformer (ViT) — 22M parameters, 336×336 input, pretrained with MIM (Masked Image Modeling) on ImageNet-22k
- **EdgeNeXt**: CNN-ViT Hybrid — 18M parameters, 384×384 input, combines ConvNeXt-style CNN stages with Transformer blocks
- **Prediction correlation: 0.12** — very low correlation means models capture complementary information
- **Ensemble diversity principle**: Ensembles work best when individual models make different errors
- Individual CV AUCs: EVA02 ~0.92, EdgeNeXt ~0.90

### Narration
> "Why these two specific models? EVA02 is a pure Vision Transformer — excellent at capturing global context through self-attention. EdgeNeXt is a hybrid — CNN stages capture local texture details, then Transformer blocks integrate globally. The key metric: their predictions have only 0.12 correlation. Low correlation means they're capturing different aspects of the data. For ensembles, diversity matters more than individual performance — two 0.90 models with uncorrelated errors can ensemble to 0.95+."

---

## Slide 21: Stacking Strategy — XGBoost + MLP

### Goal
Explain why we use two different stacking models.

### Key Information
- **XGBoost**: Gradient-boosted decision trees — excels at finding optimal splits on raw feature values; n_estimators=1000, max_depth=4, early_stopping_rounds=50
- **MLP (Multi-Layer Perceptron)**: 256→128→1 architecture with ReLU and Dropout(0.3)
- **DAE (Denoising Autoencoder)**: Neural network trained to reconstruct metadata from corrupted input — learns robust feature representations
- **Why both?**: XGBoost works well on raw splits; MLP benefits from learned representations (DAE latent). Different "reasoning" styles.
- CV AUC: XGBoost 0.928, MLP 0.941

### Narration
> "For stacking, we use both XGBoost and an MLP. XGBoost is the heavy lifter — tree-based models are excellent at finding optimal decision boundaries on raw feature values. The MLP has a different inductive bias — it benefits from the DAE's learned latent representation. The DAE is trained unsupervised on all metadata to learn a smooth manifold; noisy input features get transformed into a cleaner 64-dimensional space. Ensembling XGBoost and MLP covers different 'reasoning' styles."

---

## Slide 22: The "Golden Split" Discovery

### Goal
Explain our key empirical finding about fold quality.

### Key Information
- **5-Fold CV**: Standard cross-validation with 5 train/val splits
- **Fold 4 anomaly**: Models trained on folds {0,1,2,3} (validating on fold 4) achieved 0.990 pAUC — significantly higher than other splits
- **"Toxic Fold 4" hypothesis**: Whenever fold 4 is included in TRAINING, performance drops (0.947-0.972)
- **Golden Split strategy**: Train exclusively on folds {0,1,2,3}; do NOT include fold 4 in training
- **Result**: Single Golden Split model (0.990) outperforms 5-fold ensemble (0.982)
- **Root cause**: Fold 4 likely contains noisy, mislabeled, or out-of-distribution samples

### Narration
> "Here's our most surprising finding. We noticed that one specific cross-validation split — training on folds 0, 1, 2, 3 and validating on fold 4 — achieved 0.990 pAUC, significantly higher than any other configuration. When we analyzed the pattern, we found that fold 4 is 'toxic': whenever it's included in the TRAINING set, performance drops. The single 'Golden Split' model actually outperformed our full 5-fold ensemble. Lesson: data quality trumps data quantity. Sometimes excluding noisy samples helps more than including more data."

---

## Slide 23: Model Selection Summary

### Goal
Show the progression of improvements.

### Key Information
- **Vision Only (EVA02)**: 0.932 pAUC — baseline
- **+XGBoost Stacking**: 0.980 pAUC — +0.048 from adding metadata
- **+MLP Stacking**: 0.982 pAUC (ensemble) — +0.002 from diversity
- **+Golden Split**: 0.990 pAUC — +0.008 from data quality
- **Biggest gain**: Vision → Stacking (+0.048) — metadata is crucial
- **Key insight**: Metadata adds critical context that images miss

### Narration
> "Let me quantify each improvement. Starting from vision-only EVA02 at 0.932, adding XGBoost stacking with metadata jumps us to 0.980 — that's the biggest single gain, demonstrating that metadata is crucial for this task. Adding the MLP to the ensemble gives another 0.002. But the Golden Split discovery adds 0.008 more, bringing us to 0.990. The takeaway: fusing image and tabular data is the key architectural decision, and data quality optimization can rival model engineering in impact."

---

## Slide 24: Section Header — Explainability

### Goal
Transition to explainability analysis.

### Narration
> "Finally, let's open the black box and understand how our model makes predictions."

---

## Slide 25: Stage 1 — Vision Model Attention

### Goal
Compare attention patterns of EVA02 vs EdgeNeXt.

### Key Information
- **Attention/Activation maps**: Visualize which image regions the model focuses on when making predictions
- **EVA02 (Transformer)**: Self-attention captures long-range dependencies; tends to distribute attention across the lesion body
- **EdgeNeXt (Hybrid)**: CNN stages capture high-frequency details; tends to focus sharply on borders and texture irregularities
- **Sample shown**: ISIC_0096034 — a confirmed malignant case
- **Complementary signals**: EVA02 sees the "forest" (global asymmetry), EdgeNeXt sees the "trees" (border irregularity)

### Narration
> "These activation maps show where each model focuses. EVA02, being a pure Transformer, distributes attention broadly across the lesion — it's capturing global context, overall asymmetry, color distribution. EdgeNeXt, with its CNN backbone, focuses sharply on edges and local texture — exactly where border irregularity manifests. This visualization perfectly explains why ensembling works: they're literally looking at different features."

---

## Slide 26: Stage 2 — Stacking Decisions (SHAP)

### Goal
Quantify feature importance in the stacking model.

### Key Information
- **SHAP (SHapley Additive exPlanations)**: Game-theoretic approach to explain model predictions; decomposes each prediction into contributions from each feature
- **SHAP summary (beeswarm) plot**: Each dot is a sample; color = feature value (red=high, blue=low); x-axis = impact on prediction
- **Top features**:
  1. Vision probabilities (EVA02, EdgeNeXt) — strongest individual features
  2. age_approx — older patients have higher risk
  3. tbp_tile_type — body location context
  4. Patient-relative/erratic features — "ugly duckling" detection
- **"Metadata is King"**: While vision is #1 individually, cumulative metadata impact often outweighs vision in edge cases

### Narration
> "SHAP analysis reveals what drives the stacking model's decisions. Vision probabilities dominate at the top — unsurprisingly, if the image model says malignant, the stacker agrees. But look at the cumulative impact of metadata: age, location, size, color features — together, they often outweigh vision for borderline cases. This validates our architectural choice: the fusion of both modalities is stronger than either alone."

---

## Slide 27: Deep Dive — The DAE Latent Space

### Goal
Explain how the DAE organizes the data and why it helps the MLP.

### Key Information
- **DAE (Denoising Autoencoder)**: Neural network trained to reconstruct input from corrupted version; learns robust low-dimensional representation
- **Latent space**: The 64-dimensional bottleneck representation
- **UMAP projection**: Dimensionality reduction technique that preserves local structure for visualization
- **Observation**: Malignant cases (red) cluster in specific regions — they're not uniformly distributed in latent space
- **Benefit for MLP**: Decision boundaries in latent space are smoother and easier to learn than in raw feature space

### Narration
> "The DAE is trained unsupervised on 400,000+ samples to learn a robust 64-dimensional representation. This UMAP projection shows the latent space structure. Notice that malignant cases — the red points — cluster in specific regions. The DAE has learned, without any labels, to organize the data in a way that groups similar risk profiles together. This is why the MLP benefits from DAE features: the latent space has smoother decision boundaries than the noisy raw features."

---

## Slide 28: Thank You / Questions

### Goal
Close the presentation and open for discussion.

### Key Information
- Summary of contributions:
  1. **Two-stage stacking** architecture fusing vision and tabular modalities
  2. **Weighted sampling** as primary imbalance solution
  3. **ABCDE-based feature engineering** with patient-relative features
  4. **"Golden Split" discovery** — data quality over quantity
  5. **Comprehensive explainability** — vision attention + SHAP + DAE analysis

### Narration
> "To summarize: we built a two-stage stacking system that fuses vision models with clinical metadata. Our key innovations were applying the ABCDE clinical framework to feature engineering, discovering the 'Golden Split' for optimal data selection, and providing comprehensive explainability through attention maps, SHAP analysis, and latent space visualization. Happy to take questions."

---

## Appendix: Key Terms Glossary

| Term | Definition |
|------|------------|
| **AUC** | Area Under ROC Curve — measures discrimination ability |
| **pAUC** | Partial AUC — competition metric focusing on high-sensitivity region |
| **GBDT** | Gradient Boosted Decision Trees (XGBoost, CatBoost, LightGBM) |
| **ViT** | Vision Transformer — applies self-attention to image patches |
| **MIM** | Masked Image Modeling — pretraining by predicting masked patches |
| **OOF** | Out-of-Fold — predictions on validation fold during CV |
| **CV** | Cross-Validation — estimate generalization by training on subsets |
| **DAE** | Denoising Autoencoder — learns robust representations via reconstruction |
| **LOF** | Local Outlier Factor — density-based anomaly detection |
| **SHAP** | SHapley Additive exPlanations — game-theoretic feature attribution |
| **TBP** | Total Body Photography — 3D skin imaging system |
| **LAB color space** | Perceptually uniform color space (L=lightness, A=green-red, B=blue-yellow) |
| **Breslow thickness** | Depth of melanoma invasion into skin — key staging metric |
| **ABCDE Rule** | Clinical mnemonic: Asymmetry, Border, Color, Diameter, Evolution |

---

## Appendix: Competition Context

- **Competition**: ISIC 2024 - Skin Cancer Detection with 3D-TBP
- **Host**: Kaggle + ISIC
- **Task**: Binary classification of skin lesions (malignant vs benign)
- **Metric**: Partial AUC at 80% True Positive Rate
- **Data**: ~400k training samples, ~100 test samples
- **1st Place Solution highlights**:
  - Same architectures (EVA02, EdgeNeXt)
  - Patient-relative features (LOF-based "Ugly Duckling")
  - Rank normalization for vision predictions
  - Gaussian noise injection during GBDT training
  - Multi-seed ensembling (150 GBDT models total)

---

## Appendix: What Didn't Work

1. **Focal Loss alone** — weighted sampling was more effective
2. **SMOTE** — not suitable for image data
3. **Synthetic data in stacking** — no metadata available for synthetic images
4. **Heavy MLP heads** — simple linear head worked best for vision models
5. **Feature engineering on "Golden Split"** — hurt performance; simple features won
6. **Full 5-fold ensemble** — single Golden Split model outperformed
7. **TTA (Test-Time Augmentation)** — added complexity without improvement

---

## Timing Guidance

| Slide | Estimated Time | Cumulative |
|-------|---------------|------------|
| 1-2 (Title + Overview) | 2 min | 2 min |
| 3-8 (Imbalance + Synthetic) | 5 min | 7 min |
| 9-17 (Preprocessing + FE) | 6 min | 13 min |
| 18-23 (Architecture) | 6 min | 19 min |
| 24-27 (Explainability) | 4 min | 23 min |
| 28 (Q&A buffer) | 2 min | 25 min |

**Total**: ~25 minutes presentation + Q&A
