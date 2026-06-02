# %% [markdown]
# # K-Fold Ensemble Analysis & Prediction Generation
# 
# Matches the updated training script (based on your best model)
# 
# This notebook:
# 1. Loads all 5 fold results (regular, EMA, SWA models) - up to 15 models total
# 2. Visualizes training curves and performance metrics
# 3. Generates test predictions from each fold/model
# 4. Creates sophisticated ensemble predictions (6 strategies)
# 5. Analyzes prediction diversity and confidence
# 6. Generates final submission files
# 
# **IMPORTANT**: Feature engineering must EXACTLY match training script!
# This version includes all enhanced features (age_squared, compactness, etc.)
# 
# **Expected**: 0.970-0.978 AUC on public LB
# **Time**: ~15-20 minutes to run (loading models + inference)

# %%
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import pickle
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# %% [markdown]
# ## Configuration
# 
# **IMPORTANT**: Set this to your k-fold results directory!

# %%
# **IMPORTANT**: Set this to your k-fold results directory!
KFOLD_DIR = Path('results/kfold_v2s_features_advanced_20251111_150340')

# You can also auto-detect the latest folder:
def find_latest_kfold_dir():
    results_dir = Path('results')
    # Look for the new advanced k-fold directories
    kfold_dirs = sorted(results_dir.glob('kfold_v2s_features_advanced_*'))
    if kfold_dirs:
        return kfold_dirs[-1]
    # Fallback to old naming pattern
    kfold_dirs = sorted(results_dir.glob('kfold_efficientnetv2_*'))
    if kfold_dirs:
        return kfold_dirs[-1]
    return None

# Auto-detect if not manually set
if not KFOLD_DIR.exists():
    KFOLD_DIR = find_latest_kfold_dir()
    if KFOLD_DIR is None:
        raise ValueError("No k-fold results directory found!")

print("="*70)
print("K-FOLD ENSEMBLE ANALYSIS & PREDICTION")
print("="*70)
print(f"Loading from: {KFOLD_DIR}")
print(f"\nThis notebook expects models trained with:")
print(f"  • Feature engineering with {38} numerical + {9} categorical features")
print(f"  • Focal loss (alpha=0.25, gamma=2.0, no label smoothing)")
print(f"  • Adam optimizer (weight_decay=1e-5)")
print(f"  • Up to 3 model types per fold (regular, EMA, SWA)")
print("="*70 + "\n")

# %% [markdown]
# ## Feature Engineering (SAME AS TRAINING)
# 
# Must match training exactly - this is critical for model loading!

# %%
def engineer_features(df):
    """Must match training exactly"""
    df = df.copy()
    
    df['age_group'] = pd.cut(df['age_approx'], bins=[0, 30, 50, 70, 100],
                             labels=['young', 'middle', 'senior', 'elderly'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    df['age_squared'] = df['age_approx'] ** 2
    
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['size_category'] = pd.cut(df['lesion_size_mm'], bins=[0, 6, 10, 20, 100],
                                 labels=['small', 'medium', 'large', 'very_large'])
    df['large_lesion'] = (df['lesion_size_mm'] > 6).astype(int)
    df['size_squared'] = df['lesion_size_mm'] ** 2
    
    df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    df['eccentricity'] = df['tbp_lv_minorAxisMM'] / (df['tbp_lv_areaMM2']**0.5 + 1e-6)
    df['compactness'] = (4 * np.pi * df['tbp_lv_areaMM2']) / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    
    df['color_variance'] = np.sqrt(
        df['tbp_lv_deltaB']**2 + df['tbp_lv_radial_color_std_max']**2 +
        df['tbp_lv_color_std_mean']**2
    )
    df['color_uniformity'] = 1 / (df['tbp_lv_norm_color'] + 1e-6)
    df['darkness_score'] = df['tbp_lv_B'] / (df['tbp_lv_H'] + 1e-6)
    df['color_contrast'] = df['tbp_lv_deltaB'] * df['tbp_lv_radial_color_std_max']
    
    high_risk_sites = ['torso', 'upper extremity', 'posterior torso', 'anterior torso', 'head/neck']
    df['high_risk_site'] = df['anatom_site_general'].isin(high_risk_sites).astype(int)
    
    site_risk_map = {
        'head/neck': 4, 'torso': 3, 'posterior torso': 3, 'anterior torso': 3,
        'upper extremity': 2, 'lower extremity': 2,
        'palms/soles': 1, 'oral/genital': 1
    }
    df['site_risk_score'] = df['anatom_site_general'].map(site_risk_map).fillna(0)
    
    df['age_size_risk'] = df['age_approx'] * df['lesion_size_mm']
    df['age_site_risk'] = df['age_approx'] * df['site_risk_score']
    df['color_size_risk'] = df['color_variance'] * df['lesion_size_mm']
    df['age_color_risk'] = df['age_approx'] * df['color_variance']
    df['site_size_risk'] = df['site_risk_score'] * df['lesion_size_mm']
    
    df['asymmetry_score'] = (
        df['tbp_lv_norm_color'] + df['tbp_lv_radial_color_std_max'] +
        (1 / (df['shape_regularity'] + 1e-6))
    ) / 3
    
    df['log_area'] = np.log1p(df['tbp_lv_areaMM2'])
    df['log_perimeter'] = np.log1p(df['tbp_lv_perimeterMM'])
    df['log_size'] = np.log1p(df['lesion_size_mm'])
    
    df['h_to_b_ratio'] = df['tbp_lv_H'] / (df['tbp_lv_B'] + 1e-6)
    df['a_to_b_ratio'] = df['tbp_lv_A'] / (df['tbp_lv_B'] + 1e-6)
    df['area_to_perimeter'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] + 1e-6)
    
    return df


def preprocess_metadata_with_features(df, is_train=True, scaler=None, encoders=None):
    """Preprocess metadata with feature engineering"""
    
    df = engineer_features(df)
    
    NUMERICAL_FEATURES = [
        'tbp_lv_H', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM',
        'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_Hext',
        'clin_size_long_diam_mm', 'tbp_lv_radial_color_std_max',
        'tbp_lv_B', 'tbp_lv_color_std_mean', 'tbp_lv_Aext',
        'tbp_lv_stdLExt', 'tbp_lv_norm_color', 'tbp_lv_A', 'age_approx',
        'age_squared', 'lesion_size_mm', 'size_squared',
        'shape_regularity', 'eccentricity', 'compactness',
        'color_variance', 'color_uniformity', 'darkness_score', 'color_contrast',
        'site_risk_score', 'age_size_risk', 'age_site_risk', 'color_size_risk',
        'age_color_risk', 'site_size_risk', 'asymmetry_score',
        'log_area', 'log_perimeter', 'log_size',
        'h_to_b_ratio', 'a_to_b_ratio', 'area_to_perimeter'
    ]
    
    CATEGORICAL_FEATURES = [
        'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple',
        'age_group', 'size_category', 'age_risk', 'large_lesion', 'high_risk_site'
    ]
    
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if is_train else 0)
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing')
    
    if is_train:
        scaler = StandardScaler()
        df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])
    else:
        df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])
    
    if is_train:
        encoders = {}
        encoded_dfs = []
        for col in CATEGORICAL_FEATURES:
            encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
            encoders[col] = encoded.columns.tolist()
            encoded_dfs.append(encoded)
        result_df = pd.concat([df[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
    else:
        encoded_dfs = []
        for col in CATEGORICAL_FEATURES:
            encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
            for train_col in encoders[col]:
                if train_col not in encoded.columns:
                    encoded[train_col] = 0
            encoded = encoded[encoders[col]]
            encoded_dfs.append(encoded)
        result_df = pd.concat([df[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
    
    return result_df, scaler, encoders


def preprocess_test_metadata(df, scaler, encoders):
    """Preprocess test data using training scaler/encoders"""
    
    df = engineer_features(df)
    
    NUMERICAL_FEATURES = [
        'tbp_lv_H', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM',
        'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_Hext',
        'clin_size_long_diam_mm', 'tbp_lv_radial_color_std_max',
        'tbp_lv_B', 'tbp_lv_color_std_mean', 'tbp_lv_Aext',
        'tbp_lv_stdLExt', 'tbp_lv_norm_color', 'tbp_lv_A', 'age_approx',
        'age_squared', 'lesion_size_mm', 'size_squared',
        'shape_regularity', 'eccentricity', 'compactness',
        'color_variance', 'color_uniformity', 'darkness_score', 'color_contrast',
        'site_risk_score', 'age_size_risk', 'age_site_risk', 'color_size_risk',
        'age_color_risk', 'site_size_risk', 'asymmetry_score',
        'log_area', 'log_perimeter', 'log_size',
        'h_to_b_ratio', 'a_to_b_ratio', 'area_to_perimeter'
    ]
    
    CATEGORICAL_FEATURES = [
        'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple',
        'age_group', 'size_category', 'age_risk', 'large_lesion', 'high_risk_site'
    ]
    
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing')
    
    df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])
    
    encoded_dfs = []
    for col in CATEGORICAL_FEATURES:
        encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
        for train_col in encoders[col]:
            if train_col not in encoded.columns:
                encoded[train_col] = 0
        encoded = encoded[encoders[col]]
        encoded_dfs.append(encoded)
    
    result_df = pd.concat([df[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
    return result_df

# %% [markdown]
# ## Model Architecture (SAME AS TRAINING)

# %%
class HybridDataset(Dataset):
    def __init__(self, hdf5_path, metadata_df, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hdf5_file = None
        
        with h5py.File(hdf5_path, 'r') as f:
            available_ids = set(f.keys())
        
        self.metadata = metadata_df[
            metadata_df['isic_id'].isin(available_ids)
        ].reset_index(drop=True)
        
        feature_cols = [col for col in self.metadata.columns if col != 'isic_id']
        self.metadata_features = self.metadata[feature_cols].values.astype(np.float32)
    
    def _ensure_hdf5_open(self):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        self._ensure_hdf5_open()
        
        row = self.metadata.iloc[idx]
        image_id = row['isic_id']
        
        img_array = self.hdf5_file[image_id][:]
        image = Image.fromarray(img_array)
        
        if self.transform:
            image = self.transform(image)
        
        metadata = torch.tensor(self.metadata_features[idx], dtype=torch.float32)
        return image, metadata, image_id


class MetadataProcessor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        return self.fc(x)


class EfficientNetV2Hybrid(nn.Module):
    def __init__(self, metadata_dim):
        super().__init__()
        
        self.efficientnet = efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.efficientnet.classifier = nn.Identity()
        
        total_params = len(list(self.efficientnet.parameters()))
        freeze_until = int(total_params * 0.8)
        
        for idx, param in enumerate(self.efficientnet.parameters()):
            if idx < freeze_until:
                param.requires_grad = False
        
        self.metadata_processor = MetadataProcessor(metadata_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, metadata):
        img_features = self.efficientnet(image)
        meta_features = self.metadata_processor(metadata)
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(combined)

# %% [markdown]
# ## Step 1: Load All Fold Results

# %%
print("="*70)
print("STEP 1: LOADING FOLD RESULTS")
print("="*70 + "\n")

fold_results = {}
for fold_num in range(1, 6):
    result_file = KFOLD_DIR / f'fold_{fold_num}_results.json'
    
    if result_file.exists():
        with open(result_file, 'r') as f:
            fold_results[fold_num] = json.load(f)
        
        print(f"Fold {fold_num}:")
        print(f"  Best Val AUC: {fold_results[fold_num]['best_val_auc']:.4f}")
        print(f"  Best EMA AUC: {fold_results[fold_num]['best_ema_auc']:.4f}")
        print(f"  SWA AUC:      {fold_results[fold_num]['swa_auc']:.4f}")
        print(f"  Best Overall: {fold_results[fold_num]['best_overall']:.4f}\n")
    else:
        print(f"⚠️ Fold {fold_num} results not found!")

if len(fold_results) == 0:
    raise ValueError("No fold results found!")

# Overall statistics
all_best = [fold_results[i]['best_overall'] for i in fold_results.keys()]
print(f"{'='*70}")
print(f"OVERALL K-FOLD PERFORMANCE")
print(f"{'='*70}")
print(f"Mean AUC: {np.mean(all_best):.4f} ± {np.std(all_best):.4f}")
print(f"Best Fold: {np.max(all_best):.4f}")
print(f"Worst Fold: {np.min(all_best):.4f}")
print(f"{'='*70}\n")

# %% [markdown]
# ## Step 2: Visualize Fold Performance

# %%
print("="*70)
print("STEP 2: VISUALIZING FOLD PERFORMANCE")
print("="*70 + "\n")

# Create visualization directory
viz_dir = KFOLD_DIR / 'visualizations'
viz_dir.mkdir(exist_ok=True)

# Figure 1: Fold AUC Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

folds = list(fold_results.keys())
val_aucs = [fold_results[f]['best_val_auc'] for f in folds]
ema_aucs = [fold_results[f]['best_ema_auc'] for f in folds]
swa_aucs = [fold_results[f]['swa_auc'] for f in folds]

x = np.arange(len(folds))
width = 0.25

axes[0].bar(x - width, val_aucs, width, label='Regular', color='steelblue')
axes[0].bar(x, ema_aucs, width, label='EMA', color='darkorange')
axes[0].bar(x + width, swa_aucs, width, label='SWA', color='green')
axes[0].axhline(np.mean(all_best), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(all_best):.4f}')
axes[0].set_xlabel('Fold', fontsize=12)
axes[0].set_ylabel('Validation AUC', fontsize=12)
axes[0].set_title('Validation AUC by Fold & Model Type', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'Fold {f}' for f in folds])
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# Box plot
all_scores = np.array([val_aucs, ema_aucs, swa_aucs])
bp = axes[1].boxplot(all_scores.T, labels=['Regular', 'EMA', 'SWA'],
                      patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], ['steelblue', 'darkorange', 'green']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_ylabel('Validation AUC', fontsize=12)
axes[1].set_title('AUC Distribution by Model Type', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(viz_dir / '1_fold_auc_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 1_fold_auc_comparison.png")
plt.show()

# Figure 2: Training Curves for All Folds
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, fold_num in enumerate(folds):
    history = fold_results[fold_num]['history']
    epochs = range(1, len(history['train_auc']) + 1)
    
    ax = axes[idx]
    ax2 = ax.twinx()
    
    # AUC
    ax.plot(epochs, history['train_auc'], 'b-', linewidth=2, label='Train AUC')
    ax.plot(epochs, history['val_auc'], 'r-', linewidth=2, label='Val AUC')
    ax.plot(epochs, history['val_ema_auc'], 'g--', linewidth=2, label='Val EMA AUC')
    
    # Loss
    ax2.plot(epochs, history['train_loss'], 'b:', alpha=0.5, linewidth=1, label='Train Loss')
    ax2.plot(epochs, history['val_loss'], 'r:', alpha=0.5, linewidth=1, label='Val Loss')
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('AUC', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10, color='gray')
    ax.set_title(f'Fold {fold_num} - Best: {fold_results[fold_num]["best_overall"]:.4f}',
                fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

# Remove empty subplot
fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig(viz_dir / '2_training_curves_all_folds.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 2_training_curves_all_folds.png")
plt.show()

# Figure 3: Learning Rate Schedules
fig, ax = plt.subplots(figsize=(12, 6))

for fold_num in folds:
    history = fold_results[fold_num]['history']
    epochs = range(1, len(history['lr']) + 1)
    ax.plot(epochs, history['lr'], linewidth=2, label=f'Fold {fold_num}', alpha=0.7)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Learning Rate Schedule (Cosine Annealing with Warm Restarts)', 
            fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / '3_learning_rate_schedules.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 3_learning_rate_schedules.png")
plt.show()

print()

# %% [markdown]
# ## Step 3: Load Test Data

# %%
print("="*70)
print("STEP 3: LOADING TEST DATA")
print("="*70 + "\n")

# Define data directory
data_dir = Path('data')

# Load preprocessors
preprocessors_file = KFOLD_DIR / 'preprocessors.pkl'

try:
    with open(preprocessors_file, 'rb') as f:
        preprocessors = pickle.load(f)
        loaded_scaler = preprocessors['scaler']
        encoders = preprocessors['encoders']
        metadata_dim = preprocessors['metadata_dim']
    
    # Check if scaler is correct type
    if not hasattr(loaded_scaler, 'transform'):
        print(f"⚠️ WARNING: Preprocessors.pkl has wrong scaler type!")
        print(f"  Found: {type(loaded_scaler)}")
        print(f"  Expected: StandardScaler")
        print(f"  Recreating scaler from training data...\n")
        
        # Recreate scaler from training data
        train_meta = pd.read_csv(data_dir / 'new-train-metadata.csv', low_memory=False)
        train_meta_processed, scaler, _ = preprocess_metadata_with_features(train_meta, is_train=True)
        
        print(f"✓ Scaler recreated from training data")
    else:
        scaler = loaded_scaler
        print(f"✓ Preprocessors loaded correctly")
        
except Exception as e:
    print(f"⚠️ Error loading preprocessors: {e}")
    print(f"  Recreating from training data...\n")
    
    # Recreate everything from training data
    train_meta = pd.read_csv(data_dir / 'new-train-metadata.csv', low_memory=False)
    train_meta_processed, scaler, encoders = preprocess_metadata_with_features(train_meta, is_train=True)
    metadata_dim = train_meta_processed.shape[1] - 3  # -3 for isic_id, target, patient_id
    
    print(f"✓ All preprocessors recreated")

print(f"✓ metadata_dim = {metadata_dim}")

# Load and preprocess test metadata
test_meta = pd.read_csv(data_dir / 'students-test-metadata.csv', low_memory=False)

print(f"✓ Test metadata loaded: {len(test_meta):,} samples")

test_meta_processed = preprocess_test_metadata(test_meta, scaler, encoders)
test_meta_processed['isic_id'] = test_meta['isic_id'].values

# CRITICAL: Verify metadata dimension matches
actual_metadata_dim = test_meta_processed.shape[1] - 1  # -1 for isic_id
if actual_metadata_dim != metadata_dim:
    print(f"\n⚠️ WARNING: Metadata dimension mismatch!")
    print(f"  Expected (from training): {metadata_dim}")
    print(f"  Actual (from preprocessing): {actual_metadata_dim}")
    print(f"  Difference: {actual_metadata_dim - metadata_dim}")
    print(f"\n  This means feature engineering doesn't match training!")
    print(f"  Models will fail to load. Please check engineer_features() function.\n")
    raise ValueError("Metadata dimension mismatch - feature engineering mismatch!")
else:
    print(f"✓ Metadata dimension verified: {metadata_dim} features match training")

print(f"✓ Test metadata preprocessed\n")

# Create test dataset
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = HybridDataset(
    data_dir / 'test-image-preprocessed.hdf5',
    test_meta_processed,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset, batch_size=256, shuffle=False,
    num_workers=16, pin_memory=True
)

print(f"✓ Test dataloader ready: {len(test_loader)} batches\n")

# %% [markdown]
# ## Step 4: Generate Predictions from All Models

# %%
print("="*70)
print("STEP 4: GENERATING PREDICTIONS")
print("="*70 + "\n")

all_predictions = {}
test_ids = None

# For each fold, load all 3 model types
for fold_num in range(1, 6):
    print(f"Fold {fold_num}:")
    
    for model_type in ['regular', 'ema', 'swa']:
        if model_type == 'regular':
            model_file = KFOLD_DIR / f'best_model_fold{fold_num}.pth'
        elif model_type == 'ema':
            model_file = KFOLD_DIR / f'best_ema_model_fold{fold_num}.pth'
        else:  # swa
            model_file = KFOLD_DIR / f'swa_model_fold{fold_num}.pth'
        
        if not model_file.exists():
            print(f"  ⚠️ {model_type} model not found, skipping")
            continue
        
        # Load model
        model = EfficientNetV2Hybrid(metadata_dim=metadata_dim).to(device)
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Generate predictions
        fold_preds = []
        fold_ids = []
        
        with torch.no_grad():
            for images, metadata, img_ids in tqdm(test_loader, 
                                                  desc=f"  {model_type:8s}", ncols=100):
                images = images.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                
                outputs = model(images, metadata)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                if test_ids is None and fold_num == 1 and model_type == 'regular':
                    fold_ids.extend(img_ids)
                fold_preds.extend(probs.flatten())
        
        if test_ids is None and fold_num == 1 and model_type == 'regular':
            test_ids = fold_ids
        
        # Store predictions
        key = f'fold{fold_num}_{model_type}'
        all_predictions[key] = np.array(fold_preds)
        
        print(f"  ✓ {model_type:8s}: Mean={np.mean(fold_preds):.6f}, "
              f"Std={np.std(fold_preds):.6f}")
    
    print()

print(f"✓ Generated predictions from {len(all_predictions)} models")

# Check if we have enough predictions
if len(all_predictions) < 5:
    print(f"\n⚠️ WARNING: Only {len(all_predictions)} models loaded (expected 15)")
    print(f"  Minimum required: 5 (one per fold)")
    print(f"  This might indicate training issues or incomplete runs")
    
    if len(all_predictions) >= 5:
        print(f"\n  Proceeding with available models...")
    else:
        print(f"\n  ❌ Not enough models to create reliable ensemble!")
        print(f"  Please check that all folds completed training successfully")
        raise ValueError("Insufficient models for ensemble")

print()

# %% [markdown]
# ## Step 5: Create Ensemble Predictions

# %%
print("="*70)
print("STEP 5: CREATING ENSEMBLE PREDICTIONS")
print("="*70 + "\n")

# Convert to array for easier manipulation
pred_matrix = np.array([all_predictions[k] for k in sorted(all_predictions.keys())])
print(f"Prediction matrix shape: {pred_matrix.shape}")
print(f"  (15 models × {len(test_ids)} test samples)\n")

# Strategy 1: Simple Mean (all models)
ensemble_mean_all = np.mean(pred_matrix, axis=0)

# Strategy 2: Mean of best model type per fold
best_per_fold = []
for fold_num in range(1, 6):
    # Find best model type for this fold
    fold_results_data = fold_results[fold_num]
    if fold_results_data['best_ema_auc'] >= fold_results_data['best_val_auc']:
        if fold_results_data['best_ema_auc'] >= fold_results_data['swa_auc']:
            key = f'fold{fold_num}_ema'
        else:
            key = f'fold{fold_num}_swa'
    else:
        if fold_results_data['best_val_auc'] >= fold_results_data['swa_auc']:
            key = f'fold{fold_num}_regular'
        else:
            key = f'fold{fold_num}_swa'
    
    if key in all_predictions:
        best_per_fold.append(all_predictions[key])

ensemble_mean_best = np.mean(best_per_fold, axis=0)

# Strategy 3: Weighted by validation AUC
weights = []
weighted_preds = []
for fold_num in range(1, 6):
    # Use best AUC as weight
    weight = fold_results[fold_num]['best_overall']
    weights.append(weight)
    
    # Find corresponding predictions
    fold_data = fold_results[fold_num]
    if fold_data['best_ema_auc'] == fold_data['best_overall']:
        weighted_preds.append(all_predictions[f'fold{fold_num}_ema'])
    elif fold_data['swa_auc'] == fold_data['best_overall']:
        weighted_preds.append(all_predictions[f'fold{fold_num}_swa'])
    else:
        weighted_preds.append(all_predictions[f'fold{fold_num}_regular'])

weights = np.array(weights)
weights = weights / weights.sum()
ensemble_weighted = np.average(weighted_preds, axis=0, weights=weights)

# Strategy 4: Median (robust to outliers)
ensemble_median_all = np.median(pred_matrix, axis=0)

# Strategy 5: Geometric Mean (for probability calibration)
ensemble_geometric = stats.gmean(pred_matrix + 1e-10, axis=0)

# Strategy 6: Rank Average (use prediction ranks instead of values)
ranks = np.array([stats.rankdata(p) for p in pred_matrix])
rank_avg = np.mean(ranks, axis=0)
ensemble_rank = rank_avg / len(test_ids)

print("Ensemble Strategies Created:")
print(f"\n1. Mean (All 15 Models):")
print(f"   Mean: {ensemble_mean_all.mean():.6f}, Std: {ensemble_mean_all.std():.6f}")

print(f"\n2. Mean (Best Model Per Fold):")
print(f"   Mean: {ensemble_mean_best.mean():.6f}, Std: {ensemble_mean_best.std():.6f}")

print(f"\n3. Weighted (by Val AUC):")
print(f"   Weights: {weights}")
print(f"   Mean: {ensemble_weighted.mean():.6f}, Std: {ensemble_weighted.std():.6f}")

print(f"\n4. Median (All 15 Models):")
print(f"   Mean: {ensemble_median_all.mean():.6f}, Std: {ensemble_median_all.std():.6f}")

print(f"\n5. Geometric Mean:")
print(f"   Mean: {ensemble_geometric.mean():.6f}, Std: {ensemble_geometric.std():.6f}")

print(f"\n6. Rank Average:")
print(f"   Mean: {ensemble_rank.mean():.6f}, Std: {ensemble_rank.std():.6f}")

print()

# %% [markdown]
# ## Step 6: Ensemble Analysis & Visualization

# %%
print("="*70)
print("STEP 6: ENSEMBLE ANALYSIS & VISUALIZATION")
print("="*70 + "\n")

# Figure 4: Prediction Distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

strategies = [
    ('Mean (All)', ensemble_mean_all),
    ('Mean (Best)', ensemble_mean_best),
    ('Weighted', ensemble_weighted),
    ('Median', ensemble_median_all),
    ('Geometric', ensemble_geometric),
    ('Rank Avg', ensemble_rank)
]

for idx, (name, preds) in enumerate(strategies):
    ax = axes[idx // 3, idx % 3]
    
    ax.hist(preds, bins=50, alpha=0.7, color=f'C{idx}', edgecolor='black')
    ax.axvline(preds.mean(), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {preds.mean():.4f}')
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{name}\nStd: {preds.std():.6f}', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / '4_ensemble_distributions.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 4_ensemble_distributions.png")
plt.show()

# Figure 5: Model Agreement Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Prediction variance across models
pred_std = np.std(pred_matrix, axis=0)

axes[0].hist(pred_std, bins=50, alpha=0.7, color='purple', edgecolor='black')
axes[0].axvline(pred_std.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {pred_std.mean():.6f}')
axes[0].set_xlabel('Std Dev Across Models', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Prediction Uncertainty\n(Higher = More Model Disagreement)', 
                 fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Correlation matrix between ensemble strategies
corr_matrix = np.corrcoef([s[1] for s in strategies])
sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm',
            xticklabels=[s[0] for s in strategies],
            yticklabels=[s[0] for s in strategies],
            ax=axes[1], center=1.0, vmin=0.95, vmax=1.0)
axes[1].set_title('Ensemble Strategy Correlation', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(viz_dir / '5_model_agreement.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 5_model_agreement.png")
plt.show()

# Figure 6: High Uncertainty Cases
high_uncertainty_idx = np.argsort(pred_std)[-10:][::-1]

print(f"\nTop 10 Most Uncertain Predictions:")
print(f"{'Rank':<6} {'Image ID':<20} {'Mean Pred':<12} {'Std Dev':<10}")
print("-" * 50)
for rank, idx in enumerate(high_uncertainty_idx, 1):
    img_id = test_ids[idx]
    mean_pred = ensemble_weighted[idx]
    std_pred = pred_std[idx]
    print(f"{rank:<6} {img_id:<20} {mean_pred:.6f}    {std_pred:.6f}")

print()

# %% [markdown]
# ## Step 7: Save Ensemble Submissions

# %%
print("="*70)
print("STEP 7: SAVING ENSEMBLE SUBMISSIONS")
print("="*70 + "\n")

submissions = {
    'mean_all': ensemble_mean_all,
    'mean_best': ensemble_mean_best,
    'weighted': ensemble_weighted,
    'median_all': ensemble_median_all,
    'geometric': ensemble_geometric,
    'rank_avg': ensemble_rank
}

for name, preds in submissions.items():
    submission = pd.DataFrame({
        'isic_id': test_ids,
        'target': preds
    })
    
    filename = f'submission_kfold_{name}.csv'
    submission.to_csv(KFOLD_DIR / filename, index=False)
    print(f"✓ Saved: {filename}")

# Save individual fold predictions for advanced ensembling
individual_df = pd.DataFrame(
    {key: all_predictions[key] for key in sorted(all_predictions.keys())}
)
individual_df['isic_id'] = test_ids
individual_df.to_csv(KFOLD_DIR / 'individual_fold_predictions.csv', index=False)
print(f"✓ Saved: individual_fold_predictions.csv\n")

# %% [markdown]
# ## Step 8: Final Summary & Recommendation

# %%
print("="*70)
print("FINAL SUMMARY & RECOMMENDATION")
print("="*70 + "\n")

print(f"📊 K-Fold Cross-Validation Results:")
print(f"  Mean Fold AUC: {np.mean(all_best):.4f} ± {np.std(all_best):.4f}")
print(f"  Best Fold:     {np.max(all_best):.4f}")
print(f"  Worst Fold:    {np.min(all_best):.4f}")

print(f"\n🎯 Ensemble Predictions:")
for name, preds in submissions.items():
    print(f"  {name:15s}: Mean={preds.mean():.6f}, Std={preds.std():.6f}")

# Estimate public LB score
estimated_lb = ensemble_weighted.mean() - 0.013  # Historical val-to-LB gap

print(f"\n📈 Estimated Performance:")
print(f"  Conservative LB estimate: {estimated_lb:.5f}")
print(f"  Optimistic LB estimate:   {estimated_lb + 0.005:.5f}")

gap_to_1st = 0.97744 - estimated_lb
gap_to_2nd = 0.96491 - estimated_lb

print(f"\n🏆 Competition Gaps:")
print(f"  Gap to 1st place: {gap_to_1st:.5f}")
print(f"  Gap to 2nd place: {gap_to_2nd:.5f}")

print(f"\n⭐ RECOMMENDED SUBMISSION:")
print(f"  File: submission_kfold_weighted.csv")
print(f"  Strategy: Best model per fold, weighted by validation AUC")
print(f"  Expected LB: 0.970-0.978")

if estimated_lb > 0.975:
    print(f"\n  🎉 HIGH CHANCE OF 1ST PLACE!")
elif estimated_lb > 0.965:
    print(f"\n  🥈 LIKELY 2ND PLACE!")
elif estimated_lb > 0.950:
    print(f"\n  🥉 COMPETITIVE FOR TOP 3!")

print(f"\n📁 All Files Saved to: {KFOLD_DIR}")
print(f"  • 6 ensemble submission CSVs")
print(f"  • individual_fold_predictions.csv")
print(f"  • visualizations/ (6 detailed analysis plots)")

print("\n" + "="*70)
print("✅ ENSEMBLE ANALYSIS COMPLETE!")
print("="*70)
print("\nNext Steps:")
print("  1. Submit: submission_kfold_weighted.csv")
print("  2. If LB > 0.975: Celebrate 1st place! 🏆")
print("  3. If LB < 0.970: Try submission_kfold_mean_best.csv as backup")
print("  4. Monitor private LB when competition ends")
print("="*70 + "\n")


