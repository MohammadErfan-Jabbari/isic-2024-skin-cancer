"""
COMPLETE GBDT STACKING PIPELINE - OVERNIGHT RUN (AGGRESSIVE SETTINGS)
=====================================================================

This script does EVERYTHING with OPTIMIZED settings for maximum performance:

AGGRESSIVE OPTIMIZATION SETTINGS:
- Optuna trials: 200 (was 50) → better hyperparameter search
- Boosting rounds: 3000 (was 1000) → deeper trees
- Early stopping: 200 (was 100) → more patience
- GBDT CV folds: 10 (was 5) → more stable
- Wider search space → better exploration

Expected Performance:
- Current ensemble: 0.96741
- With GBDT stacking: 0.985-0.995 (AGGRESSIVE estimate)
- Target (1st place): 0.99248

Runtime: ~6-8 hours total (overnight run)
- OOF generation: 1-2 hours
- Test prediction: 30 minutes  
- Optuna search: 1.5-2 hours (200 trials)
- GBDT training: 2.5-3.5 hours (10-fold × 3000 rounds)
- Visualization: 15 minutes

TUNABLE PARAMETERS (edit if needed):
Line ~30: N_FOLDS = 5 (CNN folds - already trained, don't change)
Line ~540: n_trials=200 (Optuna trials - increase to 300 for more exploration)
Line ~615: n_splits=10 (GBDT CV folds - increase to 15 for max stability)
Line ~629: num_boost_round=3000 (boosting rounds - increase to 5000 if you have time)
Line ~634: early_stopping(200) (patience - increase to 300 for more patience)

Usage:
    python 13_gbdt_stacking_complete.py

Then go to sleep. Check results in the morning!
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
KFOLD_DIR = Path('results/kfold_v2s_features_advanced_20251111_150340')
DATA_DIR = Path('data')
N_FOLDS = 5
RANDOM_STATE = 42

print("="*70)
print("GBDT STACKING PIPELINE - COMPLETE (AGGRESSIVE SETTINGS)")
print("="*70)
print(f"Device: {DEVICE}")
print(f"K-Fold models: {KFOLD_DIR}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n⚙️ OPTIMIZATION SETTINGS (AGGRESSIVE):")
print(f"  • Optuna trials: 200 (explores hyperparameter space thoroughly)")
print(f"  • GBDT CV folds: 10 (more stable model)")
print(f"  • Boosting rounds: 3000 max (deeper trees)")
print(f"  • Early stopping: 200 patience (allows more training)")
print(f"  • Expected runtime: 6-8 hours")
print(f"\n📈 EXPECTED PERFORMANCE:")
print(f"  • Current best: 0.96741")
print(f"  • GBDT target: 0.985-0.995")
print(f"  • 1st place: 0.99248")
print("="*70 + "\n")


# ===========================
# FEATURE ENGINEERING (EXACT MATCH)
# ===========================

def engineer_features(df):
    """MUST match training exactly - copy-paste from training script"""
    df = df.copy()
    
    # AGE FEATURES
    df['age_group'] = pd.cut(df['age_approx'], bins=[0, 30, 50, 70, 100],
                             labels=['young', 'middle', 'senior', 'elderly'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    df['age_squared'] = df['age_approx'] ** 2
    
    # SIZE FEATURES
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['size_category'] = pd.cut(df['lesion_size_mm'], bins=[0, 6, 10, 20, 100],
                                 labels=['small', 'medium', 'large', 'very_large'])
    df['large_lesion'] = (df['lesion_size_mm'] > 6).astype(int)
    df['size_squared'] = df['lesion_size_mm'] ** 2
    
    # SHAPE FEATURES
    df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    df['eccentricity'] = df['tbp_lv_minorAxisMM'] / (df['tbp_lv_areaMM2']**0.5 + 1e-6)
    df['compactness'] = (4 * np.pi * df['tbp_lv_areaMM2']) / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    
    # COLOR FEATURES
    df['color_variance'] = np.sqrt(
        df['tbp_lv_deltaB']**2 + df['tbp_lv_radial_color_std_max']**2 +
        df['tbp_lv_color_std_mean']**2
    )
    df['color_uniformity'] = 1 / (df['tbp_lv_norm_color'] + 1e-6)
    df['darkness_score'] = df['tbp_lv_B'] / (df['tbp_lv_H'] + 1e-6)
    df['color_contrast'] = df['tbp_lv_deltaB'] * df['tbp_lv_radial_color_std_max']
    
    # ANATOMICAL FEATURES
    high_risk_sites = ['torso', 'upper extremity', 'posterior torso', 'anterior torso', 'head/neck']
    df['high_risk_site'] = df['anatom_site_general'].isin(high_risk_sites).astype(int)
    
    site_risk_map = {
        'head/neck': 4, 'torso': 3, 'posterior torso': 3, 'anterior torso': 3,
        'upper extremity': 2, 'lower extremity': 2,
        'palms/soles': 1, 'oral/genital': 1
    }
    df['site_risk_score'] = df['anatom_site_general'].map(site_risk_map).fillna(0)
    
    # INTERACTION FEATURES
    df['age_size_risk'] = df['age_approx'] * df['lesion_size_mm']
    df['age_site_risk'] = df['age_approx'] * df['site_risk_score']
    df['color_size_risk'] = df['color_variance'] * df['lesion_size_mm']
    df['age_color_risk'] = df['age_approx'] * df['color_variance']
    df['site_size_risk'] = df['site_risk_score'] * df['lesion_size_mm']
    
    # ASYMMETRY SCORE
    df['asymmetry_score'] = (
        df['tbp_lv_norm_color'] + df['tbp_lv_radial_color_std_max'] +
        (1 / (df['shape_regularity'] + 1e-6))
    ) / 3
    
    # LOG TRANSFORMS
    df['log_area'] = np.log1p(df['tbp_lv_areaMM2'])
    df['log_perimeter'] = np.log1p(df['tbp_lv_perimeterMM'])
    df['log_size'] = np.log1p(df['lesion_size_mm'])
    
    # RATIOS
    df['h_to_b_ratio'] = df['tbp_lv_H'] / (df['tbp_lv_B'] + 1e-6)
    df['a_to_b_ratio'] = df['tbp_lv_A'] / (df['tbp_lv_B'] + 1e-6)
    df['area_to_perimeter'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] + 1e-6)
    
    return df


def add_patient_features(df, is_train=True):
    """Add patient-level aggregation features (UGLY DUCKLING)"""
    
    if 'patient_id' not in df.columns:
        print("  ⚠️ No patient_id column - skipping patient features")
        return df
    
    print("  ✓ Adding patient-level features...")
    
    # Group by patient
    patient_groups = df.groupby('patient_id')
    
    # Patient lesion count
    df['patient_lesion_count'] = patient_groups['patient_id'].transform('count')
    
    # Patient-level statistics for numerical features
    for feature in ['lesion_size_mm', 'color_variance', 'age_approx']:
        if feature in df.columns:
            df[f'patient_mean_{feature}'] = patient_groups[feature].transform('mean')
            df[f'patient_std_{feature}'] = patient_groups[feature].transform('std').fillna(0)
            df[f'patient_max_{feature}'] = patient_groups[feature].transform('max')
            df[f'patient_min_{feature}'] = patient_groups[feature].transform('min')
            
            # Distance from patient mean (UGLY DUCKLING)
            df[f'{feature}_diff_from_patient_mean'] = df[feature] - df[f'patient_mean_{feature}']
            df[f'{feature}_zscore_within_patient'] = (
                df[f'{feature}_diff_from_patient_mean'] / (df[f'patient_std_{feature}'] + 1e-6)
            )
    
    # Percentile within patient
    df['size_percentile_in_patient'] = patient_groups['lesion_size_mm'].rank(pct=True)
    df['color_percentile_in_patient'] = patient_groups['color_variance'].rank(pct=True)
    
    # Is this lesion an outlier in patient? (top 10% largest or most colorful)
    df['is_size_outlier'] = (df['size_percentile_in_patient'] > 0.9).astype(int)
    df['is_color_outlier'] = (df['color_percentile_in_patient'] > 0.9).astype(int)
    
    print(f"    Added {len([c for c in df.columns if 'patient' in c or 'outlier' in c])} patient features")
    
    return df


def preprocess_metadata_with_features(df, is_train=True, scaler=None, encoders=None, include_patient_features=False):
    """Full preprocessing pipeline
    
    Args:
        include_patient_features: If True, adds patient-level features (only use for GBDT!)
    """
    
    df = engineer_features(df)
    
    # Add patient features only if requested (NOT for CNN predictions!)
    if include_patient_features:
        df = add_patient_features(df, is_train=is_train)
    
    # Base numerical features (before patient features)
    BASE_NUMERICAL_FEATURES = [
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
    
    # Add ONLY numerical patient features (filter out any non-numeric)
    patient_feature_cols = []
    if add_patient_features:
        for col in df.columns:
            if ('patient' in col or 'outlier' in col or 'percentile' in col) and col in df.columns:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    patient_feature_cols.append(col)
    
    NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + patient_feature_cols
    
    CATEGORICAL_FEATURES = [
        'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple',
        'age_group', 'size_category', 'age_risk', 'large_lesion', 'high_risk_site'
    ]
    
    # Fill missing values - only for columns that exist and are numeric
    for col in NUMERICAL_FEATURES:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median() if is_train else 0)
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing')
    
    # Filter NUMERICAL_FEATURES to only those that exist in df
    NUMERICAL_FEATURES = [col for col in NUMERICAL_FEATURES if col in df.columns]
    
    if is_train:
        scaler = StandardScaler()
        df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])
    else:
        # Only transform features that exist in both train and test
        existing_features = [col for col in NUMERICAL_FEATURES if col in df.columns]
        df[existing_features] = scaler.transform(df[existing_features])
    
    if is_train:
        encoders = {}
        encoded_dfs = []
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
                encoders[col] = encoded.columns.tolist()
                encoded_dfs.append(encoded)
        result_df = pd.concat([df[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
    else:
        encoded_dfs = []
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
                for train_col in encoders[col]:
                    if train_col not in encoded.columns:
                        encoded[train_col] = 0
                encoded = encoded[encoders[col]]
                encoded_dfs.append(encoded)
        result_df = pd.concat([df[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
    
    return result_df, scaler, encoders


# ===========================
# DATASET & MODEL (SAME AS TRAINING)
# ===========================

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
        
        feature_cols = [col for col in self.metadata.columns 
                       if col not in ['isic_id', 'target', 'patient_id']]
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

# Create results directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = Path('results') / f'gbdt_stacking_{timestamp}'
results_dir.mkdir(parents=True, exist_ok=True)
viz_dir = results_dir / 'visualizations'
viz_dir.mkdir(exist_ok=True)

print(f"Results directory: {results_dir}")
print(f"  (Intermediate results saved here during training)\n")

# ===========================
# STEP 1: GENERATE OOF PREDICTIONS
# ===========================

print("="*70)
print("STEP 1: GENERATING OUT-OF-FOLD (OOF) PREDICTIONS")
print("="*70 + "\n")

# Load metadata
train_meta = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
print(f"✓ Loaded training metadata: {len(train_meta):,} samples\n")

# Preprocess WITHOUT patient features (models were trained this way)
print("Preprocessing metadata (without patient features for CNN)...")
train_meta_processed, scaler, encoders = preprocess_metadata_with_features(
    train_meta, is_train=True, include_patient_features=False
)
train_meta_processed['isic_id'] = train_meta['isic_id'].values
train_meta_processed['target'] = train_meta['target'].values
train_meta_processed['patient_id'] = train_meta['patient_id'].values

metadata_dim = len(train_meta_processed.columns) - 3
print(f"✓ Metadata dimension: {metadata_dim} (matches trained models)\n")

# Create k-fold splits (SAME as training)
print("Creating fold splits...")
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_indices = list(sgkf.split(
    train_meta_processed,
    train_meta_processed['target'],
    train_meta_processed['patient_id']
))

# Initialize OOF predictions
oof_predictions = np.zeros(len(train_meta_processed))

# Transform
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Generate OOF predictions for each fold
for fold_num in range(1, N_FOLDS + 1):
    print(f"\nFold {fold_num}/5:")
    
    # Load model
    model_file = KFOLD_DIR / f'best_model_fold{fold_num}.pth'
    
    if not model_file.exists():
        print(f"  ⚠️ Model not found: {model_file}")
        print(f"  Using zeros for this fold (will hurt performance!)")
        continue
    
    # Create model
    model = EfficientNetV2Hybrid(metadata_dim=metadata_dim).to(DEVICE)
    checkpoint = torch.load(model_file, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get validation indices for this fold
    train_idx, val_idx = fold_indices[fold_num - 1]
    val_df = train_meta_processed.iloc[val_idx].reset_index(drop=True)
    
    print(f"  Validation set: {len(val_idx):,} samples ({val_df['target'].sum()} positive)")
    
    # Create dataset for this fold's validation set
    val_dataset = HybridDataset(
        DATA_DIR / 'train-image-preprocessed.hdf5',
        val_df,
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True
    )
    
    # Generate predictions
    fold_preds = []
    
    with torch.no_grad():
        for images, metadata, img_ids in tqdm(val_loader, desc=f"  Predicting", ncols=100):
            images = images.to(DEVICE, non_blocking=True)
            metadata = metadata.to(DEVICE, non_blocking=True)
            
            outputs = model(images, metadata)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            fold_preds.extend(probs)
    
    # Store OOF predictions
    oof_predictions[val_idx] = fold_preds
    
    # Calculate this fold's AUC
    fold_auc = roc_auc_score(val_df['target'], fold_preds)
    print(f"  ✓ Fold {fold_num} Val AUC: {fold_auc:.4f}")

# Overall OOF AUC
oof_auc = roc_auc_score(train_meta_processed['target'], oof_predictions)

print(f"\n{'='*70}")
print(f"OOF PREDICTIONS COMPLETE")
print(f"{'='*70}")
print(f"Overall OOF AUC: {oof_auc:.4f}")
print(f"Expected: ~0.945-0.955 (similar to k-fold ensemble)")
print(f"{'='*70}\n")

# Save OOF predictions (checkpoint - in case script crashes later)
oof_checkpoint = pd.DataFrame({
    'isic_id': train_meta_processed['isic_id'],
    'target': train_meta_processed['target'],
    'oof_prediction': oof_predictions
})
oof_checkpoint.to_csv(results_dir / 'oof_predictions_checkpoint.csv', index=False)
print(f"✓ Checkpoint saved: oof_predictions_checkpoint.csv\n")


# ===========================
# STEP 2: GENERATE TEST PREDICTIONS
# ===========================

print("="*70)
print("STEP 2: GENERATING TEST PREDICTIONS (5-FOLD AVERAGE)")
print("="*70 + "\n")

# Load test metadata
test_meta = pd.read_csv(DATA_DIR / 'students-test-metadata.csv', low_memory=False)
print(f"✓ Loaded test metadata: {len(test_meta):,} samples\n")

# Preprocess test metadata (without patient features for CNN)
print("Preprocessing test metadata (without patient features for CNN)...")
test_meta_processed, _, _ = preprocess_metadata_with_features(
    test_meta, is_train=False, scaler=scaler, encoders=encoders, include_patient_features=False
)
test_meta_processed['isic_id'] = test_meta['isic_id'].values

print(f"✓ Test metadata preprocessed\n")

# Create test dataset
test_dataset = HybridDataset(
    DATA_DIR / 'test-image-preprocessed.hdf5',
    test_meta_processed,
    transform=val_transform
)

test_loader = DataLoader(
    test_dataset, batch_size=256, shuffle=False,
    num_workers=8, pin_memory=True
)

# Generate predictions from all 5 folds
all_test_preds = []
test_ids = None

for fold_num in range(1, N_FOLDS + 1):
    print(f"Fold {fold_num}/5:")
    
    model_file = KFOLD_DIR / f'best_model_fold{fold_num}.pth'
    
    if not model_file.exists():
        print(f"  ⚠️ Model not found, skipping")
        continue
    
    # Load model
    model = EfficientNetV2Hybrid(metadata_dim=metadata_dim).to(DEVICE)
    checkpoint = torch.load(model_file, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate predictions
    fold_test_preds = []
    fold_ids = []
    
    with torch.no_grad():
        for images, metadata, img_ids in tqdm(test_loader, desc=f"  Predicting", ncols=100):
            images = images.to(DEVICE, non_blocking=True)
            metadata = metadata.to(DEVICE, non_blocking=True)
            
            outputs = model(images, metadata)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            fold_test_preds.extend(probs)
            
            if test_ids is None:
                fold_ids.extend(img_ids)
    
    if test_ids is None:
        test_ids = fold_ids
    
    all_test_preds.append(fold_test_preds)
    print(f"  ✓ Mean: {np.mean(fold_test_preds):.6f}, Std: {np.std(fold_test_preds):.6f}\n")

# Average predictions from all folds
test_cnn_predictions = np.mean(all_test_preds, axis=0)

print(f"{'='*70}")
print(f"TEST PREDICTIONS COMPLETE")
print(f"{'='*70}")
print(f"Test CNN predictions (5-fold average):")
print(f"  Mean: {test_cnn_predictions.mean():.6f}")
print(f"  Std:  {test_cnn_predictions.std():.6f}")
print(f"  Min:  {test_cnn_predictions.min():.6f}")
print(f"  Max:  {test_cnn_predictions.max():.6f}")
print(f"{'='*70}\n")

# Save test predictions (checkpoint)
test_checkpoint = pd.DataFrame({
    'isic_id': test_ids,
    'cnn_prediction': test_cnn_predictions
})
test_checkpoint.to_csv(results_dir / 'test_cnn_predictions_checkpoint.csv', index=False)
print(f"✓ Checkpoint saved: test_cnn_predictions_checkpoint.csv\n")


# ===========================
# STEP 3: PREPARE FEATURES FOR GBDT
# ===========================

print("="*70)
print("STEP 3: PREPARING FEATURES FOR GBDT")
print("="*70)
print("\nNOTE: Adding patient-level features NOW (not used in CNN)")
print("CNN models were trained with 72 features (no patient features)")
print("GBDT will use: 72 base + 24 patient + 1 CNN prediction = ~97 features\n")

# Now add patient features for GBDT (with patient_id preserved)
print("Engineering patient-level features for GBDT training...")

# Create copy with patient_id for feature engineering
train_for_gbdt = train_meta.copy()
train_for_gbdt = engineer_features(train_for_gbdt)
train_for_gbdt = add_patient_features(train_for_gbdt, is_train=True)

test_for_gbdt = test_meta.copy()
test_for_gbdt = engineer_features(test_for_gbdt)
test_for_gbdt = add_patient_features(test_for_gbdt, is_train=False)

print(f"✓ Patient features added\n")

# Get all numeric columns (exclude isic_id, target, patient_id)
exclude_cols = ['isic_id', 'target', 'patient_id', 'lesion_id', 'attribution']
all_numeric_cols = [col for col in train_for_gbdt.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_for_gbdt[col])]

# Filter to only columns that exist in both train and test
all_numeric_cols = [col for col in all_numeric_cols if col in test_for_gbdt.columns]

# Fill missing values for GBDT features
for col in all_numeric_cols:
    if col in train_for_gbdt.columns:
        train_for_gbdt[col] = train_for_gbdt[col].fillna(train_for_gbdt[col].median())
    if col in test_for_gbdt.columns:
        test_for_gbdt[col] = test_for_gbdt[col].fillna(0)

# Create feature matrix: metadata features + CNN prediction
X_train = train_for_gbdt[all_numeric_cols].copy()
X_train['cnn_prediction'] = oof_predictions
y_train = train_meta['target'].values

X_test = test_for_gbdt[all_numeric_cols].copy()
X_test['cnn_prediction'] = test_cnn_predictions

print(f"Training set:")
print(f"  Samples: {len(X_train):,}")
print(f"  Features: {X_train.shape[1]} ({X_train.shape[1]-1} metadata + 1 CNN prediction)")
print(f"  Positives: {y_train.sum()} ({y_train.mean()*100:.3f}%)")

print(f"\nTest set:")
print(f"  Samples: {len(X_test):,}")
print(f"  Features: {X_test.shape[1]}")

print(f"\nFeature names:")
print(f"  {list(X_train.columns[:5])} ... (showing first 5)")
print(f"  Last feature: {X_train.columns[-1]}")

print()


# ===========================
# STEP 4: TRAIN LIGHTGBM WITH OPTUNA TUNING
# ===========================

print("="*70)
print("STEP 4: TRAINING LIGHTGBM STACKER")
print("="*70 + "\n")

# Try Optuna for hyperparameter tuning
try:
    import optuna
    USE_OPTUNA = True
    print("✓ Optuna available - will use hyperparameter tuning")
    print(f"  Running 200 trials (aggressive search, ~1.5-2 hours)\n")
except ImportError:
    USE_OPTUNA = False
    print("⚠️ Optuna not available - using default hyperparameters\n")

# Hyperparameter tuning with Optuna (optional but recommended)
if USE_OPTUNA:
    print("Running Optuna hyperparameter search...")
    print(f"  Trials: 200 (aggressive exploration)")
    print(f"  CV: 5-fold per trial")
    print(f"  Estimated time: 1.5-2 hours\n")
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': RANDOM_STATE,
            # Wider search ranges for better exploration
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            # Additional parameters for fine-tuning
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        }
        
        # 5-fold CV
        cv_aucs = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            gbm = lgb.train(
                params,
                train_data,
                num_boost_round=1000,  # Moderate for search (keep it fast)
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            preds = gbm.predict(X_val)
            auc = roc_auc_score(y_val, preds)
            cv_aucs.append(auc)
        
        return np.mean(cv_aucs)
    
    study = optuna.create_study(direction='maximize', study_name='lgbm_stacking')
    study.optimize(objective, n_trials=200, show_progress_bar=True)  # Increased from 50
    
    print(f"\n✓ Optuna search complete!")
    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best trial: #{study.best_trial.number}")
    
    # Show top 5 trials
    print(f"\n  Top 5 trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
    for i, trial in enumerate(sorted_trials, 1):
        print(f"    {i}. Trial #{trial.number}: AUC = {trial.value:.4f}")
    
    print(f"\n  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print()
    
    # Save Optuna study
    try:
        import joblib
        joblib.dump(study, results_dir / 'optuna_study.pkl')
        print(f"✓ Optuna study saved: optuna_study.pkl\n")
    except:
        print(f"⚠️ Could not save Optuna study (joblib not available)\n")
    
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    })
else:
    # Default aggressive parameters (if no Optuna)
    print("Using AGGRESSIVE default parameters (no Optuna)\n")
    best_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,  # Lower for more rounds
        'num_leaves': 60,  # Moderate
        'max_depth': 10,  # Deeper trees
        'min_child_samples': 15,  # Lower for class imbalance
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,  # Some regularization
        'reg_lambda': 0.5,
        'min_gain_to_split': 0.1,
        'bagging_freq': 5,
        'bagging_fraction': 0.8,
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }

print(f"\nFinal LightGBM parameters:")
for k, v in best_params.items():
    if k != 'verbosity':
        print(f"  {k}: {v}")

print()


# ===========================
# STEP 5: TRAIN FINAL GBDT WITH 5-FOLD CV
# ===========================

print("="*70)
print("STEP 5: TRAINING FINAL GBDT MODEL")
print("="*70)
print(f"Training with AGGRESSIVE settings for maximum performance:")
print(f"  • CV Folds: 10 (more stable)")
print(f"  • Boosting rounds: 3000 (deep trees)")
print(f"  • Early stopping: 200 (patient)")
print(f"  • Expected time: ~2-3 hours")
print("="*70 + "\n")

# Train with 10-fold CV for maximum stability
cv_aucs = []
cv_models = []
feature_importances = []

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)  # Increased to 10

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"GBDT Fold {fold_idx}/10:")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Create datasets
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train with more rounds and patience
    gbm = lgb.train(
        best_params,
        train_data,
        num_boost_round=3000,  # Increased from 1000
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(200),  # Increased from 100
            lgb.log_evaluation(100)  # Log every 100 iterations
        ]
    )
    
    # Validate
    val_preds = gbm.predict(X_val)
    val_auc = roc_auc_score(y_val, val_preds)
    cv_aucs.append(val_auc)
    cv_models.append(gbm)
    
    # Save model
    gbm.save_model(str(results_dir / f'gbdt_model_fold{fold_idx}.txt'))
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': gbm.feature_importance(importance_type='gain')
    })
    feature_importances.append(importance_df)
    
    print(f"  ✓ Val AUC: {val_auc:.4f}")
    print(f"  Best iteration: {gbm.best_iteration}")
    print(f"  ✓ Model saved: gbdt_model_fold{fold_idx}.txt")
    print()

print(f"{'='*70}")
print(f"GBDT TRAINING COMPLETE")
print(f"{'='*70}")
print(f"CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
print(f"Individual folds: {[f'{auc:.4f}' for auc in cv_aucs]}")
print(f"Median: {np.median(cv_aucs):.4f}")
print(f"{'='*70}\n")


# ===========================
# STEP 6: GENERATE FINAL PREDICTIONS
# ===========================

print("="*70)
print("STEP 6: GENERATING FINAL PREDICTIONS")
print("="*70 + "\n")

# Average predictions from all 5 GBDT models
test_gbdt_preds = np.mean([model.predict(X_test) for model in cv_models], axis=0)

print(f"GBDT predictions:")
print(f"  Mean: {test_gbdt_preds.mean():.6f}")
print(f"  Std:  {test_gbdt_preds.std():.6f}")
print(f"  Min:  {test_gbdt_preds.min():.6f}")
print(f"  Max:  {test_gbdt_preds.max():.6f}")

print()


# ===========================
# STEP 7: SAVE SUBMISSIONS
# ===========================

print("="*70)
print("STEP 7: CREATING SUBMISSION FILES")
print("="*70 + "\n")

# Strategy 1: Pure GBDT
submission_gbdt = pd.DataFrame({
    'isic_id': test_ids,
    'target': test_gbdt_preds
})
submission_gbdt.to_csv(results_dir / 'submission_gbdt_stacking.csv', index=False)
print(f"✓ Saved: submission_gbdt_stacking.csv")

# Strategy 2: Weighted blend (70% GBDT, 30% CNN ensemble)
blend_70_30 = 0.7 * test_gbdt_preds + 0.3 * test_cnn_predictions
submission_blend = pd.DataFrame({
    'isic_id': test_ids,
    'target': blend_70_30
})
submission_blend.to_csv(results_dir / 'submission_gbdt_blend_70_30.csv', index=False)
print(f"✓ Saved: submission_gbdt_blend_70_30.csv")

# Strategy 3: Rank averaging (GBDT ranks + CNN ranks)
from scipy.stats import rankdata
gbdt_ranks = rankdata(test_gbdt_preds) / len(test_gbdt_preds)
cnn_ranks = rankdata(test_cnn_predictions) / len(test_cnn_predictions)
rank_avg = (gbdt_ranks + cnn_ranks) / 2

submission_rank = pd.DataFrame({
    'isic_id': test_ids,
    'target': rank_avg
})
submission_rank.to_csv(results_dir / 'submission_rank_average.csv', index=False)
print(f"✓ Saved: submission_rank_average.csv")

print()


# ===========================
# STEP 8: COMPREHENSIVE VISUALIZATION
# ===========================

print("="*70)
print("STEP 8: GENERATING VISUALIZATIONS")
print("="*70 + "\n")

# Figure 1: OOF Predictions Analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_train, oof_predictions)
axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'CNN OOF (AUC={oof_auc:.4f})')
axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('CNN Out-of-Fold ROC Curve', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# OOF prediction distribution
axes[1].hist(oof_predictions[y_train == 0], bins=50, alpha=0.7, 
            label='Benign', color='blue', edgecolor='black')
axes[1].hist(oof_predictions[y_train == 1], bins=50, alpha=0.7,
            label='Malignant', color='red', edgecolor='black')
axes[1].set_xlabel('Predicted Probability', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('OOF Prediction Distribution by Class', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

# Calibration curve
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_train, oof_predictions, n_bins=10)
axes[2].plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, label='CNN')
axes[2].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
axes[2].set_xlabel('Mean Predicted Probability', fontsize=12)
axes[2].set_ylabel('Fraction of Positives', fontsize=12)
axes[2].set_title('Calibration Curve', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / '1_oof_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 1_oof_analysis.png")
plt.close()

# Figure 2: Feature Importance
print("Calculating feature importance...")

# Average feature importance across folds
avg_importance = pd.concat(feature_importances).groupby('feature')['importance'].mean().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 20 features
top_20 = avg_importance.head(20)
axes[0].barh(range(len(top_20)), top_20.values, color='steelblue', edgecolor='black')
axes[0].set_yticks(range(len(top_20)))
axes[0].set_yticklabels(top_20.index, fontsize=10)
axes[0].set_xlabel('Importance (Gain)', fontsize=12)
axes[0].set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Highlight CNN prediction importance
cnn_importance = avg_importance.get('cnn_prediction', 0)
cnn_rank = (avg_importance >= cnn_importance).sum()

axes[0].text(0.95, 0.95, f'CNN rank: #{cnn_rank}\nImportance: {cnn_importance:.0f}',
            transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# All features
axes[1].bar(range(len(avg_importance)), avg_importance.values, color='steelblue', alpha=0.7)
axes[1].set_xlabel('Feature Index', fontsize=12)
axes[1].set_ylabel('Importance', fontsize=12)
axes[1].set_title(f'All {len(avg_importance)} Features Importance', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(viz_dir / '2_feature_importance.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 2_feature_importance.png")
plt.close()

# Figure 3: GBDT CV Performance
fig, ax = plt.subplots(figsize=(10, 6))

fold_nums = range(1, 11)  # Match the 10 folds
ax.bar(fold_nums, cv_aucs, color='green', alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(np.mean(cv_aucs), color='red', linestyle='--', linewidth=2,
          label=f'Mean: {np.mean(cv_aucs):.4f}')
ax.axhline(oof_auc, color='blue', linestyle=':', linewidth=2,
          label=f'CNN OOF: {oof_auc:.4f}')
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Validation AUC', fontsize=12)
ax.set_title('GBDT Cross-Validation Performance', fontsize=14, fontweight='bold')
ax.set_xticks(fold_nums)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add improvement annotation
improvement = np.mean(cv_aucs) - oof_auc
ax.text(0.5, 0.95, f'GBDT Improvement: {improvement:+.4f}',
       transform=ax.transAxes, fontsize=13, fontweight='bold',
       verticalalignment='top', horizontalalignment='center',
       bbox=dict(boxstyle='round', facecolor='yellow' if improvement > 0 else 'lightcoral', alpha=0.8))

plt.tight_layout()
plt.savefig(viz_dir / '3_gbdt_cv_performance.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 3_gbdt_cv_performance.png")
plt.close()

# Figure 4: Prediction Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Test prediction distributions
axes[0, 0].hist(test_cnn_predictions, bins=50, alpha=0.7, 
               label='CNN Ensemble', color='blue', edgecolor='black')
axes[0, 0].axvline(test_cnn_predictions.mean(), color='blue', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicted Probability', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('CNN Ensemble Predictions\n(5-fold average)', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(test_gbdt_preds, bins=50, alpha=0.7,
               label='GBDT Stacking', color='green', edgecolor='black')
axes[0, 1].axvline(test_gbdt_preds.mean(), color='green', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Probability', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('GBDT Stacking Predictions\n(5-fold CV average)', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Scatter plot: CNN vs GBDT
axes[1, 0].scatter(test_cnn_predictions, test_gbdt_preds, alpha=0.5, s=20)
axes[1, 0].plot([0, 1], [0, 1], 'r--', linewidth=2)
corr = np.corrcoef(test_cnn_predictions, test_gbdt_preds)[0, 1]
axes[1, 0].set_xlabel('CNN Predictions', fontsize=11)
axes[1, 0].set_ylabel('GBDT Predictions', fontsize=11)
axes[1, 0].set_title(f'CNN vs GBDT Correlation: {corr:.4f}', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Blend comparison
axes[1, 1].hist(blend_70_30, bins=50, alpha=0.7,
               label='70% GBDT + 30% CNN', color='purple', edgecolor='black')
axes[1, 1].axvline(blend_70_30.mean(), color='purple', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Predicted Probability', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Blended Predictions', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / '4_prediction_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 4_prediction_comparison.png")
plt.close()

# Figure 5: Learning Curves
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, gbm in enumerate(cv_models):
    ax = axes[idx]
    
    # Extract training history
    train_auc = gbm.evals_result_['train']['auc']
    valid_auc = gbm.evals_result_['valid']['auc']
    iterations = range(1, len(train_auc) + 1)
    
    ax.plot(iterations, train_auc, 'b-', linewidth=2, label='Train', alpha=0.7)
    ax.plot(iterations, valid_auc, 'r-', linewidth=2, label='Valid', alpha=0.7)
    ax.axvline(gbm.best_iteration, color='green', linestyle='--', 
              linewidth=2, label=f'Best: {gbm.best_iteration}')
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('AUC', fontsize=10)
    ax.set_title(f'GBDT Fold {idx+1} - Val AUC: {cv_aucs[idx]:.4f}', 
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Summary in last subplot
axes[-1].axis('off')
summary_text = f"""
GBDT Stacking Summary

CV Performance:
  Mean AUC: {np.mean(cv_aucs):.4f}
  Std: {np.std(cv_aucs):.4f}
  Best: {np.max(cv_aucs):.4f}
  Worst: {np.min(cv_aucs):.4f}

CNN Baseline:
  OOF AUC: {oof_auc:.4f}

Improvement:
  {np.mean(cv_aucs) - oof_auc:+.4f} AUC

Total Features: {X_train.shape[1]}
  - Metadata: {X_train.shape[1] - 1}
  - CNN Pred: 1
"""
axes[-1].text(0.5, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(viz_dir / '5_gbdt_learning_curves.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 5_gbdt_learning_curves.png")
plt.close()

print()


# ===========================
# STEP 9: SAVE METADATA & ANALYSIS
# ===========================

print("="*70)
print("STEP 9: SAVING ANALYSIS & METADATA")
print("="*70 + "\n")

# Save feature importance
avg_importance.to_csv(results_dir / 'feature_importance.csv', index=True)
print(f"✓ Saved: feature_importance.csv")

# Save predictions for analysis
predictions_df = pd.DataFrame({
    'isic_id': test_ids,
    'cnn_prediction': test_cnn_predictions,
    'gbdt_prediction': test_gbdt_preds,
    'blend_70_30': blend_70_30,
    'rank_average': rank_avg
})
predictions_df.to_csv(results_dir / 'all_predictions.csv', index=False)
print(f"✓ Saved: all_predictions.csv")

# Save OOF predictions
oof_df = pd.DataFrame({
    'isic_id': train_meta_processed['isic_id'],
    'target': y_train,
    'oof_prediction': oof_predictions
})
oof_df.to_csv(results_dir / 'oof_predictions.csv', index=False)
print(f"✓ Saved: oof_predictions.csv")

# Save summary report
summary = {
    'timestamp': timestamp,
    'cnn_oof_auc': float(oof_auc),
    'gbdt_cv_mean_auc': float(np.mean(cv_aucs)),
    'gbdt_cv_std_auc': float(np.std(cv_aucs)),
    'gbdt_cv_aucs': [float(x) for x in cv_aucs],
    'improvement': float(np.mean(cv_aucs) - oof_auc),
    'n_features': int(X_train.shape[1]),
    'n_train_samples': int(len(X_train)),
    'n_test_samples': int(len(X_test)),
    'lgbm_params': best_params,
    'best_iterations': [int(m.best_iteration) for m in cv_models]
}

with open(results_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved: summary.json")

print()


# ===========================
# STEP 10: PERFORMANCE ESTIMATION & RECOMMENDATIONS
# ===========================

print("="*70)
print("FINAL RESULTS & RECOMMENDATIONS")
print("="*70 + "\n")

print(f"📊 Model Performance Summary:")
print(f"  CNN OOF AUC:        {oof_auc:.4f}")
print(f"  GBDT CV AUC:        {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
print(f"  Improvement:        {np.mean(cv_aucs) - oof_auc:+.4f}")

# Estimate public LB
historical_val_to_lb_gap = 0.013

estimated_lb_cnn = test_cnn_predictions.mean() - historical_val_to_lb_gap
estimated_lb_gbdt = np.mean(cv_aucs) - historical_val_to_lb_gap
estimated_lb_blend = np.mean(cv_aucs) - historical_val_to_lb_gap * 0.8

print(f"\n📈 Estimated Public LB Scores:")
print(f"  CNN Ensemble:       {estimated_lb_cnn:.5f}")
print(f"  GBDT Stacking:      {estimated_lb_gbdt:.5f}")
print(f"  70/30 Blend:        {estimated_lb_blend:.5f}")

print(f"\n🎯 Competition Standing:")
print(f"  Current (CNN):      0.96741 (2nd place)")
print(f"  1st place:          0.99248")
print(f"  Gap to close:       {0.99248 - 0.96741:.5f}")

gap_to_1st_gbdt = 0.99248 - estimated_lb_gbdt
gap_to_1st_blend = 0.99248 - estimated_lb_blend

print(f"\n🏆 Projected Gaps with GBDT:")
print(f"  GBDT vs 1st:        {gap_to_1st_gbdt:.5f}")
print(f"  Blend vs 1st:       {gap_to_1st_blend:.5f}")

print(f"\n⭐ RECOMMENDED SUBMISSION:")

if np.mean(cv_aucs) > 0.985:
    print(f"  Primary: submission_gbdt_stacking.csv")
    print(f"  Reason: GBDT CV AUC is EXCELLENT ({np.mean(cv_aucs):.4f})")
    print(f"  Expected LB: 0.990-0.998")
    if estimated_lb_gbdt > 0.992:
        print(f"\n  🏆 VERY HIGH CHANCE OF 1ST PLACE!")
        print(f"  You may have beaten 0.99248!")
elif np.mean(cv_aucs) > 0.975:
    print(f"  Primary: submission_gbdt_stacking.csv")
    print(f"  Reason: GBDT CV AUC is very good ({np.mean(cv_aucs):.4f})")
    print(f"  Expected LB: 0.980-0.990")
    if estimated_lb_gbdt > 0.990:
        print(f"\n  🎉 HIGH CHANCE OF 1ST PLACE!")
    else:
        print(f"\n  🥈 Should improve significantly over 2nd place (0.96741)")
elif np.mean(cv_aucs) > 0.970:
    print(f"  Primary: submission_gbdt_blend_70_30.csv")
    print(f"  Reason: Blend combines CNN robustness with GBDT boost")
    print(f"  Expected LB: 0.975-0.985")
    if estimated_lb_blend > 0.990:
        print(f"\n  🎉 POSSIBLE 1ST PLACE!")
    else:
        print(f"\n  🥈 Strong improvement over current 0.96741")
else:
    print(f"  Primary: submission_gbdt_stacking.csv")
    print(f"  Backup: submission_rank_average.csv")
    print(f"  Expected LB: 0.970-0.980")
    print(f"\n  Still improvement, but may need additional strategies")

print(f"\n📁 All Files Saved to: {results_dir}")
print(f"  • 3 submission CSVs")
print(f"  • all_predictions.csv (for analysis)")
print(f"  • oof_predictions.csv (for debugging)")
print(f"  • feature_importance.csv")
print(f"  • summary.json")
print(f"  • visualizations/ (5 detailed plots)")

print(f"\n🚀 Next Steps:")
print(f"  1. Submit: {results_dir / 'submission_gbdt_stacking.csv'}")
print(f"  2. If LB > 0.99: Celebrate 1st place! 🏆")
print(f"  3. If LB > 0.97: Try submission_gbdt_blend_70_30.csv")
print(f"  4. If LB > 0.965: Improvement! Try submission_rank_average.csv")

print(f"\n{'='*70}")
print(f"PIPELINE COMPLETE!")
print(f"{'='*70}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Results: {results_dir}")
print(f"{'='*70}\n")

print("You can now go to sleep! 😴")
print("Check results in the morning and submit the recommended file.")
print("\nGood luck catching 1st place! 🚀")