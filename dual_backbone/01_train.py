"""
18_1: Dual-Backbone Hybrid Model Training (End-to-End)
======================================================
EVA02 + EdgeNeXt + Metadata Fusion - Full End-to-End Training

Architecture:
    Image (384x384) ─┬─> EVA02 (336x336)   ──> [384-dim]  ─┐
                     │                                     ├──> Concat ─┐
                     └─> EdgeNeXt (384x384) ──> [584-dim] ─┘            │
                                                                        ├──> Fusion MLP ──> Prediction
    Metadata ─────────> MetadataEncoder    ──> [64-dim]  ───────────────┘

Training Strategy (from 11_1 best practices + 1st place insights):
    - 100 epochs with early stopping (patience=15)
    - Adam optimizer (lr=5e-4, weight_decay=1e-5) - proven settings
    - ReduceLROnPlateau scheduler (mode='max', patience=5, factor=0.5)
    - Model EMA (decay=0.9999)
    - Focal Loss (alpha=0.25, gamma=2.0)

Key Features:
    - End-to-end training (vision + metadata + fusion jointly optimized)
    - Dual resolution: EVA02 @ 336, EdgeNeXt @ 384
    - Balanced sampling with synthetic data (1:1 ratio)
    - 5-fold StratifiedGroupKFold CV (patient-aware)
    - AMP with NaN protection

Feature Engineering (includes 1st place patient-relative features):
    - 32+ numerical features (z-score standardized)
    - 9 categorical features (one-hot encoded)
    - Patient-relative Z-scores: (lesion_value - patient_mean) / patient_std
    - Patient-relative ratios: lesion_value / patient_mean
    - LOF (Local Outlier Factor) per patient - captures "Ugly Duckling Sign"
    - Total: ~130+ features after encoding
    - Excluded: mel_thick_mm, mel_mitotic_index, iddx_* (leakage)

Data Augmentation:
    - RandomHorizontalFlip, RandomVerticalFlip
    - RandomRotation(20°)
    - RandomAffine (translate, scale)
    - ColorJitter (brightness, contrast, saturation, hue)
    - GaussianBlur
    - RandomErasing (Cutout)

Usage:
    # Train all 5 folds on 4 L40S GPUs
    uv run python 18_1_train_dual_backbone_hybrid.py --fold 1 --gpu 0 --experiment-name dual_backbone_hybrid_v1 &
    uv run python 18_1_train_dual_backbone_hybrid.py --fold 2 --gpu 1 --experiment-name dual_backbone_hybrid_v1 &
    uv run python 18_1_train_dual_backbone_hybrid.py --fold 3 --gpu 2 --experiment-name dual_backbone_hybrid_v1 &
    uv run python 18_1_train_dual_backbone_hybrid.py --fold 4 --gpu 3 --experiment-name dual_backbone_hybrid_v1 &
    wait
    uv run python 18_1_train_dual_backbone_hybrid.py --fold 5 --gpu 0 --experiment-name dual_backbone_hybrid_v1

Expected: ~0.97+ AUC (CV with EMA + patient-relative features)
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm
import warnings
import json
import os
import argparse
from datetime import datetime
import pickle
from copy import deepcopy
import timm
# Import shared model definitions
from isic_model import DualBackboneHybrid, DualResolutionDataset, MetadataEncoder, BalancedBatchSampler

warnings.filterwarnings('ignore')


# ===========================
# CONFIGURATION
# ===========================

EVA02_MODEL = 'eva02_small_patch14_336.mim_in22k_ft_in1k'
EDGENEXT_MODEL = 'edgenext_base.in21k_ft_in1k'
EVA02_SIZE = 336
EDGENEXT_SIZE = 384

# ===========================
# COMMAND LINE ARGUMENTS
# ===========================

def parse_args():
    parser = argparse.ArgumentParser(description='Dual-Backbone Hybrid Training')
    parser.add_argument('--fold', type=int, required=True, help='Fold number (1-5)')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (1e-5 proven in 11_1)')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--synth-dir', type=str, default='generative/data')
    parser.add_argument('--num-workers', type=int, default=16, help='DataLoader workers')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='EMA decay rate')
    parser.add_argument('--accumulation-steps', type=int, default=2, help='Gradient accumulation')
    parser.add_argument('--experiment-name', type=str, default=None, 
                        help='Experiment name for results directory (shared across folds)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use AMP (default: True for speed)')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='Disable AMP for stability')
    return parser.parse_args()


# ===========================
# FEATURE ENGINEERING
# ===========================

# Features to compute patient-relative statistics for
PATIENT_RELATIVE_FEATURES = [
    'tbp_lv_areaMM2', 'tbp_lv_perimeterMM', 'tbp_lv_minorAxisMM',
    'clin_size_long_diam_mm', 'tbp_lv_H', 'tbp_lv_B', 'tbp_lv_A',
    'tbp_lv_deltaB', 'tbp_lv_norm_color', 'tbp_lv_color_std_mean',
    'tbp_lv_radial_color_std_max', 'tbp_lv_dnn_lesion_confidence'
]

# Features for LOF computation (core lesion properties)
LOF_FEATURES = [
    'tbp_lv_areaMM2', 'tbp_lv_perimeterMM', 'tbp_lv_H', 'tbp_lv_B',
    'tbp_lv_deltaB', 'tbp_lv_norm_color', 'tbp_lv_color_std_mean'
]


def compute_patient_statistics(df):
    """
    Compute and return patient-level statistics for later use in test inference.
    
    This should be called on the FULL training data and saved for test time.
    At test time, we look up each test sample's patient and use these stats.
    
    Returns:
        dict with:
            - 'patient_stats': DataFrame with patient_id, feature means, stds
            - 'global_stats': dict with global medians and stds (fallback)
    """
    print("  Computing patient statistics for inference...")
    
    # Compute global stats (for fallback when patient not found)
    global_stats = {}
    for feat in PATIENT_RELATIVE_FEATURES:
        if feat in df.columns:
            global_stats[f'{feat}_median'] = df[feat].median()
            global_stats[f'{feat}_std'] = df[feat].std()
            global_stats[f'{feat}_mean'] = df[feat].mean()
    
    # Compute per-patient stats
    patient_stats_list = []
    for feat in PATIENT_RELATIVE_FEATURES:
        if feat not in df.columns:
            continue
        
        patient_agg = df.groupby('patient_id')[feat].agg(['mean', 'std', 'count'])
        patient_agg.columns = [f'{feat}_mean', f'{feat}_std', f'{feat}_count']
        
        # Fill missing std with global std
        patient_agg[f'{feat}_std'] = patient_agg[f'{feat}_std'].fillna(global_stats[f'{feat}_std'])
        patient_agg[f'{feat}_std'] = patient_agg[f'{feat}_std'].replace(0, global_stats[f'{feat}_std'])
        
        patient_stats_list.append(patient_agg)
    
    # Merge all feature stats
    patient_stats = pd.concat(patient_stats_list, axis=1)
    patient_stats = patient_stats.reset_index()
    
    # Add lesion count per patient
    patient_counts = df.groupby('patient_id').size().reset_index(name='lesion_count')
    patient_stats = patient_stats.merge(patient_counts, on='patient_id')
    
    print(f"  Saved statistics for {len(patient_stats)} patients")
    
    return {
        'patient_stats': patient_stats,
        'global_stats': global_stats
    }


def compute_patient_relative_features(df, patient_statistics=None):
    """
    Compute patient-relative features (1st place solution key insight).
    This captures the "Ugly Duckling Sign" - lesions that are unusual for a patient.
    
    For each patient:
        - Z-score: (lesion_value - patient_mean) / patient_std
        - Ratio: lesion_value / patient_mean
        - Diff: lesion_value - patient_mean
    
    Args:
        df: DataFrame with samples
        patient_statistics: Optional dict from compute_patient_statistics() for test inference
                           If None, computes on df directly (for training)
    """
    df = df.copy()
    print("  Computing patient-relative features...")
    
    # Ensure patient_id exists
    if 'patient_id' not in df.columns:
        print("  WARNING: patient_id not found, skipping patient-relative features")
        return df
    
    # If patient_statistics provided, use them (test time inference)
    if patient_statistics is not None:
        patient_stats = patient_statistics['patient_stats']
        global_stats = patient_statistics['global_stats']
        
        # Merge patient stats into df
        df = df.merge(patient_stats, on='patient_id', how='left')
        
        for feat in tqdm(PATIENT_RELATIVE_FEATURES, desc="  Patient-relative features"):
            if feat not in df.columns:
                continue
            
            # Get values
            feat_values = df[feat].fillna(global_stats.get(f'{feat}_median', 0))
            patient_mean = df[f'{feat}_mean'].fillna(global_stats.get(f'{feat}_mean', 0))
            patient_std = df[f'{feat}_std'].fillna(global_stats.get(f'{feat}_std', 1))
            
            # Compute features
            df[f'{feat}_pat_zscore'] = ((feat_values - patient_mean) / (patient_std + 1e-6)).clip(-10, 10)
            df[f'{feat}_pat_ratio'] = (feat_values / (patient_mean.abs() + 1e-6)).clip(0.01, 100)
            df[f'{feat}_pat_diff'] = feat_values - patient_mean
            
            # Drop temp columns
            df = df.drop(columns=[f'{feat}_mean', f'{feat}_std', f'{feat}_count'], errors='ignore')
        
        # Handle lesion count
        df['patient_lesion_count'] = df['lesion_count'].fillna(1)
        df['is_single_lesion_patient'] = (df['patient_lesion_count'] == 1).astype(int)
        df = df.drop(columns=['lesion_count'], errors='ignore')
        
        return df
    
    # Otherwise compute directly on df (training time)
    for feat in tqdm(PATIENT_RELATIVE_FEATURES, desc="  Patient-relative features"):
        if feat not in df.columns:
            continue
        
        # Fill NaN with global median for computation
        global_median = df[feat].median()
        feat_values = df[feat].fillna(global_median)
        
        # Compute patient-level mean and std
        patient_mean = df.groupby('patient_id')[feat].transform('mean')
        patient_std = df.groupby('patient_id')[feat].transform('std')
        patient_count = df.groupby('patient_id')[feat].transform('count')
        
        # Replace std=0 or std=NaN with global std to avoid division issues
        global_std = df[feat].std()
        patient_std = patient_std.fillna(global_std).replace(0, global_std)
        
        # Z-score (normalized deviation from patient mean)
        df[f'{feat}_pat_zscore'] = (feat_values - patient_mean) / (patient_std + 1e-6)
        
        # Ratio (how many times larger/smaller than patient average)
        df[f'{feat}_pat_ratio'] = feat_values / (patient_mean.abs() + 1e-6)
        
        # Simple difference
        df[f'{feat}_pat_diff'] = feat_values - patient_mean
        
        # Clamp extreme values
        df[f'{feat}_pat_zscore'] = df[f'{feat}_pat_zscore'].clip(-10, 10)
        df[f'{feat}_pat_ratio'] = df[f'{feat}_pat_ratio'].clip(0.01, 100)
        df[f'{feat}_pat_diff'] = df[f'{feat}_pat_diff'].clip(
            df[f'{feat}_pat_diff'].quantile(0.01),
            df[f'{feat}_pat_diff'].quantile(0.99)
        )
    
    # Mark patients with only 1 lesion (can't compute relative features reliably)
    patient_counts = df.groupby('patient_id').size()
    df['patient_lesion_count'] = df['patient_id'].map(patient_counts)
    df['is_single_lesion_patient'] = (df['patient_lesion_count'] == 1).astype(int)
    
    return df


def compute_lof_features(df, n_neighbors=10, min_samples=5):
    """
    Compute Local Outlier Factor for each lesion within its patient group.
    
    Optimized approach:
        1. Pre-fill missing values
        2. Vectorized groupby operations
        3. Only compute LOF for patients with enough lesions
        4. Cache results
    
    This captures the "Ugly Duckling Sign" using density-based outlier detection.
    """
    df = df.copy()
    print("  Computing LOF features (optimized)...")
    
    if 'patient_id' not in df.columns:
        print("  WARNING: patient_id not found, skipping LOF")
        df['lof_score'] = 0.0
        return df
    
    # Prepare LOF features
    lof_cols = [c for c in LOF_FEATURES if c in df.columns]
    if len(lof_cols) < 3:
        print(f"  WARNING: Not enough LOF features found ({lof_cols})")
        df['lof_score'] = 0.0
        return df
    
    # Fill missing values with global median
    lof_data = df[lof_cols].copy()
    for col in lof_cols:
        lof_data[col] = lof_data[col].fillna(lof_data[col].median())
    
    # Standardize for LOF
    lof_scaled = (lof_data - lof_data.mean()) / (lof_data.std() + 1e-6)
    
    # Initialize LOF scores
    df['lof_score'] = 0.0
    
    # Get patients with enough lesions for LOF
    patient_counts = df.groupby('patient_id').size()
    valid_patients = patient_counts[patient_counts >= min_samples].index
    
    print(f"  Computing LOF for {len(valid_patients)} patients with >= {min_samples} lesions...")
    
    # Process in batches for memory efficiency
    for patient_id in tqdm(valid_patients, desc="  LOF", mininterval=1):
        mask = df['patient_id'] == patient_id
        patient_data = lof_scaled.loc[mask].values
        
        # Adjust n_neighbors based on patient size
        n_neigh = min(n_neighbors, len(patient_data) - 1)
        if n_neigh < 2:
            continue
        
        try:
            lof = LocalOutlierFactor(n_neighbors=n_neigh, contamination='auto')
            lof.fit(patient_data)
            # LOF returns negative scores, -1 is inlier, more negative = more outlier
            # We negate so higher = more outlier
            lof_scores = -lof.negative_outlier_factor_
            df.loc[mask, 'lof_score'] = lof_scores
        except Exception as e:
            # Skip on error
            pass
    
    # Clip extreme values
    df['lof_score'] = df['lof_score'].clip(0.5, 5.0)
    
    # For patients with few lesions, use normalized global features
    small_patient_mask = df['patient_id'].isin(
        patient_counts[patient_counts < min_samples].index
    )
    if small_patient_mask.sum() > 0:
        # Use global LOF as fallback (computed once)
        # novelty=True allows using decision_function on new data
        global_lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', novelty=True)
        # Sample subset for efficiency
        sample_idx = np.random.choice(len(lof_scaled), min(50000, len(lof_scaled)), replace=False)
        global_lof.fit(lof_scaled.iloc[sample_idx].values)
        
        # Predict for small patients (decision_function gives negative scores, more negative = more outlier)
        small_data = lof_scaled.loc[small_patient_mask].values
        global_scores = -global_lof.decision_function(small_data)
        df.loc[small_patient_mask, 'lof_score'] = global_scores.clip(0.5, 5.0)
    
    return df


def engineer_features(df):
    """Enhanced feature engineering with clinical domain knowledge (from 11_1 + 1st place)"""
    df = df.copy()
    
    # AGE FEATURES
    df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median())
    df['age_group'] = pd.cut(df['age_approx'], bins=[0, 30, 50, 70, 100],
                             labels=['young', 'middle', 'senior', 'elderly'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    df['age_squared'] = df['age_approx'] ** 2
    
    # SIZE FEATURES (Diameter in ABCDE)
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['size_category'] = pd.cut(df['lesion_size_mm'].fillna(0), bins=[0, 6, 10, 20, 1000],
                                 labels=['small', 'medium', 'large', 'very_large'])
    df['large_lesion'] = (df['lesion_size_mm'] > 6).astype(int)
    
    # SHAPE FEATURES (Border irregularity in ABCDE)
    df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    df['eccentricity'] = df['tbp_lv_minorAxisMM'] / (df['tbp_lv_areaMM2']**0.5 + 1e-6)
    df['compactness'] = (4 * np.pi * df['tbp_lv_areaMM2']) / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    
    # COLOR FEATURES (Color variance in ABCDE)
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
    df['age_size_risk'] = df['age_approx'] * df['lesion_size_mm'].fillna(0)
    df['age_site_risk'] = df['age_approx'] * df['site_risk_score']
    df['color_size_risk'] = df['color_variance'] * df['lesion_size_mm'].fillna(0)
    df['age_color_risk'] = df['age_approx'] * df['color_variance']
    
    # ASYMMETRY SCORE
    df['asymmetry_score'] = (
        df['tbp_lv_norm_color'] + df['tbp_lv_radial_color_std_max'] +
        (1 / (df['shape_regularity'] + 1e-6))
    ) / 3
    
    # LOG TRANSFORMS
    df['log_area'] = np.log1p(df['tbp_lv_areaMM2'])
    df['log_perimeter'] = np.log1p(df['tbp_lv_perimeterMM'])
    df['log_size'] = np.log1p(df['lesion_size_mm'].fillna(0))
    
    # RATIOS
    df['h_to_b_ratio'] = df['tbp_lv_H'] / (df['tbp_lv_B'] + 1e-6)
    df['a_to_b_ratio'] = df['tbp_lv_A'] / (df['tbp_lv_B'] + 1e-6)
    df['area_to_perimeter'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] + 1e-6)
    
    return df


def precompute_patient_features(df):
    """
    Pre-compute patient-relative features and LOF on the FULL dataset.
    
    This should be called ONCE on all data BEFORE train/val splitting because:
    1. Patient-relative features depend on ALL lesions from that patient
    2. LOF measures how unusual a lesion is within its patient group
    3. These computations don't use the target label, so no leakage
    
    Returns:
        DataFrame with isic_id and all computed features
    """
    print("  Pre-computing patient-relative features on full dataset...")
    df = df.copy()
    
    # Compute patient-relative features (Z-scores, ratios, diffs)
    df = compute_patient_relative_features(df)
    
    # Compute LOF features
    df = compute_lof_features(df, n_neighbors=10, min_samples=5)
    
    # Extract only the computed features (not original columns)
    computed_cols = ['isic_id', 'lof_score', 'patient_lesion_count', 'is_single_lesion_patient']
    
    # Add patient-relative feature columns
    for feat in PATIENT_RELATIVE_FEATURES:
        for suffix in ['_pat_zscore', '_pat_ratio', '_pat_diff']:
            col = f'{feat}{suffix}'
            if col in df.columns:
                computed_cols.append(col)
    
    result = df[computed_cols].copy()
    print(f"  Pre-computed {len(computed_cols) - 1} features for {len(result)} samples")
    
    return result


def preprocess_metadata(df, is_train=True, scaler=None, encoders=None, 
                        precomputed_features=None):
    """
    Preprocess metadata with feature engineering.
    
    Args:
        df: DataFrame with metadata
        is_train: If True, fit scaler/encoders. If False, use provided ones.
        scaler: Pre-fitted StandardScaler (for is_train=False)
        encoders: Pre-fitted category encoders (for is_train=False)
        precomputed_features: Dict with pre-computed LOF and patient-relative features
                             (computed on full dataset before train/val split)
    """
    
    df = engineer_features(df)
    
    # If precomputed features provided, merge them in (preferred approach)
    if precomputed_features is not None:
        # Merge pre-computed patient-relative and LOF features by isic_id
        precomputed_df = precomputed_features
        merge_cols = [c for c in precomputed_df.columns if c != 'isic_id']
        df = df.merge(precomputed_df[['isic_id'] + merge_cols], on='isic_id', how='left')
        print(f"  Merged {len(merge_cols)} pre-computed features")
    else:
        # Fallback: compute on this subset (less accurate for val/test)
        df = compute_patient_relative_features(df)
        if is_train:
            df = compute_lof_features(df, n_neighbors=10, min_samples=5)
        elif 'lof_score' not in df.columns:
            df['lof_score'] = 1.0  # Default neutral value
    
    # Exclude leaky features identified in metadata investigation
    EXCLUDE_FEATURES = [
        'mel_thick_mm', 'mel_mitotic_index',  # Post-biopsy leakage
        'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5',  # Diagnosis codes
        'lesion_id',  # Identifier only
    ]
    
    # Build NUMERICAL_FEATURES dynamically including patient-relative features
    BASE_NUMERICAL = [
        # Original features
        'tbp_lv_H', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM',
        'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_Hext',
        'clin_size_long_diam_mm', 'tbp_lv_radial_color_std_max',
        'tbp_lv_B', 'tbp_lv_color_std_mean', 'tbp_lv_Aext',
        'tbp_lv_stdLExt', 'tbp_lv_norm_color', 'tbp_lv_A', 'age_approx',
        # DNN features (verified safe)
        'tbp_lv_dnn_lesion_confidence', 'tbp_lv_nevi_confidence',
        # Engineered features
        'age_squared', 'lesion_size_mm',
        'shape_regularity', 'eccentricity', 'compactness',
        'color_variance', 'color_uniformity', 'darkness_score', 'color_contrast',
        'site_risk_score', 'age_size_risk', 'age_site_risk', 'color_size_risk',
        'age_color_risk', 'asymmetry_score',
        'log_area', 'log_perimeter', 'log_size',
        'h_to_b_ratio', 'a_to_b_ratio', 'area_to_perimeter',
        # LOF score
        'lof_score',
        # Patient context
        'patient_lesion_count', 'is_single_lesion_patient'
    ]
    
    # Add patient-relative features (Z-scores, ratios, diffs)
    PATIENT_REL_SUFFIXES = ['_pat_zscore', '_pat_ratio', '_pat_diff']
    PATIENT_RELATIVE_NUMERICAL = []
    for feat in PATIENT_RELATIVE_FEATURES:
        for suffix in PATIENT_REL_SUFFIXES:
            col_name = f'{feat}{suffix}'
            if col_name in df.columns:
                PATIENT_RELATIVE_NUMERICAL.append(col_name)
    
    NUMERICAL_FEATURES = BASE_NUMERICAL + PATIENT_RELATIVE_NUMERICAL
    
    CATEGORICAL_FEATURES = [
        'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple',
        'age_group', 'size_category', 'age_risk', 'large_lesion', 'high_risk_site'
    ]
    
    # Keep only features that exist
    NUMERICAL_FEATURES = [f for f in NUMERICAL_FEATURES if f in df.columns]
    CATEGORICAL_FEATURES = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    
    print(f"  Total numerical features: {len(NUMERICAL_FEATURES)}")
    print(f"  Patient-relative features: {len(PATIENT_RELATIVE_NUMERICAL)}")
    
    # Fill missing values
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            median_val = df[col].median() if is_train else 0
            df[col] = df[col].fillna(median_val)
            # Also handle inf values
            df[col] = df[col].replace([np.inf, -np.inf], median_val)
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing')
    
    # Standardize numerical features
    if is_train:
        scaler = StandardScaler()
        df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])
    else:
        df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])
    
    # One-hot encode categorical features
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


# ===========================
# DATASET CLASS (IMPORTED)
# ===========================
# DualResolutionDataset is imported from isic_model.py



# ===========================
# BALANCED SAMPLER (IMPORTED)
# ===========================
# BalancedBatchSampler is imported from isic_model.py



# ===========================
# MODEL COMPONENTS
# ===========================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance with numerical stability"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # Clamp inputs for numerical stability
        inputs = torch.clamp(inputs, min=-50, max=50)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Clamp BCE loss to prevent extreme values
        bce_loss = torch.clamp(bce_loss, max=100)
        
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_term * bce_loss
        
        # Check for NaN and replace with 0
        focal_loss = torch.where(torch.isnan(focal_loss), torch.zeros_like(focal_loss), focal_loss)
        
        return focal_loss.mean()


# ===========================
# MODEL COMPONENTS (IMPORTED)
# ===========================
# Classes are imported from isic_model.py:
# - MetadataEncoder
# - DualBackboneHybrid



class ModelEMA:
    """Exponential Moving Average of model weights"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ===========================
# TRAINING FUNCTIONS
# ===========================

def train_epoch(model, loader, criterion, optimizer, amp_scaler, device, 
                accumulation_steps=1, model_ema=None, use_amp=True):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    nan_count = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Train", ncols=100)
    for batch_idx, (img_336, img_384, metadata, labels) in enumerate(pbar):
        img_336 = img_336.to(device, non_blocking=True)
        img_384 = img_384.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        
        # Mixed precision forward (with optional disable)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(img_336, img_384, metadata)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            print(f"\n⚠️ NaN/Inf loss detected at batch {batch_idx}, skipping...")
            optimizer.zero_grad()
            continue
        
        # Backward with gradient scaling
        amp_scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale before clipping
            amp_scaler.unscale_(optimizer)
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                nan_count += 1
                print(f"\n⚠️ NaN gradient detected at batch {batch_idx}, skipping step...")
                optimizer.zero_grad()
                # CRITICAL: Must call amp_scaler.update() after unscale_() even when skipping
                amp_scaler.update()
                continue
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step optimizer (amp_scaler will skip if inf/nan)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()
            
            if model_ema is not None:
                model_ema.update()
        
        running_loss += loss.item() * accumulation_steps
        
        # Safe sigmoid with clamping
        with torch.no_grad():
            preds = torch.sigmoid(torch.clamp(outputs, min=-50, max=50))
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{running_loss/(batch_idx+1):.4f}', 'nan': nan_count})
    
    epoch_loss = running_loss / max(len(loader), 1)
    
    if len(all_labels) > 0:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    else:
        epoch_auc = 0.0
    
    if nan_count > 0:
        print(f"  ⚠️ Total NaN/Inf events this epoch: {nan_count}")
    
    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device, use_amp=True):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img_336, img_384, metadata, labels in tqdm(loader, desc="Val", ncols=100):
            img_336 = img_336.to(device, non_blocking=True)
            img_384 = img_384.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(img_336, img_384, metadata)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Safe sigmoid
            preds = torch.sigmoid(torch.clamp(outputs, min=-50, max=50))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / max(len(loader), 1)
    epoch_auc = roc_auc_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    
    return epoch_loss, epoch_auc, np.array(all_preds).flatten(), np.array(all_labels).flatten()


# ===========================
# MAIN TRAINING
# ===========================

def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print(f"DUAL-BACKBONE HYBRID TRAINING - FOLD {args.fold}/5")
    print("="*70)
    print(f"GPU: {args.gpu}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Accumulation steps: {args.accumulation_steps}")
    print(f"Effective batch: {args.batch_size * args.accumulation_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Models: {EVA02_MODEL} + {EDGENEXT_MODEL}")
    print("="*70)
    
    # Paths
    data_dir = Path(args.data_dir)
    synth_dir = Path(args.synth_dir)
    
    # 1. Load metadata
    print("\n[1/8] Loading metadata...")
    train_meta = pd.read_csv(data_dir / 'new-train-metadata.csv', low_memory=False)
    print(f"  Total samples: {len(train_meta):,}")
    print(f"  Malignant: {train_meta['target'].sum()}")
    
    # Load enriched synthetic metadata
    synth_meta_path = synth_dir / 'synthetic_malignant_metadata_enriched.csv'
    if synth_meta_path.exists():
        synth_meta = pd.read_csv(synth_meta_path)
        print(f"  Synthetic samples: {len(synth_meta)}")
    else:
        print(f"  WARNING: Enriched synthetic metadata not found at {synth_meta_path}")
        print(f"  Run 18_0_prepare_synthetic_metadata.py first!")
        synth_meta = None
    
    # 2. Pre-compute patient-relative features on FULL dataset
    # This is done BEFORE splitting because:
    #   - Patient-relative features depend on ALL lesions from that patient
    #   - LOF measures how unusual a lesion is within patient group
    #   - These don't use target labels, so no data leakage
    print("\n[2/8] Pre-computing patient features on full dataset...")
    precomputed_features = precompute_patient_features(train_meta)
    
    # Also compute patient statistics for test-time inference
    # This will be saved and used when making predictions on test data
    patient_statistics = compute_patient_statistics(train_meta)
    
    # Also compute for synthetic data (each synthetic sample is its own "patient")
    synth_precomputed = None
    if synth_meta is not None:
        print("  Computing for synthetic samples...")
        synth_precomputed = precompute_patient_features(synth_meta)
    
    # 3. Create fold split
    print("\n[3/8] Creating fold split...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(train_meta, train_meta['target'], train_meta['patient_id']))
    train_idx, val_idx = splits[args.fold - 1]
    
    train_df = train_meta.iloc[train_idx].copy()
    val_df = train_meta.iloc[val_idx].copy()
    
    print(f"  Fold {args.fold}:")
    print(f"    Train: {len(train_df):,} ({train_df['target'].sum()} positive)")
    print(f"    Val:   {len(val_df):,} ({val_df['target'].sum()} positive)")
    
    # 4. Preprocess metadata (use pre-computed features)
    print("\n[4/8] Preprocessing metadata...")
    train_processed, scaler, encoders = preprocess_metadata(
        train_df, is_train=True, precomputed_features=precomputed_features
    )
    val_processed, _, _ = preprocess_metadata(
        val_df, is_train=False, scaler=scaler, encoders=encoders, 
        precomputed_features=precomputed_features
    )
    
    # Add identifiers back
    train_processed['isic_id'] = train_df['isic_id'].values
    train_processed['target'] = train_df['target'].values
    train_processed['patient_id'] = train_df['patient_id'].values
    
    val_processed['isic_id'] = val_df['isic_id'].values
    val_processed['target'] = val_df['target'].values
    val_processed['patient_id'] = val_df['patient_id'].values
    
    metadata_dim = len([c for c in train_processed.columns if c not in ['isic_id', 'target', 'patient_id']])
    print(f"  Metadata dimension: {metadata_dim}")
    
    # Process synthetic metadata if available
    synth_processed = None
    if synth_meta is not None:
        synth_processed, _, _ = preprocess_metadata(
            synth_meta, is_train=False, scaler=scaler, encoders=encoders,
            precomputed_features=synth_precomputed
        )
        synth_processed['isic_id'] = synth_meta['isic_id'].values
        synth_processed['target'] = synth_meta['target'].values
        synth_processed['patient_id'] = synth_meta['patient_id'].values
    
    # 5. Create transforms
    print("\n[5/8] Creating transforms...")
    
    # Normalization (ImageNet stats)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # ===== STRONG AUGMENTATION (from 1st place solution insights) =====
    # Medical imaging benefits from aggressive augmentation
    
    # EVA02 transforms (336x336) - ViT-based, benefits from cutout
    train_transform_336 = transforms.Compose([
        transforms.Resize((EVA02_SIZE, EVA02_SIZE)),
        # Geometric transforms
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),  # Increased from 15
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Up to 10% translation
            scale=(0.9, 1.1),      # 90-110% scale
            shear=5                 # Slight shear
        ),
        # Color transforms
        transforms.ColorJitter(
            brightness=0.2,   # Increased from 0.1
            contrast=0.2,
            saturation=0.2,
            hue=0.05          # Slight hue shift
        ),
        # Blur (simulates focus issues)
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
        # Convert to tensor
        transforms.ToTensor(),
        normalize,
        # Cutout (Random Erasing) - proven effective for skin lesions
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.1),
            ratio=(0.5, 2.0),
            value='random'
        )
    ])
    
    val_transform_336 = transforms.Compose([
        transforms.Resize((EVA02_SIZE, EVA02_SIZE)),
        transforms.ToTensor(),
        normalize
    ])
    
    # EdgeNeXt transforms (384x384) - ConvNet, slightly different aug
    train_transform_384 = transforms.Compose([
        transforms.Resize((EDGENEXT_SIZE, EDGENEXT_SIZE)),
        # Geometric transforms
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
        # Color transforms
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        # Blur
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
        # Convert to tensor
        transforms.ToTensor(),
        normalize,
        # Random Erasing
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.1),
            ratio=(0.5, 2.0),
            value='random'
        )
    ])
    
    val_transform_384 = transforms.Compose([
        transforms.Resize((EDGENEXT_SIZE, EDGENEXT_SIZE)),
        transforms.ToTensor(),
        normalize
    ])
    
    # 5. Create datasets and loaders
    print("\n[6/8] Creating datasets...")
    
    hdf5_path = data_dir / 'train-image-384.hdf5'
    synth_hdf5_path = synth_dir / 'synthetic_malignant_384.hdf5'
    
    if not synth_hdf5_path.exists():
        print(f"  WARNING: Synthetic HDF5 not found at {synth_hdf5_path}")
        synth_hdf5_path = None
        synth_processed = None
    
    train_dataset = DualResolutionDataset(
        hdf5_path=hdf5_path,
        metadata_df=train_processed,
        transform_336=train_transform_336,
        transform_384=train_transform_384,
        synth_hdf5_path=synth_hdf5_path,
        synth_metadata_df=synth_processed
    )
    
    val_dataset = DualResolutionDataset(
        hdf5_path=hdf5_path,
        metadata_df=val_processed,
        transform_336=val_transform_336,
        transform_384=val_transform_384
    )
    
    # Balanced sampler for training
    sampler = BalancedBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        length=len(train_dataset) // args.batch_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # 6. Create model and training components
    print("\n[7/8] Creating model...")
    
    model = DualBackboneHybrid(metadata_dim=metadata_dim).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Use Adam (proven in 11_1 to work better than AdamW for this task)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # ReduceLROnPlateau - proven to work better than CosineAnnealing in 11_1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-7
    )
    
    # GradScaler with growth interval for stability
    amp_scaler = torch.cuda.amp.GradScaler(
        init_scale=2.**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=args.amp
    )
    model_ema = ModelEMA(model, decay=args.ema_decay)
    
    # 7. Training loop
    print("\n[8/8] Starting training...")
    print(f"  AMP enabled: {args.amp}")
    
    # Create results directory with proper naming convention
    script_dir = Path(__file__).parent
    
    if args.experiment_name:
        # Use provided experiment name (shared across folds)
        results_dir = script_dir / 'results' / args.experiment_name
    else:
        # Auto-generate with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = script_dir / 'results' / f'dual_backbone_hybrid_{timestamp}'
    
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Results directory: {results_dir}")
    
    # Save config
    config = vars(args)
    config['eva02_model'] = EVA02_MODEL
    config['edgenext_model'] = EDGENEXT_MODEL
    config['metadata_dim'] = metadata_dim
    config['total_params'] = total_params
    config['trainable_params'] = trainable_params
    config['timestamp'] = datetime.now().isoformat()
    
    with open(results_dir / f'config_fold{args.fold}.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # =====================================================================
    # SAVE PREPROCESSING ARTIFACTS EARLY (crash protection)
    # These are needed for inference and won't change during training
    # =====================================================================
    print("  Saving preprocessing artifacts (crash protection)...")
    
    # Save scaler (for numerical feature standardization at test time)
    # IMPORTANT: Verify we're saving the StandardScaler, not the GradScaler
    if not hasattr(scaler, 'transform'):
        raise ValueError(f"ERROR: scaler is {type(scaler)}, expected StandardScaler! "
                        "This is a bug - check variable naming conflicts.")
    with open(results_dir / f'scaler_fold{args.fold}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save encoders (for categorical one-hot encoding at test time)
    with open(results_dir / f'encoders_fold{args.fold}.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save patient statistics (for patient-relative features at test time)
    # This allows computing Z-scores, ratios, diffs for test samples
    with open(results_dir / f'patient_statistics_fold{args.fold}.pkl', 'wb') as f:
        pickle.dump(patient_statistics, f)
    
    # Save precomputed features (LOF, patient-relative) for reference
    precomputed_features.to_pickle(results_dir / f'precomputed_features_fold{args.fold}.pkl')
    
    # Save feature lists for inference script
    feature_info = {
        'metadata_dim': metadata_dim,
        'numerical_features': [c for c in train_processed.columns if c not in ['isic_id', 'target', 'patient_id']],
        'patient_relative_features': PATIENT_RELATIVE_FEATURES,
        'lof_features': LOF_FEATURES,
    }
    with open(results_dir / f'feature_info_fold{args.fold}.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"  ✓ Saved: scaler, encoders, patient_statistics, precomputed_features, feature_info")
    
    # Training history
    history = {
        'train_loss': [], 'train_auc': [],
        'val_loss': [], 'val_auc': [],
        'val_loss_ema': [], 'val_auc_ema': [],
        'learning_rate': [],
        'epoch': []
    }
    
    best_auc = 0.0
    best_auc_ema = 0.0
    best_epoch = 0
    best_epoch_ema = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, amp_scaler, device,
            accumulation_steps=args.accumulation_steps,
            model_ema=model_ema,
            use_amp=args.amp
        )
        
        # Validate (regular model)
        val_loss, val_auc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, use_amp=args.amp
        )
        
        # Validate (EMA model)
        model_ema.apply_shadow()
        val_loss_ema, val_auc_ema, val_preds_ema, _ = validate(
            model, val_loader, criterion, device, use_amp=args.amp
        )
        model_ema.restore()
        
        # Update learning rate scheduler
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_loss_ema'].append(val_loss_ema)
        history['val_auc_ema'].append(val_auc_ema)
        history['learning_rate'].append(current_lr)
        
        print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
        print(f"Val (EMA) - Loss: {val_loss_ema:.4f}, AUC: {val_auc_ema:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # Save best models
        improved = False
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), results_dir / f'best_model_fold{args.fold}.pth')
            
            # Save OOF predictions for stacking
            # Use dataset's actual IDs (filters by available HDF5 keys)
            oof_df = pd.DataFrame({
                'isic_id': val_dataset.ids,
                'target': val_dataset.targets,
                'pred': val_preds
            })
            oof_df.to_csv(results_dir / f'oof_fold{args.fold}.csv', index=False)
            
            print(f"  ✓ New best model (regular): {best_auc:.4f}")
            improved = True
        
        if val_auc_ema > best_auc_ema:
            best_auc_ema = val_auc_ema
            best_epoch_ema = epoch + 1
            model_ema.apply_shadow()
            torch.save(model.state_dict(), results_dir / f'best_model_ema_fold{args.fold}.pth')
            
            # Save EMA OOF predictions
            oof_ema_df = pd.DataFrame({
                'isic_id': val_dataset.ids,
                'target': val_dataset.targets,
                'pred': val_preds_ema
            })
            oof_ema_df.to_csv(results_dir / f'oof_ema_fold{args.fold}.csv', index=False)
            
            model_ema.restore()
            print(f"  ✓ New best model (EMA): {best_auc_ema:.4f}")
            improved = True
        
        # Early stopping
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
        
        # Overfitting warning
        if epoch > 5:
            train_val_gap = train_auc - val_auc
            if train_val_gap > 0.15:
                print(f"  ⚠️ SEVERE overfitting! Gap: {train_val_gap:.4f}")
            elif train_val_gap > 0.08:
                print(f"  ⚠️ Overfitting warning! Gap: {train_val_gap:.4f}")
    
    # Add summary stats to history
    history['best_val_auc'] = best_auc
    history['best_val_auc_ema'] = best_auc_ema
    history['best_epoch'] = best_epoch
    history['best_epoch_ema'] = best_epoch_ema
    history['total_epochs'] = epoch + 1
    history['early_stopped'] = patience_counter >= args.patience
    
    # Save final results
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Val AUC (regular): {best_auc:.4f} @ epoch {best_epoch}")
    print(f"Best Val AUC (EMA):     {best_auc_ema:.4f} @ epoch {best_epoch_ema}")
    print(f"Total epochs:           {epoch + 1}")
    
    # Save history
    with open(results_dir / f'history_fold{args.fold}.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Note: scaler, encoders, patient_statistics already saved at start (crash protection)
    
    print(f"\nResults saved to: {results_dir}")
    print(f"  - best_model_fold{args.fold}.pth (regular)")
    print(f"  - best_model_ema_fold{args.fold}.pth (EMA)")
    print(f"  - oof_fold{args.fold}.csv / oof_ema_fold{args.fold}.csv")
    print(f"  - history_fold{args.fold}.json")
    print(f"  - config_fold{args.fold}.json")
    print(f"  - scaler_fold{args.fold}.pkl (saved at start)")
    print(f"  - encoders_fold{args.fold}.pkl (saved at start)")
    print(f"  - patient_statistics_fold{args.fold}.pkl (saved at start)")
    print(f"  - precomputed_features_fold{args.fold}.pkl (saved at start)")
    print(f"  - feature_info_fold{args.fold}.json (saved at start)")
    print("="*70)


if __name__ == '__main__':
    main()
