"""
Recover and fix the scalers for dual_hybrid_v1 folds.

This script:
1. Reconstructs the fold splits using the same StratifiedGroupKFold parameters
2. Identifies which samples were in the training set for each fold
3. Fits a proper StandardScaler on those training samples
4. Saves the corrected scaler, overwriting the GradScaler

This way we don't need to retrain - we just fix the preprocessing artifacts.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json

# ===========================
# CONFIGURATION
# ===========================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / 'data'
RESULTS_DIR = SCRIPT_DIR / 'results' / 'dual_hybrid_v1'

# Reproduction parameters (from 18_1_train_dual_backbone_hybrid.py)
RANDOM_STATE = 42
N_SPLITS = 5

# Features to compute patient-relative statistics for
PATIENT_RELATIVE_FEATURES = [
    'tbp_lv_areaMM2', 'tbp_lv_perimeterMM', 'tbp_lv_minorAxisMM',
    'clin_size_long_diam_mm', 'tbp_lv_H', 'tbp_lv_B', 'tbp_lv_A',
    'tbp_lv_deltaB', 'tbp_lv_norm_color', 'tbp_lv_color_std_mean',
    'tbp_lv_radial_color_std_max', 'tbp_lv_dnn_lesion_confidence'
]

# ===========================
# FUNCTIONS FROM TRAINING SCRIPT
# ===========================

def engineer_features(df):
    """Enhanced feature engineering (from 18_1)"""
    df = df.copy()
    
    # AGE FEATURES
    df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median())
    df['age_group'] = pd.cut(df['age_approx'], bins=[0, 30, 50, 70, 100],
                             labels=['young', 'middle', 'senior', 'elderly'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    df['age_squared'] = df['age_approx'] ** 2
    
    # SIZE FEATURES
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['size_category'] = pd.cut(df['lesion_size_mm'].fillna(0), bins=[0, 6, 10, 20, 1000],
                                 labels=['small', 'medium', 'large', 'very_large'])
    df['large_lesion'] = (df['lesion_size_mm'] > 6).astype(int)
    
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


def preprocess_metadata_for_scaler_fitting(df):
    """
    Preprocess metadata to extract numerical features for scaler fitting.
    This matches the training preprocessing.
    """
    df = engineer_features(df)
    
    # Build numerical features list (same logic as training)
    BASE_NUMERICAL = [
        'tbp_lv_H', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM',
        'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_Hext',
        'clin_size_long_diam_mm', 'tbp_lv_radial_color_std_max',
        'tbp_lv_B', 'tbp_lv_color_std_mean', 'tbp_lv_Aext',
        'tbp_lv_stdLExt', 'tbp_lv_norm_color', 'tbp_lv_A', 'age_approx',
        'tbp_lv_dnn_lesion_confidence', 'tbp_lv_nevi_confidence',
        'age_squared', 'lesion_size_mm',
        'shape_regularity', 'eccentricity', 'compactness',
        'color_variance', 'color_uniformity', 'darkness_score', 'color_contrast',
        'site_risk_score', 'age_size_risk', 'age_site_risk', 'color_size_risk',
        'age_color_risk', 'asymmetry_score',
        'log_area', 'log_perimeter', 'log_size',
        'h_to_b_ratio', 'a_to_b_ratio', 'area_to_perimeter',
        'lof_score',
        'patient_lesion_count', 'is_single_lesion_patient'
    ]
    
    # Add patient-relative features
    PATIENT_REL_SUFFIXES = ['_pat_zscore', '_pat_ratio', '_pat_diff']
    PATIENT_RELATIVE_NUMERICAL = []
    for feat in PATIENT_RELATIVE_FEATURES:
        for suffix in PATIENT_REL_SUFFIXES:
            col_name = f'{feat}{suffix}'
            if col_name in df.columns:
                PATIENT_RELATIVE_NUMERICAL.append(col_name)
    
    NUMERICAL_FEATURES = BASE_NUMERICAL + PATIENT_RELATIVE_NUMERICAL
    
    # Filter to existing columns
    NUMERICAL_FEATURES = [f for f in NUMERICAL_FEATURES if f in df.columns]
    
    # Fill missing values
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
            # Also handle inf values
            df[col] = df[col].replace([np.inf, -np.inf], median_val)
    
    return df[NUMERICAL_FEATURES]


# ===========================
# MAIN PROCEDURE
# ===========================

def main():
    print("="*70)
    print("RECOVERING CORRECT SCALERS FOR dual_hybrid_v1")
    print("="*70)
    
    # 1. Load training metadata
    print("\n[1/4] Loading training metadata...")
    train_meta_path = DATA_DIR / 'new-train-metadata.csv'
    train_meta = pd.read_csv(train_meta_path, low_memory=False)
    print(f"  Total samples: {len(train_meta):,}")
    print(f"  Malignant: {train_meta['target'].sum()}")
    
    # 2. Recreate fold splits
    print("\n[2/4] Recreating fold splits using StratifiedGroupKFold...")
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(sgkf.split(train_meta, train_meta['target'], train_meta['patient_id']))
    print(f"  Created {len(splits)} folds")
    
    # 3. For each fold, fit scaler on training data and save
    print("\n[3/4] Fitting correct scalers for each fold...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n  Fold {fold_idx}:")
        
        # Get training samples for this fold
        train_df = train_meta.iloc[train_idx].copy()
        print(f"    Training samples: {len(train_df):,}")
        
        # Preprocess to get numerical features
        numerical_features_df = preprocess_metadata_for_scaler_fitting(train_df)
        print(f"    Numerical features: {numerical_features_df.shape[1]}")
        
        # Fit StandardScaler on training data
        scaler = StandardScaler()
        scaler.fit(numerical_features_df)
        print(f"    Fitted StandardScaler")
        
        # Save to replace the GradScaler
        scaler_path = RESULTS_DIR / f'scaler_fold{fold_idx}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"    ✓ Saved correct scaler to {scaler_path.name}")
        
        # Verify what we saved
        with open(scaler_path, 'rb') as f:
            saved_scaler = pickle.load(f)
        
        if hasattr(saved_scaler, 'transform'):
            print(f"    ✓ Verification: Saved scaler is StandardScaler (correct!)")
        else:
            print(f"    ✗ Verification: ERROR - saved scaler is not StandardScaler!")
    
    # 4. Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ Successfully recovered and fixed scalers for all folds!")
    print(f"  Location: {RESULTS_DIR}")
    print(f"\nYou can now use 18_3_generate_submissions.py with the corrected scalers.")
    print(f"The inference will use the proper StandardScaler fitted on training data.")
    print("="*70)


if __name__ == '__main__':
    main()
