#!/usr/bin/env python3
"""
Corrected Stacking Submission Script

This is the corrected version of 16_5_submission_stacking.py that fixes the 
training/inference preprocessing mismatches identified in Phase 4 analysis.

FIXES APPLIED:
1. ✅ Loads standardization_stats.pkl for consistent preprocessing
2. ✅ Uses z-score standardization (not rank normalization)
3. ✅ Includes all required exclude_cols (mel_thick_mm, etc.)
4. ✅ Outputs raw GBDT probabilities (no rank normalization)
5. ✅ Maintains same feature engineering as training

Author: Kilo Code (Corrected)
Date: 2025-11-26
Original: 16_5_submission_stacking.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import argparse
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Set up paths
BASE_DIR = Path('.')
RESULTS_DIR = BASE_DIR / 'results' / 'stacking_final_v1'
DATA_DIR = BASE_DIR / 'data'

def load_standardization_stats():
    """Load standardization statistics saved during training"""
    stats_path = RESULTS_DIR / 'standardization_stats.pkl'
    if not stats_path.exists():
        raise FileNotFoundError(f"Standardization stats not found: {stats_path}")
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    print(f"Loaded standardization stats:")
    for key, value in stats.items():
        print(f"  {key}: mean={value['mean']:.6f}, std={value['std']:.6f}")
    
    return stats

def apply_zscore_standardization(df, stats):
    """Apply z-score standardization using saved training statistics"""
    for col in ['eva02_pred', 'edgenext_pred']:
        if col in df.columns and col in stats:
            mean = stats[col]['mean']
            std = stats[col]['std']
            df[f'{col}_standardized'] = (df[col] - mean) / (std + 1e-8)
            print(f"Applied z-score to {col}: mean={mean:.6f}, std={std:.6f}")
    
    return df

def get_exclude_cols():
    """Get exclude columns list - must match training script 16_3"""
    return [
        'isic_id', 'patient_id', 'target', 'image_type', 'attribution', 'copyright_license',
        'mel_thick_mm', 'mel_mitotic_index',  # LEAKY FEATURES - EXCLUDE
        'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5',  # DIAGNOSIS LEAKS
    ]

def load_test_data():
    """Load test metadata and vision predictions"""
    # Load test metadata
    test_meta_path = DATA_DIR / 'new-test-metadata.csv'
    if not test_meta_path.exists():
        test_meta_path = DATA_DIR / 'test-metadata.csv'
    
    df = pd.read_csv(test_meta_path, low_memory=False)
    print(f"Loaded test metadata: {len(df)} samples")
    
    # Load vision predictions
    test_preds_path = RESULTS_DIR / 'test_vision_preds.csv'
    if test_preds_path.exists():
        test_preds = pd.read_csv(test_preds_path)
        df = df.merge(test_preds, on='isic_id', how='left')
        print(f"Merged vision predictions: {len(df)} samples")
    else:
        raise FileNotFoundError(f"Test vision predictions not found: {test_preds_path}")
    
    return df

def engineer_features(df):
    """Apply feature engineering - must match training script"""
    # Patient-level features
    patient_stats = df.groupby('patient_id').agg({
        'tbp_lv_areaMM2': ['mean', 'std', 'count'],
        'tbp_lv_perimeterMM': ['mean', 'std'],
        'age_approx': 'mean'
    }).reset_index()
    
    # Flatten column names
    patient_stats.columns = ['patient_id'] + [f'patient_{col[0]}_{col[1]}' for col in patient_stats.columns[1:]]
    
    # Merge back
    df = df.merge(patient_stats, on='patient_id', how='left')
    
    # Vision model features
    df['mean_vision'] = (df['eva02_pred'] + df['edgenext_pred']) / 2
    df['vision_diff'] = df['eva02_pred'] - df['edgenext_pred']
    df['vision_ratio'] = df['eva02_pred'] / (df['edgenext_pred'] + 1e-8)
    
    # Patient-relative features (z-scores)
    numeric_cols = ['tbp_lv_areaMM2', 'tbp_lv_perimeterMM', 'tbp_lv_minorAxisMM']
    for col in numeric_cols:
        if col in df.columns:
            patient_mean = df.groupby('patient_id')[col].transform('mean')
            patient_std = df.groupby('patient_id')[col].transform('std')
            df[f'{col}_zscore'] = (df[col] - patient_mean) / (patient_std + 1e-8)
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-vision', action='store_true', help='Skip vision predictions')
    parser.add_argument('--output-name', default='submission_corrected', help='Output file name')
    args = parser.parse_args()
    
    print("=== Corrected Stacking Submission Pipeline ===")
    print(f"Args: {args}")
    
    # Load data
    df = load_test_data()
    
    # Load standardization stats
    std_stats = load_standardization_stats()
    
    # Apply z-score standardization (NOT rank normalization)
    if not args.skip_vision:
        df = apply_zscore_standardization(df, std_stats)
        print("✅ Applied z-score standardization to vision predictions")
    else:
        print("⚠️ Skipping vision predictions as requested")
    
    # Feature engineering
    df = engineer_features(df)
    print(f"✅ Feature engineering completed: {df.shape}")
    
    # Get exclude columns
    exclude_cols = get_exclude_cols()
    print(f"✅ Using exclude_cols: {exclude_cols}")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category')
    
    print(f"✅ Feature matrix prepared: {X.shape}")
    print(f"   Numeric features: {len(numeric_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")
    
    # Load trained models and make predictions
    predictions = []
    model_files = list((RESULTS_DIR / 'models').glob('lgbm_fold*.joblib'))
    
    if not model_files:
        raise FileNotFoundError("No trained models found in {RESULTS_DIR / 'models'}")
    
    print(f"Found {len(model_files)} trained models")
    
    for model_file in sorted(model_files):
        print(f"Loading model: {model_file.name}")
        model = pickle.load(open(model_file, 'rb'))
        
        # Make prediction
        pred = model.predict_proba(X)[:, 1]
        predictions.append(pred)
    
    # Average predictions across folds
    stack_preds = np.mean(predictions, axis=0)
    
    print(f"✅ Stacking predictions completed")
    print(f"   Prediction range: [{stack_preds.min():.6f}, {stack_preds.max():.6f}]")
    print(f"   Prediction mean: {stack_preds.mean():.6f}")
    
    # Create submission (RAW probabilities - NO rank normalization)
    submission = pd.DataFrame({
        'isic_id': df['isic_id'],
        'target': stack_preds  # RAW probabilities, not ranked
    })
    
    # Save submission
    output_path = BASE_DIR / 'submissions' / f'{args.output_name}.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"✅ Submission saved: {output_path}")
    print(f"   File size: {output_path.stat().st_size} bytes")
    
    # Validation checks
    print(f"\n=== Validation Checks ===")
    print(f"Prediction range: [{submission['target'].min():.6f}, {submission['target'].max():.6f}]")
    print(f"Prediction mean: {submission['target'].mean():.6f}")
    print(f"Any NaN values: {submission['target'].isna().any()}")
    print(f"Any infinite values: {np.isinf(submission['target']).any()}")
    
    # Compare to training OOF if available
    oof_path = RESULTS_DIR / 'stacking_oof.csv'
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        print(f"\nTraining OOF comparison:")
        print(f"  OOF range: [{oof['stack_pred'].min():.6f}, {oof['stack_pred'].max():.6f}]")
        print(f"  OOF mean: {oof['stack_pred'].mean():.6f}")
        
        # Check if ranges are reasonable
        test_range = submission['target'].max() - submission['target'].min()
        oof_range = oof['stack_pred'].max() - oof['stack_pred'].min()
        range_ratio = test_range / oof_range if oof_range > 0 else 0
        
        print(f"  Range ratio (test/oof): {range_ratio:.2f}")
        if 0.5 <= range_ratio <= 2.0:
            print("  ✅ Range ratio is reasonable")
        else:
            print("  ⚠️ Range ratio may indicate issues")

if __name__ == "__main__":
    main()
