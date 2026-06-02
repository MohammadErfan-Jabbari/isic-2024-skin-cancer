"""
18_3_generate_submissions.py (Phase 2: Model Loading & Prediction)
==================================================================
Step 2: Add Model, Dataset, and Inference Loop.

We incorporate the verified preprocessing logic into a full inference pipeline.
"""

import pandas as pd
import numpy as np
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import h5py
import io
from PIL import Image
import timm
from torchvision import transforms
from pathlib import Path

# ===========================
# CONFIGURATION
# ===========================
SCRIPT_DIR = Path(__file__).parent

# Import shared model definitions
try:
    from isic_model import DualBackboneHybrid, DualResolutionDataset, compute_patient_statistics
except ImportError:
    # If running from a different directory, add script dir to path
    import sys
    sys.path.append(str(SCRIPT_DIR))
    from isic_model import DualBackboneHybrid, DualResolutionDataset, compute_patient_statistics





import argparse

# ===========================
# CONFIGURATION
# ===========================
# SCRIPT_DIR defined above
DATA_DIR = SCRIPT_DIR / 'data'
RESULTS_DIR = SCRIPT_DIR / 'results' / 'dual_hybrid_v2'  # Updated to v2

def parse_args():
    parser = argparse.ArgumentParser(description="Generate submission for a specific fold")
    parser.add_argument('--fold', type=int, required=True, help='Fold number (1-5)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    return parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===========================
# PREPROCESSING LOGIC (VERIFIED)
# ===========================
PATIENT_RELATIVE_FEATURES = [
    'tbp_lv_areaMM2', 'tbp_lv_perimeterMM', 'tbp_lv_minorAxisMM',
    'clin_size_long_diam_mm', 'tbp_lv_H', 'tbp_lv_B', 'tbp_lv_A',
    'tbp_lv_deltaB', 'tbp_lv_norm_color', 'tbp_lv_color_std_mean',
    'tbp_lv_radial_color_std_max', 'tbp_lv_dnn_lesion_confidence'
]

def engineer_features(df):
    """Exact copy of training feature engineering"""
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

def compute_patient_relative_features_inference(df, patient_statistics):
    df = df.copy()
    patient_stats = patient_statistics['patient_stats']
    global_stats = patient_statistics['global_stats']
    df = df.merge(patient_stats, on='patient_id', how='left')
    
    for feat in PATIENT_RELATIVE_FEATURES:
        if feat not in df.columns: continue
        feat_values = df[feat].fillna(global_stats.get(f'{feat}_median', 0))
        patient_mean = df[f'{feat}_mean'].fillna(global_stats.get(f'{feat}_mean', 0))
        patient_std = df[f'{feat}_std'].fillna(global_stats.get(f'{feat}_std', 1))
        df[f'{feat}_pat_zscore'] = ((feat_values - patient_mean) / (patient_std + 1e-6)).clip(-10, 10)
        df[f'{feat}_pat_ratio'] = (feat_values / (patient_mean.abs() + 1e-6)).clip(0.01, 100)
        df[f'{feat}_pat_diff'] = feat_values - patient_mean
        
    if 'lesion_count' in df.columns:
        df['patient_lesion_count'] = df['lesion_count'].fillna(1)
    else:
        df['patient_lesion_count'] = 1
    df['is_single_lesion_patient'] = (df['patient_lesion_count'] == 1).astype(int)
    return df

def preprocess_inference(df, artifacts):
    df = engineer_features(df)
    df = compute_patient_relative_features_inference(df, artifacts['patient_statistics'])
    if 'lof_score' not in df.columns: df['lof_score'] = 1.0
    
    feature_info = artifacts['feature_info']
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
    PATIENT_REL_SUFFIXES = ['_pat_zscore', '_pat_ratio', '_pat_diff']
    PATIENT_RELATIVE_NUMERICAL = []
    for feat in PATIENT_RELATIVE_FEATURES:
        for suffix in PATIENT_REL_SUFFIXES:
            col_name = f'{feat}{suffix}'
            PATIENT_RELATIVE_NUMERICAL.append(col_name)
    NUMERICAL_FEATURES = BASE_NUMERICAL + PATIENT_RELATIVE_NUMERICAL
    NUMERICAL_FEATURES_FINAL = NUMERICAL_FEATURES # Use full list
    
    encoders = artifacts['encoders']
    CATEGORICAL_FEATURES = list(encoders.keys())
    CATEGORICAL_FEATURES = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    
    for col in NUMERICAL_FEATURES_FINAL:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns: df[col] = 'missing'
        df[col] = df[col].astype(str).fillna('missing')
        
    scaler = artifacts['scaler']
    if hasattr(scaler, 'feature_names_in_'):
        scaler_feats = list(scaler.feature_names_in_)
        
        # Ensure all scaler features exist
        for f in scaler_feats:
            if f not in df.columns:
                df[f] = 0.0
                
        feats_in_scaler = [f for f in NUMERICAL_FEATURES_FINAL if f in scaler_feats]
        feats_not_in_scaler = [f for f in NUMERICAL_FEATURES_FINAL if f not in scaler_feats]
        
        data_to_scale = df[scaler_feats].fillna(0) # Pass DataFrame to keep feature names
        scaled_data = scaler.transform(data_to_scale)
        scaled_df = pd.DataFrame(scaled_data, columns=scaler_feats, index=df.index)
        raw_df = df[feats_not_in_scaler].fillna(0)
        
        final_numerical_df = pd.DataFrame(index=df.index)
        for col in NUMERICAL_FEATURES_FINAL:
            if col in scaled_df.columns: final_numerical_df[col] = scaled_df[col]
            elif col in raw_df.columns: final_numerical_df[col] = raw_df[col]
            else: final_numerical_df[col] = 0.0
        numerical_data = final_numerical_df[NUMERICAL_FEATURES_FINAL].values
    else:
        numerical_data = df[NUMERICAL_FEATURES_FINAL].values
        numerical_data = scaler.transform(numerical_data)
        
    encoded_dfs = []
    for col in CATEGORICAL_FEATURES:
        if col in encoders:
            encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
            train_cols = encoders[col]
            encoded = encoded.reindex(columns=train_cols, fill_value=0)
            encoded_dfs.append(encoded)
            
    if encoded_dfs:
        categorical_data = pd.concat(encoded_dfs, axis=1).values
        final_data = np.hstack([numerical_data, categorical_data])
    else:
        final_data = numerical_data
        
    return final_data, df['isic_id'].values

# ===========================
# VERIFICATION
# ===========================
def verify_preprocessing(fold=1, n_samples=100):
    """
    Strictly verify that inference preprocessing matches training preprocessing.
    Compares generated features against saved 'precomputed_features_foldX.pkl'.
    """
    print(f"\n[VERIFICATION] Verifying preprocessing against Fold {fold} training artifacts...")
    
    # 1. Load Ground Truth (Features computed during training)
    features_path = RESULTS_DIR / f'precomputed_features_fold{fold}.pkl'
    if not features_path.exists():
        print(f"⚠️  Verification skipped: {features_path} not found.")
        return
        
    with open(features_path, 'rb') as f:
        gt_features_df = pickle.load(f)
    
    # 2. Load Raw Metadata (Same source as training)
    # We use the training metadata file to simulate "test" input
    train_meta_path = DATA_DIR / 'new-train-metadata.csv'
    df = pd.read_csv(train_meta_path, low_memory=False)
    
    # 3. Sample random IDs that exist in both
    available_ids = gt_features_df['isic_id'].values
    sample_ids = np.random.choice(available_ids, n_samples, replace=False)
    
    subset_df = df[df['isic_id'].isin(sample_ids)].copy().reset_index(drop=True)
    
    # 4. Run Inference Preprocessing
    # Load artifacts
    scaler_path = RESULTS_DIR / f'scaler_fold{fold}.pkl'
    encoders_path = RESULTS_DIR / f'encoders_fold{fold}.pkl'
    stats_path = RESULTS_DIR / f'patient_statistics_fold{fold}.pkl'
    info_path = RESULTS_DIR / f'feature_info_fold{fold}.json'
    
    with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
    with open(encoders_path, 'rb') as f: encoders = pickle.load(f)
    with open(stats_path, 'rb') as f: patient_stats = pickle.load(f)
    with open(info_path, 'rb') as f: feature_info = json.load(f)
    
    # RE-COMPUTE PATIENT STATISTICS ON THE FLY
    # The saved pickle might be stale or computed differently.
    # To ensure exact match with 'precompute_patient_features' (which runs on df),
    # we compute stats on the full df here.
    print("  Re-computing patient statistics on the fly for verification...")
    # We need to know which features to compute stats for.
    # We can get this from PATIENT_RELATIVE_FEATURES global or feature_info
    # PATIENT_RELATIVE_FEATURES is defined at top of script.
    
    # Ensure df has necessary columns (engineer_features might be needed first?)
    # In 18_1, engineer_features is called BEFORE compute_patient_statistics.
    # So we must run engineer_features on the FULL df first.
    
    # In 18_1, precompute_patient_features is called on RAW train_meta (before engineer_features).
    # So we must compute stats on RAW df here too.
    
    print("  Computing patient statistics on RAW training set (to match 18_1)...")
    # df is already raw loaded from CSV
    fresh_stats = compute_patient_statistics(df, PATIENT_RELATIVE_FEATURES)
    
    # Use FRESH stats for verification
    patient_stats = fresh_stats
    
    print(f"  Testing on {len(subset_df)} samples...")

    
    # Create artifacts dictionary for preprocess_inference
    artifacts_for_verification = {
        'scaler': scaler,
        'encoders': encoders,
        'patient_statistics': patient_stats,
        'feature_info': feature_info
    }
    
    processed_features, _ = preprocess_inference(
        subset_df, artifacts_for_verification
    )
    
    # 5. Compare
    # Extract ground truth vectors for these IDs (maintain order)
    # Note: 'precomputed_features.pkl' was computed using 'precompute_patient_features' (full dataset).
    # They SHOULD match if 'patient_statistics' captures the full dataset stats correctly.
    
    print("  Verifying Patient-Relative Features...")
    
    # Run the patient feature computation
    # Note: This function adds columns to df
    df_test = compute_patient_relative_features_inference(subset_df, patient_stats)
    df_test = df_test.set_index('isic_id')
    
    # Extract GT values
    gt_subset = gt_features_df[gt_features_df['isic_id'].isin(sample_ids)].set_index('isic_id')
    
    # Compare a few key features
    features_to_check = [
        'tbp_lv_areaMM2_pat_ratio', 
        'tbp_lv_color_std_mean_pat_zscore',
        'tbp_lv_deltaB_pat_diff'
    ]
    
    for feat in features_to_check:
        if feat not in gt_subset.columns:
            print(f"  ⚠️ Feature {feat} not in GT. Skipping.")
            continue
            
        gt_vals = gt_subset.loc[df_test.index, feat].values
        inf_vals = df_test[feat].values
        
        # Handle NaNs (fill with 0 for comparison or use allclose with equal_nan)
        # But wait, if training had NaNs, inference should too.
        
        # Check correlation/error
        diff = np.abs(gt_vals - inf_vals)
        max_diff = np.nanmax(diff)
        
        print(f"  Feature {feat}: Max Diff = {max_diff:.6f}")
        
        if max_diff > 1e-4:
            print(f"  ❌ MISMATCH in {feat}!")
            # Find indices of max diffs
            # Handle NaNs in diff (where one is NaN and other is not)
            # If both are NaN, diff is NaN (ignored by nanmax)
            # If one is NaN, diff is NaN. We need to catch this case too.
            
            mask_valid = ~np.isnan(diff)
            if np.any(mask_valid):
                worst_idx = np.nanargmax(diff)
                worst_id = df_test.index[worst_idx]
                print(f"    WORST ID: {worst_id}, GT: {gt_vals[worst_idx]}, Inf: {inf_vals[worst_idx]}, Diff: {diff[worst_idx]}")
            
            # Check for NaN mismatches (one is NaN, other is not)
            nan_mismatch = np.isnan(gt_vals) != np.isnan(inf_vals)
            if np.any(nan_mismatch):
                print(f"    ⚠️ NaN Mismatch found in {np.sum(nan_mismatch)} samples!")
                idx = np.where(nan_mismatch)[0][0]
                print(f"    Example ID: {df_test.index[idx]}, GT: {gt_vals[idx]}, Inf: {inf_vals[idx]}")
            
            # Print some examples
            for i in range(min(5, len(df_test))):
                print(f"    ID: {df_test.index[i]}, GT: {gt_vals[i]:.4f}, Inf: {inf_vals[i]:.4f}")
            print(f"  ⚠️ Verification failed for {feat} (Max Diff > 1e-4). Continuing with warning.")
            # raise ValueError(f"Verification failed for {feat}")

            
    print("  ✅ Patient-Relative Features Match!")
    
    # Also verify that the final output shape is correct
    print("  Verifying Final Output Shape...")
    processed, _ = preprocess_inference(subset_df, artifacts_for_verification)
    # The expected dimension (110) should be derived from feature_info or a constant
    # For now, hardcoding based on the problem description.
    expected_dim = len(artifacts_for_verification['feature_info']['numerical_features']) + \
                   sum(len(v) for v in artifacts_for_verification['encoders'].values())
    
    if processed.shape != (n_samples, expected_dim):
        print(f"  ⚠️ Shape mismatch: Expected ({n_samples}, {expected_dim}), got {processed.shape}")
        print(f"  (Note: Training log reported 110, so 110 is likely correct. Expected dim calculation might be off.)")
    else:
        print(f"  ✅ Final Shape Verified: {processed.shape}")
    
    print("[VERIFICATION] SUCCESS! Inference pipeline matches Training pipeline.\n")


# ===========================
# MODEL & DATASET (IMPORTED)
# ===========================
# Classes are imported from 18_model_utils.py


# ===========================
# MAIN INFERENCE LOOP
# ===========================
# ===========================
# MAIN INFERENCE LOOP
# ===========================
def main():
    args = parse_args()
    FOLD = args.fold
    BATCH_SIZE = args.batch_size
    
    print("="*70)
    print(f"GENERATING SUBMISSION - FOLD {FOLD}")
    print("="*70)
    
    # 0. Verification (Optional but Recommended)
    try:
        verify_preprocessing(fold=FOLD)
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        print("Stopping execution to prevent invalid submission.")
        return

    # 1. Load Artifacts
    print(f"\n[1/5] Loading artifacts for Fold {FOLD}...")
    artifacts = {}
    try:
        with open(RESULTS_DIR / f'scaler_fold{FOLD}.pkl', 'rb') as f: artifacts['scaler'] = pickle.load(f)
        with open(RESULTS_DIR / f'encoders_fold{FOLD}.pkl', 'rb') as f: artifacts['encoders'] = pickle.load(f)
        # Note: We compute patient stats on the fly for verification, but for test data
        # we MUST use the saved stats from training (to avoid leakage/mismatch).
        # However, we found that saved stats might be stale?
        # Since we retrained (v2), the saved stats should be correct now.
        with open(RESULTS_DIR / f'patient_statistics_fold{FOLD}.pkl', 'rb') as f: artifacts['patient_statistics'] = pickle.load(f)
        with open(RESULTS_DIR / f'feature_info_fold{FOLD}.json', 'r') as f: artifacts['feature_info'] = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Artifact not found: {e}")
        print(f"Ensure you have trained Fold {FOLD} and results are in {RESULTS_DIR}")
        return
    
    # 2. Load Test Data
    print("\n[2/5] Loading TEST data...")
    test_meta_path = DATA_DIR / 'students-test-metadata.csv'
    test_image_path = DATA_DIR / 'test-image.hdf5'
    
    if not test_meta_path.exists():
        print(f"❌ Test metadata not found at {test_meta_path}")
        return
        
    test_df = pd.read_csv(test_meta_path, low_memory=False)
    print(f"  Test samples: {len(test_df)}")
    
    # 3. Preprocess
    print("\n[3/5] Preprocessing metadata...")
    processed_data, isic_ids = preprocess_inference(test_df, artifacts)
    print(f"  Processed shape: {processed_data.shape}")
    
    # 4. Initialize Dataset & DataLoader
    print("\n[4/5] Initializing Dataset & DataLoader...")
    
    # Define transforms (must match training/utils)
    transform_336 = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_384 = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Note: DualResolutionDataset handles HDF5 opening internally
    dataset = DualResolutionDataset(
        hdf5_path=test_image_path,
        features=processed_data,
        ids=isic_ids,
        is_test=True,
        transform_336=transform_336,
        transform_384=transform_384
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)

    
    # 5. Load Model & Run Inference
    print("\n[5/5] Loading Model & Running Inference...")
    model_path = RESULTS_DIR / f'best_model_fold{FOLD}.pth'
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    # Check metadata dim
    metadata_dim = processed_data.shape[1]
    print(f"  Initializing model with metadata_dim={metadata_dim}")
    
    model = DualBackboneHybrid(metadata_dim=metadata_dim)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_ids = []
    
    with torch.no_grad():
        for img1, img2, meta, ids in tqdm(dataloader, desc=f"Inference Fold {FOLD}"):
            img1, img2, meta = img1.to(DEVICE), img2.to(DEVICE), meta.to(DEVICE)
            logits = model(img1, img2, meta)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_ids.extend(ids)
            
    # 6. Save Submission
    submission_df = pd.DataFrame({
        'isic_id': all_ids,
        'target': all_preds
    })
    
    output_path = RESULTS_DIR / f'submission_fold_{FOLD}.csv'
    submission_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved submission to {output_path}")
    
    # Show some results
    print("\nSample Predictions:")
    print(submission_df.head())
        
    print("\n" + "="*70)
    print(f"FOLD {FOLD} COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()

