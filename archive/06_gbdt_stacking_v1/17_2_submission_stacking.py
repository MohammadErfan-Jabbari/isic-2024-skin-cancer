#!/usr/bin/env python3
"""
17_2_submission_stacking.py - Submission Script for GBDT Stacking Model
========================================================================

This script generates submissions using the trained GBDT stacking model.
CRITICAL: This script uses the EXACT SAME preprocessing as training to avoid
the distribution shift issues identified in the investigation.

Key Fixes Applied:
1. Uses Z-Score standardization (matching training)
2. Loads saved standardization_stats from training
3. Excludes the same leakage features
4. Uses saved label encoders for categorical features
5. Outputs RAW probabilities (no final rank normalization)

Author: Data Science Pipeline
Date: 2025-11-26
Version: 2.0 (Matching 17_1_train_stacking_gbdt.py)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import argparse
from pathlib import Path
import pickle
import joblib
import json
import h5py
import warnings
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / 'data'

# ===========================
# FEATURE CONFIGURATION
# Must match training exactly
# ===========================

# LEAKAGE FEATURES - MUST EXCLUDE (same as training)
LEAKAGE_FEATURES = [
    'mel_thick_mm',
    'mel_mitotic_index',
    'iddx_full',
    'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5',
]

# NON-INFORMATIVE FEATURES - EXCLUDE
NON_INFORMATIVE_FEATURES = [
    'isic_id', 'patient_id', 'target',
    'image_type', 'attribution', 'copyright_license',
    'tbp_lv_location',
]

# CATEGORICAL FEATURES
CATEGORICAL_FEATURES = [
    'sex', 
    'anatom_site_general', 
    'tbp_tile_type', 
    'tbp_lv_location_simple'
]

# Features for patient-relative calculations
RELATIVE_FEATURE_COLS = [
    'tbp_lv_areaMM2', 'tbp_lv_deltaB', 'clin_size_long_diam_mm',
    'tbp_lv_minorAxisMM', 'tbp_lv_eccentricity', 'tbp_lv_norm_color',
    'tbp_lv_radial_color_std_max', 'tbp_lv_color_std_mean',
    'eva02_pred', 'edgenext_pred'
]

# Features for LOF calculation
LOF_FEATURES = [
    'tbp_lv_areaMM2', 'tbp_lv_deltaB', 'clin_size_long_diam_mm',
    'tbp_lv_eccentricity', 'tbp_lv_norm_color', 'tbp_lv_radial_color_std_max'
]


# ===========================
# VISION MODEL INFERENCE
# ===========================

def run_vision_inference(model_dir, hdf5_path, image_size, batch_size=32, device='cuda'):
    """
    Run inference using a vision model on test images.
    
    Args:
        model_dir: Directory containing vision model folds
        hdf5_path: Path to HDF5 file with test images
        image_size: Image size (224, 336, 384)
        batch_size: Batch size for inference
        device: Device to use
        
    Returns:
        DataFrame with isic_id and predictions
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import timm
    
    class HDF5TestDataset(Dataset):
        def __init__(self, hdf5_path, image_size):
            self.hdf5_path = hdf5_path
            self.image_size = image_size
            
            with h5py.File(hdf5_path, 'r') as f:
                self.isic_ids = [id.decode() if isinstance(id, bytes) else id 
                                 for id in f['isic_id'][:]]
                
        def __len__(self):
            return len(self.isic_ids)
            
        def __getitem__(self, idx):
            with h5py.File(self.hdf5_path, 'r') as f:
                img = f['images'][idx]
                
            # Convert to float and normalize
            img = img.astype(np.float32) / 255.0
            
            # Resize if needed
            from PIL import Image
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
            img = np.array(img_pil, dtype=np.float32) / 255.0
            
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            
            # HWC -> CHW
            img = img.transpose(2, 0, 1)
            
            return torch.tensor(img, dtype=torch.float32), self.isic_ids[idx]
    
    # Load models
    model_paths = sorted(Path(model_dir).glob('*_fold*.pth')) + \
                  sorted(Path(model_dir).glob('best_model*.pth'))
    
    if not model_paths:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    print(f"   Found {len(model_paths)} model files")
    
    models = []
    for path in model_paths:
        # Detect model architecture from path
        if 'eva02' in str(path).lower():
            model = timm.create_model('eva02_small_patch14_336.mim_in22k_ft_in1k', 
                                      pretrained=False, num_classes=1)
        elif 'edgenext' in str(path).lower():
            model = timm.create_model('edgenext_base', pretrained=False, num_classes=1)
        else:
            # Try to detect from directory name
            if 'eva' in str(model_dir).lower():
                model = timm.create_model('eva02_small_patch14_336.mim_in22k_ft_in1k',
                                          pretrained=False, num_classes=1)
            else:
                model = timm.create_model('edgenext_base', pretrained=False, num_classes=1)
        
        state_dict = torch.load(path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
    
    # Create dataset and loader
    dataset = HDF5TestDataset(hdf5_path, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Run inference
    all_preds = []
    all_ids = []
    
    with torch.no_grad():
        for batch_imgs, batch_ids in tqdm(loader, desc="   Inference"):
            batch_imgs = batch_imgs.to(device)
            
            # Average predictions from all folds
            batch_preds = []
            for model in models:
                preds = torch.sigmoid(model(batch_imgs)).cpu().numpy().squeeze()
                batch_preds.append(preds)
            
            avg_preds = np.mean(batch_preds, axis=0)
            
            all_preds.extend(avg_preds if len(avg_preds.shape) > 0 else [avg_preds])
            all_ids.extend(batch_ids)
    
    return pd.DataFrame({'isic_id': all_ids, 'pred': all_preds})


# ===========================
# FEATURE ENGINEERING
# Must match training exactly
# ===========================

def calculate_patient_relative_features(df):
    """
    Calculates patient-relative features.
    MUST MATCH TRAINING EXACTLY.
    """
    print("    - Calculating Patient-Relative Statistics...")
    
    cols_to_process = [c for c in RELATIVE_FEATURE_COLS if c in df.columns]
    
    grouped = df.groupby('patient_id')[cols_to_process]
    means = grouped.transform('mean')
    stds = grouped.transform('std')
    mins = grouped.transform('min')
    maxs = grouped.transform('max')
    counts = df.groupby('patient_id')['isic_id'].transform('count')
    
    for col in cols_to_process:
        df[f'{col}_ratio_mean'] = df[col] / (means[col] + 1e-6)
        df[f'{col}_diff_mean'] = df[col] - means[col]
        z_score = (df[col] - means[col]) / (stds[col] + 1e-6)
        df[f'{col}_zscore'] = z_score.fillna(0)
        df[f'{col}_ratio_max'] = df[col] / (maxs[col] + 1e-6)
        df[f'{col}_ratio_min'] = df[col] / (mins[col] + 1e-6)

    df['patient_lesion_count'] = counts
    
    return df


def calculate_lof(df):
    """
    Calculates Local Outlier Factor.
    MUST MATCH TRAINING EXACTLY.
    """
    print("    - Calculating Local Outlier Factor (LOF)...")
    
    lof_features = [c for c in LOF_FEATURES if c in df.columns]
    df['patient_lof'] = np.nan
    
    patient_counts = df['patient_id'].value_counts()
    valid_patients = patient_counts[patient_counts >= 5].index
    
    df_filled = df.copy()
    for col in lof_features:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
    def get_lof(group):
        if len(group) < 5:
            return np.full(len(group), -1.0)
        try:
            clf = LocalOutlierFactor(n_neighbors=min(len(group)-1, 20), novelty=False)
            X = group[lof_features].values
            clf.fit_predict(X)
            return clf.negative_outlier_factor_
        except:
            return np.full(len(group), -1.0)

    lof_map = {}
    valid_df = df_filled[df_filled['patient_id'].isin(valid_patients)]
    
    for pid, group in tqdm(valid_df.groupby('patient_id'), desc="LOF Calculation", leave=False):
        lof_scores = get_lof(group)
        for i, isic_id in enumerate(group['isic_id'].values):
            lof_map[isic_id] = lof_scores[i]
            
    df['patient_lof'] = df['isic_id'].map(lof_map).fillna(-1.0)
    
    return df


def engineer_features(df):
    """
    Master feature engineering function.
    MUST MATCH TRAINING EXACTLY.
    """
    df = df.copy()
    
    # Basic derived features
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    
    # Shape regularity
    df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    
    # Color variance
    df['color_variance'] = np.sqrt(
        df['tbp_lv_deltaB']**2 + 
        df['tbp_lv_radial_color_std_max']**2 +
        df['tbp_lv_color_std_mean']**2
    )
    
    # Patient-relative features
    df = calculate_patient_relative_features(df)
    
    # Local Outlier Factor
    df = calculate_lof(df)
    
    # Vision ensemble features
    if 'eva02_pred' in df.columns and 'edgenext_pred' in df.columns:
        df['mean_vision_pred'] = (df['eva02_pred'] + df['edgenext_pred']) / 2
        df['vision_pred_diff'] = df['eva02_pred'] - df['edgenext_pred']
        df['vision_pred_max'] = df[['eva02_pred', 'edgenext_pred']].max(axis=1)
        df['vision_pred_min'] = df[['eva02_pred', 'edgenext_pred']].min(axis=1)
        
        # Patient-relative vision features
        grouped = df.groupby('patient_id')['mean_vision_pred']
        means = grouped.transform('mean')
        stds = grouped.transform('std')
        df['mean_vision_pred_zscore'] = ((df['mean_vision_pred'] - means) / (stds + 1e-6)).fillna(0)
        df['mean_vision_pred_ratio'] = df['mean_vision_pred'] / (means + 1e-6)
        
    return df


def preprocess_for_gbdt(df, label_encoders):
    """
    Preprocesses test data for GBDT inference.
    Uses label_encoders from training.
    """
    df = engineer_features(df)
    
    # Build exclude list
    exclude_cols = LEAKAGE_FEATURES + NON_INFORMATIVE_FEATURES
    
    # Get numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in exclude_cols]
    
    # Get categorical columns
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    
    # Handle categorical encoding using saved encoders
    for col in cat_cols:
        if col not in df.columns:
            continue
            
        if col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].fillna('MISSING').astype(str)
            # Handle unseen categories
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'MISSING')
            # Add MISSING to classes if not present
            if 'MISSING' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'MISSING')
            df[col] = le.transform(df[col])
    
    return df, num_cols, cat_cols


# ===========================
# MAIN INFERENCE FUNCTION
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing trained GBDT models')
    parser.add_argument('--eva-dir', type=str, default=None,
                        help='Directory containing EVA02 models (if running vision inference)')
    parser.add_argument('--edgenext-dir', type=str, default=None,
                        help='Directory containing EdgeNeXt models (if running vision inference)')
    parser.add_argument('--eva-preds', type=str, default=None,
                        help='Pre-computed EVA02 test predictions CSV')
    parser.add_argument('--edgenext-preds', type=str, default=None,
                        help='Pre-computed EdgeNeXt test predictions CSV')
    parser.add_argument('--test-hdf5', type=str, 
                        default='data/test-images-processed.hdf5',
                        help='Test HDF5 file')
    parser.add_argument('--test-meta', type=str,
                        default='data/students-test-metadata.csv',
                        help='Test metadata CSV')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output submission file')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 GBDT STACKING SUBMISSION GENERATION")
    print("   Using MATCHING preprocessing from training")
    print("=" * 70)
    
    model_dir = Path(args.model_dir)
    
    # 1. Load training artifacts
    print("\n📂 Loading training artifacts...")
    
    # Load standardization stats
    stats_path = model_dir / 'standardization_stats.pkl'
    if not stats_path.exists():
        raise FileNotFoundError(f"❌ standardization_stats.pkl not found in {model_dir}")
    
    with open(stats_path, 'rb') as f:
        standardization_stats = pickle.load(f)
    print(f"   ✅ Loaded standardization stats")
    for col, stats in standardization_stats.items():
        print(f"      {col}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    
    # Load label encoders
    le_path = model_dir / 'label_encoders.pkl'
    if not le_path.exists():
        raise FileNotFoundError(f"❌ label_encoders.pkl not found in {model_dir}")
        
    with open(le_path, 'rb') as f:
        label_encoders = pickle.load(f)
    print(f"   ✅ Loaded label encoders for: {list(label_encoders.keys())}")
    
    # Load feature list
    feat_path = model_dir / 'feature_list.json'
    if not feat_path.exists():
        raise FileNotFoundError(f"❌ feature_list.json not found in {model_dir}")
        
    with open(feat_path, 'r') as f:
        feature_config = json.load(f)
    features = feature_config['features']
    print(f"   ✅ Loaded {len(features)} features")
    
    # 2. Load GBDT models
    print("\n🤖 Loading GBDT models...")
    
    models_subdir = model_dir / 'models'
    
    lgbm_paths = sorted(models_subdir.glob('lgbm_fold*.joblib'))
    xgb_paths = sorted(models_subdir.glob('xgb_fold*.joblib'))
    
    lgbm_models = [joblib.load(p) for p in lgbm_paths]
    xgb_models = [joblib.load(p) for p in xgb_paths]
    
    print(f"   ✅ Loaded {len(lgbm_models)} LightGBM models")
    print(f"   ✅ Loaded {len(xgb_models)} XGBoost models")
    
    # 3. Load test metadata
    print("\n📂 Loading test metadata...")
    test_meta_path = SCRIPT_DIR / args.test_meta
    test_meta = pd.read_csv(test_meta_path, low_memory=False)
    print(f"   Loaded {len(test_meta):,} test samples")
    
    # 4. Get vision predictions
    print("\n🔮 Getting vision predictions...")
    
    if args.eva_preds and Path(args.eva_preds).exists():
        print("   Loading pre-computed EVA02 predictions...")
        eva_preds = pd.read_csv(args.eva_preds)
        eva_preds.rename(columns={'pred': 'eva02_pred', 'target': 'eva02_pred'}, inplace=True)
        if 'eva02_pred' not in eva_preds.columns:
            # Try to find the prediction column
            pred_cols = [c for c in eva_preds.columns if 'pred' in c.lower() or c == 'target']
            if pred_cols:
                eva_preds.rename(columns={pred_cols[0]: 'eva02_pred'}, inplace=True)
    elif args.eva_dir:
        print("   Running EVA02 inference...")
        eva_preds = run_vision_inference(
            args.eva_dir, 
            SCRIPT_DIR / args.test_hdf5,
            336, args.batch_size, args.device
        )
        eva_preds.rename(columns={'pred': 'eva02_pred'}, inplace=True)
    else:
        raise ValueError("❌ Must provide either --eva-preds or --eva-dir")
    
    if args.edgenext_preds and Path(args.edgenext_preds).exists():
        print("   Loading pre-computed EdgeNeXt predictions...")
        edgenext_preds = pd.read_csv(args.edgenext_preds)
        edgenext_preds.rename(columns={'pred': 'edgenext_pred', 'target': 'edgenext_pred'}, inplace=True)
        if 'edgenext_pred' not in edgenext_preds.columns:
            pred_cols = [c for c in edgenext_preds.columns if 'pred' in c.lower() or c == 'target']
            if pred_cols:
                edgenext_preds.rename(columns={pred_cols[0]: 'edgenext_pred'}, inplace=True)
    elif args.edgenext_dir:
        print("   Running EdgeNeXt inference...")
        edgenext_preds = run_vision_inference(
            args.edgenext_dir,
            SCRIPT_DIR / args.test_hdf5,
            384, args.batch_size, args.device
        )
        edgenext_preds.rename(columns={'pred': 'edgenext_pred'}, inplace=True)
    else:
        raise ValueError("❌ Must provide either --edgenext-preds or --edgenext-dir")
    
    print(f"   EVA02: {len(eva_preds):,} predictions, mean={eva_preds['eva02_pred'].mean():.6f}")
    print(f"   EdgeNeXt: {len(edgenext_preds):,} predictions, mean={edgenext_preds['edgenext_pred'].mean():.6f}")
    
    # 5. Merge predictions with metadata
    print("\n🔗 Merging predictions...")
    test_df = test_meta.merge(eva_preds[['isic_id', 'eva02_pred']], on='isic_id', how='left')
    test_df = test_df.merge(edgenext_preds[['isic_id', 'edgenext_pred']], on='isic_id', how='left')
    
    # Check for missing predictions
    eva_missing = test_df['eva02_pred'].isna().sum()
    edge_missing = test_df['edgenext_pred'].isna().sum()
    
    if eva_missing > 0 or edge_missing > 0:
        print(f"   ⚠️ Missing predictions: EVA02={eva_missing}, EdgeNeXt={edge_missing}")
        print(f"      Filling with mean...")
        test_df['eva02_pred'] = test_df['eva02_pred'].fillna(test_df['eva02_pred'].mean())
        test_df['edgenext_pred'] = test_df['edgenext_pred'].fillna(test_df['edgenext_pred'].mean())
    
    # 6. Apply Z-Score Standardization using TRAINING stats
    print("\n⚖️  Applying Z-Score Standardization (using TRAINING stats)...")
    
    for col in ['eva02_pred', 'edgenext_pred']:
        if col in standardization_stats:
            train_mean = standardization_stats[col]['mean']
            train_std = standardization_stats[col]['std']
            
            # Keep raw for debugging
            test_df[f'{col}_raw'] = test_df[col].copy()
            
            # Apply z-score using TRAINING statistics
            test_df[col] = (test_df[col] - train_mean) / (train_std + 1e-8)
            
            print(f"   {col}:")
            print(f"      Raw mean: {test_df[f'{col}_raw'].mean():.6f}")
            print(f"      Using training mean={train_mean:.6f}, std={train_std:.6f}")
            print(f"      Standardized range: [{test_df[col].min():.2f}, {test_df[col].max():.2f}]")
        else:
            print(f"   ⚠️ No training stats for {col}, skipping standardization")
    
    # 7. Feature Engineering
    print("\n🛠️  Engineering Features...")
    test_df, num_cols, cat_cols = preprocess_for_gbdt(test_df, label_encoders)
    
    # Ensure we have all required features
    missing_features = [f for f in features if f not in test_df.columns]
    if missing_features:
        print(f"   ⚠️ Missing features: {missing_features[:10]}...")
        for f in missing_features:
            test_df[f] = 0  # Fill with 0
    
    # 8. Run GBDT Inference
    print("\n🎯 Running GBDT Inference...")
    
    X_test = test_df[features]
    
    # LightGBM predictions
    lgbm_preds = np.zeros(len(test_df))
    for i, model in enumerate(lgbm_models):
        preds = model.predict_proba(X_test)[:, 1]
        lgbm_preds += preds / len(lgbm_models)
        print(f"   LightGBM fold {i+1}: mean={preds.mean():.4f}")
    
    print(f"   LightGBM ensemble: mean={lgbm_preds.mean():.4f}, std={lgbm_preds.std():.4f}")
    
    # XGBoost predictions (if available)
    if xgb_models:
        xgb_preds = np.zeros(len(test_df))
        for i, model in enumerate(xgb_models):
            preds = model.predict_proba(X_test)[:, 1]
            xgb_preds += preds / len(xgb_models)
            print(f"   XGBoost fold {i+1}: mean={preds.mean():.4f}")
        
        print(f"   XGBoost ensemble: mean={xgb_preds.mean():.4f}, std={xgb_preds.std():.4f}")
        
        # Ensemble LightGBM + XGBoost
        final_preds = (lgbm_preds + xgb_preds) / 2
    else:
        final_preds = lgbm_preds
    
    print(f"\n   Final predictions: mean={final_preds.mean():.4f}, std={final_preds.std():.4f}")
    print(f"   Prediction range: [{final_preds.min():.6f}, {final_preds.max():.6f}]")
    
    # 9. Create submission
    print("\n📝 Creating submission...")
    
    submission = pd.DataFrame({
        'isic_id': test_df['isic_id'],
        'target': final_preds
    })
    
    # Output to model directory's submissions subfolder
    submissions_dir = model_dir / 'submissions'
    submissions_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = submissions_dir / args.output
    else:
        output_path = submissions_dir / f'submission_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n✅ Submission saved to: {output_path}")
    print(f"   Shape: {submission.shape}")
    print(f"   Target mean: {submission['target'].mean():.6f}")
    print(f"   Target std: {submission['target'].std():.6f}")
    print(f"   Target range: [{submission['target'].min():.6f}, {submission['target'].max():.6f}]")
    
    # Also save debug info
    debug_df = test_df[['isic_id', 'eva02_pred', 'edgenext_pred', 
                        'eva02_pred_raw', 'edgenext_pred_raw']].copy()
    debug_df['lgbm_pred'] = lgbm_preds
    if xgb_models:
        debug_df['xgb_pred'] = xgb_preds
    debug_df['final_pred'] = final_preds
    
    debug_path = submissions_dir / f"{output_path.stem}_debug.csv"
    debug_df.to_csv(debug_path, index=False)
    print(f"   Debug info saved to: {debug_path}")
    
    # 10. Summary statistics
    print("\n" + "=" * 70)
    print("📊 SUBMISSION SUMMARY")
    print("=" * 70)
    print(f"\n   Samples: {len(submission):,}")
    print(f"\n   Prediction Distribution:")
    print(f"      Min:    {final_preds.min():.6f}")
    print(f"      25%:    {np.percentile(final_preds, 25):.6f}")
    print(f"      50%:    {np.percentile(final_preds, 50):.6f}")
    print(f"      75%:    {np.percentile(final_preds, 75):.6f}")
    print(f"      90%:    {np.percentile(final_preds, 90):.6f}")
    print(f"      95%:    {np.percentile(final_preds, 95):.6f}")
    print(f"      99%:    {np.percentile(final_preds, 99):.6f}")
    print(f"      Max:    {final_preds.max():.6f}")
    
    print("\n✅ Submission generation complete!")
    print("   Ready to upload to Kaggle.")


if __name__ == '__main__':
    main()
