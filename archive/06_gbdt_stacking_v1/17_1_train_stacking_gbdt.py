#!/usr/bin/env python3
"""
17_1_train_stacking_gbdt.py - Comprehensive GBDT Stacking Model
================================================================

This script trains a GBDT stacking model using insights from:
1. Metadata investigation (6 phases)
2. Post-feature analysis (6 phases)  
3. 1st place solution techniques

Key Features:
- Multi-model ensemble (LightGBM + CatBoost + XGBoost)
- All leakage features excluded
- Z-score standardization (matching 1st place)
- Noise injection to prevent overfitting
- Patient-relative feature engineering
- Local Outlier Factor (LOF) features
- Comprehensive artifact saving for consistent inference

Author: Data Science Pipeline
Date: 2025-11-26
Version: 2.0 (Complete rewrite based on investigation findings)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
import argparse
from pathlib import Path
import pickle
import joblib
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / 'data'
EVA02_RESULTS_DIR = SCRIPT_DIR / 'results/gen-train-run-eva-v2'
EDGENEXT_RESULTS_DIR = SCRIPT_DIR / 'results/gen-train-run-edgenext-v2'

# ===========================
# FEATURE CONFIGURATION
# Based on metadata investigation and post-feature analysis
# ===========================

# LEAKAGE FEATURES - MUST EXCLUDE
# These contain post-diagnosis/post-biopsy information
LEAKAGE_FEATURES = [
    'mel_thick_mm',        # 100% malignant when present (51/51)
    'mel_mitotic_index',   # 100% malignant when present (43/43)
    'iddx_full',           # Diagnosis code (post-diagnosis)
    'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5',  # Diagnosis codes
]

# NON-INFORMATIVE FEATURES - EXCLUDE
NON_INFORMATIVE_FEATURES = [
    'isic_id', 'patient_id', 'target',
    'image_type', 'attribution', 'copyright_license',
    'tbp_lv_location',  # Redundant with tbp_lv_location_simple
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
# UTILITY FUNCTIONS
# ===========================

def score_pauc(y_true, y_pred, min_tpr=0.80):
    """Calculates partial AUC above a minimum TPR threshold."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        mask = tpr >= min_tpr
        if mask.sum() < 2:
            return 0.0
        return auc(fpr[mask], tpr[mask])
    except:
        return 0.0


def load_oofs(dir_path, model_prefix):
    """Loads OOF predictions from a directory."""
    path = Path(dir_path)
    oof_files = sorted(list(path.glob('oof_fold*.csv')))
    
    if not oof_files:
        print(f"⚠️ No OOF files found in {dir_path}")
        return None
        
    dfs = []
    for f in oof_files:
        df = pd.read_csv(f)
        if df['pred'].dtype == object:
            df['pred'] = df['pred'].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)
        dfs.append(df[['isic_id', 'pred']])
        
    all_oofs = pd.concat(dfs)
    all_oofs.rename(columns={'pred': f'{model_prefix}_pred'}, inplace=True)
    return all_oofs


# ===========================
# FEATURE ENGINEERING
# ===========================

def calculate_patient_relative_features(df):
    """
    Calculates Z-scores, Ratios, and Differences for features relative to patient's portfolio.
    This captures the "Ugly Duckling" sign - lesions that differ from patient's other lesions.
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
        # Ratio to patient mean
        df[f'{col}_ratio_mean'] = df[col] / (means[col] + 1e-6)
        
        # Difference from patient mean
        df[f'{col}_diff_mean'] = df[col] - means[col]
        
        # Z-Score within patient
        z_score = (df[col] - means[col]) / (stds[col] + 1e-6)
        df[f'{col}_zscore'] = z_score.fillna(0)
        
        # Min/Max ratios (how extreme is this lesion?)
        df[f'{col}_ratio_max'] = df[col] / (maxs[col] + 1e-6)
        df[f'{col}_ratio_min'] = df[col] / (mins[col] + 1e-6)

    df['patient_lesion_count'] = counts
    
    return df


def calculate_lof(df):
    """
    Calculates Local Outlier Factor (LOF) for each patient's lesions.
    Identifies outlier lesions within the patient's own context.
    """
    print("    - Calculating Local Outlier Factor (LOF)...")
    
    lof_features = [c for c in LOF_FEATURES if c in df.columns]
    df['patient_lof'] = np.nan
    
    patient_counts = df['patient_id'].value_counts()
    valid_patients = patient_counts[patient_counts >= 5].index
    
    # Pre-fill missing values
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
    """Master feature engineering function."""
    df = df.copy()
    
    # Basic derived features
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    
    # Shape regularity (circle has ratio of 1/(4*pi) ≈ 0.0796)
    df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    
    # Color variance (combined color features)
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


def preprocess_for_gbdt(df, label_encoders=None, is_train=True):
    """
    Preprocesses data for GBDT training/inference.
    
    Args:
        df: DataFrame with features
        label_encoders: Dict of LabelEncoders for categorical features (for inference)
        is_train: Whether this is training data
        
    Returns:
        df: Processed DataFrame
        num_cols: List of numerical column names
        cat_cols: List of categorical column names
        label_encoders: Dict of fitted LabelEncoders
    """
    df = engineer_features(df)
    
    # Build exclude list
    exclude_cols = LEAKAGE_FEATURES + NON_INFORMATIVE_FEATURES
    
    # Get numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in exclude_cols]
    
    # Get categorical columns
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    
    # Handle categorical encoding
    if label_encoders is None:
        label_encoders = {}
        
    for col in cat_cols:
        if col not in df.columns:
            continue
            
        if is_train:
            le = LabelEncoder()
            # Handle NaN by converting to string
            df[col] = df[col].fillna('MISSING').astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].fillna('MISSING').astype(str)
                # Handle unseen categories
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'MISSING')
                # Add MISSING to classes if not present
                if 'MISSING' not in le.classes_:
                    le.classes_ = np.append(le.classes_, 'MISSING')
                df[col] = le.transform(df[col])
    
    return df, num_cols, cat_cols, label_encoders


# ===========================
# MODEL TRAINING
# ===========================

def get_lgbm_params(n_estimators=3000):
    """LightGBM parameters based on 1st place solution."""
    return {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.005,
        'n_estimators': n_estimators,
        'num_leaves': 64,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }


def get_xgb_params(n_estimators=3000):
    """XGBoost parameters."""
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.005,
        'n_estimators': n_estimators,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'tree_method': 'hist'  # Fast histogram-based training
    }


def inject_noise(X, columns, noise_std=0.1):
    """
    Injects Gaussian noise to prevent overfitting.
    Following 1st place solution: noise_std=0.1
    """
    X = X.copy()
    for col in columns:
        if col in X.columns:
            noise = np.random.normal(0, noise_std, size=len(X))
            X[col] = X[col] + noise
    return X


def train_lgbm_fold(X_train, y_train, X_val, y_val, features, params, noise_cols, noise_std=0.1):
    """Train a single LightGBM fold."""
    # Inject noise to training data only
    X_train_noisy = inject_noise(X_train[features], noise_cols, noise_std)
    
    model = lgb.LGBMClassifier(**params)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=500)
    ]
    
    model.fit(
        X_train_noisy, y_train,
        eval_set=[(X_val[features], y_val)],
        callbacks=callbacks
    )
    
    val_pred = model.predict_proba(X_val[features])[:, 1]
    return model, val_pred


def train_xgb_fold(X_train, y_train, X_val, y_val, features, params, noise_cols, noise_std=0.1):
    """Train a single XGBoost fold."""
    # Inject noise to training data only
    X_train_noisy = inject_noise(X_train[features], noise_cols, noise_std)
    
    # Set early stopping in params for new XGBoost API
    params_with_es = params.copy()
    params_with_es['early_stopping_rounds'] = 150
    
    model = xgb.XGBClassifier(**params_with_es)
    
    model.fit(
        X_train_noisy, y_train,
        eval_set=[(X_val[features], y_val)],
        verbose=False
    )
    
    val_pred = model.predict_proba(X_val[features])[:, 1]
    return model, val_pred


# ===========================
# ANALYSIS & LOGGING
# ===========================

def save_training_config(save_dir, config):
    """Save training configuration for reproducibility."""
    with open(save_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)


def save_feature_importance(models, features, save_dir, prefix='lgbm'):
    """Save feature importance from trained models."""
    importances = pd.DataFrame()
    
    for i, model in enumerate(models):
        if hasattr(model, 'feature_importances_'):
            fold_imp = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_,
                'fold': i + 1
            })
            importances = pd.concat([importances, fold_imp])
    
    if len(importances) > 0:
        avg_imp = importances.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
        avg_imp.to_csv(save_dir / f'{prefix}_feature_importance.csv', index=False)
        return avg_imp
    return None


def save_error_analysis(df, target_col, pred_col, save_dir):
    """Identifies and saves the worst predictions."""
    print("📊 Generating Error Analysis...")
    
    # Top False Positives
    fp = df[df[target_col] == 0].nlargest(100, pred_col)
    fp.to_csv(save_dir / 'top_100_false_positives.csv', index=False)
    
    # Top False Negatives  
    fn = df[df[target_col] == 1].nsmallest(100, pred_col)
    fn.to_csv(save_dir / 'top_100_false_negatives.csv', index=False)


# ===========================
# MAIN TRAINING FUNCTION
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--no-xgb', action='store_true', help='Skip XGBoost training')
    parser.add_argument('--noise-std', type=float, default=0.1, help='Noise std for injection')
    parser.add_argument('--output-dir', type=str, default='stacking_v2', help='Output directory name')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 COMPREHENSIVE GBDT STACKING TRAINING")
    print("   Based on metadata investigation + post-feature analysis")
    print("=" * 70)
    
    if args.debug:
        print("🐞 DEBUG MODE: Using reduced settings")
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = SCRIPT_DIR / 'results' / f'{args.output_dir}_{timestamp}'
    if args.debug:
        results_dir = SCRIPT_DIR / 'results' / 'stacking_debug'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {results_dir}")
    
    # 1. Load Metadata
    print("\n📂 Loading Metadata...")
    train_meta = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
    print(f"   Loaded {len(train_meta):,} samples")
    
    # 2. Load Vision OOF Predictions
    print("\n🔮 Loading Vision OOF Predictions...")
    eva_oofs = load_oofs(EVA02_RESULTS_DIR, 'eva02')
    edgenext_oofs = load_oofs(EDGENEXT_RESULTS_DIR, 'edgenext')
    
    if eva_oofs is None or edgenext_oofs is None:
        print("❌ Missing OOFs. Cannot proceed.")
        return
    
    print(f"   EVA02: {len(eva_oofs):,} predictions")
    print(f"   EdgeNeXt: {len(edgenext_oofs):,} predictions")
    
    # 3. Merge Predictions
    print("\n🔗 Merging Predictions...")
    train_meta = train_meta.merge(eva_oofs, on='isic_id', how='left')
    train_meta = train_meta.merge(edgenext_oofs, on='isic_id', how='left')
    
    # Fill missing predictions with mean
    for col in ['eva02_pred', 'edgenext_pred']:
        missing = train_meta[col].isna().sum()
        if missing > 0:
            print(f"   ⚠️ {col}: {missing} missing, filling with mean")
            train_meta[col] = train_meta[col].fillna(train_meta[col].mean())
    
    # 4. Z-Score Standardization
    print("\n⚖️  Applying Z-Score Standardization...")
    standardization_stats = {}
    
    for col in ['eva02_pred', 'edgenext_pred']:
        mean = train_meta[col].mean()
        std = train_meta[col].std()
        
        # Keep raw predictions for analysis
        train_meta[f'{col}_raw'] = train_meta[col].copy()
        
        # Apply z-score
        train_meta[col] = (train_meta[col] - mean) / (std + 1e-8)
        
        standardization_stats[col] = {'mean': float(mean), 'std': float(std)}
        print(f"   {col}: mean={mean:.6f}, std={std:.6f}")
        print(f"   {col} standardized: [{train_meta[col].quantile(0.01):.2f}, {train_meta[col].quantile(0.99):.2f}]")
    
    # 5. Feature Engineering
    print("\n🛠️  Engineering Features...")
    train_df, num_cols, cat_cols, label_encoders = preprocess_for_gbdt(train_meta, is_train=True)
    
    features = num_cols + cat_cols
    target = 'target'
    
    print(f"   Total features: {len(features)}")
    print(f"   Numerical: {len(num_cols)}")
    print(f"   Categorical: {len(cat_cols)}")
    
    # Columns to inject noise into
    noise_cols = ['eva02_pred', 'edgenext_pred', 'mean_vision_pred', 
                  'eva02_pred_zscore', 'edgenext_pred_zscore', 'mean_vision_pred_zscore']
    noise_cols = [c for c in noise_cols if c in features]
    
    # 6. Save preprocessing artifacts
    print("\n💾 Saving preprocessing artifacts...")
    
    # Save standardization stats
    with open(results_dir / 'standardization_stats.pkl', 'wb') as f:
        pickle.dump(standardization_stats, f)
    
    # Save label encoders
    with open(results_dir / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save feature list
    with open(results_dir / 'feature_list.json', 'w') as f:
        json.dump({'features': features, 'num_cols': num_cols, 'cat_cols': cat_cols}, f, indent=2)
    
    # Save configuration
    config = {
        'timestamp': timestamp,
        'n_samples': len(train_df),
        'n_features': len(features),
        'noise_std': args.noise_std,
        'leakage_features_excluded': LEAKAGE_FEATURES,
        'categorical_features': cat_cols,
        'standardization_stats': standardization_stats,
        'debug_mode': args.debug
    }
    save_training_config(results_dir, config)
    
    # 7. Train Models
    print("\n🏋️ Training Models...")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    n_estimators = 100 if args.debug else 3000
    
    # LightGBM
    print("\n--- LightGBM Training ---")
    lgbm_params = get_lgbm_params(n_estimators)
    lgbm_models = []
    lgbm_oof_preds = np.zeros(len(train_df))
    lgbm_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, train_df[target], train_df['patient_id'])):
        print(f"\n  Fold {fold+1}/5")
        
        X_train = train_df.iloc[train_idx]
        y_train = train_df.iloc[train_idx][target]
        X_val = train_df.iloc[val_idx]
        y_val = train_df.iloc[val_idx][target]
        
        model, val_pred = train_lgbm_fold(
            X_train, y_train, X_val, y_val, 
            features, lgbm_params, noise_cols, args.noise_std
        )
        
        lgbm_oof_preds[val_idx] = val_pred
        lgbm_models.append(model)
        
        auc_score = roc_auc_score(y_val, val_pred)
        pauc_score = score_pauc(y_val, val_pred)
        lgbm_scores.append(auc_score)
        
        print(f"    ✅ AUC: {auc_score:.5f} | pAUC: {pauc_score:.5f}")
        
        # Save model
        model_dir = results_dir / 'models'
        model_dir.mkdir(exist_ok=True)
        joblib.dump(model, model_dir / f'lgbm_fold{fold+1}.joblib')
    
    # LightGBM results
    lgbm_overall_auc = roc_auc_score(train_df[target], lgbm_oof_preds)
    lgbm_overall_pauc = score_pauc(train_df[target], lgbm_oof_preds)
    print(f"\n  📊 LightGBM Overall - AUC: {lgbm_overall_auc:.5f} | pAUC: {lgbm_overall_pauc:.5f}")
    
    # XGBoost
    xgb_oof_preds = np.zeros(len(train_df))
    xgb_models = []
    xgb_scores = []
    
    if not args.no_xgb:
        print("\n--- XGBoost Training ---")
        xgb_params = get_xgb_params(n_estimators)
        
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, train_df[target], train_df['patient_id'])):
            print(f"\n  Fold {fold+1}/5")
            
            X_train = train_df.iloc[train_idx]
            y_train = train_df.iloc[train_idx][target]
            X_val = train_df.iloc[val_idx]
            y_val = train_df.iloc[val_idx][target]
            
            model, val_pred = train_xgb_fold(
                X_train, y_train, X_val, y_val,
                features, xgb_params, noise_cols, args.noise_std
            )
            
            xgb_oof_preds[val_idx] = val_pred
            xgb_models.append(model)
            
            auc_score = roc_auc_score(y_val, val_pred)
            pauc_score = score_pauc(y_val, val_pred)
            xgb_scores.append(auc_score)
            
            print(f"    ✅ AUC: {auc_score:.5f} | pAUC: {pauc_score:.5f}")
            
            # Save model
            joblib.dump(model, model_dir / f'xgb_fold{fold+1}.joblib')
        
        xgb_overall_auc = roc_auc_score(train_df[target], xgb_oof_preds)
        xgb_overall_pauc = score_pauc(train_df[target], xgb_oof_preds)
        print(f"\n  📊 XGBoost Overall - AUC: {xgb_overall_auc:.5f} | pAUC: {xgb_overall_pauc:.5f}")
    
    # 8. Ensemble predictions
    print("\n🎯 Creating Ensemble...")
    
    if args.no_xgb:
        ensemble_oof_preds = lgbm_oof_preds
    else:
        # Simple average of LightGBM and XGBoost
        ensemble_oof_preds = (lgbm_oof_preds + xgb_oof_preds) / 2
    
    ensemble_auc = roc_auc_score(train_df[target], ensemble_oof_preds)
    ensemble_pauc = score_pauc(train_df[target], ensemble_oof_preds)
    
    print(f"   📊 Ensemble - AUC: {ensemble_auc:.5f} | pAUC: {ensemble_pauc:.5f}")
    
    # 9. Save Results
    print("\n💾 Saving Results...")
    
    # Save OOF predictions
    train_df['lgbm_pred'] = lgbm_oof_preds
    if not args.no_xgb:
        train_df['xgb_pred'] = xgb_oof_preds
    train_df['ensemble_pred'] = ensemble_oof_preds
    
    oof_cols = ['isic_id', 'patient_id', 'target', 'lgbm_pred', 'ensemble_pred']
    if 'eva02_pred' in train_df.columns:
        oof_cols.extend(['eva02_pred', 'edgenext_pred'])
    if not args.no_xgb:
        oof_cols.append('xgb_pred')
    
    train_df[oof_cols].to_csv(results_dir / 'oof_predictions.csv', index=False)
    
    # Save feature importance
    lgbm_imp = save_feature_importance(lgbm_models, features, results_dir, 'lgbm')
    if not args.no_xgb:
        xgb_imp = save_feature_importance(xgb_models, features, results_dir, 'xgb')
    
    # Save error analysis
    save_error_analysis(train_df, target, 'ensemble_pred', results_dir)
    
    # Save final metrics
    final_metrics = {
        'lgbm': {
            'auc': float(lgbm_overall_auc),
            'pauc': float(lgbm_overall_pauc),
            'fold_scores': [float(s) for s in lgbm_scores]
        },
        'ensemble': {
            'auc': float(ensemble_auc),
            'pauc': float(ensemble_pauc)
        }
    }
    
    if not args.no_xgb:
        final_metrics['xgb'] = {
            'auc': float(xgb_overall_auc),
            'pauc': float(xgb_overall_pauc),
            'fold_scores': [float(s) for s in xgb_scores]
        }
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # 10. Final Summary
    print("\n" + "=" * 70)
    print("🏁 TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"   LightGBM  - AUC: {lgbm_overall_auc:.5f} | pAUC: {lgbm_overall_pauc:.5f}")
    if not args.no_xgb:
        print(f"   XGBoost   - AUC: {xgb_overall_auc:.5f} | pAUC: {xgb_overall_pauc:.5f}")
    print(f"   Ensemble  - AUC: {ensemble_auc:.5f} | pAUC: {ensemble_pauc:.5f}")
    
    print(f"\n📁 Results saved to: {results_dir}")
    print(f"\n📋 Top 20 Features (LightGBM):")
    if lgbm_imp is not None:
        print(lgbm_imp.head(20).to_string(index=False))
    
    print("\n✅ Ready for inference with 17_2_submission_stacking.py")


if __name__ == '__main__':
    main()
