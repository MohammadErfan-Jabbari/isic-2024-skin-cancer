import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import argparse
from pathlib import Path
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
SCRIPT_DIR = Path(__file__).parent
EVA02_RESULTS_DIR = SCRIPT_DIR / 'results/gen-train-run-eva-v2'
EDGENEXT_RESULTS_DIR = SCRIPT_DIR / 'results/gen-train-run-edgenext-v2'
DATA_DIR = SCRIPT_DIR / 'data'

# Features to calculate patient-relative statistics for
RELATIVE_FEATURE_COLS = [
    'tbp_lv_areaMM2', 'tbp_lv_deltaB', 'clin_size_long_diam_mm',
    'tbp_lv_minorAxisMM', 'tbp_lv_eccentricity', 'tbp_lv_norm_color',
    'tbp_lv_radial_color_std_max', 'tbp_lv_color_std_mean',
    'eva02_pred', 'edgenext_pred' # Include vision predictions!
]

def score_pauc(y_true, y_pred, min_tpr=0.80):
    """Calculates pAUC above a minimum TPR threshold."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        mask = tpr >= min_tpr
        if mask.sum() < 2: return 0.0
        return auc(fpr[mask], tpr[mask])
    except:
        return 0.0

def load_oofs(dir_path, model_prefix):
    """Loads OOF predictions from a directory and returns a DataFrame."""
    path = Path(dir_path)
    oof_files = sorted(list(path.glob('oof_fold*.csv')))
    if not oof_files:
        print(f"⚠️ No OOF files found in {dir_path}")
        return None
        
    dfs = []
    for f in oof_files:
        df = pd.read_csv(f)
        # Clean prediction column
        if df['pred'].dtype == object:
            df['pred'] = df['pred'].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)
        dfs.append(df[['isic_id', 'pred']])
        
    all_oofs = pd.concat(dfs)
    all_oofs.rename(columns={'pred': f'{model_prefix}_pred'}, inplace=True)
    return all_oofs

def calculate_patient_relative_features(df):
    """
    Calculates Z-scores, Ratios, and Differences for features relative to the patient's portfolio.
    """
    print("    - Calculating Patient-Relative Statistics...")
    
    # Ensure we have the columns we want to process
    cols_to_process = [c for c in RELATIVE_FEATURE_COLS if c in df.columns]
    
    # Group by patient
    # We use transform to keep the original shape
    grouped = df.groupby('patient_id')[cols_to_process]
    
    means = grouped.transform('mean')
    stds = grouped.transform('std')
    mins = grouped.transform('min')
    maxs = grouped.transform('max')
    counts = df.groupby('patient_id')['isic_id'].transform('count')
    
    for col in cols_to_process:
        # 1. Ratio to Mean
        # Add epsilon to avoid division by zero
        df[f'{col}_ratio_mean'] = df[col] / (means[col] + 1e-6)
        
        # 2. Difference from Mean
        df[f'{col}_diff_mean'] = df[col] - means[col]
        
        # 3. Z-Score
        # Handle single-lesion patients (std=0 or NaN) -> Z-score = 0
        z_score = (df[col] - means[col]) / (stds[col] + 1e-6)
        df[f'{col}_zscore'] = z_score.fillna(0)
        
        # 4. Min/Max Ratios (How extreme is this lesion?)
        df[f'{col}_ratio_max'] = df[col] / (maxs[col] + 1e-6)
        df[f'{col}_ratio_min'] = df[col] / (mins[col] + 1e-6)

    # Add Patient Lesion Count
    df['patient_lesion_count'] = counts
    
    return df

def calculate_lof(df):
    """
    Calculates Local Outlier Factor (LOF) for each patient's lesions.
    Identifies 'Ugly Ducklings' within the patient's own context.
    """
    print("    - Calculating Local Outlier Factor (LOF)...")
    
    # Features to use for LOF (Shape, Color, Size)
    lof_features = [
        'tbp_lv_areaMM2', 'tbp_lv_deltaB', 'clin_size_long_diam_mm',
        'tbp_lv_eccentricity', 'tbp_lv_norm_color', 'tbp_lv_radial_color_std_max'
    ]
    # Filter to existing columns
    lof_features = [c for c in lof_features if c in df.columns]
    
    # Initialize result column
    df['patient_lof'] = np.nan
    
    # We can only calculate LOF for patients with enough samples (e.g., >= 5)
    # For efficiency, we iterate only over patients with enough data
    patient_counts = df['patient_id'].value_counts()
    valid_patients = patient_counts[patient_counts >= 5].index
    
    # Pre-fill missing values for LOF calculation
    df_filled = df.copy()
    for col in lof_features:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
    # Iterate (This can be slow, so we use tqdm)
    # Optimization: Group by patient and apply
    
    def get_lof(group):
        if len(group) < 5:
            return np.full(len(group), -1.0) # Default for small groups
        
        try:
            clf = LocalOutlierFactor(n_neighbors=min(len(group)-1, 20), novelty=False)
            # LOF returns negative values for outliers (lower is more outlier)
            # We want a score where higher = more outlier, so we flip sign or take inverse
            # Standard LOF: -1 is normal, large negative is outlier.
            # Let's use negative_outlier_factor_ directly.
            X = group[lof_features].values
            clf.fit_predict(X)
            return clf.negative_outlier_factor_
        except:
            return np.full(len(group), -1.0)

    # Apply per patient
    # Note: This loop is still the safest way to handle variable group sizes with sklearn
    # To speed up, we only process valid patients
    
    # Create a mapping
    lof_map = {}
    
    # Filter df to valid patients
    valid_df = df_filled[df_filled['patient_id'].isin(valid_patients)]
    
    for pid, group in tqdm(valid_df.groupby('patient_id'), desc="LOF Calculation"):
        lof_scores = get_lof(group)
        # Map isic_id to score
        for i, isic_id in enumerate(group['isic_id'].values):
            lof_map[isic_id] = lof_scores[i]
            
    # Map back to main DF
    df['patient_lof'] = df['isic_id'].map(lof_map).fillna(-1.0)
    
    return df

def engineer_features(df):
    """
    Master feature engineering function.
    """
    df = df.copy()
    
    # 1. Basic Metadata Features
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    
    # Shape/Color
    df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    df['color_variance'] = np.sqrt(
        df['tbp_lv_deltaB']**2 + df['tbp_lv_radial_color_std_max']**2 +
        df['tbp_lv_color_std_mean']**2
    )
    
    # 2. Patient-Relative Features (The "Ugly Duckling" Sign)
    df = calculate_patient_relative_features(df)
    
    # 3. Local Outlier Factor
    df = calculate_lof(df)
    
    # 4. Vision Ensemble Feature
    if 'eva02_pred' in df.columns and 'edgenext_pred' in df.columns:
        df['mean_vision_pred'] = (df['eva02_pred'] + df['edgenext_pred']) / 2
        # Add relative features for the mean prediction too
        # (We do this manually since calculate_patient_relative_features is already called)
        grouped = df.groupby('patient_id')['mean_vision_pred']
        means = grouped.transform('mean')
        stds = grouped.transform('std')
        df['mean_vision_pred_zscore'] = ((df['mean_vision_pred'] - means) / (stds + 1e-6)).fillna(0)
        
    return df

def preprocess_for_gbdt(df, is_train=True):
    df = engineer_features(df)
    
    # Select Features
    # We include all numerical columns except IDs, Targets, and LEAKY columns
    # mel_thick_mm: LEAKAGE CONFIRMED - 100% of non-missing are malignant (post-biopsy data)
    # mel_mitotic_index: 99.99% missing, likely post-biopsy data
    # iddx_* columns: diagnostic codes (post-diagnosis)
    exclude_cols = [
        'isic_id', 'patient_id', 'target', 'image_type', 'attribution', 'copyright_license',
        'mel_thick_mm', 'mel_mitotic_index',  # LEAKY: post-biopsy measurements
        'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5',  # Diagnostic codes
    ]
    
    # Identify numerical and categorical
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in exclude_cols]
    
    cat_cols = ['sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple']
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    # Handle Categorical
    for col in cat_cols:
        df[col] = df[col].astype('category')
        
    return df, num_cols, cat_cols

def analyze_and_log_data(df, features, target, save_dir):
    """
    Performs deep-dive analysis on the prepared data before training.
    """
    print("🔍 Running Pre-Training Data Analysis...")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Statistics
    print("    - Saving feature statistics...")
    stats = df[features].describe()
    stats.to_csv(save_dir / 'feature_stats.csv')

    # 2. Target Correlations
    print("    - Calculating correlations with target...")
    # Handle non-numeric columns gracefully
    numeric_df = df[features + [target]].select_dtypes(include=[np.number])
    corrs = numeric_df.corrwith(df[target]).sort_values(ascending=False)
    corrs.to_csv(save_dir / 'target_correlations.csv')
    
    # 3. Vision Model Correlation
    if 'eva02_pred' in df.columns and 'edgenext_pred' in df.columns:
        vision_corr = df['eva02_pred'].corr(df['edgenext_pred'])
        print(f"    - Vision Model Correlation: {vision_corr:.4f}")
        with open(save_dir / 'vision_correlation.txt', 'w') as f:
            f.write(f"Correlation between EVA02 and EdgeNeXt: {vision_corr}\n")

    # 4. Sample Data
    print("    - Saving sample processed data...")
    df.head(100).to_csv(save_dir / 'processed_data_sample.csv', index=False)
    
    # 5. Check for NaNs/Infs
    null_counts = df[features].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        print("    ⚠️ WARNING: The following columns have NaNs:")
        print(null_cols)
        null_cols.to_csv(save_dir / 'nan_columns.csv')
    else:
        print("    ✅ No NaNs found in features.")

def save_error_analysis(df, target_col, pred_col, save_dir):
    """
    Identifies and saves the worst predictions (False Positives and False Negatives).
    """
    print("🔍 Generating Error Analysis...")
    save_dir = Path(save_dir)
    
    # Top False Positives (Target=0, Pred High) - "False Alarm"
    fp = df[df[target_col] == 0].sort_values(by=pred_col, ascending=False).head(100)
    fp.to_csv(save_dir / 'top_100_false_positives.csv', index=False)
    
    # Top False Negatives (Target=1, Pred Low) - "Missed Cancer"
    fn = df[df[target_col] == 1].sort_values(by=pred_col, ascending=True).head(100)
    fn.to_csv(save_dir / 'top_100_false_negatives.csv', index=False)
    
    print("    - Saved top 100 False Positives and False Negatives.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (fast trial run)')
    return parser.parse_args()

def main():
    args = parse_args()
    print("🚀 Starting Advanced GBDT Stacking (Phase 3 & 4)...")
    if args.debug:
        print("🐞 DEBUG MODE ACTIVE: Using subset of data and fewer estimators.")
    
    # 1. Load Metadata
    print("📂 Loading Metadata...")
    train_meta = pd.read_csv(Path(DATA_DIR) / 'new-train-metadata.csv', low_memory=False)
    
    # 2. Load OOF Predictions
    print("🔮 Loading Vision OOF Predictions...")
    eva_oofs = load_oofs(EVA02_RESULTS_DIR, 'eva02')
    edgenext_oofs = load_oofs(EDGENEXT_RESULTS_DIR, 'edgenext')
    
    if eva_oofs is None or edgenext_oofs is None:
        print("❌ Missing OOFs. Cannot proceed.")
        return

    # 3. Merge Predictions
    print("🔗 Merging Predictions...")
    train_meta = train_meta.merge(eva_oofs, on='isic_id', how='left')
    train_meta = train_meta.merge(edgenext_oofs, on='isic_id', how='left')
    
    # Fill missing predictions (if any)
    train_meta['eva02_pred'] = train_meta['eva02_pred'].fillna(train_meta['eva02_pred'].mean())
    train_meta['edgenext_pred'] = train_meta['edgenext_pred'].fillna(train_meta['edgenext_pred'].mean())
    
    # 4. Z-Score Standardization (Following 1st Place Solution)
    # Use z-score instead of rank to preserve relative distances and handle distribution shift
    print("⚖️  Applying Z-Score Standardization...")
    
    standardization_stats = {}
    for col in ['eva02_pred', 'edgenext_pred']:
        mean = train_meta[col].mean()
        std = train_meta[col].std()
        train_meta[f'{col}_raw'] = train_meta[col].copy()  # Keep raw for analysis
        train_meta[col] = (train_meta[col] - mean) / (std + 1e-8)
        standardization_stats[col] = {'mean': mean, 'std': std}
        print(f"    {col}: mean={mean:.6f}, std={std:.6f}")
        print(f"    {col} (std): [{train_meta[col].quantile(0.01):.2f}, {train_meta[col].quantile(0.99):.2f}]")
    
    # 5. Feature Engineering
    print("🛠️  Engineering Features (Patient-Relative & LOF)...")
    train_df, num_cols, cat_cols = preprocess_for_gbdt(train_meta, is_train=True)
    
    features = num_cols + cat_cols
    target = 'target'
    print(f"✅ Total Features: {len(features)}")

    # --- NEW: ANALYSIS ---
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results' / 'stacking_final_v1'
    if args.debug:
        results_dir = script_dir / 'results' / 'stacking_debug'
    
    analyze_and_log_data(train_df, features, target, results_dir / 'analysis')
    # ---------------------
    
    # 6. Train LGBM with Noise Injection
    print("🏋️ Training LightGBM Stacker...")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train_df))
    models = []
    scores = []
    
    n_estimators = 100 if args.debug else 3000
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.005, # Slower learning for better generalization
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
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, train_df[target], train_df['patient_id'])):
        print(f"\n--- Fold {fold+1} ---")
        
        X_train = train_df.iloc[train_idx][features].copy()
        y_train = train_df.iloc[train_idx][target]
        X_val = train_df.iloc[val_idx][features].copy()
        y_val = train_df.iloc[val_idx][target]
        
        # --- NOISE INJECTION (CRITICAL) ---
        # Add Gaussian noise to vision predictions in TRAINING set only
        # This prevents the GBDT from overfitting to the vision scores
        noise_std = 0.1
        print(f"    💉 Injecting Gaussian Noise (sigma={noise_std}) to Vision Preds...")
        
        for col in ['eva02_pred', 'edgenext_pred', 'mean_vision_pred']:
            if col in X_train.columns:
                noise = np.random.normal(0, noise_std, size=len(X_train))
                X_train[col] = X_train[col] + noise
                # Clip to keep within reasonable bounds (though rank is 0-1, noise can push out)
                X_train[col] = np.clip(X_train[col], 0, 1)
                
        # Also inject noise to the Z-score features derived from vision
        z_cols = [c for c in X_train.columns if 'pred_zscore' in c]
        for col in z_cols:
             noise = np.random.normal(0, noise_std, size=len(X_train))
             X_train[col] = X_train[col] + noise
        
        # Train
        model = lgb.LGBMClassifier(**lgb_params)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=500)
        ]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=callbacks
        )
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        
        auc_score = roc_auc_score(y_val, val_pred)
        pauc_score = score_pauc(y_val, val_pred)
        
        print(f"    ✅ Fold {fold+1} AUC: {auc_score:.5f} | pAUC: {pauc_score:.5f}")
        scores.append(auc_score)
        models.append(model)
        
        # Save Model
        model_save_path = results_dir / 'models'
        model_save_path.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(model, model_save_path / f'lgbm_fold{fold+1}.joblib')
        
    # 7. Final Results
    overall_auc = roc_auc_score(train_df[target], oof_preds)
    overall_pauc = score_pauc(train_df[target], oof_preds)
    
    print("\n" + "="*30)
    print(f"🏁 FINAL STACKING RESULTS")
    print(f"Overall AUC:  {overall_auc:.5f}")
    print(f"Overall pAUC: {overall_pauc:.5f}")
    print(f"Avg Fold AUC: {np.mean(scores):.5f}")
    print("="*30)
    
    # 8. Save Results
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Standardization Stats (CRITICAL for inference)
    with open(results_dir / 'standardization_stats.pkl', 'wb') as f:
        pickle.dump(standardization_stats, f)
    print(f"💾 Saved standardization stats to {results_dir / 'standardization_stats.pkl'}")
    
    # Assign OOF predictions to dataframe
    train_df['stack_pred'] = oof_preds

    # Save OOF
    # Include vision preds and patient_id for error analysis
    cols_to_save = ['isic_id', 'patient_id', 'target', 'stack_pred']
    if 'eva02_pred' in train_df.columns: cols_to_save.append('eva02_pred')
    if 'edgenext_pred' in train_df.columns: cols_to_save.append('edgenext_pred')
    
    train_df[cols_to_save].to_csv(results_dir / 'stacking_oof.csv', index=False)
    
    # Save Feature Importance
    importances = pd.DataFrame()
    for i, model in enumerate(models):
        fold_imp = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_,
            'fold': i+1
        })
        importances = pd.concat([importances, fold_imp])
        
    avg_imp = importances.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
    avg_imp.to_csv(results_dir / 'feature_importance.csv', index=False)
    
    # 9. Error Analysis
    save_error_analysis(train_df, target, 'stack_pred', results_dir)

    print(f"\nSaved results to {results_dir}")
    print("Top 20 Features:")
    print(avg_imp.head(20))

if __name__ == '__main__':
    main()
