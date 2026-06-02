"""
Step 14.3: GBDT Stacking Ensemble
Models: Eva02 Small + EdgeNeXt Base
Method: LightGBM Stacking on OOF Predictions

This script:
1. Loads OOF predictions from the specified result directories.
2. Merges them into a single training set based on 'isic_id'.
3. Trains a LightGBM meta-learner using StratifiedGroupKFold.
4. Outputs the CV AUC of the ensemble.

Usage:
    uv run python DeepLearning/Kaggle/14_3_ensemble_stacking_gbdt.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import argparse
import warnings
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
# Paths to the experiments containing oof_predictions_fold*.csv
MODEL_DIRS = {
    'eva02': 'DeepLearning/Kaggle/results/eva02_exp_v1',
    'edgenext': 'DeepLearning/Kaggle/results/edgenext_exp_v1'
}

# Path to original metadata for GroupKFold (patient_id)
METADATA_PATH = 'DeepLearning/Kaggle/data/new-train-metadata.csv'

def score_pauc(y_true, y_pred, min_tpr=0.80):
    """Calculates pAUC above a minimum TPR threshold."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        mask = tpr >= min_tpr
        if mask.sum() < 2: return 0.0
        return auc(fpr[mask], tpr[mask])
    except:
        return 0.0

def load_and_merge_oofs(model_dirs):
    """
    Loads OOF predictions from multiple models and merges them into a single DataFrame.
    Ensures alignment by isic_id.
    """
    print("Loading and merging OOF predictions...")
    
    # 1. Load Metadata (we need patient_id for proper splitting later)
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")
    
    # Load only necessary metadata for splitting
    usecols = ['isic_id', 'patient_id', 'target']
    meta_df = pd.read_csv(METADATA_PATH, usecols=usecols)
    print(f"Loaded metadata: {len(meta_df)} rows")
    
    # 2. Load OOFs for each model
    for model_name, dir_path in model_dirs.items():
        dir_path = Path(dir_path)
        print(f"Processing {model_name} from {dir_path}...")
        
        model_oofs = []
        for fold in range(1, 6):
            oof_path = dir_path / f"oof_predictions_fold{fold}.csv"
            if not oof_path.exists():
                print(f"  Warning: {oof_path} not found! Skipping fold {fold}.")
                continue
            
            # Read OOF
            df = pd.read_csv(oof_path)
            
            # Keep only necessary columns to avoid conflicts
            if 'prediction' in df.columns:
                df = df[['isic_id', 'prediction']]
            else:
                # Fallback if column name is different
                pred_col = [c for c in df.columns if 'pred' in c][0]
                df = df[['isic_id', pred_col]]
                df = df.rename(columns={pred_col: 'prediction'})
            
            # Ensure prediction is numeric (handle string representation of list like "[0.123]")
            if df['prediction'].dtype == 'object':
                df['prediction'] = df['prediction'].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
            
            df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce')
            
            # Rank Normalization (Per Fold)
            # This fixes the distribution shift between folds
            df['prediction'] = df['prediction'].rank(pct=True)
                
            model_oofs.append(df)
        
        if not model_oofs:
            raise ValueError(f"No OOF files found for {model_name}")
            
        # Concatenate all folds for this model
        full_model_df = pd.concat(model_oofs).reset_index(drop=True)
        
        # Check for duplicates
        if full_model_df['isic_id'].duplicated().any():
            print(f"CRITICAL WARNING: Duplicate ISIC_IDs found in {model_name} OOFs!")
            # Keep first occurrence
            full_model_df = full_model_df.drop_duplicates(subset='isic_id', keep='first')
        
        # Rename prediction column
        full_model_df = full_model_df.rename(columns={'prediction': f'pred_{model_name}'})
        
        # Merge into main dataframe
        # We use LEFT JOIN on metadata to ensure we keep the original dataset structure
        meta_df = meta_df.merge(full_model_df, on='isic_id', how='left')
        
        # Check for missing predictions
        missing = meta_df[f'pred_{model_name}'].isna().sum()
        if missing > 0:
            print(f"  Warning: {missing} images are missing predictions for {model_name}.")
            # Fill missing with average (neutral) prediction or 0
            meta_df[f'pred_{model_name}'] = meta_df[f'pred_{model_name}'].fillna(meta_df[f'pred_{model_name}'].mean())

    return meta_df

def plot_results(df, feature_cols, target_col, output_dir):
    """Generates and saves visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROC Curves
    plt.figure(figsize=(10, 8))
    
    # Plot Base Models
    for col in feature_cols:
        if col.startswith('pred_'):
            fpr, tpr, _ = roc_curve(df[target_col], df[col])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{col} (AUC = {roc_auc:.4f})', alpha=0.7)
            
    # Plot Stacking
    if 'stacking_prediction' in df.columns:
        fpr, tpr, _ = roc_curve(df[target_col], df['stacking_prediction'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Stacking Ensemble (AUC = {roc_auc:.4f})', linewidth=3, color='black')
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'roc_comparison.png', dpi=300)
    plt.close()
    
    # 2. Prediction Distributions
    plt.figure(figsize=(12, 6))
    for col in feature_cols:
        if col.startswith('pred_'):
            sns.kdeplot(df[col], label=col, fill=True, alpha=0.3)
    
    if 'stacking_prediction' in df.columns:
        sns.kdeplot(df['stacking_prediction'], label='Stacking', fill=True, alpha=0.3, color='black')
        
    plt.title('Distribution of Predictions (Rank Normalized)')
    plt.xlabel('Predicted Rank/Probability')
    plt.legend()
    plt.savefig(output_dir / 'prediction_distributions.png', dpi=300)
    plt.close()
    
    # 3. Correlation Matrix
    pred_cols = [c for c in df.columns if c.startswith('pred_') or c == 'stacking_prediction']
    if pred_cols:
        plt.figure(figsize=(10, 8))
        corr = df[pred_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=0, vmax=1, fmt='.3f')
        plt.title('Prediction Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=300)
        plt.close()

def train_stacking(is_test=False):
    # 1. Prepare Data
    df = load_and_merge_oofs(MODEL_DIRS)
    target_col = 'target'
    
    if is_test:
        print("\n!!! TEST MODE: Using balanced subset (50 pos + 50 neg) and 10 estimators !!!")
        pos = df[df[target_col] == 1].head(50)
        neg = df[df[target_col] == 0].head(50)
        df = pd.concat([pos, neg]).reset_index(drop=True)
    
    # Drop rows where we might have failed to get predictions (if any critical errors)
    initial_len = len(df)
    df = df.dropna(subset=[f'pred_{name}' for name in MODEL_DIRS.keys()])
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows due to missing predictions.")
    
    print(f"Final Stacking Dataset: {len(df)} samples.")
    
    # Define Features (Predictions Only)
    feature_cols = [f'pred_{name}' for name in MODEL_DIRS.keys()]
    
    print(f"Input Features: {feature_cols}")
    
    # 2. Cross-Validation
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(df))
    models = []
    
    # LightGBM Parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 8,          # Small number of leaves
        'max_depth': 3,           # Shallow trees
        'min_child_samples': 100, # Prevent fitting to noise
        'subsample': 0.8,
        'colsample_bytree': 1.0,  # Use all features (since we only have 2)
        'n_jobs': -1,
        'verbosity': -1
    }
    
    print("\nStarting Stacking Training...")
    
    results_dir = Path("DeepLearning/Kaggle/results/gbdt_stacking_v1")
    models_dir = results_dir / "models"
    viz_dir = results_dir / "visualizations"
    sub_dir = results_dir / "submissions"
    
    for d in [models_dir, viz_dir, sub_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df[target_col], df['patient_id']), 1):
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_col]
        X_val = df.iloc[val_idx][feature_cols]
        y_val = df.iloc[val_idx][target_col]
        
        # Train
        n_estimators = 10 if is_test else 1000
        model = lgb.LGBMClassifier(**lgb_params, n_estimators=n_estimators)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predict
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        # Score
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_pauc = score_pauc(y_val, val_preds)
        print(f"Fold {fold} | AUC: {fold_auc:.5f} | pAUC: {fold_pauc:.5f}")
        fold_scores.append(fold_auc)
        
        models.append(model)
        
        # Save Stacker
        model.booster_.save_model(models_dir / f"gbdt_stacker_fold{fold}.txt")
        
    # 3. Overall Score
    total_auc = roc_auc_score(df[target_col], oof_preds)
    total_pauc = score_pauc(df[target_col], oof_preds)
    
    print(f"\nOverall Stacking AUC: {total_auc:.5f}")
    print(f"Overall Stacking pAUC: {total_pauc:.5f}")
    print(f"Average Fold AUC: {np.mean(fold_scores):.5f}")
    
    # 4. Save Results & Visualizations
    if not is_test:
        df['stacking_prediction'] = oof_preds
        
        # Save Submission CSV
        sub_name = f"oof_stacking_eva02_edgenext_gbdt_auc{total_auc:.4f}.csv"
        df[['isic_id', 'target', 'stacking_prediction'] + feature_cols].to_csv(
            sub_dir / sub_name, index=False
        )
        print(f"Saved OOF submission to: {sub_dir / sub_name}")
        
        # Generate Plots
        print("Generating visualizations...")
        plot_results(df, feature_cols, target_col, viz_dir)
    
    # 5. Comparison
    print("\n=== Model Comparison ===")
    for model_name in MODEL_DIRS.keys():
        auc_score = roc_auc_score(df[target_col], df[f'pred_{model_name}'])
        print(f"{model_name.ljust(10)}: {auc_score:.5f}")
    print(f"{'Stacking'.ljust(10)}: {total_auc:.5f}")
    
    print(f"\nResults saved to: {results_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in fast test mode')
    args = parser.parse_args()
    
    train_stacking(args.test)

if __name__ == "__main__":
    main()
