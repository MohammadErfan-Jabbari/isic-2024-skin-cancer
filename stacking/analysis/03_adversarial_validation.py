import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Config
DATA_DIR = Path('./data')
RESULTS_DIR = Path('./last_run/analysis')

def run_adversarial_validation():
    print("Loading metadata...")
    train_df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv')
    test_df = pd.read_csv(DATA_DIR / 'students-test-metadata.csv')
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # 1. Prepare Data
    # Drop target and ID columns
    drop_cols = ['isic_id', 'patient_id', 'target', 'image_type', 'tbp_tile_type', 'attribution', 'copyright_license', 'lesion_id']
    # Drop leakage/missing columns identified earlier
    drop_cols.extend(['mel_thick_mm', 'mel_mitotic_index', 'tbp_lv_dnn_lesion_confidence'])
    drop_cols.extend([c for c in train_df.columns if c.startswith('iddx_')])
    
    # Align columns
    common_cols = [c for c in train_df.columns if c in test_df.columns and c not in drop_cols]
    print(f"\nUsing {len(common_cols)} common features for Adversarial Validation.")
    
    train_X = train_df[common_cols].copy()
    test_X = test_df[common_cols].copy()
    
    # Add Adversarial Target
    train_X['adv_target'] = 0
    test_X['adv_target'] = 1
    
    # Combine
    combined = pd.concat([train_X, test_X], axis=0).reset_index(drop=True)
    y = combined['adv_target'].values
    X = combined.drop(columns=['adv_target'])
    
    # Handle Categoricals (Label Encoding for XGBoost)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    # 2. Train Classifier (CV)
    print("\nTraining Adversarial Classifier (XGBoost)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    feature_importance = np.zeros(X.shape[1])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
        feature_importance += clf.feature_importances_
        
        auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"  Fold {fold+1} AUC: {auc:.4f}")
        
    total_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall Adversarial AUC: {total_auc:.4f}")
    
    # 3. Analyze Results
    if total_auc > 0.7:
        print("\nWARNING: Significant Covariate Shift Detected!")
        print("Top Drifting Features:")
        feature_importance /= 5
        imp_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
        imp_df = imp_df.sort_values('importance', ascending=False).head(10)
        print(imp_df)
    else:
        print("\nSUCCESS: Train and Test distributions are reasonably similar.")

if __name__ == "__main__":
    run_adversarial_validation()
