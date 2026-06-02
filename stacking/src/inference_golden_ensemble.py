import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import argparse
from feature_engineering import engineer_features, NEW_FEATURES
import joblib
import json
from catboost import CatBoostClassifier

# Config
DATA_DIR = Path('./data')
LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
SUBMISSION_DIR = LAST_RUN_DIR / 'submissions'
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Base Features
SAFE_FEATURES = [
    'age_approx', 'sex', 'anatom_site_general',
    'clin_size_long_diam_mm', 'tbp_lv_nevi_confidence',
    'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext',
    'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext',
    'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean',
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
    'tbp_lv_eccentricity', 'tbp_lv_location', 'tbp_lv_location_simple',
    'tbp_lv_minorAxisMM', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
    'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max',
    'tbp_lv_stdL', 'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
    'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z'
]

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

def inference_golden_ensemble(args):
    print("--- Golden Split Grand Ensemble Inference ---")
    
    # 1. Load Test Data
    print("Loading Test Metadata...")
    df_test = pd.read_csv(DATA_DIR / 'students-test-metadata.csv', low_memory=False)
    
    # Apply Feature Engineering (for FE models)
    print("Applying Feature Engineering...")
    df_test_fe = engineer_features(df_test.copy())
    
    # Load DAE Latent Features
    print("Loading DAE Latent Features...")
    dae_latent = np.load(RESULTS_DIR / 'dae_latent_test.npy')
    
    # Load Vision Outputs (Probs + Embeddings)
    # We need to load fold-specific outputs?
    # For Golden Split, we typically use the models trained on Folds 0-3.
    # But for Test Inference, we need to process the Test Images.
    # We have `test_probs_{model}_fold{fold}.csv`.
    # Since we trained on Folds 0-3 (Golden), we should ideally use an ensemble of vision models from Folds 0-3?
    # Or just average all 5 folds of vision models?
    # The Stacking models (XGB/MLP) were trained using OOFs from Folds 0-3.
    # For Inference, we should use the average of all vision models to get the best signal.
    # Let's use the average of all 5 folds for vision inputs.
    
    def load_avg_vision(model_name):
        probs_list = []
        embs_list = []
        for fold in range(5):
            p = pd.read_csv(RESULTS_DIR / f"test_probs_{model_name}_fold{fold}.csv")['target'].values
            e = np.load(RESULTS_DIR / f"test_emb_{model_name}_fold{fold}.npy")
            probs_list.append(p)
            embs_list.append(e)
        return np.mean(probs_list, axis=0), np.mean(embs_list, axis=0)

    print("Loading Vision Outputs (Averaged)...")
    eva_p, eva_e = load_avg_vision('eva02_small_patch14_336.mim_in22k_ft_in1k')
    edge_p, edge_e = load_avg_vision('edgenext_base')
    
    # PCA Transform
    # We need to use the PCA fitted on the Golden Split.
    # We saved `golden_pca_eva_no_fe.pkl` and `golden_pca_eva_fe.pkl`.
    # They should be similar, let's use `no_fe` for consistency with XGB/MLP baseline.
    # Actually, XGB/MLP used `pca_eva_fold4.pkl` (fitted on Folds 0-3).
    # Let's use `pca_eva_fold4.pkl` for XGB/MLP and `golden_pca_eva_fe.pkl` for CatBoost FE.
    
    print("Loading PCAs...")
    pca_eva_base = joblib.load(RESULTS_DIR / 'pca_eva_fold4.pkl')
    pca_edge_base = joblib.load(RESULTS_DIR / 'pca_edge_fold4.pkl')
    
    pca_eva_fe = joblib.load(RESULTS_DIR / 'golden_pca_eva_fe.pkl')
    pca_edge_fe = joblib.load(RESULTS_DIR / 'golden_pca_edge_fe.pkl')
    
    # Transform Embeddings
    eva_e_pca_base = pca_eva_base.transform(eva_e)
    edge_e_pca_base = pca_edge_base.transform(edge_e)
    
    eva_e_pca_fe = pca_eva_fe.transform(eva_e)
    edge_e_pca_fe = pca_edge_fe.transform(edge_e)
    
    # Preprocessing Metadata
    # XGB/MLP used `stacking_preprocessor_fold4.pkl`
    print("Loading Preprocessors...")
    preprocessor = joblib.load(RESULTS_DIR / 'stacking_preprocessor_fold4.pkl')
    X_meta_base = preprocessor.transform(df_test_fe)
    
    # --- PREDICTIONS ---
    
    # 1. XGBoost (Fold 4 Model - No FE)
    print("Predicting XGBoost (No FE)...")
    X_xgb = np.hstack([
        X_meta_base, 
        eva_p.reshape(-1,1), edge_p.reshape(-1,1),
        eva_e_pca_base, edge_e_pca_base
    ]).astype(np.float32)
    
    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model(RESULTS_DIR / "xgb_fold4.json")
    pred_xgb = clf_xgb.predict_proba(X_xgb)[:, 1]
    
    # 2. MLP (Fold 4 Model - No FE)
    print("Predicting MLP (No FE)...")
    X_mlp = np.hstack([
        X_meta_base, dae_latent,
        eva_p.reshape(-1,1), edge_p.reshape(-1,1),
        eva_e_pca_base, edge_e_pca_base
    ]).astype(np.float32)
    
    mlp = SimpleMLP(X_mlp.shape[1]).to(DEVICE)
    mlp.load_state_dict(torch.load(RESULTS_DIR / "mlp_fold4.pth", map_location=DEVICE))
    mlp.eval()
    with torch.no_grad():
        X_mlp_t = torch.tensor(X_mlp, dtype=torch.float32).to(DEVICE)
        pred_mlp = mlp(X_mlp_t).sigmoid().cpu().numpy().flatten()
        
    # 3. CatBoost (Golden - No FE)
    print("Predicting CatBoost (No FE)...")
    # CatBoost needs raw dataframe with categoricals
    def make_cat_dataset(df_in, eva_p, edge_p, eva_pca, edge_pca, use_fe=False):
        cols = SAFE_FEATURES + (NEW_FEATURES if use_fe else [])
        X_meta = df_in[cols].copy()
        
        # Handle Categoricals
        cat_cols = ['sex', 'anatom_site_general', 'tbp_lv_location', 'tbp_lv_location_simple']
        if use_fe:
            cat_cols += ['age_group', 'size_cat']
            
        for col in cat_cols:
            if col in X_meta.columns:
                X_meta[col] = X_meta[col].astype(object).fillna('missing').astype(str).astype('category')
                
        # Vision Features
        vision_df = pd.DataFrame()
        vision_df['eva_prob'] = eva_p
        vision_df['edge_prob'] = edge_p
        for i in range(eva_pca.shape[1]):
            vision_df[f'eva_pca_{i}'] = eva_pca[:, i]
        for i in range(edge_pca.shape[1]):
            vision_df[f'edge_pca_{i}'] = edge_pca[:, i]
            
        return pd.concat([X_meta.reset_index(drop=True), vision_df], axis=1)

    X_cat_no_fe = make_cat_dataset(df_test, eva_p, edge_p, eva_e_pca_base, edge_e_pca_base, use_fe=False)
    
    clf_cat_no_fe = CatBoostClassifier()
    clf_cat_no_fe.load_model(RESULTS_DIR / "golden_catboost_no_fe.cbm")
    pred_cat_no_fe = clf_cat_no_fe.predict_proba(X_cat_no_fe)[:, 1]
    
    # 4. CatBoost (Golden - With FE)
    print("Predicting CatBoost (With FE)...")
    X_cat_fe = make_cat_dataset(df_test_fe, eva_p, edge_p, eva_e_pca_fe, edge_e_pca_fe, use_fe=True)
    
    clf_cat_fe = CatBoostClassifier()
    clf_cat_fe.load_model(RESULTS_DIR / "golden_catboost_fe.cbm")
    pred_cat_fe = clf_cat_fe.predict_proba(X_cat_fe)[:, 1]
    
    # --- ENSEMBLING ---
    # We have 4 predictions.
    # 1. XGB (No FE) - Strong Baseline
    # 2. MLP (No FE) - Strong Baseline
    # 3. Cat (No FE) - Good (0.960)
    # 4. Cat (FE) - Good (0.957)
    
    # Simple Average
    final_pred = (pred_xgb + pred_mlp + pred_cat_no_fe + pred_cat_fe) / 4
    
    print(f"\nPredictions Mean: {final_pred.mean():.4f}")
    
    # Save Submission
    sub_df = pd.DataFrame({
        'isic_id': df_test['isic_id'],
        'target': final_pred
    })
    out_path = SUBMISSION_DIR / 'submission_golden_grand_ensemble.csv'
    sub_df.to_csv(out_path, index=False)
    print(f"✅ Saved Grand Ensemble: {out_path}")
    
    # Save Individual Predictions for Analysis
    debug_df = pd.DataFrame({
        'isic_id': df_test['isic_id'],
        'xgb_no_fe': pred_xgb,
        'mlp_no_fe': pred_mlp,
        'cat_no_fe': pred_cat_no_fe,
        'cat_fe': pred_cat_fe,
        'grand_ensemble': final_pred
    })
    debug_df.to_csv(RESULTS_DIR / 'golden_predictions_debug.csv', index=False)
    print(f"Saved Debug Predictions: {RESULTS_DIR / 'golden_predictions_debug.csv'}")

if __name__ == "__main__":
    inference_golden_ensemble(None)
