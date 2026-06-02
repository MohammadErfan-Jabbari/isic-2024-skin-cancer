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

# Config
DATA_DIR = Path('./data')
LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
SUBMISSION_DIR = LAST_RUN_DIR / 'submissions'
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature List (The 38 Safe Features)
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
] + NEW_FEATURES

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

def inference_stacking(args):
    print("--- Step 4: Final Inference (Stacking) ---")
    
    # 1. Load Test Data
    print("Loading Test Metadata...")
    df_test = pd.read_csv(DATA_DIR / 'students-test-metadata.csv', low_memory=False)
    
    # Apply Feature Engineering
    print("Applying Feature Engineering...")
    df_test = engineer_features(df_test)
    
    # Load DAE Latent
    print("Loading DAE Latent Features...")
    dae_latent = np.load(RESULTS_DIR / 'dae_latent_test.npy')
    
    # Load Vision Probs & Embeddings
    print("Loading Vision Outputs...")
    # Helper to load fold outputs
    def load_fold_vision(model_name, fold):
        probs = pd.read_csv(RESULTS_DIR / f"test_probs_{model_name}_fold{fold}.csv")['target'].values
        embs = np.load(RESULTS_DIR / f"test_emb_{model_name}_fold{fold}.npy")
        return probs, embs
        
    # 2. Loop Folds and Predict
    all_preds = []
    
    folds_to_run = args.folds if args.folds else [0, 1, 2, 3, 4]
    print(f"Running for Folds: {folds_to_run}")
    
    for fold in folds_to_run:
        print(f"\n=== Predicting Fold {fold} ===")
        
        # A. Load Preprocessors (Scaler/PCA)
        preprocessor = joblib.load(RESULTS_DIR / f'stacking_preprocessor_fold{fold}.pkl')
        pca_eva = joblib.load(RESULTS_DIR / f'pca_eva_fold{fold}.pkl')
        pca_edge = joblib.load(RESULTS_DIR / f'pca_edge_fold{fold}.pkl')
        
        # B. Transform Metadata
        X_meta = preprocessor.transform(df_test[SAFE_FEATURES])
        
        # C. Load Vision Outputs for this Fold
        # Note: We use the vision model trained on Fold X to predict Test
        eva_p, eva_e = load_fold_vision('eva02_small_patch14_336.mim_in22k_ft_in1k', fold)
        edge_p, edge_e = load_fold_vision('edgenext_base', fold)
        
        # D. Transform Embeddings
        eva_e_pca = pca_eva.transform(eva_e)
        edge_e_pca = pca_edge.transform(edge_e)
        
        # E. Construct Inputs
        # XGBoost Input
        X_xgb = np.hstack([
            X_meta, 
            eva_p.reshape(-1,1), edge_p.reshape(-1,1),
            eva_e_pca, edge_e_pca
        ]).astype(np.float32)
        
        # MLP Input
        X_mlp = np.hstack([
            X_meta, dae_latent,
            eva_p.reshape(-1,1), edge_p.reshape(-1,1),
            eva_e_pca, edge_e_pca
        ]).astype(np.float32)
        
        # F. Predict XGBoost
        pred_xgb = None
        if args.model_type in ['ensemble', 'xgb']:
            clf_xgb = xgb.XGBClassifier()
            clf_xgb.load_model(RESULTS_DIR / f"xgb_fold{fold}.json")
            pred_xgb = clf_xgb.predict_proba(X_xgb)[:, 1]
        
        # G. Predict MLP
        pred_mlp = None
        if args.model_type in ['ensemble', 'mlp']:
            mlp = SimpleMLP(X_mlp.shape[1]).to(DEVICE)
            mlp.load_state_dict(torch.load(RESULTS_DIR / f"mlp_fold{fold}.pth", map_location=DEVICE))
            mlp.eval()
            with torch.no_grad():
                X_mlp_t = torch.tensor(X_mlp, dtype=torch.float32).to(DEVICE)
                pred_mlp = mlp(X_mlp_t).sigmoid().cpu().numpy().flatten()
            
        # Ensemble for this fold
        if args.model_type == 'ensemble':
            pred_fold = (pred_xgb + pred_mlp) / 2
        elif args.model_type == 'xgb':
            pred_fold = pred_xgb
        elif args.model_type == 'mlp':
            pred_fold = pred_mlp
            
        all_preds.append(pred_fold)
        
        print(f"  Fold {fold} Predictions: Mean={pred_fold.mean():.4f}")
        
    # 3. Average Across Folds
    final_preds = np.mean(all_preds, axis=0)
    print(f"\nFinal Predictions: Mean={final_preds.mean():.4f}")
    
    # 4. Save Submission
    sub_df = pd.DataFrame({
        'isic_id': df_test['isic_id'],
        'target': final_preds
    })
    
    # Determine output filename
    if args.output:
        out_name = args.output
    else:
        folds_str = "".join(map(str, folds_to_run))
        out_name = f"submission_{args.model_type}_folds{folds_str}.csv"
        
    out_path = SUBMISSION_DIR / out_name
    sub_df.to_csv(out_path, index=False)
    print(f"✅ Saved Submission: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='ensemble', choices=['ensemble', 'xgb', 'mlp'])
    parser.add_argument('--folds', type=int, nargs='+', help='List of folds to use (e.g. 0 1 2)')
    parser.add_argument('--output', type=str, help='Custom output filename')
    args = parser.parse_args()
    
    inference_stacking(args)
