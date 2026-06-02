import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import argparse
import joblib
import json
from feature_engineering import engineer_features, NEW_FEATURES

# Config
DATA_DIR = Path('./data')
LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
FOLDS_PATH = LAST_RUN_DIR / 'data/folds.csv'
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

def train_stacking(args):
    print("--- Step 3: Training Stacking Models (XGBoost & MLP) ---")
    
    # 1. Load Data & Folds
    print("Loading Metadata and Folds...")
    df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
    
    # Apply Feature Engineering
    print("Applying Feature Engineering...")
    df = engineer_features(df)
    
    folds_df = pd.read_csv(FOLDS_PATH)
    
    # Merge folds info if not present (assuming isic_id matches)
    if 'fold' not in df.columns:
        df = df.merge(folds_df[['isic_id', 'fold']], on='isic_id', how='left')
        
    # Load DAE Latent Features
    print("Loading DAE Latent Features...")
    dae_latent = np.load(RESULTS_DIR / 'dae_latent_train.npy')
    if len(dae_latent) != len(df):
        # This might happen if DAE was trained on Train+Test but we only have Train here
        # We need to slice the Train part.
        # Check if DAE was trained on concatenated [Train, Test]
        # In train_dae.py we did: X_all = pd.concat([X_train_raw, X_test_raw])
        # So the first len(df) rows are Train.
        print(f"  Slicing DAE Latent (Total: {len(dae_latent)}) to match Train ({len(df)})")
        dae_latent = dae_latent[:len(df)]
        
    # 2. Prepare Metadata
    print("Preparing Metadata...")
    X_meta_raw = df[SAFE_FEATURES].copy()
    y = df['target'].values
    folds = df['fold'].values
    
    # Identify Column Types
    num_cols = X_meta_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_meta_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 3. CV Loop
    oof_preds_xgb = np.zeros(len(df))
    oof_preds_mlp = np.zeros(len(df))
    
    for fold in range(5):
        print(f"\n=== Processing Fold {fold} ===")
        
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        if args.debug:
            print("!! DEBUG MODE: Using small subset !!")
            # Ensure we have positives
            y_train_full = y[train_idx]
            pos_idx = train_idx[y_train_full == 1]
            neg_idx = train_idx[y_train_full == 0]
            
            # Take all positives (usually few) and some negatives
            # If no positives in this fold's train (unlikely), take what we have
            n_pos = len(pos_idx)
            n_neg = 1000 - n_pos
            if n_neg < 0: n_neg = 0
            
            debug_train_idx = np.concatenate([pos_idx, neg_idx[:n_neg]])
            np.random.shuffle(debug_train_idx)
            train_idx = debug_train_idx
            
            # Same for val
            y_val_full = y[val_idx]
            pos_v_idx = val_idx[y_val_full == 1]
            neg_v_idx = val_idx[y_val_full == 0]
            n_pos_v = len(pos_v_idx)
            n_neg_v = 200 - n_pos_v
            if n_neg_v < 0: n_neg_v = 0
            
            debug_val_idx = np.concatenate([pos_v_idx, neg_v_idx[:n_neg_v]])
            np.random.shuffle(debug_val_idx)
            val_idx = debug_val_idx
            
        # A. Metadata Preprocessing (Per-Fold)
        # Fit on TRAIN, Transform VAL
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), num_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), cat_cols)
            ]
        )
        
        X_meta_train = preprocessor.fit_transform(X_meta_raw.iloc[train_idx])
        X_meta_val = preprocessor.transform(X_meta_raw.iloc[val_idx])
        
        # Save Preprocessor for Inference
        joblib.dump(preprocessor, RESULTS_DIR / f'stacking_preprocessor_fold{fold}.pkl')
        
        # B. Load Vision OOFs
        # We need to load OOFs for THIS fold.
        # Note: OOFs are predictions for the VAL set of this fold.
        # But for Stacking Training, we need:
        #   - Input for Stacking Train: Vision Predictions for the Stacking Train set (which were Val sets in other folds)
        #   - Input for Stacking Val: Vision Predictions for the Stacking Val set (which is Val set in this fold)
        # Actually, it's simpler: We have OOF predictions for the ENTIRE training set (accumulated from 5 folds).
        # Let's load the full OOFs first.
        
        # Helper to load full OOFs
        def load_full_oofs(model_name):
            # We have 5 files: oof_{model}_fold0.csv, etc.
            # We need to concatenate them to reconstruct the full dataset order?
            # Or better: Create a dataframe with isic_id and merge.
            dfs = []
            embs = []
            for f in range(5):
                d = pd.read_csv(RESULTS_DIR / f"oof_{model_name}_fold{f}.csv")
                e = np.load(RESULTS_DIR / f"oof_emb_{model_name}_fold{f}.npy")
                d['fold_origin'] = f
                # We need to keep track of embeddings. 
                # Let's add an index to d to map to e
                d['emb_idx'] = range(len(d))
                dfs.append((d, e))
                
            # We can't just concat because the order might be different from df.
            # But df has 'isic_id'.
            # Let's make a big dictionary or map.
            full_map = {} # isic_id -> (pred, emb)
            for d, e in dfs:
                for i, row in d.iterrows():
                    full_map[row['isic_id']] = (row['pred'], e[row['emb_idx']])
            
            # Now map to current df order
            preds = []
            ordered_embs = []
            for iso in df['isic_id']:
                if iso in full_map:
                    p, emb = full_map[iso]
                    preds.append(p)
                    ordered_embs.append(emb)
                else:
                    # This shouldn't happen if folds match
                    print(f"Warning: {iso} not found in OOFs!")
                    preds.append(0.5)
                    ordered_embs.append(np.zeros(e.shape[1]))
                    
            return np.array(preds), np.array(ordered_embs)

        # Load Vision Data (Cached or loaded once)
        if fold == 0: # Load once
            print("  Loading Vision OOFs...")
            # EVA02
            eva_preds, eva_embs = load_full_oofs('eva02_small_patch14_336.mim_in22k_ft_in1k')
            # EdgeNeXt
            edge_preds, edge_embs = load_full_oofs('edgenext_base')
            
            # Global variables to hold this
            global G_EVA_PREDS, G_EVA_EMBS, G_EDGE_PREDS, G_EDGE_EMBS
            G_EVA_PREDS, G_EVA_EMBS = eva_preds, eva_embs
            G_EDGE_PREDS, G_EDGE_EMBS = edge_preds, edge_embs
        else:
            eva_preds, eva_embs = G_EVA_PREDS, G_EVA_EMBS
            edge_preds, edge_embs = G_EDGE_PREDS, G_EDGE_EMBS
            
        # Slice for this fold
        eva_p_train, eva_p_val = eva_preds[train_idx], eva_preds[val_idx]
        edge_p_train, edge_p_val = edge_preds[train_idx], edge_preds[val_idx]
        
        eva_e_train, eva_e_val = eva_embs[train_idx], eva_embs[val_idx]
        edge_e_train, edge_e_val = edge_embs[train_idx], edge_embs[val_idx]
        
        # C. PCA on Embeddings (Per-Fold)
        print("  Fitting PCA on Embeddings...")
        pca_eva = PCA(n_components=50, random_state=42)
        eva_e_train_pca = pca_eva.fit_transform(eva_e_train)
        eva_e_val_pca = pca_eva.transform(eva_e_val)
        joblib.dump(pca_eva, RESULTS_DIR / f'pca_eva_fold{fold}.pkl')
        
        pca_edge = PCA(n_components=50, random_state=42)
        edge_e_train_pca = pca_edge.fit_transform(edge_e_train)
        edge_e_val_pca = pca_edge.transform(edge_e_val)
        joblib.dump(pca_edge, RESULTS_DIR / f'pca_edge_fold{fold}.pkl')
        
        # D. Construct Inputs
        # XGBoost Input: Raw Meta (Encoded) + Probs + PCA Embs
        # We use X_meta_train (which is encoded/scaled) for simplicity and consistency
        # XGBoost doesn't strictly need scaling but it doesn't hurt.
        X_xgb_train = np.hstack([
            X_meta_train, 
            eva_p_train.reshape(-1,1), edge_p_train.reshape(-1,1),
            eva_e_train_pca, edge_e_train_pca
        ])
        X_xgb_val = np.hstack([
            X_meta_val, 
            eva_p_val.reshape(-1,1), edge_p_val.reshape(-1,1),
            eva_e_val_pca, edge_e_val_pca
        ])
        
        # MLP Input: Scaled Meta + DAE Latent + Probs + PCA Embs
        dae_train, dae_val = dae_latent[train_idx], dae_latent[val_idx]
        X_mlp_train = np.hstack([
            X_meta_train, dae_train,
            eva_p_train.reshape(-1,1), edge_p_train.reshape(-1,1),
            eva_e_train_pca, edge_e_train_pca
        ])
        X_mlp_val = np.hstack([
            X_meta_val, dae_val,
            eva_p_val.reshape(-1,1), edge_p_val.reshape(-1,1),
            eva_e_val_pca, edge_e_val_pca
        ])
        
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Ensure float32
        X_xgb_train = X_xgb_train.astype(np.float32)
        X_xgb_val = X_xgb_val.astype(np.float32)
        
        # E. Train XGBoost
        print("  Training XGBoost...")
        clf_xgb = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=4
        )
        clf_xgb.fit(X_xgb_train, y_train, eval_set=[(X_xgb_val, y_val)], verbose=False)
        xgb_preds = clf_xgb.predict_proba(X_xgb_val)[:, 1]
        xgb_auc = roc_auc_score(y_val, xgb_preds)
        print(f"    XGBoost AUC: {xgb_auc:.5f}")
        
        clf_xgb.save_model(RESULTS_DIR / f"xgb_fold{fold}.json")
        oof_preds_xgb[val_idx] = xgb_preds
        
        # F. Train MLP
        print("  Training MLP...")
        X_mlp_train_t = torch.tensor(X_mlp_train, dtype=torch.float32).to(DEVICE)
        y_mlp_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        X_mlp_val_t = torch.tensor(X_mlp_val, dtype=torch.float32).to(DEVICE)
        
        mlp = SimpleMLP(X_mlp_train.shape[1]).to(DEVICE)
        opt = optim.Adam(mlp.parameters(), lr=1e-3)
        crit = nn.BCEWithLogitsLoss()
        
        ds = TensorDataset(X_mlp_train_t, y_mlp_train_t)
        dl = DataLoader(ds, batch_size=256, shuffle=True)
        
        best_mlp_auc = 0
        for ep in range(10): # Quick training for MLP
            mlp.train()
            for bx, by in dl:
                opt.zero_grad()
                out = mlp(bx)
                loss = crit(out, by)
                loss.backward()
                opt.step()
                
            # Val
            mlp.eval()
            with torch.no_grad():
                val_out = mlp(X_mlp_val_t).sigmoid().cpu().numpy().flatten()
            val_auc = roc_auc_score(y_val, val_out)
            
            if val_auc > best_mlp_auc:
                best_mlp_auc = val_auc
                torch.save(mlp.state_dict(), RESULTS_DIR / f"mlp_fold{fold}.pth")
                oof_preds_mlp[val_idx] = val_out
                
        print(f"    MLP AUC: {best_mlp_auc:.5f}")
        
    # Final Score
    total_auc_xgb = roc_auc_score(y, oof_preds_xgb)
    total_auc_mlp = roc_auc_score(y, oof_preds_mlp)
    ensemble_preds = (oof_preds_xgb + oof_preds_mlp) / 2
    total_auc_ens = roc_auc_score(y, ensemble_preds)
    
    print("\n=== Final CV Results ===")
    print(f"XGBoost CV: {total_auc_xgb:.5f}")
    print(f"MLP CV:     {total_auc_mlp:.5f}")
    print(f"Ensemble:   {total_auc_ens:.5f}")
    
    # Save OOFs
    oof_df = pd.DataFrame({
        'isic_id': df['isic_id'],
        'target': y,
        'xgb_pred': oof_preds_xgb,
        'mlp_pred': oof_preds_mlp,
        'ensemble_pred': ensemble_preds
    })
    oof_df.to_csv(RESULTS_DIR / 'oof_stacking.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    train_stacking(args)
