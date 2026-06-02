import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from feature_engineering import engineer_features, NEW_FEATURES

# Config
DATA_DIR = Path('./data')
LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
FOLDS_PATH = LAST_RUN_DIR / 'data/folds.csv'

# Base Features (Safe)
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

def train_golden_split(args):
    print(f"--- Golden Split Training (FE={args.use_fe}) ---")
    
    # 1. Load Data & Folds
    print("Loading Metadata and Folds...")
    df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
    folds_df = pd.read_csv(FOLDS_PATH)
    
    # Merge Folds
    df = df.merge(folds_df[['isic_id', 'fold']], on='isic_id', how='left')
    
    # 2. Feature Engineering (Optional)
    features_to_use = SAFE_FEATURES.copy()
    if args.use_fe:
        print("Applying Feature Engineering...")
        df = engineer_features(df)
        features_to_use += NEW_FEATURES
        
    # 3. Preprocessing
    print("Preprocessing...")
    # Handle Categoricals
    cat_cols = ['sex', 'anatom_site_general', 'tbp_lv_location', 'tbp_lv_location_simple']
    if args.use_fe:
        cat_cols += ['age_group', 'size_cat'] # Add FE categoricals
        
    # Convert to category type for LGBM/CatBoost
    for col in cat_cols:
        if col in df.columns:
            # Fill NaNs with 'missing' to avoid CatBoost errors
            df[col] = df[col].astype(object).fillna('missing').astype(str).astype('category')
            
    # 4. Prepare Golden Split
    # Train: Folds 0, 1, 2, 3
    # Val: Fold 4
    train_idx = df['fold'].isin([0, 1, 2, 3])
    val_idx = df['fold'] == 4
    
    X_train_meta = df.loc[train_idx, features_to_use]
    y_train = df.loc[train_idx, 'target']
    X_val_meta = df.loc[val_idx, features_to_use]
    y_val = df.loc[val_idx, 'target']
    
    print(f"Train Shape: {X_train_meta.shape}, Val Shape: {X_val_meta.shape}")
    
    # 5. Load Vision Outputs (OOFs)
    print("Loading Vision OOFs...")
    # We need to load OOFs for ALL folds and concatenate them correctly
    
    def load_vision_data(model_name):
        # Load all OOFs
        probs_list = []
        emb_list = []
        ids_list = []
        
        for fold in range(5):
            p = pd.read_csv(RESULTS_DIR / f"oof_{model_name}_fold{fold}.csv")
            e = np.load(RESULTS_DIR / f"oof_emb_{model_name}_fold{fold}.npy")
            probs_list.append(p)
            emb_list.append(e)
            
        probs_all = pd.concat(probs_list, ignore_index=True)
        emb_all = np.vstack(emb_list)
        
        # We need to align this with 'df'. 
        # The 'df' is loaded from metadata, 'probs_all' is from OOFs.
        # They *should* be aligned if sorted by isic_id, but OOFs might be shuffled.
        # Safer to merge on isic_id.
        
        # Create a mapping dictionary
        prob_map = dict(zip(probs_all['isic_id'], probs_all['pred']))
        # For embeddings, it's harder to map directly in pandas. 
        # Let's assume OOF generation preserved order within folds, but let's be careful.
        # Actually, best way is to re-index.
        
        # Let's trust the 'df' order and map probabilities.
        # For embeddings, we need an index map.
        id_to_idx = {id_: i for i, id_ in enumerate(probs_all['isic_id'])}
        
        # Get indices for our train/val split
        train_isic_ids = df.loc[train_idx, 'isic_id'].values
        val_isic_ids = df.loc[val_idx, 'isic_id'].values
        
        train_indices = [id_to_idx[id_] for id_ in train_isic_ids if id_ in id_to_idx]
        val_indices = [id_to_idx[id_] for id_ in val_isic_ids if id_ in id_to_idx]
        
        # Extract Probs
        train_probs = np.array([prob_map[id_] for id_ in train_isic_ids if id_ in prob_map])
        val_probs = np.array([prob_map[id_] for id_ in val_isic_ids if id_ in prob_map])
        
        # Extract Embeddings
        train_emb = emb_all[train_indices]
        val_emb = emb_all[val_indices]
        
        return train_probs, train_emb, val_probs, val_emb

    print("  Loading EVA02...")
    eva_tr_p, eva_tr_e, eva_val_p, eva_val_e = load_vision_data('eva02_small_patch14_336.mim_in22k_ft_in1k')
    print("  Loading EdgeNeXt...")
    edge_tr_p, edge_tr_e, edge_val_p, edge_val_e = load_vision_data('edgenext_base')
    
    # 6. PCA on Embeddings
    print("Fitting PCA...")
    pca_eva = PCA(n_components=50, random_state=42)
    pca_edge = PCA(n_components=50, random_state=42)
    
    eva_tr_pca = pca_eva.fit_transform(eva_tr_e)
    eva_val_pca = pca_eva.transform(eva_val_e)
    
    edge_tr_pca = pca_edge.fit_transform(edge_tr_e)
    edge_val_pca = pca_edge.transform(edge_val_e)
    
    # Save PCAs
    suffix = "_fe" if args.use_fe else "_no_fe"
    joblib.dump(pca_eva, RESULTS_DIR / f'golden_pca_eva{suffix}.pkl')
    joblib.dump(pca_edge, RESULTS_DIR / f'golden_pca_edge{suffix}.pkl')
    
    # 7. Construct Final Feature Matrix
    # We need to concat metadata + vision probs + vision pca
    # Note: CatBoost/LGBM handle categoricals, so we pass X_meta directly (with categories)
    
    # Helper to concat
    def make_dataset(X_meta, eva_p, edge_p, eva_pca, edge_pca):
        # Reset index to align
        X_meta = X_meta.reset_index(drop=True)
        
        # Create DataFrame for Vision Features
        vision_df = pd.DataFrame()
        vision_df['eva_prob'] = eva_p
        vision_df['edge_prob'] = edge_p
        
        # Add PCA features
        for i in range(eva_pca.shape[1]):
            vision_df[f'eva_pca_{i}'] = eva_pca[:, i]
        for i in range(edge_pca.shape[1]):
            vision_df[f'edge_pca_{i}'] = edge_pca[:, i]
            
        # Concat
        return pd.concat([X_meta, vision_df], axis=1)

    X_train = make_dataset(X_train_meta, eva_tr_p, edge_tr_p, eva_tr_pca, edge_tr_pca)
    X_val = make_dataset(X_val_meta, eva_val_p, edge_val_p, eva_val_pca, edge_val_pca)
    
    print(f"Final Train Matrix: {X_train.shape}")
    
    # 8. Train CatBoost
    print("\nTraining CatBoost...")
    cat_features = [c for c in X_train.columns if X_train[c].dtype.name == 'category']
    print(f"Categorical Features: {cat_features}")
    
    clf_cat = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100,
        allow_writing_files=False
    )
    
    clf_cat.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True
    )
    
    cat_score = clf_cat.best_score_['validation']['AUC']
    print(f"CatBoost Best AUC: {cat_score:.5f}")
    clf_cat.save_model(RESULTS_DIR / f"golden_catboost{suffix}.cbm")
    
    # 9. Train LightGBM
    print("\nTraining LightGBM...")
    # LGBM needs simple callbacks
    # callbacks = [
    #    lgb.early_stopping(stopping_rounds=100),
    #    lgb.log_evaluation(period=100)
    # ]
    print("\nSkipping LightGBM (Stability Issues)...")
    lgb_score = 0.0
    # clf_lgb = lgb.LGBMClassifier(...)
    # clf_lgb.fit(...)
    # lgb_score = clf_lgb.best_score_['valid']['auc']
    # print(f"LightGBM Best AUC: {lgb_score:.5f}")
    # joblib.dump(clf_lgb, RESULTS_DIR / f"golden_lgbm{suffix}.pkl")
    
    # 10. Save Scores
    scores = {
        'catboost': cat_score,
        'lgbm': lgb_score,
        'use_fe': args.use_fe
    }
    with open(RESULTS_DIR / f"golden_scores{suffix}.json", 'w') as f:
        json.dump(scores, f, indent=4)
        
    print(f"\n✅ Golden Split Training Complete (FE={args.use_fe})")
    print(f"CatBoost: {cat_score:.5f}")
    print(f"LightGBM: {lgb_score:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fe', action='store_true', help='Enable Feature Engineering')
    args = parser.parse_args()
    
    train_golden_split(args)
