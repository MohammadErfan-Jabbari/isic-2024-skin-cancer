
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from sklearn.decomposition import PCA

# Config
DATA_DIR = Path('./data')
LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
FOLDS_PATH = LAST_RUN_DIR / 'data/folds.csv'
OUTPUT_DIR = Path('public/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add src to path
sys.path.append(str(LAST_RUN_DIR / 'src'))
try:
    from feature_engineering import engineer_features, NEW_FEATURES
except ImportError:
    print(f"Could not import feature_engineering from {LAST_RUN_DIR}/src")
    sys.exit(1)

# Feature List (Copied from train_stacking.py)
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

def main():
    print("Loading Data for SHAP Analysis (Fold 0 Validation Set)...")
    
    # 1. Load Metadata
    df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
    folds_df = pd.read_csv(FOLDS_PATH)
    if 'fold' not in df.columns:
        df = df.merge(folds_df[['isic_id', 'fold']], on='isic_id', how='left')
        
    # Filter for Fold 0 (Validation only)
    val_idx = df[df['fold'] == 0].index
    val_df = df.iloc[val_idx].copy()
    print(f"Validation Samples: {len(val_df)}")
    
    # 2. Feature Engineering
    print("Applying Feature Engineering...")
    val_df = engineer_features(val_df)
    
    # 3. Preprocessing (Metadata)
    print("Preprocessing Metadata...")
    preprocessor = joblib.load(RESULTS_DIR / 'stacking_preprocessor_fold0.pkl')
    X_meta = preprocessor.transform(val_df[SAFE_FEATURES])
    
    try:
        meta_feature_names = list(preprocessor.get_feature_names_out())
    except Exception as e:
        print(f"Error getting feature names: {e}")
        meta_feature_names = [f"Meta_{i}" for i in range(X_meta.shape[1])]
        
    print(f"Metadata Features: {len(meta_feature_names)}")
    print(f"X_meta Shape: {X_meta.shape}")

    # 4. Load & Prepare Vision Features
    print("Loading Vision OOFs for Fold 0...")
    # Load predictions
    eva_df = pd.read_csv(RESULTS_DIR / 'oof_eva02_small_patch14_336.mim_in22k_ft_in1k_fold0.csv')
    edge_df = pd.read_csv(RESULTS_DIR / 'oof_edgenext_base_fold0.csv')
    
    # Load embeddings
    eva_emb = np.load(RESULTS_DIR / 'oof_emb_eva02_small_patch14_336.mim_in22k_ft_in1k_fold0.npy')
    edge_emb = np.load(RESULTS_DIR / 'oof_emb_edgenext_base_fold0.npy')
    
    # Align by isic_id
    # Create map
    eva_map = {row['isic_id']: (row['pred'], eva_emb[i]) for i, row in eva_df.iterrows()}
    edge_map = {row['isic_id']: (row['pred'], edge_emb[i]) for i, row in edge_df.iterrows()}
    
    eva_preds = []
    edge_preds = []
    eva_embs_list = []
    edge_embs_list = []
    
    # Filter validation df to only those present in OOFs (sanity check)
    valid_ids_mask = val_df['isic_id'].isin(eva_map.keys()) & val_df['isic_id'].isin(edge_map.keys())
    val_df = val_df[valid_ids_mask]
    # Also filter X_meta
    X_meta = X_meta[valid_ids_mask.values]
    
    for iso in val_df['isic_id']:
        ep, ee = eva_map[iso]
        edp, ede = edge_map[iso]
        eva_preds.append(ep)
        edge_preds.append(edp)
        eva_embs_list.append(ee)
        edge_embs_list.append(ede)
        
    eva_preds = np.array(eva_preds).reshape(-1, 1)
    edge_preds = np.array(edge_preds).reshape(-1, 1)
    eva_embs = np.array(eva_embs_list)
    edge_embs = np.array(edge_embs_list)
    
    # PCA Transform
    print("Applying PCA to Embeddings...")
    pca_eva = joblib.load(RESULTS_DIR / 'pca_eva_fold0.pkl')
    pca_edge = joblib.load(RESULTS_DIR / 'pca_edge_fold0.pkl')
    
    eva_pca = pca_eva.transform(eva_embs)
    edge_pca = pca_edge.transform(edge_embs)
    
    print(f"EVA PCA Shape: {eva_pca.shape}")
    print(f"Edge PCA Shape: {edge_pca.shape}")
    
    # 5. Concatenate All Features
    X_full = np.hstack([X_meta, eva_preds, edge_preds, eva_pca, edge_pca])
    
    # Feature Names
    # Note: ColumnTransformer names might look like 'num__age' or 'cat__sex_male'
    # Clean them up for prettier plot
    clean_meta_names = [n.split('__')[-1] for n in meta_feature_names]
    
    feature_names = clean_meta_names + \
                    ['EVA02_Prob', 'EdgeNeXt_Prob'] + \
                    [f'EVA_PCA_{i}' for i in range(eva_pca.shape[1])] + \
                    [f'Edge_PCA_{i}' for i in range(edge_pca.shape[1])]
                    
    print(f"X_full Shape: {X_full.shape}")
    print(f"Total Feature Names: {len(feature_names)}")
    
    if len(feature_names) != X_full.shape[1]:
        print(f"⚠️ Mismatch! Generating {X_full.shape[1]} generic names.")
        feature_names = [f"Feature_{i}" for i in range(X_full.shape[1])]
    
    # 6. Load Model
    print("Loading XGBoost Model...")
    bst = xgb.Booster()
    bst.load_model(RESULTS_DIR / 'xgb_fold0.json')
    
    # 7. Run SHAP
    print("Running SHAP TreeExplainer...")
    # Sample down for speed if large
    if len(X_full) > 2000:
        indices = np.random.choice(len(X_full), 2000, replace=False)
        X_shap = X_full[indices]
    else:
        X_shap = X_full
        
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_shap)
    
    # 8. Plot Summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False, max_display=15)
    plt.title('Top 15 Features Driving Stacking Decisions (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {OUTPUT_DIR / 'shap_summary.png'}")

if __name__ == "__main__":
    main()
