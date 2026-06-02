import sys
import os
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
from pathlib import Path
import io
from PIL import Image
import joblib
from sklearn.neighbors import LocalOutlierFactor
import warnings

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
DATA_DIR = Path('./data')
RESULTS_DIR = Path('./results/stacking_final_v1')
EVA02_DIR = Path('./results/gen-train-run-eva-v2')
EDGENEXT_DIR = Path('./results/gen-train-run-edgenext-v2')

EVA02_MODEL_NAME = 'eva02_small_patch14_336.mim_in22k_ft_in1k'
EDGENEXT_MODEL_NAME = 'edgenext_base.in21k_ft_in1k'

BATCH_SIZE = 64
NUM_WORKERS = 4

# ===========================
# VISION MODEL COMPONENTS
# ===========================
class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.model(x)

class ISICTestDataset(Dataset):
    def __init__(self, hdf5_path, ids, transform=None):
        self.hdf5_path = hdf5_path
        self.ids = ids
        self.transform = transform
        self.hdf5_file = None

    def _ensure_open(self):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self._ensure_open()
        img_id = self.ids[idx]
        
        try:
            img_data = self.hdf5_file[img_id][:]
            if isinstance(img_data, np.ndarray) and img_data.ndim == 3:
                image = Image.fromarray(img_data)
            else:
                image = Image.open(io.BytesIO(img_data))
        except Exception as e:
            # Fallback for missing images
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, img_id

def run_vision_inference(model_name, image_size, model_dir, test_df, hdf5_path, device):
    print(f"🔮 Running Inference: {model_name} ({image_size}px)")
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ISICTestDataset(hdf5_path, test_df['isic_id'].values, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    ensemble_preds = np.zeros(len(test_df))
    models_found = 0
    
    for fold in range(1, 6):
        model_path = model_dir / f'best_model_fold{fold}.pth'
        if not model_path.exists():
            print(f"⚠️ Model not found: {model_path}")
            continue
            
        print(f"  - Fold {fold}...")
        model = ISICModel(model_name, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc=f"Fold {fold}", ncols=80):
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)
                
        ensemble_preds += np.array(fold_preds)
        models_found += 1
        
    if models_found > 0:
        return ensemble_preds / models_found
    else:
        raise RuntimeError(f"No models found in {model_dir}")

# ===========================
# FEATURE ENGINEERING (MATCHING 16_3)
# ===========================
def calculate_patient_relative_features(df):
    print("    - Calculating Patient-Relative Statistics...")
    RELATIVE_FEATURE_COLS = [
        'tbp_lv_areaMM2', 'tbp_lv_deltaB', 'clin_size_long_diam_mm',
        'tbp_lv_minorAxisMM', 'tbp_lv_eccentricity', 'tbp_lv_norm_color',
        'tbp_lv_radial_color_std_max', 'tbp_lv_color_std_mean',
        'eva02_pred', 'edgenext_pred'
    ]
    
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
    print("    - Calculating Local Outlier Factor (LOF)...")
    lof_features = [
        'tbp_lv_areaMM2', 'tbp_lv_deltaB', 'clin_size_long_diam_mm',
        'tbp_lv_eccentricity', 'tbp_lv_norm_color', 'tbp_lv_radial_color_std_max'
    ]
    lof_features = [c for c in lof_features if c in df.columns]
    
    df['patient_lof'] = np.nan
    patient_counts = df['patient_id'].value_counts()
    valid_patients = patient_counts[patient_counts >= 5].index
    
    df_filled = df.copy()
    for col in lof_features:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
    lof_map = {}
    valid_df = df_filled[df_filled['patient_id'].isin(valid_patients)]
    
    for pid, group in tqdm(valid_df.groupby('patient_id'), desc="LOF"):
        if len(group) < 5:
            scores = np.full(len(group), -1.0)
        else:
            try:
                clf = LocalOutlierFactor(n_neighbors=min(len(group)-1, 20), novelty=False)
                clf.fit_predict(group[lof_features].values)
                scores = clf.negative_outlier_factor_
            except:
                scores = np.full(len(group), -1.0)
                
        for i, isic_id in enumerate(group['isic_id'].values):
            lof_map[isic_id] = scores[i]
            
    df['patient_lof'] = df['isic_id'].map(lof_map).fillna(-1.0)
    return df

def engineer_features(df):
    df = df.copy()
    # Basic Metadata
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    df['color_variance'] = np.sqrt(
        df['tbp_lv_deltaB']**2 + df['tbp_lv_radial_color_std_max']**2 +
        df['tbp_lv_color_std_mean']**2
    )
    
    # Patient Relative
    df = calculate_patient_relative_features(df)
    
    # LOF
    df = calculate_lof(df)
    
    # Vision Ensemble
    if 'eva02_pred' in df.columns and 'edgenext_pred' in df.columns:
        df['mean_vision_pred'] = (df['eva02_pred'] + df['edgenext_pred']) / 2
        grouped = df.groupby('patient_id')['mean_vision_pred']
        means = grouped.transform('mean')
        stds = grouped.transform('std')
        df['mean_vision_pred_zscore'] = ((df['mean_vision_pred'] - means) / (stds + 1e-6)).fillna(0)
        
    return df

def preprocess_for_gbdt(df):
    df = engineer_features(df)
    
    exclude_cols = ['isic_id', 'patient_id', 'target', 'image_type', 'attribution', 'copyright_license']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in exclude_cols]
    
    cat_cols = ['sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple']
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    for col in cat_cols:
        df[col] = df[col].astype('category')
        
    return df, num_cols, cat_cols

# ===========================
# RANK NORMALIZATION UTILS
# ===========================
def load_reference_oofs(model_dir, model_prefix):
    """Loads OOF predictions to build a reference distribution for ranking."""
    print(f"    - Loading reference OOFs from {model_dir}...")
    path = Path(model_dir)
    oof_files = sorted(list(path.glob('oof_fold*.csv')))
    
    all_preds = []
    for f in oof_files:
        df = pd.read_csv(f)
        # Clean prediction column
        if df['pred'].dtype == object:
            df['pred'] = df['pred'].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)
        all_preds.extend(df['pred'].values)
        
    return np.sort(np.array(all_preds))

def get_reference_rank(val_preds, ref_preds_sorted):
    """
    Maps validation predictions to their rank in the reference distribution.
    Equivalent to: rank(pct=True) but against the training set.
    """
    # searchsorted returns the index where val_preds would be inserted
    # index / len = percentile
    ranks = np.searchsorted(ref_preds_sorted, val_preds) / len(ref_preds_sorted)
    return ranks

# ===========================
# MAIN
# ===========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip-vision', action='store_true', help='Skip vision inference if preds exist')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Submission Generation on {device}")
    
    # 1. Load Test Metadata
    test_csv = DATA_DIR / 'students-test-metadata.csv'
    # Fallback to absolute path if relative fails
    if not test_csv.exists():
        test_csv = Path('./data/students-test-metadata.csv')
        
    test_hdf5 = DATA_DIR / 'test-image.hdf5'
    
    if not test_csv.exists():
        print(f"❌ Test metadata not found: {test_csv}")
        return
        
    df = pd.read_csv(test_csv)
    print(f"📊 Loaded {len(df)} test samples")
    
    # 2. Vision Inference
    vision_preds_path = RESULTS_DIR / 'test_vision_preds.csv'
    
    if args.skip_vision and vision_preds_path.exists():
        print("⏩ Skipping Vision Inference (Loading cached predictions)...")
        vision_df = pd.read_csv(vision_preds_path)
        df['eva02_pred'] = vision_df['eva02_pred']
        df['edgenext_pred'] = vision_df['edgenext_pred']
    else:
        # EVA02
        df['eva02_pred'] = run_vision_inference(
            EVA02_MODEL_NAME, 336, EVA02_DIR, df, test_hdf5, device
        )
        
        # EdgeNeXt
        df['edgenext_pred'] = run_vision_inference(
            EDGENEXT_MODEL_NAME, 384, EDGENEXT_DIR, df, test_hdf5, device
        )
        
        # Cache predictions
        df[['isic_id', 'eva02_pred', 'edgenext_pred']].to_csv(vision_preds_path, index=False)
        print(f"✅ Saved vision predictions to {vision_preds_path}")

    # 3. Rank Normalization (Reference-Based)
    print("⚖️  Applying Reference-Based Rank Normalization...")
    
    # Load Reference Distributions (Training OOFs)
    eva_ref = load_reference_oofs(EVA02_DIR, 'eva02')
    edgenext_ref = load_reference_oofs(EDGENEXT_DIR, 'edgenext')
    
    # Apply Ranking
    df['eva02_pred'] = get_reference_rank(df['eva02_pred'].values, eva_ref)
    df['edgenext_pred'] = get_reference_rank(df['edgenext_pred'].values, edgenext_ref)
    
    print(f"    - EVA02 Rank Range: {df['eva02_pred'].min():.4f} - {df['eva02_pred'].max():.4f}")
    print(f"    - EdgeNeXt Rank Range: {df['edgenext_pred'].min():.4f} - {df['edgenext_pred'].max():.4f}")
    
    # 4. Feature Engineering
    print("🛠️  Engineering Features...")
    df_processed, num_cols, cat_cols = preprocess_for_gbdt(df)
    
    # --- FIX: Align features with Training Data ---
    # Load feature importance to get the exact list of 100 features
    fi_path = RESULTS_DIR / 'feature_importance.csv'
    if not fi_path.exists():
        print("❌ Feature importance file not found. Cannot align features.")
        return
        
    fi_df = pd.read_csv(fi_path)
    train_features = fi_df['feature'].tolist()
    print(f"📋 Aligning with {len(train_features)} training features...")
    
    # Add missing columns
    for col in train_features:
        if col not in df_processed.columns:
            print(f"    ⚠️ Missing column: {col} (Filling with NaN)")
            df_processed[col] = np.nan
            
    # Select and Reorder
    X_test = df_processed[train_features]
    
    # 5. Stacking Inference
    print("🏋️ Running Stacking Inference...")
    stack_preds = np.zeros(len(df))
    models_found = 0
    
    models_dir = RESULTS_DIR / 'models'
    for fold in range(1, 6):
        model_path = models_dir / f'lgbm_fold{fold}.joblib'
        if not model_path.exists():
            print(f"⚠️ Stacking model fold {fold} not found")
            continue
            
        print(f"  - Fold {fold}...")
        model = joblib.load(model_path)
        stack_preds += model.predict_proba(X_test)[:, 1]
        models_found += 1
        
    if models_found > 0:
        stack_preds /= models_found
    else:
        print("❌ No stacking models found!")
        return
        
    # --- NEW: Rank Normalize Final Output ---
    print("⚖️  Rank Normalizing Final Predictions...")
    oof_path = RESULTS_DIR / 'stacking_oof.csv'
    if oof_path.exists():
        stack_oofs_df = pd.read_csv(oof_path)
        # Ensure we use the correct column 'stack_pred'
        stack_ref = np.sort(stack_oofs_df['stack_pred'].values)
        
        # Apply Ranking
        stack_preds_ranked = get_reference_rank(stack_preds, stack_ref)
        
        print(f"    - Raw Range: {stack_preds.min():.4f} - {stack_preds.max():.4f}")
        print(f"    - Ranked Range: {stack_preds_ranked.min():.4f} - {stack_preds_ranked.max():.4f}")
        
        stack_preds = stack_preds_ranked
    else:
        print("⚠️ Stacking OOFs not found. Skipping final rank normalization.")
    # ----------------------------------------
        
    # 6. Save Submission
    submission_dir = RESULTS_DIR / 'submission_file'
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    submission = pd.DataFrame({
        'isic_id': df['isic_id'],
        'target': stack_preds
    })
    
    save_path = submission_dir / 'submission_stacking_v1.csv'
    submission.to_csv(save_path, index=False)
    
    print("\n" + "="*30)
    print(f"✅ Submission Generated: {save_path}")
    print(submission.head())
    print("="*30)

if __name__ == '__main__':
    main()