"""
Step 14.4: Submission Inference Script
Models: Eva02 Small + EdgeNeXt Base -> LightGBM Stacking
Target: Kaggle Competition Submission

This script:
1. Loads the test dataset (images + metadata).
2. Runs inference with 5 folds of Eva02 (TTA: None).
3. Runs inference with 5 folds of EdgeNeXt (TTA: None).
4. Aggregates predictions (Average).
5. Applies Rank Normalization.
6. Runs inference with 5 folds of LightGBM Stacker.
7. Generates submission.csv.

Usage:
    uv run python DeepLearning/Kaggle/14_4_submission_inference.py
"""

import pandas as pd
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import lightgbm as lgb
from pathlib import Path
from tqdm import tqdm
import pickle
import warnings
import gc

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===========================
# CONFIGURATION
# ===========================
TEST_IMAGE_PATH = 'DeepLearning/Kaggle/data/test-image-384.hdf5'
TEST_METADATA_PATH = 'DeepLearning/Kaggle/data/students-test-metadata.csv'
EVA02_DIR = 'DeepLearning/Kaggle/results/eva02_exp_v1'
EDGENEXT_DIR = 'DeepLearning/Kaggle/results/edgenext_exp_v1'
STACKER_DIR = 'DeepLearning/Kaggle/results/gbdt_stacking_v1/models'
OUTPUT_DIR = 'DeepLearning/Kaggle/results/gbdt_stacking_v1/submissions'

# ===========================
# FEATURE LISTS (MUST MATCH TRAINING)
# ===========================
NUMERICAL_FEATURES = [
    'tbp_lv_H', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM',
    'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_Hext',
    'clin_size_long_diam_mm', 'tbp_lv_radial_color_std_max',
    'tbp_lv_B', 'tbp_lv_color_std_mean', 'tbp_lv_Aext',
    'tbp_lv_stdLExt', 'tbp_lv_norm_color', 'tbp_lv_A', 'age_approx',
    'age_squared', 'lesion_size_mm', 'size_squared',
    'shape_regularity', 'eccentricity', 'compactness',
    'color_variance', 'color_uniformity', 'darkness_score', 'color_contrast',
    'site_risk_score', 'age_size_risk', 'age_site_risk', 'color_size_risk',
    'age_color_risk', 'site_size_risk', 'asymmetry_score',
    'log_area', 'log_perimeter', 'log_size',
    'h_to_b_ratio', 'a_to_b_ratio', 'area_to_perimeter'
]

CATEGORICAL_FEATURES = [
    'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple',
    'age_group', 'size_category', 'age_risk', 'large_lesion', 'high_risk_site'
]

# ===========================
# FEATURE ENGINEERING
# ===========================
def engineer_features(df):
    """Matches the training script feature engineering."""
    df = df.copy()
    
    # Handle missing values for critical columns if they exist in test data
    if 'clin_size_long_diam_mm' in df.columns:
        df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df.get('tbp_lv_minorAxisMM', 0))
    else:
        df['lesion_size_mm'] = df.get('tbp_lv_minorAxisMM', 0)

    # AGE FEATURES
    if 'age_approx' in df.columns:
        df['age_group'] = pd.cut(df['age_approx'], bins=[0, 30, 50, 70, 100],
                                 labels=['young', 'middle', 'senior', 'elderly'])
        df['age_risk'] = (df['age_approx'] > 50).astype(int)
        df['age_squared'] = df['age_approx'] ** 2
    
    # SIZE FEATURES
    df['size_category'] = pd.cut(df['lesion_size_mm'], bins=[0, 6, 10, 20, 100],
                                 labels=['small', 'medium', 'large', 'very_large'])
    df['large_lesion'] = (df['lesion_size_mm'] > 6).astype(int)
    df['size_squared'] = df['lesion_size_mm'] ** 2
    
    # SHAPE FEATURES
    if all(c in df.columns for c in ['tbp_lv_areaMM2', 'tbp_lv_perimeterMM', 'tbp_lv_minorAxisMM']):
        df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
        df['eccentricity'] = df['tbp_lv_minorAxisMM'] / (df['tbp_lv_areaMM2']**0.5 + 1e-6)
        df['compactness'] = (4 * np.pi * df['tbp_lv_areaMM2']) / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    
    # COLOR FEATURES
    if all(c in df.columns for c in ['tbp_lv_deltaB', 'tbp_lv_radial_color_std_max', 'tbp_lv_color_std_mean', 'tbp_lv_norm_color', 'tbp_lv_B', 'tbp_lv_H']):
        df['color_variance'] = np.sqrt(
            df['tbp_lv_deltaB']**2 + df['tbp_lv_radial_color_std_max']**2 +
            df['tbp_lv_color_std_mean']**2
        )
        df['color_uniformity'] = 1 / (df['tbp_lv_norm_color'] + 1e-6)
        df['darkness_score'] = df['tbp_lv_B'] / (df['tbp_lv_H'] + 1e-6)
        df['color_contrast'] = df['tbp_lv_deltaB'] * df['tbp_lv_radial_color_std_max']
        
    # ANATOMICAL FEATURES
    high_risk_sites = ['torso', 'upper extremity', 'posterior torso', 'anterior torso', 'head/neck']
    df['high_risk_site'] = df['anatom_site_general'].isin(high_risk_sites).astype(int)
    
    site_risk_map = {
        'head/neck': 4, 'torso': 3, 'posterior torso': 3, 'anterior torso': 3,
        'upper extremity': 2, 'lower extremity': 2,
        'palms/soles': 1, 'oral/genital': 1
    }
    df['site_risk_score'] = df['anatom_site_general'].map(site_risk_map).fillna(0)
    
    # INTERACTION FEATURES
    df['age_size_risk'] = df['age_approx'] * df['lesion_size_mm']
    df['age_site_risk'] = df['age_approx'] * df['site_risk_score']
    df['color_size_risk'] = df['color_variance'] * df['lesion_size_mm']
    df['age_color_risk'] = df['age_approx'] * df['color_variance']
    df['site_size_risk'] = df['site_risk_score'] * df['lesion_size_mm']
    
    # ASYMMETRY SCORE
    df['asymmetry_score'] = (
        df['tbp_lv_norm_color'] + df['tbp_lv_radial_color_std_max'] +
        (1 / (df['shape_regularity'] + 1e-6))
    ) / 3
    
    # LOG TRANSFORMS
    df['log_area'] = np.log1p(df['tbp_lv_areaMM2'])
    df['log_perimeter'] = np.log1p(df['tbp_lv_perimeterMM'])
    df['log_size'] = np.log1p(df['lesion_size_mm'])
    
    # RATIOS
    df['h_to_b_ratio'] = df['tbp_lv_H'] / (df['tbp_lv_B'] + 1e-6)
    df['a_to_b_ratio'] = df['tbp_lv_A'] / (df['tbp_lv_B'] + 1e-6)
    df['area_to_perimeter'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] + 1e-6)
        
    return df

# ===========================
# DATASET & MODELS
# ===========================
class HybridDataset(Dataset):
    def __init__(self, hdf5_path, metadata_df, transform=None, feature_cols=None):
        self.hdf5_path = hdf5_path
        self.metadata = metadata_df
        self.transform = transform
        self.feature_cols = feature_cols
        self.hdf5_file = None
        
        # Pre-extract features to numpy for speed
        if self.feature_cols:
            # Ensure columns exist
            missing = [c for c in self.feature_cols if c not in self.metadata.columns]
            if missing:
                print(f"Warning: Missing columns in metadata: {missing}")
                # Fill missing with 0
                for c in missing:
                    self.metadata[c] = 0
            self.meta_features = self.metadata[self.feature_cols].values.astype(np.float32)
        else:
            self.meta_features = np.zeros((len(self.metadata), 1), dtype=np.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
            
        row = self.metadata.iloc[idx]
        isic_id = row['isic_id']
        
        # Load Image
        try:
            img_data = self.hdf5_file[isic_id][()]
            image = Image.fromarray(img_data)
        except KeyError:
            # Fallback for missing images (should not happen in valid test set)
            image = Image.new('RGB', (384, 384), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        # Load Metadata
        meta = torch.tensor(self.meta_features[idx], dtype=torch.float32)
        
        return image, meta, isic_id

class GenericHybridModel(nn.Module):
    def __init__(self, model_name, metadata_dim, pretrained=False):
        super().__init__()
        
        # Create backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,
            global_pool='avg'
        )
        
        # Determine feature dimension dynamically
        if hasattr(self.backbone, 'num_features'):
            img_dim = self.backbone.num_features
        else:
            with torch.no_grad():
                dummy = torch.randn(2, 3, 224, 224)
                feats = self.backbone(dummy)
                img_dim = feats.shape[1]
        
        print(f"Model {model_name} feature dim: {img_dim}")
        
        # Metadata Head
        self.meta_net = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined Head
        self.head = nn.Sequential(
            nn.Linear(img_dim + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, img, meta):
        img_feat = self.backbone(img)
        meta_feat = self.meta_net(meta)
        combined = torch.cat([img_feat, meta_feat], dim=1)
        return self.head(combined)

# ===========================
# UTILS
# ===========================
def run_inference(model_name, model_dir, test_df, image_size, batch_size=64):
    """Runs inference for all 5 folds of a model type."""
    print(f"Running inference for {model_name} from {model_dir}...")
    model_dir = Path(model_dir)
    
    fold_preds = []
    
    for fold in range(1, 6):
        print(f"  Processing Fold {fold}...")
        
        # 1. Load Preprocessors
        try:
            with open(model_dir / f'scaler_fold{fold}.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open(model_dir / f'encoders_fold{fold}.pkl', 'rb') as f:
                encoders = pickle.load(f)
        except FileNotFoundError:
            print(f"    Warning: Preprocessors for fold {fold} not found. Skipping.")
            continue

        # 2. Preprocess Metadata
        df_fold = engineer_features(test_df)
        
        # Fill NaNs for Numericals
        for col in NUMERICAL_FEATURES:
            if col in df_fold.columns:
                df_fold[col] = df_fold[col].fillna(0) # Use 0 for test
        
        # Fill NaNs for Categoricals
        for col in CATEGORICAL_FEATURES:
            if col in df_fold.columns:
                df_fold[col] = df_fold[col].astype(str).fillna('missing')

        # Apply Scaler
        # We must ensure we pass exactly the columns the scaler expects.
        # The training script did: df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])
        # So we pass NUMERICAL_FEATURES
        try:
            df_fold[NUMERICAL_FEATURES] = scaler.transform(df_fold[NUMERICAL_FEATURES])
        except ValueError as e:
            print(f"    Scaler Error: {e}")
            # Try to fix if feature count mismatch (e.g. if scaler has fewer features)
            # But usually it should match if we use the same list.
            continue

        # Apply Encoders (One-Hot)
        # Training: pd.get_dummies(df[col], prefix=col)
        # We need to match the columns produced by get_dummies in training.
        # The training script saved `encoders` as a dict: {col: [list_of_columns]}
        encoded_dfs = []
        for col in CATEGORICAL_FEATURES:
            encoded = pd.get_dummies(df_fold[col], prefix=col, dtype=float)
            # Align with training columns
            if col in encoders:
                train_cols = encoders[col]
                for train_col in train_cols:
                    if train_col not in encoded.columns:
                        encoded[train_col] = 0
                encoded = encoded[train_cols] # Reorder and select
                encoded_dfs.append(encoded)
            else:
                # Should not happen if encoders dict is correct
                print(f"    Warning: No encoder found for {col}")
        
        # Concatenate Features
        # Result must be: [NUMERICAL_FEATURES] + [Encoded Categoricals]
        df_processed = pd.concat([df_fold[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
        
        # Add isic_id back for Dataset
        df_processed['isic_id'] = df_fold['isic_id']
        
        feature_cols = [c for c in df_processed.columns if c != 'isic_id']
        
        # 3. Load Model
        metadata_dim = len(feature_cols)
        model = GenericHybridModel(model_name, metadata_dim).to(device)
        
        checkpoint_path = model_dir / f'best_model_fold{fold}.pth'
        if not checkpoint_path.exists():
            print(f"    Checkpoint not found: {checkpoint_path}")
            continue
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"    Error loading state dict: {e}")
            continue
            
        model.eval()
        
        # 4. Dataset & Loader
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = HybridDataset(TEST_IMAGE_PATH, df_processed, transform=transform, feature_cols=feature_cols)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 5. Predict
        preds = []
        with torch.no_grad():
            for images, meta, _ in tqdm(loader, desc=f"    Infering Fold {fold}", leave=False):
                images, meta = images.to(device), meta.to(device)
                outputs = model(images, meta)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds.extend(probs)
        
        fold_preds.append(np.array(preds))
        
        # Cleanup
        del model, dataset, loader
        torch.cuda.empty_cache()
        gc.collect()
        
    if not fold_preds:
        return None
        
    # Average Folds
    avg_preds = np.mean(fold_preds, axis=0)
    return avg_preds

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. Load Test Metadata
    print("Loading Test Metadata...")
    test_df = pd.read_csv(TEST_METADATA_PATH)
    print(f"Test Samples: {len(test_df)}")
    
    # 2. Run Eva02 Inference
    print("\n=== Eva02 Inference ===")
    pred_eva02 = run_inference('eva02_small_patch14_336.mim_in22k_ft_in1k', EVA02_DIR, test_df, image_size=336, batch_size=32)
    
    # 3. Run EdgeNeXt Inference
    print("\n=== EdgeNeXt Inference ===")
    pred_edgenext = run_inference('edgenext_base.in21k_ft_in1k', EDGENEXT_DIR, test_df, image_size=384, batch_size=48)
    
    if pred_eva02 is None or pred_edgenext is None:
        print("Error: Failed to generate predictions for one or more models.")
        return

    # 4. Prepare Stacking Data
    print("\n=== Stacking Inference ===")
    stack_df = pd.DataFrame({
        'isic_id': test_df['isic_id'],
        'pred_eva02': pred_eva02,
        'pred_edgenext': pred_edgenext
    })
    
    # Rank Normalization (Critical!)
    stack_df['pred_eva02'] = stack_df['pred_eva02'].rank(pct=True)
    stack_df['pred_edgenext'] = stack_df['pred_edgenext'].rank(pct=True)
    
    feature_cols = ['pred_eva02', 'pred_edgenext']
    X_test = stack_df[feature_cols]
    
    # 5. Run Stacker Inference
    stacker_preds = []
    stacker_dir = Path(STACKER_DIR)
    
    for fold in range(1, 6):
        model_path = stacker_dir / f"gbdt_stacker_fold{fold}.txt"
        if not model_path.exists():
            print(f"  Stacker model {model_path} not found. Skipping.")
            continue
            
        model = lgb.Booster(model_file=str(model_path))
        preds = model.predict(X_test)
        stacker_preds.append(preds)
        
    if not stacker_preds:
        print("Error: No stacker models found.")
        return
        
    final_preds = np.mean(stacker_preds, axis=0)
    
    # 6. Save Submission
    submission = pd.DataFrame({
        'isic_id': test_df['isic_id'],
        'target': final_preds
    })
    
    sub_path = Path(OUTPUT_DIR) / 'submission.csv'
    submission.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to: {sub_path}")
    print(submission.head())

if __name__ == "__main__":
    main()
