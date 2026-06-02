"""
Step 14.2: EdgeNeXt Visualization & Prediction Script
Model: edgenext_base.in21k_ft_in1k
Resolution: 384x384

This script:
1. Loads training results and visualizes performance.
2. Generates predictions on the test set using trained models.
3. Creates submission files.
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import json
import sys

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ===========================
# CONFIGURATION
# ===========================
MODEL_NAME = 'edgenext_base.in21k_ft_in1k'
IMAGE_SIZE = 384
RESULTS_DIR = Path('results/edgenext_exp_v1')
DATA_DIR = Path('data')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Results Directory: {RESULTS_DIR}")

if not RESULTS_DIR.exists():
    raise ValueError(f"Results directory not found: {RESULTS_DIR}")

# ===========================
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# ===========================
def engineer_features(df):
    """Enhanced feature engineering matching the advanced pipeline"""
    df = df.copy()
    
    # AGE FEATURES
    df['age_group'] = pd.cut(df['age_approx'], bins=[0, 30, 50, 70, 100],
                             labels=['young', 'middle', 'senior', 'elderly'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    df['age_squared'] = df['age_approx'] ** 2
    
    # SIZE FEATURES
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['size_category'] = pd.cut(df['lesion_size_mm'], bins=[0, 6, 10, 20, 100],
                                 labels=['small', 'medium', 'large', 'very_large'])
    df['large_lesion'] = (df['lesion_size_mm'] > 6).astype(int)
    df['size_squared'] = df['lesion_size_mm'] ** 2
    
    # SHAPE FEATURES
    df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    df['eccentricity'] = df['tbp_lv_minorAxisMM'] / (df['tbp_lv_areaMM2']**0.5 + 1e-6)
    df['compactness'] = (4 * np.pi * df['tbp_lv_areaMM2']) / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    
    # COLOR FEATURES
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

def preprocess_test_metadata(df, scaler, encoders):
    """Preprocess test data using training scaler/encoders"""
    
    df = engineer_features(df)
    
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
    
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing')
    
    df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])
    
    encoded_dfs = []
    for col in CATEGORICAL_FEATURES:
        encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
        for train_col in encoders[col]:
            if train_col not in encoded.columns:
                encoded[train_col] = 0
        encoded = encoded[encoders[col]]
        encoded_dfs.append(encoded)
    
    result_df = pd.concat([df[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
    return result_df

# ===========================
# DATASET & MODEL
# ===========================
class HybridDataset(Dataset):
    def __init__(self, hdf5_path, metadata_df, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hdf5_file = None
        
        with h5py.File(hdf5_path, 'r') as f:
            available_ids = set(f.keys())
        
        self.metadata = metadata_df[
            metadata_df['isic_id'].isin(available_ids)
        ].reset_index(drop=True)
        
        feature_cols = [col for col in self.metadata.columns if col not in ['isic_id', 'target', 'patient_id']]
        self.metadata_features = self.metadata[feature_cols].values.astype(np.float32)
    
    def _ensure_hdf5_open(self):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        self._ensure_hdf5_open()
        
        row = self.metadata.iloc[idx]
        image_id = row['isic_id']
        
        img_array = self.hdf5_file[image_id][:]
        image = Image.fromarray(img_array)
        
        if self.transform:
            image = self.transform(image)
        
        metadata = torch.tensor(self.metadata_features[idx], dtype=torch.float32)
        return image, metadata, image_id

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
        with torch.no_grad():
            dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
            feats = self.backbone(dummy)
            img_dim = feats.shape[1]
            
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
# MAIN EXECUTION
# ===========================
def main():
    print("="*70)
    print(f"VISUALIZATION & PREDICTION: {MODEL_NAME}")
    print("="*70 + "\n")
    
    # 1. Load Fold Results
    fold_results = {}
    for fold_num in range(1, 6):
        result_file = RESULTS_DIR / f'training_results_fold{fold_num}.pkl'
        if result_file.exists():
            with open(result_file, 'rb') as f:
                fold_results[fold_num] = pickle.load(f)
            print(f"✓ Loaded Fold {fold_num} results")
        else:
            print(f"⚠️ Fold {fold_num} results not found")
            
    if not fold_results:
        print("No results found! Exiting.")
        return

    # 2. Visualize Performance
    viz_dir = RESULTS_DIR / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Plot Training Curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, fold_num in enumerate(sorted(fold_results.keys())):
        history = fold_results[fold_num]
        epochs = range(1, len(history['train_auc']) + 1)
        
        ax = axes[idx]
        ax2 = ax.twinx()
        
        ax.plot(epochs, history['train_auc'], 'b-', label='Train AUC')
        ax.plot(epochs, history['val_auc'], 'r-', label='Val AUC')
        ax.plot(epochs, history['val_ema_auc'], 'g--', label='Val EMA AUC')
        
        ax2.plot(epochs, history['train_loss'], 'b:', alpha=0.3, label='Train Loss')
        ax2.plot(epochs, history['val_loss'], 'r:', alpha=0.3, label='Val Loss')
        
        ax.set_title(f'Fold {fold_num}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax2.set_ylabel('Loss')
        ax.legend(loc='lower right')
        
    plt.tight_layout()
    plt.savefig(viz_dir / 'training_curves.png')
    print(f"✓ Saved training curves to {viz_dir}")
    
    # 3. Load Test Data
    print("\nLoading Test Data...")
    test_meta = pd.read_csv(DATA_DIR / 'students-test-metadata.csv', low_memory=False)
    
    # Load Preprocessors from Fold 1 (assuming consistent across folds)
    with open(RESULTS_DIR / 'scaler_fold1.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(RESULTS_DIR / 'encoders_fold1.pkl', 'rb') as f:
        encoders = pickle.load(f)
        
    test_meta_processed = preprocess_test_metadata(test_meta, scaler, encoders)
    test_meta_processed['isic_id'] = test_meta['isic_id'].values
    
    metadata_dim = test_meta_processed.shape[1] - 1 # -1 for isic_id
    print(f"Metadata Dim: {metadata_dim}")
    
    # Transforms
    data_config = timm.data.resolve_data_config({}, model=MODEL_NAME)
    mean = data_config['mean']
    std = data_config['std']
    
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    test_dataset = HybridDataset(
        DATA_DIR / 'test-image-384.hdf5',
        test_meta_processed,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, 
        num_workers=8, pin_memory=True
    )
    
    # 4. Generate Predictions
    print("\nGenerating Predictions...")
    all_preds = []
    test_ids = []
    
    for fold_num in sorted(fold_results.keys()):
        print(f"Processing Fold {fold_num}...")
        
        # Load Best Model
        model_path = RESULTS_DIR / f'best_model_fold{fold_num}.pth'
        if not model_path.exists():
            print(f"  ⚠️ Model not found: {model_path}")
            continue
            
        model = GenericHybridModel(MODEL_NAME, metadata_dim, pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        fold_preds = []
        fold_ids = []
        
        with torch.no_grad():
            for images, metadata, ids in tqdm(test_loader, desc=f"Fold {fold_num}"):
                images = images.to(device)
                metadata = metadata.to(device)
                
                outputs = model(images, metadata)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                
                fold_preds.extend(probs)
                if fold_num == 1:
                    fold_ids.extend(ids)
        
        all_preds.append(fold_preds)
        
        # Save individual fold predictions
        fold_df = pd.DataFrame({'isic_id': fold_ids if fold_num==1 else test_ids, 'target': fold_preds})
        fold_df.to_csv(RESULTS_DIR / f'predictions_fold{fold_num}.csv', index=False)
        
        if fold_num == 1:
            test_ids = fold_ids

    # 5. Ensemble & Submit
    if all_preds:
        # Create submissions directory
        sub_dir = RESULTS_DIR / 'submissions'
        sub_dir.mkdir(exist_ok=True)
        
        # Convert to numpy array for easier manipulation
        pred_matrix = np.array(all_preds) # shape: (n_folds, n_samples)
        
        # Calculate weights based on validation AUC
        weights = []
        for fold_num in sorted(fold_results.keys()):
            # Get best AUC from history
            history = fold_results[fold_num]
            # Handle potential missing keys or different structures
            try:
                best_fold_auc = max(max(history.get('val_auc', [0])), max(history.get('val_ema_auc', [0])))
            except:
                best_fold_auc = 1.0 # Fallback
            weights.append(best_fold_auc)
        
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum() # Normalize
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        print(f"\nEnsemble Weights (based on Val AUC): {weights}")
        
        # Sanitize model name for filenames
        safe_model_name = MODEL_NAME.replace('.', '_').replace('-', '_')

        # Strategy 1: Mean
        mean_preds = np.mean(pred_matrix, axis=0)
        pd.DataFrame({'isic_id': test_ids, 'target': mean_preds}).to_csv(
            sub_dir / f'submission_{safe_model_name}_mean.csv', index=False
        )
        print(f"✓ Saved Mean ensemble: submission_{safe_model_name}_mean.csv")

        # Strategy 2: Median
        median_preds = np.median(pred_matrix, axis=0)
        pd.DataFrame({'isic_id': test_ids, 'target': median_preds}).to_csv(
            sub_dir / f'submission_{safe_model_name}_median.csv', index=False
        )
        print(f"✓ Saved Median ensemble: submission_{safe_model_name}_median.csv")
        
        # Strategy 3: Weighted Mean
        weighted_preds = np.average(pred_matrix, axis=0, weights=weights)
        pd.DataFrame({'isic_id': test_ids, 'target': weighted_preds}).to_csv(
            sub_dir / f'submission_{safe_model_name}_weighted.csv', index=False
        )
        print(f"✓ Saved Weighted ensemble: submission_{safe_model_name}_weighted.csv")

        # Strategy 4: Min (Conservative)
        min_preds = np.min(pred_matrix, axis=0)
        pd.DataFrame({'isic_id': test_ids, 'target': min_preds}).to_csv(
            sub_dir / f'submission_{safe_model_name}_min.csv', index=False
        )
        print(f"✓ Saved Min ensemble: submission_{safe_model_name}_min.csv")

        # Strategy 5: Max (Aggressive)
        max_preds = np.max(pred_matrix, axis=0)
        pd.DataFrame({'isic_id': test_ids, 'target': max_preds}).to_csv(
            sub_dir / f'submission_{safe_model_name}_max.csv', index=False
        )
        print(f"✓ Saved Max ensemble: submission_{safe_model_name}_max.csv")
        
        # Save raw probabilities for stacking
        prob_df = pd.DataFrame(pred_matrix.T, columns=[f'fold{i}' for i in sorted(fold_results.keys())])
        prob_df['isic_id'] = test_ids
        prob_df.to_csv(sub_dir / 'ensemble_probabilities.csv', index=False)
        print(f"✓ Saved ensemble probabilities to {sub_dir}")

if __name__ == '__main__':
    main()
