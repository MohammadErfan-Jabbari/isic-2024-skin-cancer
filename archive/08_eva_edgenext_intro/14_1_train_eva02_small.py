"""
Step 14.1: EVA-02 Small Training Script (384px)
Model: eva02_small_patch14_336.mim_in22k_ft_in1k
Dataset: train-image-384.hdf5 (Upscaled)

This script adapts the advanced training pipeline (SWA, EMA, Focal Loss) 
for the EVA-02 Vision Transformer.

Usage:
    python 14_1_train_eva02_small.py --fold 1 --gpu 0
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
import warnings
import json
import os
import argparse
from datetime import datetime
import pickle
import sys

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
MODEL_NAME = 'eva02_small_patch14_336.mim_in22k_ft_in1k'
IMAGE_SIZE = 336

def score_pauc(y_true, y_pred, min_tpr=0.80):
    """Calculates pAUC above a minimum TPR threshold."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        mask = tpr >= min_tpr
        if mask.sum() < 2: return 0.0
        return auc(fpr[mask], tpr[mask])
    except:
        return 0.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, help='Fold number (1-5)')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID (0-3)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (ViTs converge faster)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (Lower for ViT/384px)')
    parser.add_argument('--accumulation-steps', type=int, default=2, help='Gradient accumulation')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate (Lower for ViT)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name for grouping folds')
    return parser.parse_args()

# ===========================
# FEATURE ENGINEERING
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

def preprocess_metadata_with_features(df, is_train=True, scaler=None, encoders=None):
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
            df[col] = df[col].fillna(df[col].median() if is_train else 0)
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing')
    
    if is_train:
        scaler = StandardScaler()
        df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])
    else:
        df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])
    
    if is_train:
        encoders = {}
        encoded_dfs = []
        for col in CATEGORICAL_FEATURES:
            encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
            encoders[col] = encoded.columns.tolist()
            encoded_dfs.append(encoded)
        result_df = pd.concat([df[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
    else:
        encoded_dfs = []
        for col in CATEGORICAL_FEATURES:
            encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
            for train_col in encoders[col]:
                if train_col not in encoded.columns:
                    encoded[train_col] = 0
            encoded = encoded[encoders[col]]
            encoded_dfs.append(encoded)
        result_df = pd.concat([df[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
    
    return result_df, scaler, encoders

# ===========================
# DATASET CLASS
# ===========================
class HybridDataset(Dataset):
    def __init__(self, hdf5_path, metadata_df, transform=None, is_test=False):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.is_test = is_test
        self.hdf5_file = None
        
        with h5py.File(hdf5_path, 'r') as f:
            available_ids = set(f.keys())
        
        self.metadata = metadata_df[
            metadata_df['isic_id'].isin(available_ids)
        ].reset_index(drop=True)
        
        feature_cols = [col for col in self.metadata.columns 
                       if col not in ['isic_id', 'target', 'patient_id']]
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
        
        if self.is_test:
            return image, metadata, image_id
        else:
            label = row['target']
            return image, metadata, label, image_id

# ===========================
# MODEL ARCHITECTURE
# ===========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_term * bce_loss
        return focal_loss.mean()

class GenericHybridModel(nn.Module):
    def __init__(self, model_name, metadata_dim, pretrained=True):
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
# MODEL EMA
# ===========================
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ===========================
# TRAINING FUNCTIONS
# ===========================
def train_epoch(model, loader, criterion, optimizer, amp_scaler, device, 
                accumulation_steps=1, model_ema=None):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    optimizer.zero_grad()
    
    for batch_idx, (images, metadata, labels, _) in enumerate(tqdm(loader, desc="Training")):
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        amp_scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()
            if model_ema: model_ema.update()
        
        running_loss += loss.item() * accumulation_steps
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(loader), roc_auc_score(all_labels, all_preds)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_ids = [], [], []
    
    with torch.no_grad():
        for images, metadata, labels, ids in tqdm(loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(ids)
    
    return running_loss / len(loader), roc_auc_score(all_labels, all_preds), all_labels, all_preds, all_ids

# ===========================
# MAIN
# ===========================
def train_fold(fold_num, gpu_id, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training {MODEL_NAME} | Fold {fold_num} | GPU {gpu_id}")
    
    # Load Data
    data_dir = Path(args.data_dir)
    train_meta = pd.read_csv(data_dir / 'new-train-metadata.csv', low_memory=False)
    
    # Feature Engineering
    train_meta_processed, scaler, encoders = preprocess_metadata_with_features(train_meta, is_train=True)
    train_meta_processed['isic_id'] = train_meta['isic_id'].values
    train_meta_processed['target'] = train_meta['target'].values
    train_meta_processed['patient_id'] = train_meta['patient_id'].values
    
    metadata_dim = len(train_meta_processed.columns) - 3
    
    # Split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(sgkf.split(train_meta_processed, train_meta_processed['target'], train_meta_processed['patient_id']))
    train_idx, val_idx = fold_indices[fold_num - 1]
    
    # Transforms (Optimized for 384px)
    data_config = timm.data.resolve_data_config({}, model=MODEL_NAME)
    mean = data_config['mean']
    std = data_config['std']
    
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Datasets
    full_dataset = HybridDataset(data_dir / 'train-image-384.hdf5', train_meta_processed, transform=None)
    
    train_subset = Subset(full_dataset, train_idx)
    train_subset.dataset.transform = train_transform
    val_subset = Subset(full_dataset, val_idx)
    val_subset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    # Model Setup
    model = GenericHybridModel(MODEL_NAME, metadata_dim).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    amp_scaler = torch.cuda.amp.GradScaler()
    model_ema = ModelEMA(model)
    
    # SWA
    swa_model = optim.swa_utils.AveragedModel(model)
    swa_start = int(args.epochs * 0.7)
    
    # Results Setup
    if args.experiment_name:
        results_dir = Path(f"results/{args.experiment_name}")
        timestamp = args.experiment_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path(f"results/eva02_small_384_{timestamp}")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save Config
    config = vars(args)
    config['model_name'] = MODEL_NAME
    config['image_size'] = IMAGE_SIZE
    config['timestamp'] = timestamp
    config['libraries'] = {
        'torch': torch.__version__,
        'timm': timm.__version__,
        'numpy': np.__version__
    }
    with open(results_dir / f'config_fold{fold_num}.json', 'w') as f:
        json.dump(config, f, indent=4)

    # 2. Save Preprocessing
    with open(results_dir / f'scaler_fold{fold_num}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(results_dir / f'encoders_fold{fold_num}.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    best_auc = 0
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': [], 'val_pauc': [], 'val_ema_auc': [], 'val_ema_pauc': []}
    
    for epoch in range(args.epochs):
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, amp_scaler, device, args.accumulation_steps, model_ema)
        val_loss, val_auc, val_labels, val_preds, val_ids = validate(model, val_loader, criterion, device)
        val_pauc = score_pauc(val_labels, val_preds)
        
        model_ema.apply_shadow()
        _, val_ema_auc, val_ema_labels, val_ema_preds, val_ema_ids = validate(model, val_loader, criterion, device)
        val_ema_pauc = score_pauc(val_ema_labels, val_ema_preds)
        model_ema.restore()
        
        print(f"Epoch {epoch+1} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Val pAUC: {val_pauc:.4f} | EMA AUC: {val_ema_auc:.4f}")
        
        # Update History
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_pauc'].append(val_pauc)
        history['val_ema_auc'].append(val_ema_auc)
        history['val_ema_pauc'].append(val_ema_pauc)
        
        if val_ema_auc > best_auc:
            best_auc = val_ema_auc
            model_ema.apply_shadow()
            torch.save(model.state_dict(), results_dir / f"best_model_fold{fold_num}.pth")
            
            # 3. Save OOF & Worst Errors for Best Model
            oof_df = pd.DataFrame({
                'isic_id': val_ema_ids,
                'target': [x[0] for x in val_ema_labels],
                'prediction': val_ema_preds
            })
            # Add patient_id mapping
            oof_df = oof_df.merge(train_meta[['isic_id', 'patient_id']], on='isic_id', how='left')
            oof_df.to_csv(results_dir / f"oof_predictions_fold{fold_num}.csv", index=False)
            
            # Worst Errors
            oof_df['error'] = np.abs(oof_df['target'] - oof_df['prediction'])
            worst_errors = oof_df.sort_values('error', ascending=False).head(100)
            worst_errors.to_csv(results_dir / f"worst_errors_fold{fold_num}.csv", index=False)
            
            model_ema.restore()
            print(f"--> Saved Best EMA Model: {best_auc:.4f}")
            
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            
        scheduler.step()
        
    # Save SWA
    torch.save(swa_model.module.state_dict(), results_dir / f"swa_model_fold{fold_num}.pth")
    
    # 4. Save History
    with open(results_dir / f'training_results_fold{fold_num}.pkl', 'wb') as f:
        pickle.dump(history, f)
        
    print(f"Fold {fold_num} Finished. Best AUC: {best_auc:.4f}")

if __name__ == '__main__':
    args = parse_args()
    train_fold(args.fold, args.gpu, args)
