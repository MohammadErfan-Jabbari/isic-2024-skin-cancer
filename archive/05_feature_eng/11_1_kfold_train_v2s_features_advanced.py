"""
5-Fold Cross-Validation Training - Optimized for Your Dataset
EfficientNetV2-S + Feature Engineering + Proven Techniques

BASED ON YOUR BEST MODEL (0.96491) with these additions:
1. Stratified Group K-Fold - prevents patient leakage ✓
2. Model EMA - smooth weight averaging ✓
3. Stochastic Weight Averaging (SWA) - ensemble-in-a-model ✓
4. Mixed Precision Training - faster training ✓
5. Early stopping - prevents overfitting ✓

MATCHES YOUR BEST MODEL:
- Same weight decay (1e-5)
- Same dropout (0.5, 0.3)
- Same augmentation strength
- Same optimizer (Adam)
- Same focal loss (no label smoothing)

CONFIGURABLE:
- Can use ReduceLROnPlateau (safe, default) or CosineAnnealing (experimental)

Usage:
    # Train all 5 folds in parallel on 4 GPUs (SAFE settings)
    python 11_kfold_train_v2s_features_advanced.py --fold 1 --gpu 0 &
    python 11_kfold_train_v2s_features_advanced.py --fold 2 --gpu 1 &
    python 11_kfold_train_v2s_features_advanced.py --fold 3 --gpu 2 &
    python 11_kfold_train_v2s_features_advanced.py --fold 4 --gpu 3 &
    python 11_kfold_train_v2s_features_advanced.py --fold 5 --gpu 0
    
    # Or try experimental cosine scheduler
    python 11_kfold_train_v2s_features_advanced.py --fold 1 --gpu 0 --scheduler cosine
    
Expected: 0.970-0.978 AUC (ensemble of 5 folds)
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
from torchvision.models import efficientnet_v2_s
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import warnings
import json
import os
import argparse
from datetime import datetime
import pickle
from copy import deepcopy

warnings.filterwarnings('ignore')


# ===========================
# COMMAND LINE ARGUMENTS
# ===========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, help='Fold number (1-5)')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID (0-3)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--scheduler', type=str, default='plateau', 
                       choices=['plateau', 'cosine'],
                       help='LR scheduler: plateau (safe) or cosine (experimental)')
    parser.add_argument('--swa-start', type=int, default=20, help='Epoch to start SWA')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    return parser.parse_args()


# ===========================
# FEATURE ENGINEERING
# ===========================

def engineer_features(df):
    """Enhanced feature engineering with clinical domain knowledge"""
    df = df.copy()
    
    # AGE FEATURES
    df['age_group'] = pd.cut(df['age_approx'], bins=[0, 30, 50, 70, 100],
                             labels=['young', 'middle', 'senior', 'elderly'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    df['age_squared'] = df['age_approx'] ** 2  # Non-linear age effect
    
    # SIZE FEATURES (Diameter in ABCDE)
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['size_category'] = pd.cut(df['lesion_size_mm'], bins=[0, 6, 10, 20, 100],
                                 labels=['small', 'medium', 'large', 'very_large'])
    df['large_lesion'] = (df['lesion_size_mm'] > 6).astype(int)
    df['size_squared'] = df['lesion_size_mm'] ** 2  # Non-linear size effect
    
    # SHAPE FEATURES (Border irregularity in ABCDE)
    df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    df['eccentricity'] = df['tbp_lv_minorAxisMM'] / (df['tbp_lv_areaMM2']**0.5 + 1e-6)
    df['compactness'] = (4 * np.pi * df['tbp_lv_areaMM2']) / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    
    # COLOR FEATURES (Color variance in ABCDE)
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
    
    # INTERACTION FEATURES (combining risk factors)
    df['age_size_risk'] = df['age_approx'] * df['lesion_size_mm']
    df['age_site_risk'] = df['age_approx'] * df['site_risk_score']
    df['color_size_risk'] = df['color_variance'] * df['lesion_size_mm']
    df['age_color_risk'] = df['age_approx'] * df['color_variance']
    df['site_size_risk'] = df['site_risk_score'] * df['lesion_size_mm']
    
    # ASYMMETRY SCORE (Asymmetry in ABCDE)
    df['asymmetry_score'] = (
        df['tbp_lv_norm_color'] + df['tbp_lv_radial_color_std_max'] +
        (1 / (df['shape_regularity'] + 1e-6))
    ) / 3
    
    # LOG TRANSFORMS (handle skewness)
    df['log_area'] = np.log1p(df['tbp_lv_areaMM2'])
    df['log_perimeter'] = np.log1p(df['tbp_lv_perimeterMM'])
    df['log_size'] = np.log1p(df['lesion_size_mm'])
    
    # RATIOS
    df['h_to_b_ratio'] = df['tbp_lv_H'] / (df['tbp_lv_B'] + 1e-6)
    df['a_to_b_ratio'] = df['tbp_lv_A'] / (df['tbp_lv_B'] + 1e-6)
    df['area_to_perimeter'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] + 1e-6)
    
    return df


def preprocess_metadata_with_features(df, is_train=True, scaler=None, encoders=None):
    """Preprocess metadata with feature engineering"""
    
    df = engineer_features(df)
    
    NUMERICAL_FEATURES = [
        # Original features
        'tbp_lv_H', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM',
        'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_Hext',
        'clin_size_long_diam_mm', 'tbp_lv_radial_color_std_max',
        'tbp_lv_B', 'tbp_lv_color_std_mean', 'tbp_lv_Aext',
        'tbp_lv_stdLExt', 'tbp_lv_norm_color', 'tbp_lv_A', 'age_approx',
        # Engineered features
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
            return image, metadata, label


# ===========================
# MODEL ARCHITECTURE
# ===========================

class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """Binary Cross Entropy with Label Smoothing"""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # Apply label smoothing: 0 -> smoothing, 1 -> 1-smoothing
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, target)


class FocalLoss(nn.Module):
    """Focal Loss with optional Label Smoothing"""
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        # Apply label smoothing only if smoothing > 0
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_term * bce_loss
        return focal_loss.mean()


class MetadataProcessor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        return self.fc(x)


class EfficientNetV2Hybrid(nn.Module):
    def __init__(self, metadata_dim):
        super().__init__()
        
        self.efficientnet = efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.efficientnet.classifier = nn.Identity()
        
        total_params = len(list(self.efficientnet.parameters()))
        freeze_until = int(total_params * 0.8)
        
        for idx, param in enumerate(self.efficientnet.parameters()):
            if idx < freeze_until:
                param.requires_grad = False
        
        self.metadata_processor = MetadataProcessor(metadata_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Match best model (was 0.6)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Match best model (was 0.5)
            nn.Linear(128, 1)
        )
    
    def forward(self, image, metadata):
        img_features = self.efficientnet(image)
        meta_features = self.metadata_processor(metadata)
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(combined)


# ===========================
# MODEL EMA
# ===========================

class ModelEMA:
    """Exponential Moving Average of model weights"""
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
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, (images, metadata, labels) in enumerate(tqdm(loader, desc="Training")):
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        # Backward pass with gradient scaling
        amp_scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()
            
            # Update EMA
            if model_ema is not None:
                model_ema.update()
        
        running_loss += loss.item() * accumulation_steps
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_auc = roc_auc_score(all_labels, all_preds)
    
    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, metadata, labels in tqdm(loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_auc = roc_auc_score(all_labels, all_preds)
    
    return epoch_loss, epoch_auc


# ===========================
# MAIN TRAINING FUNCTION
# ===========================

def train_fold(fold_num, gpu_id, args):
    """Train a single fold with SOTA techniques"""
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"TRAINING FOLD {fold_num}/5 - ADVANCED")
    print(f"{'='*70}")
    print(f"GPU: {gpu_id}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Accumulation steps: {args.accumulation_steps}")
    print(f"Effective batch: {args.batch_size * args.accumulation_steps}")
    print(f"\nOPTIMIZATION SETTINGS (based on your best model):")
    print(f"  ✓ Focal Loss (alpha=0.25, gamma=2.0, NO label smoothing)")
    print(f"  ✓ Adam optimizer (lr=0.0005, weight_decay=1e-5)")
    print(f"  ✓ {args.scheduler.capitalize()} scheduler")
    print(f"  ✓ Dropout: [0.5, 0.3] (proven to work)")
    print(f"\nNEW K-FOLD ENHANCEMENTS:")
    print(f"  ✓ Stratified Group K-Fold (prevents patient leakage)")
    print(f"  ✓ Model EMA (decay=0.9999)")
    print(f"  ✓ Stochastic Weight Averaging (starts epoch {args.swa_start})")
    print(f"  ✓ Mixed Precision Training")
    print(f"  ✓ Early Stopping (patience=8)")
    print("="*70 + "\n")
    
    # Load data
    data_dir = Path(args.data_dir)
    train_meta = pd.read_csv(data_dir / 'new-train-metadata.csv', low_memory=False)
    
    # Preprocess with feature engineering
    print("Engineering features...")
    train_meta_processed, scaler, encoders = preprocess_metadata_with_features(
        train_meta, is_train=True
    )
    train_meta_processed['isic_id'] = train_meta['isic_id'].values
    train_meta_processed['target'] = train_meta['target'].values
    train_meta_processed['patient_id'] = train_meta['patient_id'].values
    
    metadata_dim = len(train_meta_processed.columns) - 3  # -3 for isic_id, target, patient_id
    print(f"✓ Metadata dimension: {metadata_dim}\n")
    
    # Stratified Group K-Fold (prevents patient leakage)
    print("Creating fold splits with Stratified Group K-Fold...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_indices = list(sgkf.split(
        train_meta_processed,
        train_meta_processed['target'],
        train_meta_processed['patient_id']
    ))
    
    train_idx, val_idx = fold_indices[fold_num - 1]
    
    print(f"Fold {fold_num}:")
    print(f"  Train: {len(train_idx):,} samples ({train_meta_processed.iloc[train_idx]['target'].sum()} positive)")
    print(f"  Val:   {len(val_idx):,} samples ({train_meta_processed.iloc[val_idx]['target'].sum()} positive)\n")
    
    # Transforms - match best model (milder augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),  # Changed from 20
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Milder
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Removed RandomPerspective and RandomErasing - too aggressive
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = HybridDataset(
        data_dir / 'train-image-preprocessed.hdf5',
        train_meta_processed, transform=None, is_test=False
    )
    
    train_subset = Subset(full_dataset, train_idx)
    train_subset.dataset.transform = train_transform
    
    val_subset = Subset(full_dataset, val_idx)
    val_subset.dataset.transform = val_transform
    
    # DataLoaders
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=16, pin_memory=True, persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=16, pin_memory=True, persistent_workers=True
    )
    
    print(f"✓ DataLoaders ready\n")
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    kfold_marker_file = results_dir / '.kfold_advanced_current'
    
    if kfold_marker_file.exists():
        with open(kfold_marker_file, 'r') as f:
            kfold_dir_name = f.read().strip()
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        kfold_dir_name = f'kfold_v2s_features_advanced_{timestamp}'
        with open(kfold_marker_file, 'w') as f:
            f.write(kfold_dir_name)
    
    kfold_dir = results_dir / kfold_dir_name
    kfold_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = EfficientNetV2Hybrid(metadata_dim=metadata_dim).to(device)
    
    # Loss WITHOUT label smoothing (match best model)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, smoothing=0.0)  # Changed from 0.05
    
    # Optimizer - match best model settings
    optimizer = optim.Adam(  # Changed from AdamW
        model.parameters(),
        lr=0.0005,
        weight_decay=1e-5,  # Changed from 5e-4 (back to working value!)
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler - configurable
    if args.scheduler == 'plateau':
        # Match best model (SAFE - proven to work)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        print(f"Using ReduceLROnPlateau (same as best model)")
    else:
        # Experimental (might help, might hurt)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        print(f"Using CosineAnnealingWarmRestarts (experimental)")
    
    # Mixed precision scaler (RENAME to avoid collision with StandardScaler)
    amp_scaler = torch.cuda.amp.GradScaler()  # Changed from 'scaler'
    
    # Model EMA
    model_ema = ModelEMA(model, decay=0.9999)
    
    # SWA model
    swa_model = optim.swa_utils.AveragedModel(model)
    swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=0.0001)
    
    # Training loop
    best_auc = 0.0
    best_ema_auc = 0.0
    patience_counter = 0
    max_patience = 8  # Early stopping patience
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': [],
               'val_ema_auc': [], 'lr': []}
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, amp_scaler, device,
            args.accumulation_steps, model_ema
        )
        
        # Validate with regular model
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        
        # Validate with EMA model
        model_ema.apply_shadow()
        _, val_ema_auc = validate(model, val_loader, criterion, device)
        model_ema.restore()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(float(train_loss))
        history['train_auc'].append(float(train_auc))
        history['val_loss'].append(float(val_loss))
        history['val_auc'].append(float(val_auc))
        history['val_ema_auc'].append(float(val_ema_auc))
        history['lr'].append(float(current_lr))
        
        print(f"  Train: Loss={train_loss:.4f}, AUC={train_auc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, AUC={val_auc:.4f}")
        print(f"  EMA:   AUC={val_ema_auc:.4f}")
        print(f"  LR:    {current_lr:.6f}")
        
        # Save best regular model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0  # Reset patience
            torch.save({
                'fold': fold_num,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': float(val_auc),
            }, kfold_dir / f'best_model_fold{fold_num}.pth')
            print(f"  ✓ Best model: {best_auc:.4f}")
        else:
            patience_counter += 1
        
        # Save best EMA model
        if val_ema_auc > best_ema_auc:
            best_ema_auc = val_ema_auc
            model_ema.apply_shadow()
            torch.save({
                'fold': fold_num,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': float(val_ema_auc),
            }, kfold_dir / f'best_ema_model_fold{fold_num}.pth')
            model_ema.restore()
            print(f"  ✓ Best EMA: {best_ema_auc:.4f}")
        
        # Early stopping check
        if patience_counter >= max_patience:
            print(f"\n  ⚠️ Early stopping triggered (patience={max_patience})")
            print(f"  Best val AUC hasn't improved for {max_patience} epochs")
            print(f"  Stopping to prevent overfitting...")
            break
        
        # Overfitting warning
        if epoch > 5:
            train_val_gap = train_auc - val_auc
            if train_val_gap > 0.15:
                print(f"  ⚠️ SEVERE overfitting! Gap: {train_val_gap:.4f}")
            elif train_val_gap > 0.08:
                print(f"  ⚠️ Overfitting warning! Gap: {train_val_gap:.4f}")
        
        # SWA and scheduler
        if epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"  ✓ SWA updated")
        else:
            if args.scheduler == 'plateau':
                scheduler.step(val_auc)
            else:
                scheduler.step()
        
        print()
    
    # Finalize SWA - Custom BN update for hybrid model
    print("Finalizing SWA model...")
    
    # Custom batch norm update for models with multiple inputs
    @torch.no_grad()
    def update_bn_hybrid(loader, model, device):
        """Update batch norm stats for hybrid model (image + metadata)"""
        model.train()
        for images, metadata, labels in tqdm(loader, desc="Updating BN", ncols=100):
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            # Forward pass to update running stats
            _ = model(images, metadata)
    
    update_bn_hybrid(train_loader, swa_model, device=device)
    swa_val_loss, swa_val_auc = validate(swa_model, val_loader, criterion, device)
    
    print(f"\nSWA Val AUC: {swa_val_auc:.4f}")
    
    # Save SWA model
    torch.save({
        'fold': fold_num,
        'model_state_dict': swa_model.module.state_dict(),
        'val_auc': float(swa_val_auc),
    }, kfold_dir / f'swa_model_fold{fold_num}.pth')
    
    # Save results
    results = {
        'fold': fold_num,
        'best_val_auc': float(best_auc),
        'best_ema_auc': float(best_ema_auc),
        'swa_auc': float(swa_val_auc),
        'best_overall': float(max(best_auc, best_ema_auc, swa_val_auc)),
        'history': history,
        'gpu_id': gpu_id,
    }
    
    with open(kfold_dir / f'fold_{fold_num}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save preprocessors (fold 1 only)
    if fold_num == 1:
        with open(kfold_dir / 'preprocessors.pkl', 'wb') as f:
            pickle.dump({
                'scaler': scaler,  # This is StandardScaler from preprocessing
                'encoders': encoders,
                'metadata_dim': metadata_dim
            }, f)
        print(f"\n✓ Preprocessors saved (StandardScaler + encoders)")
    
    print(f"\n{'='*70}")
    print(f"FOLD {fold_num} COMPLETE")
    print(f"{'='*70}")
    print(f"Best Val AUC: {best_auc:.4f}")
    print(f"Best EMA AUC: {best_ema_auc:.4f}")
    print(f"SWA AUC: {swa_val_auc:.4f}")
    print(f"Best Overall: {max(best_auc, best_ema_auc, swa_val_auc):.4f}")
    print(f"Results: {kfold_dir}")
    print(f"{'='*70}\n")
    
    return fold_num, max(best_auc, best_ema_auc, swa_val_auc)


# ===========================
# MAIN ENTRY POINT
# ===========================

if __name__ == '__main__':
    args = parse_args()
    
    if args.fold < 1 or args.fold > 5:
        print("Error: Fold must be between 1 and 5")
        exit(1)
    
    print(f"\n{'='*70}")
    print("5-FOLD CV - EFFICIENTNETV2-S + FEATURES + SOTA")
    print(f"{'='*70}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    fold_num, best_auc = train_fold(args.fold, args.gpu, args)
    
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✓ Fold {fold_num} complete: {best_auc:.4f}")