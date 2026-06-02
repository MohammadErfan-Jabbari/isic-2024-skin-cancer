"""
5-Fold Cross-Validation Training Script - Single Fold per GPU

This script trains ONE fold at a time, designed to run on multiple GPUs in parallel.

Usage:
    # Train fold 1 on GPU 0
    python 8_hybrid_cnn_metadata_crpss-valid_ensemble.py --fold 1 --gpu 0
    
    # Train fold 2 on GPU 1 (in another terminal)
    python 8_hybrid_cnn_metadata_crpss-valid_ensemble.py --fold 2 --gpu 1
    
    # Train fold 3 on GPU 2
    python 8_hybrid_cnn_metadata_crpss-valid_ensemble.py --fold 3 --gpu 2
    
    # Train fold 4 on GPU 3
    python 8_hybrid_cnn_metadata_crpss-valid_ensemble.py --fold 4 --gpu 3
    
    # Train fold 5 on any GPU (after one of the above completes)
    python 8_hybrid_cnn_metadata_crpss-valid_ensemble.py --fold 5 --gpu 0

Expected training time per fold: ~4-5 hours on a good GPU
Total time on 4 GPUs: ~6-8 hours (5th fold runs sequentially after one GPU frees up)
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
import json
import os
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')


# ===========================
# MODEL ARCHITECTURE
# ===========================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
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
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, metadata):
        img_features = self.efficientnet(image)
        meta_features = self.metadata_processor(metadata)
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(combined)


# ===========================
# HYBRID DATASET CLASS
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
                       if col not in ['isic_id', 'target']]
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
# TRAINING FUNCTIONS
# ===========================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, metadata, labels in tqdm(loader, desc="Training", ncols=100):
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
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
        for images, metadata, labels in tqdm(loader, desc="Validation", ncols=100):
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
    
    return epoch_loss, epoch_auc, all_preds, all_labels


# ===========================
# METADATA PREPROCESSING
# ===========================

def preprocess_metadata(df, is_train=True, scaler=None, encoders=None):
    """Preprocess metadata with numerical and categorical features."""
    
    NUMERICAL_FEATURES = [
        'tbp_lv_H', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM',
        'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_Hext',
        'clin_size_long_diam_mm', 'tbp_lv_radial_color_std_max',
        'tbp_lv_B', 'tbp_lv_color_std_mean', 'tbp_lv_Aext',
        'tbp_lv_stdLExt', 'tbp_lv_norm_color', 'tbp_lv_A',
        'age_approx'
    ]
    
    CATEGORICAL_FEATURES = [
        'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple'
    ]
    
    df = df.copy()
    
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if is_train else 0)
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('missing')
    
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
# MAIN TRAINING FUNCTION
# ===========================

def train_fold(fold_num, gpu_id, num_epochs=25):
    """Train a single fold on a specific GPU."""
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"TRAINING FOLD {fold_num}/5")
    print(f"{'='*70}")
    print(f"GPU ID: {gpu_id}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Configuration
    BATCH_SIZE = 256
    NUM_WORKERS = 16
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # Load data
    data_dir = Path('data')
    train_meta = pd.read_csv(data_dir / 'new-train-metadata.csv', low_memory=False)
    
    print(f"Data loaded: {len(train_meta):,} samples\n")
    
    # Preprocess metadata
    train_meta_processed, scaler, encoders = preprocess_metadata(train_meta, is_train=True)
    train_meta_processed['isic_id'] = train_meta['isic_id'].values
    train_meta_processed['target'] = train_meta['target'].values
    
    metadata_dim = len(train_meta_processed.columns) - 2
    print(f"Metadata dimension: {metadata_dim}\n")
    
    # Create K-fold splits
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(skf.split(train_meta_processed, train_meta_processed['target']))
    
    # Get the fold we need to train
    train_idx, val_idx = fold_indices[fold_num - 1]
    
    print(f"Fold {fold_num} split:")
    fold_targets = train_meta_processed.iloc[train_idx]['target']
    val_targets = train_meta_processed.iloc[val_idx]['target']
    print(f"  Train: {len(train_idx):,} samples ({fold_targets.sum()} positive)")
    print(f"  Val:   {len(val_idx):,} samples ({val_targets.sum()} positive)")
    print(f"  Val positive rate: {val_targets.mean():.4f}\n")
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = HybridDataset(
        hdf5_path=data_dir / 'train-image-preprocessed.hdf5',
        metadata_df=train_meta_processed,
        transform=None,
        is_test=False
    )
    
    train_subset = Subset(full_dataset, train_idx)
    train_subset.dataset.transform = train_transform
    
    val_subset = Subset(full_dataset, val_idx)
    val_subset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    
    print(f"Dataloaders created: {len(train_loader)} train batches, {len(val_loader)} val batches\n")
    
    # Create results directory with timestamped kfold subdirectory
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Read or create kfold experiment directory name
    kfold_marker_file = results_dir / '.kfold_current'
    
    if kfold_marker_file.exists():
        # Read existing kfold directory name
        with open(kfold_marker_file, 'r') as f:
            kfold_dir_name = f.read().strip()
    else:
        # Create new timestamped kfold directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        kfold_dir_name = f'kfold_efficientnetv2_{timestamp}'
        # Store the directory name for other folds to use
        with open(kfold_marker_file, 'w') as f:
            f.write(kfold_dir_name)
    
    kfold_dir = results_dir / kfold_dir_name
    kfold_dir.mkdir(parents=True, exist_ok=True)
    
    # Create fold-specific results file
    fold_results_file = kfold_dir / f'fold_{fold_num}_results.json'
    fold_model_file = kfold_dir / f'best_model_fold{fold_num}.pth'
    
    # Create model
    model = EfficientNetV2Hybrid(metadata_dim=metadata_dim).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Training loop
    best_fold_auc = 0.0
    fold_history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}
    
    print(f"{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        print(f"Fold {fold_num} - Epoch {epoch+1}/{num_epochs}")
        
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, _, _ = validate(model, val_loader, criterion, device)
        
        fold_history['train_loss'].append(float(train_loss))
        fold_history['train_auc'].append(float(train_auc))
        fold_history['val_loss'].append(float(val_loss))
        fold_history['val_auc'].append(float(val_auc))
        
        print(f"  Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
        
        if val_auc > best_fold_auc:
            best_fold_auc = val_auc
            torch.save({
                'fold': fold_num,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': float(val_auc),
                'scaler': scaler,
                'encoders': encoders
            }, fold_model_file)
            print(f"  ✓ Saved best model (AUC: {best_fold_auc:.4f})")
        
        scheduler.step(val_auc)
        
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print(f"  LR too small, stopping early...")
            break
    
    # Save fold results
    results = {
        'fold': fold_num,
        'best_val_auc': float(best_fold_auc),
        'best_epoch': epoch,
        'history': fold_history,
        'gpu_id': gpu_id,
        'training_time_per_epoch': f"~{(num_epochs * 4.5 / num_epochs):.1f} min"
    }
    
    with open(fold_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save preprocessors (scaler and encoders) for fold 1 (same for all folds)
    if fold_num == 1:
        import pickle
        preprocessors = {
            'scaler': scaler,
            'encoders': encoders,
            'metadata_dim': metadata_dim
        }
        preprocessors_file = kfold_dir / 'preprocessors.pkl'
        with open(preprocessors_file, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"Preprocessors saved to: {preprocessors_file}")
    
    # Save training config
    if fold_num == 1:
        config = {
            'model': 'EfficientNetV2-S Hybrid',
            'batch_size': BATCH_SIZE,
            'num_workers': NUM_WORKERS,
            'num_epochs': num_epochs,
            'learning_rate': 0.0005,
            'weight_decay': 1e-5,
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2.0,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_patience': 5,
            'scheduler_factor': 0.5,
            'n_folds': N_FOLDS,
            'random_state': RANDOM_STATE
        }
        config_file = kfold_dir / 'training_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Training config saved to: {config_file}")
    
    print(f"\n{'='*70}")
    print(f"FOLD {fold_num} COMPLETE")
    print(f"{'='*70}")
    print(f"Best Val AUC: {best_fold_auc:.4f}")
    print(f"Results saved to: {fold_results_file}")
    print(f"Model saved to: {fold_model_file}")
    print(f"{'='*70}\n")
    
    return fold_num, best_fold_auc


# ===========================
# MAIN ENTRY POINT
# ===========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a single fold on a specific GPU')
    parser.add_argument('--fold', type=int, required=True, help='Fold number (1-5)')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID (0-3)')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs per fold')
    
    args = parser.parse_args()
    
    if args.fold < 1 or args.fold > 5:
        print("Error: Fold must be between 1 and 5")
        exit(1)
    
    if args.gpu < 0 or args.gpu > 3:
        print("Error: GPU ID must be between 0 and 3")
        exit(1)
    
    print(f"\n{'='*70}")
    print(f"5-FOLD CROSS-VALIDATION - SINGLE FOLD TRAINING")
    print(f"{'='*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    fold_num, best_auc = train_fold(args.fold, args.gpu, num_epochs=args.epochs)
    
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✓ Training complete for fold {fold_num} with AUC: {best_auc:.4f}")
