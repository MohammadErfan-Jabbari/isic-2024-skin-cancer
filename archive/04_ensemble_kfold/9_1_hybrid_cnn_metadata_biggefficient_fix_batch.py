"""
EfficientNetV2-M Hybrid Model - CORRECTED VERSION

Fixes:
1. ✅ Batch size: 128 (was incorrectly 512)
2. ✅ Stronger regularization (dropout 0.6/0.5)
3. ✅ Better augmentation (rotation, color jitter)
4. ✅ Mixup data augmentation
5. ✅ Early stopping
6. ✅ GPU selection via command line

Usage:
    python train_efficientnetv2_m_fixed.py --gpu 0
    python train_efficientnetv2_m_fixed.py --gpu 1 --epochs 30
    python train_efficientnetv2_m_fixed.py --gpu 2 --batch-size 128 --no-mixup
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_m
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings
import pickle
from datetime import datetime
import json
import argparse
import os

warnings.filterwarnings('ignore')


# ===========================
# COMMAND LINE ARGUMENTS
# ===========================

def parse_args():
    parser = argparse.ArgumentParser(description='Train EfficientNetV2-M with corrected settings')
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU ID to use (0-3)')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128, FIXED from 512)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-5,
                       help='Weight decay (INCREASED from 2e-5)')
    
    # Augmentation
    parser.add_argument('--no-mixup', action='store_true',
                       help='Disable mixup augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                       help='Mixup alpha parameter')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    
    return parser.parse_args()


# ===========================
# METADATA PREPROCESSING
# ===========================

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


def preprocess_metadata(df, is_train=True, scaler=None, encoders=None):
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
                       if col not in ['isic_id', 'target']]
        self.metadata_features = self.metadata[feature_cols].values.astype(np.float32)
        
        if not is_test and 'target' in self.metadata.columns:
            print(f"  ✓ {len(self.metadata)} samples, "
                  f"distribution: {self.metadata['target'].value_counts().to_dict()}")
        else:
            print(f"  ✓ {len(self.metadata)} samples")
    
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
# MIXUP AUGMENTATION
# ===========================

def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ===========================
# FOCAL LOSS
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


# ===========================
# MODEL ARCHITECTURE
# ===========================

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


class EfficientNetV2MHybrid(nn.Module):
    """EfficientNetV2-M with STRONGER regularization"""
    def __init__(self, metadata_dim):
        super().__init__()
        
        self.efficientnet = efficientnet_v2_m(weights='IMAGENET1K_V1')
        self.efficientnet.classifier = nn.Identity()
        
        # Freeze 80% of parameters
        total_params = len(list(self.efficientnet.parameters()))
        freeze_until = int(total_params * 0.8)
        
        for idx, param in enumerate(self.efficientnet.parameters()):
            if idx < freeze_until:
                param.requires_grad = False
        
        self.metadata_processor = MetadataProcessor(metadata_dim)
        
        # STRONGER dropout: 0.6 and 0.5 (was 0.5 and 0.3)
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.6),  # Increased from 0.5
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased from 0.3
            nn.Linear(128, 1)
        )
    
    def forward(self, image, metadata):
        img_features = self.efficientnet(image)
        meta_features = self.metadata_processor(metadata)
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(combined)


# ===========================
# TRAINING FUNCTIONS
# ===========================

def train_epoch(model, loader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.2):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    for images, metadata, labels in tqdm(loader, desc="Training", ncols=100):
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        
        # Apply mixup
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            metadata = metadata  # Don't mixup metadata
        
        optimizer.zero_grad()
        outputs = model(images, metadata)
        
        # Calculate loss
        if use_mixup:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # For AUC calculation (use original labels)
        if use_mixup:
            all_labels.extend(labels_a.cpu().numpy())
        else:
            all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
    
    elapsed = time.time() - start_time
    epoch_loss = running_loss / len(loader)
    epoch_auc = roc_auc_score(all_labels, all_preds)
    
    return epoch_loss, epoch_auc, elapsed


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
# MAIN TRAINING LOOP
# ===========================

def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("EFFICIENTNETV2-M HYBRID - CORRECTED VERSION")
    print("="*70)
    print(f"GPU: {args.gpu} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Batch size: {args.batch_size} (FIXED from 512)")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay} (INCREASED from 2e-5)")
    print(f"Dropout: [0.6, 0.5] (INCREASED from [0.5, 0.3])")
    print(f"Mixup: {'Enabled' if not args.no_mixup else 'Disabled'}")
    print(f"Early stopping patience: {args.patience}")
    print("="*70 + "\n")
    
    # Load data
    data_dir = Path(args.data_dir)
    train_meta = pd.read_csv(data_dir / 'new-train-metadata.csv', low_memory=False)
    test_meta = pd.read_csv(data_dir / 'students-test-metadata.csv', low_memory=False)
    
    print(f"Metadata loaded:")
    print(f"  Train: {len(train_meta):,} samples")
    print(f"  Test: {len(test_meta):,} samples\n")
    
    # Preprocess metadata
    print("Preprocessing metadata...")
    train_meta_processed, scaler, encoders = preprocess_metadata(train_meta, is_train=True)
    test_meta_processed, _, _ = preprocess_metadata(test_meta, is_train=False, 
                                                    scaler=scaler, encoders=encoders)
    
    train_meta_processed['isic_id'] = train_meta['isic_id'].values
    train_meta_processed['target'] = train_meta['target'].values
    test_meta_processed['isic_id'] = test_meta['isic_id'].values
    
    metadata_dim = len(train_meta_processed.columns) - 2
    print(f"✓ Metadata dimension: {metadata_dim}\n")
    
    # BETTER augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),  # NEW
        transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                             saturation=0.2, hue=0.1),  # NEW
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Train/val split
    train_df, val_df = train_test_split(
        train_meta_processed, test_size=0.2, random_state=42,
        stratify=train_meta_processed['target']
    )
    
    print(f"Split: {len(train_df):,} train / {len(val_df):,} val\n")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = HybridDataset(
        hdf5_path=data_dir / 'train-image-preprocessed.hdf5',
        metadata_df=train_df, transform=train_transform, is_test=False
    )
    
    val_dataset = HybridDataset(
        hdf5_path=data_dir / 'train-image-preprocessed.hdf5',
        metadata_df=val_df, transform=val_transform, is_test=False
    )
    
    test_dataset = HybridDataset(
        hdf5_path=data_dir / 'test-image-preprocessed.hdf5',
        metadata_df=test_meta_processed, transform=val_transform, is_test=True
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=16, pin_memory=True, persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=16, pin_memory=True, persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=16, pin_memory=True, persistent_workers=True
    )
    
    print(f"\nDataLoaders ready: {len(train_loader)} train batches, "
          f"{len(val_loader)} val batches\n")
    
    # Create model
    print("Creating model...")
    model = EfficientNetV2MHybrid(metadata_dim=metadata_dim).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}\n")
    
    # Training setup
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    # Training loop
    best_auc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_auc': [], 'train_time': [],
        'val_loss': [], 'val_auc': []
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('results') / f'efficientnetv2_m_fixed_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    total_start = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_auc, train_time = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            use_mixup=not args.no_mixup, mixup_alpha=args.mixup_alpha
        )
        
        # Validate
        val_loss, val_auc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['train_time'].append(train_time)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Time: {train_time:.1f}s")
        print(f"  Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
        
        # Check overfitting
        if epoch > 5:
            gap = train_auc - val_auc
            if gap > 0.08:
                print(f"  ⚠️ Large train-val gap: {gap:.4f} (overfitting!)")
            elif gap > 0.05:
                print(f"  ⚠ Moderate train-val gap: {gap:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, results_dir / 'best_efficientnetv2_m_fixed.pth')
            print(f"  ✓ Saved best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
        
        # Learning rate
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.6f}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  Early stopping triggered (patience={args.patience})")
            break
        
        if current_lr < 1e-6:
            print(f"\n  LR too small, stopping...")
            break
    
    total_time = time.time() - total_start
    
    # Save results
    with open(results_dir / 'training_results.pkl', 'wb') as f:
        pickle.dump({
            'timestamp': timestamp,
            'model': 'EfficientNetV2-M Fixed',
            'best_auc': best_auc,
            'history': history,
            'total_time': total_time,
            'batch_size': args.batch_size,
            'args': vars(args),
        }, f)
    
    with open(results_dir / 'preprocessors.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler, 'encoders': encoders}, f)
    
    # Generate test predictions
    print("\n" + "="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70 + "\n")
    
    checkpoint = torch.load(results_dir / 'best_efficientnetv2_m_fixed.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_ids = []
    test_preds = []
    
    with torch.no_grad():
        for images, metadata, img_ids in tqdm(test_loader, desc="Testing", ncols=100):
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            outputs = model(images, metadata)
            probs = torch.sigmoid(outputs).cpu().numpy()
            test_ids.extend(img_ids)
            test_preds.extend(probs.flatten())
    
    submission = pd.DataFrame({'isic_id': test_ids, 'target': test_preds})
    submission.to_csv(results_dir / 'submission_efficientnetv2_m_fixed.csv', index=False)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Improvement vs V2-S: {best_auc - 0.9508:+.4f}")
    print(f"Results saved to: {results_dir}")
    
    if best_auc > 0.95:
        print(f"\n🎉 Excellent! Much better than before (0.9199)")
    elif best_auc > 0.945:
        print(f"\n✓ Good improvement!")
    else:
        print(f"\n⚠ Still needs work")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()