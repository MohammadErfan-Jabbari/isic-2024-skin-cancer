"""
Vision Transformer (ViT-B/16) Hybrid with Feature Engineering
MAXIMUM ensemble diversity - pure attention-based architecture

Key differences from CNNs:
- No convolutions (uses self-attention)
- Processes image as patches (16x16)
- Global receptive field from layer 1
- Different inductive biases → excellent ensemble diversity

Usage:
    python train_vit_with_features.py --gpu 2
    python train_vit_with_features.py --gpu 3 --epochs 30
    
Expected:
- Training time: ~6-8 hours (slower than CNNs)
- Val AUC: 0.935-0.945 (might not beat CNNs alone)
- Ensemble boost: +0.008-0.015 (HIGH diversity!)
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
from torchvision.models import vit_b_16
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2, help='GPU ID (0-3)')
    parser.add_argument('--batch-size', type=int, default=96, help='Batch size (smaller for ViT)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs (ViT needs more)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (lower for ViT)')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Warmup epochs')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    return parser.parse_args()


# ===========================
# FEATURE ENGINEERING (SAME AS OTHERS)
# ===========================

def engineer_features(df):
    df = df.copy()
    
    df['age_group'] = pd.cut(df['age_approx'], bins=[0, 30, 50, 70, 100],
                             labels=['young', 'middle', 'senior', 'elderly'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['size_category'] = pd.cut(df['lesion_size_mm'], bins=[0, 6, 10, 20, 100],
                                 labels=['small', 'medium', 'large', 'very_large'])
    df['large_lesion'] = (df['lesion_size_mm'] > 6).astype(int)
    
    df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    df['eccentricity'] = df['tbp_lv_minorAxisMM'] / (df['tbp_lv_areaMM2']**0.5 + 1e-6)
    
    df['color_variance'] = np.sqrt(
        df['tbp_lv_deltaB']**2 + df['tbp_lv_radial_color_std_max']**2 +
        df['tbp_lv_color_std_mean']**2
    )
    df['color_uniformity'] = 1 / (df['tbp_lv_norm_color'] + 1e-6)
    df['darkness_score'] = df['tbp_lv_B'] / (df['tbp_lv_H'] + 1e-6)
    
    high_risk_sites = ['torso', 'upper extremity', 'posterior torso', 'anterior torso']
    df['high_risk_site'] = df['anatom_site_general'].isin(high_risk_sites).astype(int)
    
    site_risk_map = {
        'torso': 3, 'posterior torso': 3, 'anterior torso': 3,
        'upper extremity': 2, 'lower extremity': 2, 'head/neck': 2,
        'palms/soles': 1, 'oral/genital': 1
    }
    df['site_risk_score'] = df['anatom_site_general'].map(site_risk_map).fillna(0)
    
    df['age_size_risk'] = df['age_approx'] * df['lesion_size_mm']
    df['age_site_risk'] = df['age_approx'] * df['high_risk_site']
    df['color_size_risk'] = df['color_variance'] * df['lesion_size_mm']
    
    df['asymmetry_score'] = (
        df['tbp_lv_norm_color'] + df['tbp_lv_radial_color_std_max'] +
        (1 / (df['shape_regularity'] + 1e-6))
    ) / 3
    
    df['log_area'] = np.log1p(df['tbp_lv_areaMM2'])
    df['log_perimeter'] = np.log1p(df['tbp_lv_perimeterMM'])
    df['log_size'] = np.log1p(df['lesion_size_mm'])
    
    df['h_to_b_ratio'] = df['tbp_lv_H'] / (df['tbp_lv_B'] + 1e-6)
    df['a_to_b_ratio'] = df['tbp_lv_A'] / (df['tbp_lv_B'] + 1e-6)
    
    return df


def preprocess_metadata_with_features(df, is_train=True, scaler=None, encoders=None):
    df = engineer_features(df)
    
    NUMERICAL_FEATURES = [
        'tbp_lv_H', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM',
        'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_Hext',
        'clin_size_long_diam_mm', 'tbp_lv_radial_color_std_max',
        'tbp_lv_B', 'tbp_lv_color_std_mean', 'tbp_lv_Aext',
        'tbp_lv_stdLExt', 'tbp_lv_norm_color', 'tbp_lv_A', 'age_approx',
        'lesion_size_mm', 'shape_regularity', 'eccentricity',
        'color_variance', 'color_uniformity', 'darkness_score',
        'site_risk_score', 'age_size_risk', 'age_site_risk',
        'color_size_risk', 'asymmetry_score',
        'log_area', 'log_perimeter', 'log_size',
        'h_to_b_ratio', 'a_to_b_ratio'
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
                       if col not in ['isic_id', 'target']]
        self.metadata_features = self.metadata[feature_cols].values.astype(np.float32)
        
        if not is_test and 'target' in self.metadata.columns:
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
# VISION TRANSFORMER MODEL
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


class ViTHybrid(nn.Module):
    """
    Vision Transformer (ViT-B/16) hybrid model
    
    Architecture:
      - ViT-B/16 (pretrained) → 768 features
      - Metadata MLP → 64 features
      - Concatenate → 832 features
      - Classifier → 1 output
    
    ViT advantages:
    - Pure attention (no convolutions!)
    - Global receptive field from layer 1
    - Different learning patterns than CNNs
    - MAXIMUM ensemble diversity
    
    ViT challenges:
    - Needs more data (we have 400k - should be OK)
    - Slower training
    - Lower LR required
    - Longer warmup needed
    """
    def __init__(self, metadata_dim):
        super().__init__()
        
        # Load pretrained ViT-B/16
        vit = vit_b_16(weights='IMAGENET1K_V1')
        
        # Extract encoder (everything except final head)
        self.conv_proj = vit.conv_proj
        self.encoder = vit.encoder
        
        # ViT outputs 768 features
        # Don't use the heads - we'll build our own
        
        # Freeze early layers (first 6 of 12 transformer blocks)
        for idx, block in enumerate(self.encoder.layers):
            if idx < 6:
                for param in block.parameters():
                    param.requires_grad = False
        
        # Metadata processor
        self.metadata_processor = MetadataProcessor(metadata_dim)
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, metadata):
        # ViT forward pass
        # 1. Patch embedding
        x = self.conv_proj(image)  # [B, 768, 14, 14] (224/16 = 14 patches)
        
        # 2. Reshape to sequence
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 768] (14*14 = 196 patches)
        
        # 3. Add class token
        batch_class_token = self.encoder.pos_embedding[:, :1, :].expand(B, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)  # [B, 197, 768]
        
        # 4. Add positional embedding
        x = x + self.encoder.pos_embedding
        
        # 5. Dropout
        x = self.encoder.dropout(x)
        
        # 6. Transformer blocks
        x = self.encoder.layers(x)  # [B, 197, 768]
        
        # 7. Layer norm
        x = self.encoder.ln(x)
        
        # 8. Extract class token (first token contains image representation)
        img_features = x[:, 0]  # [B, 768]
        
        # Process metadata
        meta_features = self.metadata_processor(metadata)
        
        # Combine and classify
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(combined)


# ===========================
# LEARNING RATE WARMUP SCHEDULER
# ===========================

class WarmupScheduler:
    """Linear warmup followed by ReduceLROnPlateau"""
    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
    
    def step(self, val_auc=None):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif val_auc is not None:
            # After warmup, use plateau scheduler
            self.plateau_scheduler.step(val_auc)
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# ===========================
# TRAINING FUNCTIONS
# ===========================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    for images, metadata, labels in tqdm(loader, desc="Training", ncols=100):
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping (important for ViT)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
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
# MAIN
# ===========================

def main():
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("VISION TRANSFORMER (ViT-B/16) + FEATURE ENGINEERING")
    print("="*70)
    print(f"GPU: {args.gpu}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {args.batch_size} (smaller - ViT uses more memory)")
    print(f"Epochs: {args.epochs} (more - ViT needs longer training)")
    print(f"Learning rate: {args.lr} (lower - ViT is sensitive)")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"\n⭐ Architecture: PURE ATTENTION (no convolutions!)")
    print(f"⭐ Maximum diversity for ensemble with CNNs")
    print("="*70 + "\n")
    
    # Load data
    data_dir = Path(args.data_dir)
    train_meta = pd.read_csv(data_dir / 'new-train-metadata.csv', low_memory=False)
    test_meta = pd.read_csv(data_dir / 'students-test-metadata.csv', low_memory=False)
    
    print(f"Data loaded: {len(train_meta):,} train, {len(test_meta):,} test\n")
    
    # Preprocess with feature engineering
    print("Engineering features...")
    train_meta_processed, scaler, encoders = preprocess_metadata_with_features(
        train_meta, is_train=True
    )
    test_meta_processed, _, _ = preprocess_metadata_with_features(
        test_meta, is_train=False, scaler=scaler, encoders=encoders
    )
    
    train_meta_processed['isic_id'] = train_meta['isic_id'].values
    train_meta_processed['target'] = train_meta['target'].values
    test_meta_processed['isic_id'] = test_meta['isic_id'].values
    
    metadata_dim = len(train_meta_processed.columns) - 2
    print(f"✓ Metadata dimension: {metadata_dim}\n")
    
    # ViT-specific transforms (stronger augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),  # More rotation for ViT
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small shifts
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1))  # Random erasing
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Split
    train_df, val_df = train_test_split(
        train_meta_processed, test_size=0.2, random_state=42,
        stratify=train_meta_processed['target']
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = HybridDataset(
        data_dir / 'train-image-preprocessed.hdf5',
        train_df, train_transform, is_test=False
    )
    
    val_dataset = HybridDataset(
        data_dir / 'train-image-preprocessed.hdf5',
        val_df, val_transform, is_test=False
    )
    
    test_dataset = HybridDataset(
        data_dir / 'test-image-preprocessed.hdf5',
        test_meta_processed, val_transform, is_test=True
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
    
    print(f"✓ DataLoaders ready\n")
    
    # Create model
    model = ViTHybrid(metadata_dim=metadata_dim).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Vision Transformer: {total_params:,} params ({trainable_params:,} trainable)\n")
    
    # Training setup - ViT-specific!
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # ViT uses AdamW with lower LR and higher weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr,  # 0.0001 (lower than CNNs)
        weight_decay=0.05,  # Higher than CNNs
        betas=(0.9, 0.999)
    )
    
    # Warmup scheduler
    scheduler = WarmupScheduler(optimizer, args.warmup_epochs, args.lr)
    
    print(f"Training setup (ViT-optimized):")
    print(f"  Optimizer: AdamW (lr={args.lr}, wd=0.05)")
    print(f"  Warmup: {args.warmup_epochs} epochs")
    print(f"  Scheduler: Linear warmup → ReduceLROnPlateau")
    print(f"  Grad clipping: 1.0")
    print(f"  Note: ViT needs different hyperparams than CNNs!\n")
    
    # Training loop
    best_auc = 0.0
    patience_counter = 0
    max_patience = 12  # More patience for ViT
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('results') / f'vit_features_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    total_start = time.time()
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_auc, train_time = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_auc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Train: Loss={train_loss:.4f}, AUC={train_auc:.4f}, Time={train_time:.1f}s")
        print(f"  Val:   Loss={val_loss:.4f}, AUC={val_auc:.4f}")
        print(f"  LR:    {current_lr:.6f} ", end='')
        
        if epoch < args.warmup_epochs:
            print("(warmup)")
        else:
            print()
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
            }, results_dir / 'best_model.pth')
            print(f"  ✓ Best: {best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  Early stopping ({patience_counter}/{max_patience})")
                break
        
        # Step scheduler
        scheduler.step(val_auc if epoch >= args.warmup_epochs else None)
        
        if current_lr < 5e-6:
            print("  LR too small, stopping")
            break
        print()
    
    total_time = time.time() - total_start
    
    # Save results
    with open(results_dir / 'training_results.pkl', 'wb') as f:
        pickle.dump({
            'model': 'ViT-B/16 + Features',
            'best_auc': best_auc,
            'history': history,
            'total_time': total_time,
            'metadata_dim': metadata_dim,
        }, f)
    
    with open(results_dir / 'preprocessors.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler, 'encoders': encoders}, f)
    
    # Generate predictions
    print("\nGenerating test predictions...")
    checkpoint = torch.load(results_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_ids = []
    test_preds = []
    
    with torch.no_grad():
        for images, metadata, img_ids in tqdm(test_loader, ncols=100):
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            outputs = model(images, metadata)
            probs = torch.sigmoid(outputs).cpu().numpy()
            test_ids.extend(img_ids)
            test_preds.extend(probs.flatten())
    
    submission = pd.DataFrame({'isic_id': test_ids, 'target': test_preds})
    submission.to_csv(results_dir / 'submission_vit_features.csv', index=False)
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"Best Val AUC: {best_auc:.4f}")
    print(f"\nEnsemble Impact Analysis:")
    print(f"  Expected alone: 0.935-0.945 (might not beat CNNs)")
    print(f"  Expected in ensemble: +0.008-0.015 AUC (HIGH diversity!)")
    print(f"  Reason: Completely different architecture from CNNs")
    print(f"\nResults: {results_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()