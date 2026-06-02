"""
ConvNeXt-Base Hybrid with Feature Engineering
Different architecture for ensemble diversity

Usage:
    python train_convnext_with_features.py --gpu 1
    python train_convnext_with_features.py --gpu 2 --batch-size 128
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
from torchvision.models import convnext_base
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
    parser.add_argument('--gpu', type=int, default=1, help='GPU ID (0-3)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (smaller for larger model)')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    return parser.parse_args()


# ===========================
# FEATURE ENGINEERING (SAME AS V2-S)
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
# DATASET (SAME AS V2-S)
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
# CONVNEXT MODEL
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


class ConvNeXtHybrid(nn.Module):
    """
    ConvNeXt-Base hybrid model
    
    Architecture:
      - ConvNeXt-Base (pretrained) → 1024 features
      - Metadata MLP → 64 features
      - Concatenate → 1088 features
      - Classifier → 1 output
    
    ConvNeXt advantages:
    - Modern CNN design (2022)
    - Different from EfficientNet → excellent ensemble diversity
    - Great for texture-rich medical images
    - Competes with ViT but trains faster
    """
    def __init__(self, metadata_dim):
        super().__init__()
        
        # Load pretrained ConvNeXt-Base
        convnext = convnext_base(weights='IMAGENET1K_V1')
        
        # Extract feature extractor only (remove classifier head)
        self.features = convnext.features
        self.avgpool = convnext.avgpool
        self.layer_norm = convnext.classifier[0]  # LayerNorm layer
        
        # Freeze early layers (70% of features)
        total_blocks = len(self.features)
        freeze_until = int(total_blocks * 0.7)
        
        for idx, block in enumerate(self.features):
            if idx < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False
        
        # Metadata processor
        self.metadata_processor = MetadataProcessor(metadata_dim)
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, metadata):
        # Extract ConvNeXt features
        x = self.features(image)
        x = self.avgpool(x)
        x = self.layer_norm(x)
        img_features = x.flatten(1)  # Flatten to (batch, 1024)
        
        # Process metadata
        meta_features = self.metadata_processor(metadata)
        
        # Combine and classify
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(combined)


# ===========================
# TRAINING FUNCTIONS (SAME AS V2-S)
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
        
        # Gradient clipping for stability
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
    print("CONVNEXT-BASE + FEATURE ENGINEERING")
    print("="*70)
    print(f"GPU: {args.gpu}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Architecture: Different from EfficientNet → ensemble diversity!")
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
    
    # Transforms (ConvNeXt uses same ImageNet stats)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    model = ConvNeXtHybrid(metadata_dim=metadata_dim).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"ConvNeXt-Base: {total_params:,} params ({trainable_params:,} trainable)\n")
    
    # Training setup
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # ConvNeXt uses AdamW (better for transformers/modern CNNs)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    # Training loop
    best_auc = 0.0
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('results') / f'convnext_features_{timestamp}'
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
        
        print(f"  Train: Loss={train_loss:.4f}, AUC={train_auc:.4f}, Time={train_time:.1f}s")
        print(f"  Val:   Loss={val_loss:.4f}, AUC={val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
            }, results_dir / 'best_model.pth')
            print(f"  ✓ Best: {best_auc:.4f}")
        
        scheduler.step(val_auc)
        
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("  LR too small, stopping")
            break
        print()
    
    total_time = time.time() - total_start
    
    # Save results
    with open(results_dir / 'training_results.pkl', 'wb') as f:
        pickle.dump({
            'model': 'ConvNeXt-Base + Features',
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
    submission.to_csv(results_dir / 'submission_convnext_features.csv', index=False)
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {total_time/60:.1f} min")
    print(f"Best Val AUC: {best_auc:.4f}")
    print(f"Expected ensemble diversity: HIGH (different architecture)")
    print(f"Results: {results_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()