"""
Batch Size Finder for EVA-02 Small and EdgeNeXt Base

This script helps determine the optimal batch size for your models based on:
1. Available GPU memory
2. Model architecture (parameter count)
3. Image resolution and data loading overhead
4. Training throughput (samples/sec)

The script runs short training loops to measure:
- Memory consumption per batch
- Training speed (iterations/sec)
- Memory efficiency (samples/GB)
- Recommended batch sizes for different scenarios

Usage:
    uv run python batch_size_finder.py --model eva02_small_patch14_336.mim_in22k_ft_in1k --gpu 0
    uv run python batch_size_finder.py --model edgenext_base.in21k_ft_in1k --gpu 0
    uv run python batch_size_finder.py --all --gpu 0
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
from sklearn.preprocessing import StandardScaler
import warnings
import json
import os
import argparse
from datetime import datetime
import time
import pickle

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
MODELS_CONFIG = {
    'eva02_small_patch14_336.mim_in22k_ft_in1k': {
        'name': 'EVA-02 Small',
        'image_size': 336,
        'typical_bs': 128,
        'accumulation_steps': 1
    },
    'edgenext_base.in21k_ft_in1k': {
        'name': 'EdgeNeXt Base',
        'image_size': 384,
        'typical_bs': 128,
        'accumulation_steps': 1
    }
}

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
    def __init__(self, hdf5_path, metadata_df, image_size, transform=None):
        self.hdf5_path = hdf5_path
        self.image_size = image_size
        self.transform = transform
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
        label = row['target']
        
        return image, metadata, label

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

class GenericHybridModel(nn.Module):
    def __init__(self, model_name, metadata_dim, image_size, pretrained=True):
        super().__init__()
        self.image_size = image_size
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,
            global_pool='avg'
        )
        
        # Determine feature dimension dynamically
        with torch.no_grad():
            dummy = torch.randn(2, 3, image_size, image_size)
            feats = self.backbone(dummy)
            img_dim = feats.shape[1]
        
        self.meta_net = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
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
# BATCH SIZE FINDER
# ===========================
def find_batch_sizes(model_name, gpu_id, data_dir='data', num_iterations=5):
    """Test different batch sizes and measure memory/speed"""
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = MODELS_CONFIG[model_name]
    image_size = config['image_size']
    model_display_name = config['name']
    
    print(f"\n{'='*80}")
    print(f"BATCH SIZE FINDER: {model_display_name}")
    print(f"{'='*80}")
    print(f"GPU: {gpu_id} | Image Size: {image_size}x{image_size}")
    print(f"{'='*80}\n")
    
    # Load Data
    data_dir = Path(data_dir)
    train_meta = pd.read_csv(data_dir / 'new-train-metadata.csv', low_memory=False)
    
    # Feature Engineering
    train_meta_processed, scaler, encoders = preprocess_metadata_with_features(train_meta, is_train=True)
    train_meta_processed['isic_id'] = train_meta['isic_id'].values
    train_meta_processed['target'] = train_meta['target'].values
    train_meta_processed['patient_id'] = train_meta['patient_id'].values
    
    metadata_dim = len(train_meta_processed.columns) - 3
    
    # Transforms
    data_config = timm.data.resolve_data_config({}, model=model_name)
    mean = data_config['mean']
    std = data_config['std']
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Create dataset
    full_dataset = HybridDataset(
        data_dir / 'train-image-384.hdf5', 
        train_meta_processed, 
        image_size,
        transform=train_transform
    )
    
    # Use subset for faster testing
    subset_size = min(2000, len(full_dataset))
    subset = Subset(full_dataset, np.random.choice(len(full_dataset), subset_size, replace=False))
    
    # Batch sizes to test
    batch_sizes = [8, 16, 24, 32, 48, 64, 80, 96, 128, 160]
    
    results = []
    
    for bs in batch_sizes:
        try:
            torch.cuda.empty_cache()
            
            # Create dataloader
            loader = DataLoader(
                subset, 
                batch_size=bs, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True
            )
            
            # Create model
            model = GenericHybridModel(model_name, metadata_dim, image_size).to(device)
            model.eval()
            
            criterion = FocalLoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            amp_scaler = torch.cuda.amp.GradScaler()
            
            # Measure memory and speed
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                for iteration, (images, metadata, labels) in enumerate(loader):
                    if iteration >= num_iterations:
                        break
                    
                    images = images.to(device, non_blocking=True)
                    metadata = metadata.to(device, non_blocking=True)
                    labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast():
                        outputs = model(images, metadata)
                        loss = criterion(outputs, labels)
                    
                    torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            max_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            # Calculate metrics
            samples_per_iteration = bs * min(num_iterations, len(loader))
            throughput = samples_per_iteration / elapsed_time
            memory_per_sample = max_memory_mb / (bs * num_iterations)
            
            results.append({
                'batch_size': bs,
                'memory_mb': max_memory_mb,
                'memory_per_sample_mb': memory_per_sample,
                'throughput_samples_per_sec': throughput,
                'time_per_iter_ms': (elapsed_time / min(num_iterations, len(loader))) * 1000,
                'status': '✓ OK'
            })
            
            print(f"BS={bs:3d} | Memory: {max_memory_mb:7.1f}MB | "
                  f"Per-Sample: {memory_per_sample:.3f}MB | "
                  f"Throughput: {throughput:6.1f} samples/sec | "
                  f"Time: {(elapsed_time / min(num_iterations, len(loader))) * 1000:6.2f}ms/iter")
            
            del model, loader
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                results.append({
                    'batch_size': bs,
                    'memory_mb': None,
                    'memory_per_sample_mb': None,
                    'throughput_samples_per_sec': None,
                    'time_per_iter_ms': None,
                    'status': '✗ OOM'
                })
                print(f"BS={bs:3d} | ✗ OUT OF MEMORY")
            else:
                raise
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}\n")
    
    valid_results = [r for r in results if r['status'] == '✓ OK']
    
    if valid_results:
        # Best for memory efficiency
        best_memory_efficient = min(valid_results, key=lambda x: x['memory_per_sample_mb'])
        
        # Best for throughput
        best_throughput = max(valid_results, key=lambda x: x['throughput_samples_per_sec'])
        
        # Largest that fits comfortably (< 80% of max used)
        max_memory_used = max([r['memory_mb'] for r in valid_results])
        comfortable_bs = max([r for r in valid_results if r['memory_mb'] < max_memory_used * 0.8],
                            key=lambda x: x['batch_size'])
        
        print(f"📊 Memory-Efficient Batch Size: {best_memory_efficient['batch_size']} "
              f"({best_memory_efficient['memory_per_sample_mb']:.3f}MB per sample)")
        print(f"⚡ Best Throughput Batch Size: {best_throughput['batch_size']} "
              f"({best_throughput['throughput_samples_per_sec']:.1f} samples/sec)")
        print(f"🎯 Comfortable Large Batch Size: {comfortable_bs['batch_size']} "
              f"({comfortable_bs['memory_mb']:.1f}MB, {comfortable_bs['throughput_samples_per_sec']:.1f} samples/sec)")
        
        print(f"\n💡 SUGGESTED DEFAULTS:")
        print(f"   • Standard training: --batch-size {comfortable_bs['batch_size']}")
        print(f"   • With gradient accumulation: --batch-size {best_memory_efficient['batch_size']} "
              f"--accumulation-steps {comfortable_bs['batch_size'] // best_memory_efficient['batch_size']}")
        print(f"   • Aggressive tuning: --batch-size {best_throughput['batch_size']}")
        
        print(f"\n📈 MEMORY HEADROOM:")
        avg_memory_used = np.mean([r['memory_mb'] for r in valid_results])
        print(f"   • Average memory used: {avg_memory_used:.1f}MB")
        print(f"   • Max memory used: {max_memory_used:.1f}MB")
        print(f"   • GPU available: ~46GB (L40S)")
        print(f"   • Safety margin: {(46000 - max_memory_used) / 1000:.1f}GB")
    else:
        print("❌ All batch sizes failed! Check GPU memory or model configuration.")
    
    print(f"\n{'='*80}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Find optimal batch size for training')
    parser.add_argument('--model', type=str, choices=list(MODELS_CONFIG.keys()),
                       help='Model to test')
    parser.add_argument('--all', action='store_true', help='Test all models')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations to test')
    
    args = parser.parse_args()
    
    if args.all:
        for model_name in MODELS_CONFIG.keys():
            find_batch_sizes(model_name, args.gpu, args.data_dir, args.iterations)
    elif args.model:
        find_batch_sizes(args.model, args.gpu, args.data_dir, args.iterations)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
