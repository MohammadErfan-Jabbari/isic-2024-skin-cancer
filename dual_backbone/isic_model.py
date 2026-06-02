"""
isic_model.py
=============
Shared model, dataset, and utility definitions for the ISIC Kaggle competition.
This file serves as the single source of truth for model architecture and data loading
to ensure consistency between training (18_1) and inference (18_3).
"""

import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image
import numpy as np
import h5py
import io

# ===========================
# CONFIGURATION
# ===========================
EVA02_MODEL = 'eva02_small_patch14_336.mim_in22k_ft_in1k'
EDGENEXT_MODEL = 'edgenext_base.in21k_ft_in1k'
EVA02_SIZE = 336
EDGENEXT_SIZE = 384

# ===========================
# MODEL COMPONENTS
# ===========================
class MetadataEncoder(nn.Module):
    """Encodes metadata features into a compact representation.
    Uses LayerNorm instead of BatchNorm for stability with small/variable batches."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # LayerNorm for stability
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # Check for NaN in input and replace with 0
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        return self.encoder(x)

class DualBackboneHybrid(nn.Module):
    """
    End-to-end trainable dual-backbone hybrid model.
    
    Architecture:
        EVA02 (336x336) ──> [384-dim] ─┐
                                       ├──> Concat ──> Fusion MLP ──> Prediction
        EdgeNeXt (384x384) ──> [584-dim] ─┤
                                       │
        Metadata ──> Encoder ──> [64-dim] ─┘
    """
    def __init__(self, metadata_dim, freeze_ratio=0.7, num_classes=1):
        super().__init__()
        
        # ===== Vision Backbones =====
        # EVA02 - ViT-based, strong on fine details
        self.eva02 = timm.create_model(
            EVA02_MODEL,
            pretrained=False, # Loaded from checkpoint usually
            num_classes=0  # Remove classifier, get embeddings
        )
        self.dim1 = self.eva02.num_features  # 384
        
        # EdgeNeXt - ConvNet, good at local patterns  
        self.edgenext = timm.create_model(
            EDGENEXT_MODEL,
            pretrained=False,
            num_classes=0
        )
        self.dim2 = self.edgenext.num_features  # 584
        
        # ===== Metadata Branch =====
        self.metadata_encoder = MetadataEncoder(metadata_dim)
        self.meta_dim = 64
        
        # ===== Fusion Head =====
        self.fusion_dim = self.dim1 + self.dim2 + self.meta_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.LayerNorm(512),  # LayerNorm for numerical stability
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize fusion head weights"""
        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        for module in self.metadata_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x1, x2, meta):
        """
        Args:
            x1: Tensor [B, 3, 336, 336] image for EVA02
            x2: Tensor [B, 3, 384, 384] image for EdgeNeXt
            meta: Tensor [B, metadata_dim] preprocessed features
        """
        # Extract vision embeddings
        f1 = self.eva02(x1)      # [B, 384]
        f2 = self.edgenext(x2)   # [B, 584]
        
        # Encode metadata
        fm = self.metadata_encoder(meta)  # [B, 64]
        
        # Concatenate all embeddings
        combined = torch.cat([f1, f2, fm], dim=1)  # [B, 1032]
        
        # Fusion and classification
        logits = self.fusion(combined)  # [B, 1]
        
        return logits

# ===========================
# DATASET CLASS
# ===========================
class DualResolutionDataset(Dataset):
    """
    Dataset that provides images at two resolutions for EVA02 and EdgeNeXt.
    Supports initialization from:
    1. DataFrame (Training): Handles filtering by HDF5 keys, synthetic data merging, and feature extraction.
    2. Features Array (Inference): Uses pre-computed features and IDs directly.
    """
    def __init__(self, hdf5_path, metadata_df=None, features=None, targets=None, ids=None,
                 transform_336=None, transform_384=None, 
                 synth_hdf5_path=None, synth_metadata_df=None, is_test=False, is_synth=None):
        self.hdf5_path = hdf5_path
        self.synth_hdf5_path = synth_hdf5_path
        self.transform_336 = transform_336
        self.transform_384 = transform_384
        self.is_test = is_test
        
        self.hdf5_file = None
        self.synth_hdf5_file = None
        
        # Mode 1: Training (DataFrame provided)
        if metadata_df is not None:
            import pandas as pd # Ensure pandas is available
            
            # Get available IDs from HDF5
            with h5py.File(hdf5_path, 'r') as f:
                available_ids = set(f.keys())
            
            # Filter metadata to only include available images
            self.metadata = metadata_df[
                metadata_df['isic_id'].isin(available_ids)
            ].copy().reset_index(drop=True)
            
            n_real = len(self.metadata)
            
            # Add synthetic data if provided
            if synth_metadata_df is not None and synth_hdf5_path is not None:
                with h5py.File(synth_hdf5_path, 'r') as f:
                    synth_ids = set(f.keys())
                
                synth_filtered = synth_metadata_df[
                    synth_metadata_df['isic_id'].isin(synth_ids)
                ].copy().reset_index(drop=True)
                
                self.metadata = pd.concat([self.metadata, synth_filtered], ignore_index=True)
            
            # Mark synthetic
            self.is_synth = np.zeros(len(self.metadata), dtype=bool)
            if synth_metadata_df is not None:
                self.is_synth[n_real:] = True
                print(f"  Real samples: {n_real}, Synthetic samples: {len(self.metadata) - n_real}")
            else:
                print(f"  Real samples: {n_real}, Synthetic samples: 0")
            
            # Extract feature columns
            feature_cols = [col for col in self.metadata.columns 
                           if col not in ['isic_id', 'target', 'patient_id']]
            self.metadata_features = self.metadata[feature_cols].values.astype(np.float32)
            
            # Store targets and IDs
            self.ids = self.metadata['isic_id'].values
            if 'target' in self.metadata.columns:
                self.targets = self.metadata['target'].values.astype(np.float32)
            else:
                self.targets = None
                
        # Mode 2: Inference (Features provided directly)
        elif features is not None and ids is not None:
            self.metadata_features = features
            self.ids = ids
            self.targets = targets
            self.is_synth = is_synth if is_synth is not None else np.zeros(len(features), dtype=bool)
            
        else:
            raise ValueError("Must provide either 'metadata_df' (training) or 'features' + 'ids' (inference)")

    def _ensure_open(self):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
        if self.synth_hdf5_path is not None and self.synth_hdf5_file is None:
            self.synth_hdf5_file = h5py.File(self.synth_hdf5_path, 'r', swmr=True)

    def __len__(self):
        return len(self.metadata_features)
        
    def __getitem__(self, idx):
        self._ensure_open()
        
        img_id = self.ids[idx]
        is_synthetic = self.is_synth[idx]
        
        # Load image from appropriate HDF5
        try:
            if is_synthetic and self.synth_hdf5_file is not None:
                img_array = self.synth_hdf5_file[img_id][:]
            else:
                img_array = self.hdf5_file[img_id][:]
            
            # CRITICAL: Convert to RGB to ensure 3 channels for Normalize
            image = Image.fromarray(img_array).convert('RGB')
        except Exception as e:
            # Return dummy data
            image = Image.new('RGB', (384, 384), color='black')
        
        # Apply transforms for each resolution
        img_336 = self.transform_336(image) if self.transform_336 else image
        img_384 = self.transform_384(image) if self.transform_384 else image
        
        # Get metadata features
        metadata = torch.tensor(self.metadata_features[idx], dtype=torch.float32)
        
        if self.is_test:
            return img_336, img_384, metadata, img_id
        else:
            target = self.targets[idx]
            return img_336, img_384, metadata, target


# ===========================
# SAMPLER
# ===========================
class BalancedBatchSampler(Sampler):
    """
    Ensures each batch has ~50% positive and ~50% negative samples.
    Essential for extreme class imbalance (0.085% malignant).
    """
    def __init__(self, dataset, batch_size, length=None):
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.pos_indices = np.where(dataset.targets == 1)[0]
        self.neg_indices = np.where(dataset.targets == 0)[0]
        
        self.n_pos = batch_size // 2
        self.n_neg = batch_size - self.n_pos
        
        if length is None:
            # One epoch = seeing approximately all negatives once
            self.length = len(self.neg_indices) // self.n_neg
        else:
            self.length = length
            
    def __iter__(self):
        pos_pointer = 0
        neg_pointer = 0
        
        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)
        
        for _ in range(self.length):
            batch_indices = []
            
            # Sample positives (with replacement if needed)
            for _ in range(self.n_pos):
                if pos_pointer >= len(self.pos_indices):
                    np.random.shuffle(self.pos_indices)
                    pos_pointer = 0
                batch_indices.append(self.pos_indices[pos_pointer])
                pos_pointer += 1
            
            # Sample negatives
            for _ in range(self.n_neg):
                if neg_pointer >= len(self.neg_indices):
                    np.random.shuffle(self.neg_indices)
                    neg_pointer = 0
                batch_indices.append(self.neg_indices[neg_pointer])
                neg_pointer += 1
            
            # Shuffle within batch
            np.random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        return self.length

# ===========================
# UTILITY FUNCTIONS
# ===========================
def compute_patient_statistics(df, patient_relative_features):
    """
    Compute patient statistics (mean, std, count) for specific features.
    Used for 'Ugly Duckling' feature computation.
    """
    import pandas as pd # Ensure pandas is available
    
    print("  Computing patient statistics...")
    global_stats = {}
    for feat in patient_relative_features:
        if feat in df.columns:
            global_stats[f'{feat}_median'] = df[feat].median()
            global_stats[f'{feat}_std'] = df[feat].std()
            global_stats[f'{feat}_mean'] = df[feat].mean()
    
    patient_stats_list = []
    for feat in patient_relative_features:
        if feat not in df.columns:
            continue
        # Compute stats
        patient_agg = df.groupby('patient_id')[feat].agg(['mean', 'std', 'count'])
        patient_agg.columns = [f'{feat}_mean', f'{feat}_std', f'{feat}_count']
        
        # Fill missing std with global std (for single-lesion patients or constant values)
        # Note: std is NaN if count=1.
        patient_agg[f'{feat}_std'] = patient_agg[f'{feat}_std'].fillna(global_stats.get(f'{feat}_std', 1))
        patient_agg[f'{feat}_std'] = patient_agg[f'{feat}_std'].replace(0, global_stats.get(f'{feat}_std', 1))
        
        patient_stats_list.append(patient_agg)
    
    if not patient_stats_list:
        return {'patient_stats': pd.DataFrame(), 'global_stats': global_stats}
        
    patient_stats = pd.concat(patient_stats_list, axis=1)
    patient_stats = patient_stats.reset_index()
    
    # Add total lesion count
    patient_counts = df.groupby('patient_id').size().reset_index(name='lesion_count')
    patient_stats = patient_stats.merge(patient_counts, on='patient_id')
    
    return {
        'patient_stats': patient_stats,
        'global_stats': global_stats
    }

