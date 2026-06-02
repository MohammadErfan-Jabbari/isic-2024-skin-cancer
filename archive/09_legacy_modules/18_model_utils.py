"""
18_model_utils.py
=================
Shared model and dataset definitions for the Dual Backbone Hybrid approach.
Extracted from 18_1_train_dual_backbone_hybrid.py to ensure consistency.
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
    def __init__(self, metadata_dim, freeze_ratio=0.7):
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
            nn.Linear(128, 1)
        )
        
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
    def __init__(self, hdf5_path, metadata, targets=None, is_test=False, 
                 transform_336=None, transform_384=None, ids=None, is_synth=None, synth_hdf5_path=None):
        self.hdf5_path = hdf5_path
        self.metadata_features = metadata
        self.targets = targets
        self.is_test = is_test
        self.transform_336 = transform_336
        self.transform_384 = transform_384
        self.ids = ids
        self.is_synth = is_synth if is_synth is not None else np.zeros(len(metadata), dtype=bool)
        self.synth_hdf5_path = synth_hdf5_path
        
        self.hdf5_file = None
        self.synth_hdf5_file = None
        
    def _ensure_open(self):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
        if self.synth_hdf5_path and self.synth_hdf5_file is None:
            try:
                self.synth_hdf5_file = h5py.File(self.synth_hdf5_path, 'r', swmr=True)
            except:
                pass

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
