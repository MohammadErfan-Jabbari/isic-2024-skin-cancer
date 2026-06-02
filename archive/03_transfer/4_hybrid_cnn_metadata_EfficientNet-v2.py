# %%
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
from torchvision.models import efficientnet_v2_s
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
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():    
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}\n")


# %% [markdown]
# ## 1. Metadata Preprocessing (Same as baseline)

# %%
# Load metadata
data_dir = Path('data')
train_meta = pd.read_csv(data_dir / 'new-train-metadata.csv', low_memory=False)
test_meta = pd.read_csv(data_dir / 'students-test-metadata.csv', low_memory=False)

print("Metadata loaded:")
print(f"  Train: {len(train_meta):,} samples")
print(f"  Test: {len(test_meta):,} samples\n")

# %%
# Define features (same as baseline)
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

# %%
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

train_meta_processed, scaler, encoders = preprocess_metadata(train_meta, is_train=True)
test_meta_processed, _, _ = preprocess_metadata(test_meta, is_train=False, scaler=scaler, encoders=encoders)

train_meta_processed['isic_id'] = train_meta['isic_id'].values
train_meta_processed['target'] = train_meta['target'].values
test_meta_processed['isic_id'] = test_meta['isic_id'].values

metadata_dim = len(train_meta_processed.columns) - 2
print(f"Metadata dimension: {metadata_dim}\n")

# %% [markdown]
# ## 2. Hybrid Dataset (Same as baseline)

# %%
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
        
        print(f"✓ {len(self.metadata)} samples from {Path(hdf5_path).name}")
        
        if not is_test and 'target' in self.metadata.columns:
            print(f"  Distribution: {self.metadata['target'].value_counts().to_dict()}")
    
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

# %%
# Transforms
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

# Split
train_df, val_df = train_test_split(
    train_meta_processed, test_size=0.2, random_state=42,
    stratify=train_meta_processed['target']
)

print(f"Split: {len(train_df):,} train / {len(val_df):,} val\n")

# Create datasets
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
BATCH_SIZE = 256  # Smaller due to larger model
NUM_WORKERS = 16

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

print(f"DataLoaders: {BATCH_SIZE} batch, {NUM_WORKERS} workers")
print(f"Batches: {len(train_loader)} train, {len(val_loader)} val\n")

# %% [markdown]
# ## 3. Focal Loss (Same as baseline)

# %%
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

# %% [markdown]
# ## 4. EfficientNetV2-S Hybrid Model ⭐ UPGRADED!
# 

# %%
class MetadataProcessor(nn.Module):
    """Same metadata processor as baseline"""
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
    """
    Hybrid model with EfficientNetV2-S backbone
    
    Architecture:
      - EfficientNetV2-S (pretrained) → 1280 features
      - Metadata MLP → 64 features
      - Concatenate → 1344 features
      - Classifier → 1 output
    
    Key improvements over EfficientNet-B0:
    - Better ImageNet accuracy: 84.3% vs 77.1%
    - 2x faster training (Fused-MBConv blocks)
    - Better transfer learning (progressive training)
    - Still outputs 1280 features (same as B0)
    """
    def __init__(self, metadata_dim):
        super().__init__()
        
        # Load pretrained EfficientNetV2-S
        self.efficientnet = efficientnet_v2_s(weights='IMAGENET1K_V1')
        
        # Remove original classifier to use as feature extractor
        self.efficientnet.classifier = nn.Identity()
        
        # Freeze early layers (80%)
        total_params = len(list(self.efficientnet.parameters()))
        freeze_until = int(total_params * 0.8)
        
        for idx, param in enumerate(self.efficientnet.parameters()):
            if idx < freeze_until:
                param.requires_grad = False
        
        # Metadata processor
        self.metadata_processor = MetadataProcessor(metadata_dim)
        
        # Combined classifier
        # EfficientNetV2-S outputs 1280 features (same as B0)
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
        # Extract image features
        img_features = self.efficientnet(image)
        
        # Process metadata
        meta_features = self.metadata_processor(metadata)
        
        # Combine and classify
        combined = torch.cat([img_features, meta_features], dim=1)
        output = self.classifier(combined)
        return output


# Create model
model = EfficientNetV2Hybrid(metadata_dim=metadata_dim).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("="*70)
print("EFFICIENTNETV2-S HYBRID MODEL")
print("="*70)
print(f"Backbone: EfficientNetV2-S (ImageNet pretrained)")
print(f"ImageNet Top-1: 84.3% (vs 77.1% for EfficientNet-B0)")
print(f"Training speed: 2-4x faster than EfficientNet-B0")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")
print(f"Image features: 1280 (from EfficientNetV2-S)")
print(f"Metadata features: 64 (from MLP)")
print(f"Combined: 1344 → Classifier → 1 output")
print("="*70 + "\n")


# %% [markdown]
# ## 5. Training Setup (Same as baseline)

# %%
criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Lower LR for pretrained

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5
)

print("Training setup:")
print(f"  Loss: Focal Loss (alpha=0.25, gamma=2.0)")
print(f"  Optimizer: Adam (lr=0.0005, weight_decay=1e-5)")
print(f"  Scheduler: ReduceLROnPlateau (patience=5)\n")

# %% [markdown]
# ## 6. Training Functions (Same as baseline)

# %%
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

# %% [markdown]
# ## 7. Training Loop

# %%
NUM_EPOCHS = 25
best_auc = 0.0
history = {
    'train_loss': [], 'train_auc': [], 'train_time': [],
    'val_loss': [], 'val_auc': []
}

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = Path('results') / f'efficientnet_v2_hybrid_{timestamp}'
results_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("STARTING TRAINING - EFFICIENTNETV2-S HYBRID")
print("="*70)
print(f"Results: {results_dir}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Expected improvement: 0.927 → 0.95-0.96 (+0.02-0.03 AUC)")
print(f"Expected speed: 2x faster training than EfficientNet-B0")
print("="*70 + "\n")

total_start = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*70}")
    
    train_loss, train_auc, train_time = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    val_loss, val_auc, val_preds, val_labels = validate(
        model, val_loader, criterion, device
    )
    
    history['train_loss'].append(train_loss)
    history['train_auc'].append(train_auc)
    history['train_time'].append(train_time)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(val_auc)
    
    print(f"\nResults:")
    print(f"  Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Time: {train_time:.1f}s")
    print(f"  Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
    
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
        }, results_dir / 'best_efficientnet_v2_hybrid.pth')
        print(f"  ✓ Saved best model (AUC: {best_auc:.4f})")
    
    scheduler.step(val_auc)
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    if optimizer.param_groups[0]['lr'] < 1e-6:
        print(f"\n  LR too small, stopping...")
        break

total_time = time.time() - total_start

# Save results
with open(results_dir / 'training_results.pkl', 'wb') as f:
    pickle.dump({
        'timestamp': timestamp,
        'model': 'EfficientNetV2-S Hybrid',
        'best_auc': best_auc,
        'history': history,
        'total_time': total_time,
        'batch_size': BATCH_SIZE,
        'metadata_dim': metadata_dim,
        'total_params': total_params,
        'trainable_params': trainable_params,
    }, f)

with open(results_dir / 'preprocessors.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler, 'encoders': encoders}, f)

print(f"\n{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Best validation AUC: {best_auc:.4f}")
print(f"Improvement vs baseline: {best_auc - 0.9365:.4f}")
print(f"Improvement vs simple CNN: {best_auc - 0.51:.4f}")
print(f"\n✓ Results saved to: {results_dir}")
print(f"{'='*70}\n")


# %% [markdown]
# ## 8. Evaluation & Predictions

# %%
# --- Load previous run if available ---
import os

# Set your results directory here to load a previous run, or leave as None to rerun training
load_results_dir = "results/efficientnet_v2_hybrid_20251107_182005"  # e.g., Path('results/efficientnet_v2_hybrid_20251107_182005')

if load_results_dir is not None and os.path.exists(load_results_dir):
    print(f"Loading previous run from: {load_results_dir}")
    # Load model checkpoint
    checkpoint = torch.load(Path(load_results_dir) / 'best_efficientnet_v2_hybrid.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_auc = checkpoint.get('val_auc', None)

    # Load training history
    with open(Path(load_results_dir) / 'training_results.pkl', 'rb') as f:
        training_results = pickle.load(f)
        history = training_results['history']
        total_params = training_results.get('total_params', None)
        trainable_params = training_results.get('trainable_params', None)

    # Load preprocessors (scaler, encoders)
    with open(Path(load_results_dir) / 'preprocessors.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
        scaler = preprocessors['scaler']
        encoders = preprocessors['encoders']

    results_dir = Path(load_results_dir)
else:
    print("No previous run loaded. Will use current training results.")

# %%
# ROC curve
checkpoint = torch.load(results_dir / 'best_efficientnet_v2_hybrid.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
_, _, val_preds, val_labels = validate(model, val_loader, criterion, device)

fpr, tpr, thresholds = roc_curve(val_labels, val_preds)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {best_auc:.4f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'go', markersize=12, 
         label=f'Threshold = {optimal_threshold:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - EfficientNetV2-S Hybrid')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(results_dir / 'roc_curve.png', dpi=150)
print(f"✓ ROC curve saved\n")
plt.show()

# Training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history['train_loss'], label='Train', marker='o')
axes[0].plot(history['val_loss'], label='Val', marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_auc'], label='Train', marker='o')
axes[1].plot(history['val_auc'], label='Val', marker='o')
axes[1].axhline(y=best_auc, color='r', linestyle='--', label=f'Best: {best_auc:.4f}')
axes[1].axhline(y=0.9365, color='gray', linestyle=':', label='Baseline: 0.9365')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC')
axes[1].set_title('AUC')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'training_history.png', dpi=150)
print(f"✓ Training plots saved\n")
plt.show()

# Generate test predictions
print("Generating test predictions...")
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
submission.to_csv(results_dir / 'submission_efficientnet_v2_hybrid.csv', index=False)

print(f"\n{'='*70}")
print("SUBMISSION GENERATED")
print(f"{'='*70}")
print(f"File: submission_efficientnet_v2_hybrid.csv")
print(f"Shape: {submission.shape}")
print(f"\nPrediction statistics:")
print(submission['target'].describe())
print(f"{'='*70}\n")

# %% [markdown]
# ## 9. Comparison Summary

# %%
print("="*70)
print("MODEL COMPARISON")
print("="*70)
print("\nBaseline (Custom CNN + Metadata):")
print(f"  Val AUC: 0.9365")
print(f"  Parameters: ~27M")
print(f"  Training: From scratch")

print(f"\nEfficientNetV2-S Hybrid:")
print(f"  Val AUC: {best_auc:.4f}")
print(f"  Parameters: {total_params:,} ({trainable_params:,} trainable)")
print(f"  Training: Transfer learning (ImageNet pretrained, 84.3% accuracy)")
print(f"  Speed: 2x faster than EfficientNet-B0")
print(f"  Improvement: {best_auc - 0.9365:+.4f} ({(best_auc - 0.9365)/0.9365*100:+.1f}%)")

if best_auc > 0.95:
    print(f"\n🎉 Excellent! Likely 2nd place on leaderboard!")
elif best_auc > 0.94:
    print(f"\n✓ Good improvement! Competing for 2nd-3rd place")
else:
    print(f"\n⚠ Modest improvement. May need more training or ensemble")

print("="*70)



