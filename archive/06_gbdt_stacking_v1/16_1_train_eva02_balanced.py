import sys
import os
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import timm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path
import io
from PIL import Image
from datetime import datetime
import json
import pickle
from torch.amp import GradScaler, autocast

# ===========================
# CONFIGURATION
# ===========================
# The winner used this specific EVA02 checkpoint
MODEL_NAME = 'eva02_small_patch14_336.mim_in22k_ft_in1k'
IMAGE_SIZE = 336  # Native resolution for this model
NUM_CLASSES = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1, help='Fold number (1-5)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (Increased for L40S)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='DeepLearning/Kaggle/data', help='Real Data directory')
    parser.add_argument('--synth-dir', type=str, default='DeepLearning/Kaggle/generative/data', help='Synthetic Data directory')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name for grouping folds')
    return parser.parse_args()

# ===========================
# DATASET
# ===========================
class ISICImageDataset(Dataset):
    def __init__(self, hdf5_path, metadata_df, transform=None, synth_hdf5_path=None, synth_metadata_df=None):
        self.hdf5_path = hdf5_path
        self.metadata = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.hdf5_file = None
        
        # Synthetic Data Support
        self.synth_hdf5_path = synth_hdf5_path
        self.synth_metadata = synth_metadata_df
        self.synth_hdf5_file = None
        
        # Combine Real and Synthetic indices if provided
        self.is_synth = np.zeros(len(self.metadata), dtype=bool)
        
        if self.synth_metadata is not None:
            # Append synthetic metadata
            self.synth_metadata = self.synth_metadata.reset_index(drop=True)
            self.is_synth = np.concatenate([self.is_synth, np.ones(len(self.synth_metadata), dtype=bool)])
            self.metadata = pd.concat([self.metadata, self.synth_metadata], ignore_index=True)
            
        self.ids = self.metadata['isic_id'].values
        self.targets = self.metadata['target'].values.astype(np.float32)

    def _ensure_open(self):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
        if self.synth_hdf5_path is not None and self.synth_hdf5_file is None:
            self.synth_hdf5_file = h5py.File(self.synth_hdf5_path, 'r', swmr=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        self._ensure_open()
        
        img_id = self.ids[idx]
        is_synthetic = self.is_synth[idx]
        
        try:
            if is_synthetic:
                # Load from Synthetic HDF5
                img_data = self.synth_hdf5_file[img_id][:]
            else:
                # Load from Real HDF5
                img_data = self.hdf5_file[img_id][:]
            
            # Handle bytes vs numpy array
            if isinstance(img_data, np.ndarray) and img_data.ndim == 3:
                image = Image.fromarray(img_data)
            else:
                image = Image.open(io.BytesIO(img_data))
                
        except Exception as e:
            print(f"Error loading {img_id}: {e}")
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(self.targets[idx])

# ===========================
# BALANCED SAMPLER (The Secret Sauce)
# ===========================
class BalancedBatchSampler(Sampler):
    """
    Ensures each batch has 50% positive and 50% negative samples.
    """
    def __init__(self, dataset, batch_size, length=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_indices = np.where(dataset.targets == 1)[0]
        self.neg_indices = np.where(dataset.targets == 0)[0]
        
        self.n_pos = int(batch_size / 2)
        self.n_neg = batch_size - self.n_pos
        
        # If length is not provided, iterate over all negatives (very long epoch)
        # If length is provided (e.g. standard epoch size), we only iterate that many batches
        if length is None:
            self.length = len(self.neg_indices) // self.n_neg
        else:
            self.length = length

    def __iter__(self):
        # Shuffle indices at the start of each epoch
        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)
        
        # We cycle through positives because there are fewer of them
        pos_pointer = 0
        
        for i in range(self.length):
            # Select negatives for this batch
            # If we run out of negatives in this epoch (because length > negs/n_neg), we cycle
            # But typically length <= negs/n_neg, so we just take a slice
            
            start_idx = (i * self.n_neg) % len(self.neg_indices)
            end_idx = start_idx + self.n_neg
            
            if end_idx > len(self.neg_indices):
                # Wrap around if needed (rare if length is set correctly)
                batch_neg = np.concatenate([
                    self.neg_indices[start_idx:],
                    self.neg_indices[:end_idx - len(self.neg_indices)]
                ])
            else:
                batch_neg = self.neg_indices[start_idx:end_idx]
            
            # Select positives (cycling if needed)
            if pos_pointer + self.n_pos > len(self.pos_indices):
                # Reset pointer and reshuffle if we run out
                np.random.shuffle(self.pos_indices)
                pos_pointer = 0
                
            batch_pos = self.pos_indices[pos_pointer : pos_pointer + self.n_pos]
            pos_pointer += self.n_pos
            
            # Combine and shuffle within the batch
            batch = np.concatenate([batch_neg, batch_pos])
            np.random.shuffle(batch)
            
            yield batch.astype(int)

    def __len__(self):
        return self.length

# ===========================
# MODEL
# ===========================
class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes,
            drop_rate=0.2,
            drop_path_rate=0.1
        )
        
    def forward(self, x):
        return self.model(x)

# ===========================
# TRAINING LOOP
# ===========================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc="Train", ncols=80)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Use AMP
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / len(loader)
    epoch_auc = roc_auc_score(all_targets, all_preds)
    return epoch_loss, epoch_auc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Val", ncols=80):
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    epoch_loss = running_loss / len(loader)
    epoch_auc = roc_auc_score(all_targets, all_preds)
    return epoch_loss, epoch_auc, all_preds

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Training Fold {args.fold} | Model: {MODEL_NAME}")
    print(f"🖼️  Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    # 1. Load Metadata
    data_dir = Path(args.data_dir)
    synth_dir = Path(args.synth_dir)
    
    df = pd.read_csv(data_dir / 'new-train-metadata.csv', low_memory=False)
    synth_df = pd.read_csv(synth_dir / 'synthetic_malignant_metadata.csv')
    
    # Ensure synthetic targets are 1
    synth_df['target'] = 1
    
    # 2. Split (StratifiedGroupKFold)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(df, df['target'], df['patient_id']))
    train_idx, val_idx = splits[args.fold - 1]
    
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    print(f"📊 Fold {args.fold}: {len(train_df)} Train (Real), {len(val_df)} Val")
    print(f"🧪 Injecting {len(synth_df)} Synthetic Malignant Samples")
    
    # 3. Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Datasets & Loaders
    # Note: We use train-image-384.hdf5 if available, else preprocessed
    real_hdf5 = data_dir / 'train-image-384.hdf5'
    if not real_hdf5.exists():
        print("⚠️ 384px HDF5 not found, falling back to preprocessed...")
        real_hdf5 = data_dir / 'train-image-preprocessed.hdf5'
        
    synth_hdf5 = synth_dir / 'synthetic_malignant_384.hdf5'
    
    # Train Dataset (Real + Synthetic)
    train_ds = ISICImageDataset(
        real_hdf5, train_df, transform=train_transform,
        synth_hdf5_path=synth_hdf5, synth_metadata_df=synth_df
    )
    
    # Val Dataset (Real Only)
    val_ds = ISICImageDataset(real_hdf5, val_df, transform=val_transform)
    
    # Sampler
    # We define an epoch as "seeing the equivalent of 1 full pass of the training set"
    # But since we oversample positives, we actually see fewer unique negatives per epoch
    # This keeps epoch time reasonable (~8-10 mins) while maintaining 50/50 balance
    samples_per_epoch = len(train_ds)
    batches_per_epoch = samples_per_epoch // args.batch_size
    sampler = BalancedBatchSampler(train_ds, batch_size=args.batch_size, length=batches_per_epoch)
    
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # 5. Model Setup
    model = ISICModel(MODEL_NAME).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    
    # Initialize Scaler for AMP
    scaler = GradScaler('cuda')
    
    # 6. Training Loop
    
    # Fix: Use absolute path relative to this script for results
    # This ensures consistency with other scripts in DeepLearning/Kaggle/results
    script_dir = Path(__file__).parent
    if args.experiment_name:
        save_dir = script_dir / "results" / args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = script_dir / "results" / f"eva02_balanced_{timestamp}"
        
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Config
    config = vars(args)
    config['model_name'] = MODEL_NAME
    config['image_size'] = IMAGE_SIZE
    config['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(save_dir / f'config_fold{args.fold}.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    best_auc = 0.0
    history = {
        'train_loss': [], 'train_auc': [],
        'val_loss': [], 'val_auc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        t_loss, t_auc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Adaptive Validation: Validate less frequently at start if needed, but for now every epoch is fine
        v_loss, v_auc, v_preds = validate(model, val_loader, criterion, device)
        
        print(f"Train: Loss={t_loss:.4f} AUC={t_auc:.4f}")
        print(f"Val:   Loss={v_loss:.4f} AUC={v_auc:.4f}")
        
        # Update History
        history['train_loss'].append(t_loss)
        history['train_auc'].append(t_auc)
        history['val_loss'].append(v_loss)
        history['val_auc'].append(v_auc)
        
        scheduler.step()
        
        if v_auc > best_auc:
            best_auc = v_auc
            torch.save(model.state_dict(), save_dir / f"best_model_fold{args.fold}.pth")
            
            # Save OOF Predictions for the best model (Critical for Stacking)
            # We rely on shuffle=False in val_loader to match indices with val_df
            oof_df = val_df.copy()
            oof_df['pred'] = v_preds
            oof_df.to_csv(save_dir / f"oof_fold{args.fold}.csv", index=False)
            
            print(f"⭐ New Best AUC: {best_auc:.4f} | Saved OOF predictions")
            
        # Save History at every epoch
        with open(save_dir / f'training_results_fold{args.fold}.pkl', 'wb') as f:
            pickle.dump(history, f)
            
    print(f"✅ Training Complete. Best AUC: {best_auc:.4f}")

if __name__ == '__main__':
    main()
