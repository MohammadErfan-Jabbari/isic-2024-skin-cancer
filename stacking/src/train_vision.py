import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import timm
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import ISICDataset

# Config
DATA_DIR = Path('./data')
LAST_RUN_DIR = Path('./last_run')
FOLDS_PATH = LAST_RUN_DIR / 'data/folds.csv'
RESULTS_DIR = LAST_RUN_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transforms(img_size=336):
    return {
        'train': A.Compose([
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(std_range=(0.01, 0.05)),
            ], p=0.7),
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2()
        ]),
        'val': A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2()
        ])
    }

class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.in_features = self.backbone.num_features
        
        # Linear Head (as decided in Research Phase)
        self.head = nn.Linear(self.in_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x) # (Batch, Embed_Dim)
        logits = self.head(features)
        return logits, features
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler('cuda')
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits, _ = model(imgs)
            loss = criterion(logits, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds = []
    targets_list = []
    embeddings_list = []
    
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Val", leave=False):
            imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
            
            with torch.amp.autocast('cuda'):
                logits, features = model(imgs)
                loss = criterion(logits, targets)
            
            total_loss += loss.item()
            preds.extend(logits.sigmoid().cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            embeddings_list.extend(features.cpu().numpy())
            
    preds = np.array(preds)
    targets_list = np.array(targets_list)
    try:
        auc = roc_auc_score(targets_list, preds)
        # pAUC (TPR at FPR=0.8 is not standard pAUC, usually max FPR=0.01 or something)
        # Competition metric is pAUC > 0.8 TPR? No, it's pAUC above 80% TPR.
        # Implementation of pAUC is complex, using simple AUC for monitoring.
    except:
        auc = 0.5
        
    return total_loss / len(loader), auc, preds, np.array(embeddings_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--model', type=str, default='eva02_small_patch14_336.mim_in22k_ft_in1k')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--debug', action='store_true', help="Run on small subset for debugging")
    args = parser.parse_args()
    
    print(f"Training Fold {args.fold} with {args.model} (Debug={args.debug})")
    
    # Save Config
    config_path = RESULTS_DIR / f"config_{args.model}_fold{args.fold}.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load Folds
    df = pd.read_csv(FOLDS_PATH, low_memory=False)
    train_df = df[df['fold'] != args.fold].reset_index(drop=True)
    val_df = df[df['fold'] == args.fold].reset_index(drop=True)
    
    if args.debug:
        print("!! DEBUG MODE: Using 5% of data !!")
        train_df = train_df.sample(frac=0.05, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(frac=0.05, random_state=42).reset_index(drop=True)
    
    # Sampler for Class Imbalance
    # Calculate weights
    neg_count = (train_df['target'] == 0).sum()
    pos_count = (train_df['target'] == 1).sum()
    weight_for_0 = 1.0 / neg_count
    weight_for_1 = 1.0 / pos_count
    samples_weight = train_df['target'].map({0: weight_for_0, 1: weight_for_1}).values
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    # Datasets
    transforms = get_transforms(img_size=336) # EVA02-Small uses 336
    train_ds = ISICDataset(DATA_DIR / 'train-image-384.hdf5', train_df, transform=transforms['train'])
    val_ds = ISICDataset(DATA_DIR / 'train-image-384.hdf5', val_df, transform=transforms['val'])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = ISICModel(args.model).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_auc = 0
    history = []
    
    for epoch in range(args.epochs):
        # Linear Probing Warmup (Epoch 0)
        if epoch == 0:
            print("  [Warmup] Freezing Backbone, Training Head Only...")
            model.freeze_backbone()
        elif epoch == 1:
            print("  [Unfreeze] Unfreezing Backbone, Full Fine-tuning...")
            model.unfreeze_backbone()
            
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_auc, val_preds, val_embs = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # Log History
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc
        })
        pd.DataFrame(history).to_csv(RESULTS_DIR / f"log_{args.model}_fold{args.fold}.csv", index=False)
        
        if val_auc > best_auc:
            best_auc = val_auc
            # Save Model
            torch.save(model.state_dict(), RESULTS_DIR / f"{args.model}_fold{args.fold}.pth")
            # Save OOF (Crucial for Stacking)
            # We save: isic_id, target, pred, embeddings
            oof_df = val_df[['isic_id', 'target']].copy()
            oof_df['pred'] = val_preds
            # Save embeddings separately or as numpy?
            # Saving as numpy is better for 768 dims
            np.save(RESULTS_DIR / f"oof_emb_{args.model}_fold{args.fold}.npy", val_embs)
            oof_df.to_csv(RESULTS_DIR / f"oof_{args.model}_fold{args.fold}.csv", index=False)
            print(f"  Saved Best Model & OOF (AUC: {best_auc:.4f})")
            
    print(f"Fold {args.fold} Finished. Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
