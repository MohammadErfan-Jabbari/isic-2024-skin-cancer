import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import timm
import argparse
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import ISICDataset

# Config
DATA_DIR = Path('./data')
LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.in_features = self.backbone.num_features
        self.head = nn.Linear(self.in_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits, features

def get_transforms(img_size=336):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ])

def extract_test_embeddings(model_name, batch_size=64):
    print(f"--- Extracting Test Embeddings for {model_name} ---")
    
    # 1. Find Trained Folds
    weight_files = sorted(list(RESULTS_DIR.glob(f"{model_name}_fold*.pth")))
    if not weight_files:
        print(f"❌ No weights found for model: {model_name}")
        return
    print(f"✅ Found {len(weight_files)} trained folds.")
    
    # 2. Load Test Data
    test_df = pd.read_csv(DATA_DIR / 'students-test-metadata.csv')
    transforms = get_transforms(img_size=336)
    test_ds = ISICDataset(DATA_DIR / 'test-image.hdf5', test_df, transform=transforms, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. Loop Folds
    all_probs = []
    all_embs = []
    
    for weights_path in weight_files:
        fold = weights_path.stem.split('fold')[-1]
        print(f"  Processing Fold {fold}...")
        
        # Load Model
        model = ISICModel(model_name, pretrained=False)
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        fold_probs = []
        fold_embs = []
        
        with torch.no_grad():
            for imgs, _ in tqdm(test_loader, desc=f"Fold {fold}", leave=False):
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    logits, features = model(imgs)
                    
                fold_probs.extend(logits.sigmoid().cpu().numpy().flatten())
                fold_embs.extend(features.cpu().numpy())
                
        # Save Fold Results immediately
        fold_probs = np.array(fold_probs)
        fold_embs = np.array(fold_embs)
        
        # Save Probs
        sub_df = pd.DataFrame({'isic_id': test_df['isic_id'], 'target': fold_probs})
        sub_df.to_csv(RESULTS_DIR / f"test_probs_{model_name}_fold{fold}.csv", index=False)
        
        # Save Embeddings
        np.save(RESULTS_DIR / f"test_emb_{model_name}_fold{fold}.npy", fold_embs)
        print(f"    Saved: Probs & Emb for Fold {fold}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    extract_test_embeddings(args.model, args.batch_size)
