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
SUBMISSION_DIR = LAST_RUN_DIR / 'submissions'
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

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

def inference(model_name, batch_size=64):
    print(f"Running Inference for Model: {model_name}")
    
    # 1. Find Trained Folds
    weight_files = sorted(list(RESULTS_DIR.glob(f"{model_name}_fold*.pth")))
    if not weight_files:
        print(f"❌ No weights found for model: {model_name}")
        return
        
    print(f"✅ Found {len(weight_files)} trained folds: {[f.name for f in weight_files]}")
    
    # 2. Load Test Metadata (Only for IDs)
    test_df = pd.read_csv(DATA_DIR / 'students-test-metadata.csv')
    
    # 3. Dataset & Loader
    transforms = get_transforms(img_size=336)
    test_ds = ISICDataset(DATA_DIR / 'test-image.hdf5', test_df, transform=transforms, is_test=True)
    print(f"  Dataset: {len(test_ds)} samples from {DATA_DIR / 'test-image.hdf5'}")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. Loop through Folds
    for weights_path in weight_files:
        fold = weights_path.stem.split('fold')[-1]
        print(f"\n--- Processing Fold {fold} ---")
        print(f"  Loading Weights: {weights_path}")
        
        # Load Model
        model = ISICModel(model_name, pretrained=False)
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        # Predict
        preds = []
        ids = []
        
        with torch.no_grad():
            for imgs, batch_ids in tqdm(test_loader, desc=f"Predicting Fold {fold}"):
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    logits, _ = model(imgs)
                preds.extend(logits.sigmoid().cpu().numpy().flatten())
                ids.extend(batch_ids)
                
        # Save Submission
        sub_df = pd.DataFrame({
            'isic_id': ids,
            'target': preds
        })
        
        out_path = SUBMISSION_DIR / f"submission_vision_only_{model_name}_fold{fold}.csv"
        sub_df.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    inference(args.model, args.batch_size)

if __name__ == "__main__":
    main()
