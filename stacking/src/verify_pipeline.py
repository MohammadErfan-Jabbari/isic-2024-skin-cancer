import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import timm
from dataset import ISICDataset
from train_vision import ISICModel, get_transforms

# Config
DATA_DIR = Path('./data')
LAST_RUN_DIR = Path('./last_run')
FOLDS_PATH = LAST_RUN_DIR / 'data/folds.csv'
RESULTS_DIR = LAST_RUN_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def verify_pipeline():
    print("Verifying Training Pipeline...")
    
    # 1. Load Folds
    print("1. Loading Folds...")
    df = pd.read_csv(FOLDS_PATH)
    train_df = df[df['fold'] != 0].sample(n=50).reset_index(drop=True) # Small subset
    val_df = df[df['fold'] == 0].sample(n=50).reset_index(drop=True)
    
    # 2. Dataset & Loader
    print("2. Initializing Dataset & Loader...")
    transforms = get_transforms(img_size=336)
    train_ds = ISICDataset(DATA_DIR / 'train-image-384.hdf5', train_df, transform=transforms['train'])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    
    batch = next(iter(train_loader))
    imgs, targets = batch
    print(f"  Batch Shape: {imgs.shape}, Targets: {targets.shape}")
    assert imgs.shape == (4, 3, 336, 336), "Image batch shape mismatch!"
    
    # 3. Model Initialization
    print("3. Initializing Model (EVA02)...")
    model_name = 'eva02_small_patch14_336.mim_in22k_ft_in1k'
    model = ISICModel(model_name).to(DEVICE)
    
    # 4. Forward Pass
    print("4. Running Forward Pass...")
    imgs = imgs.to(DEVICE)
    logits, features = model(imgs)
    print(f"  Logits Shape: {logits.shape}")
    print(f"  Features Shape: {features.shape}")
    
    assert logits.shape == (4, 1), "Logits shape mismatch!"
    assert features.shape == (4, model.in_features), f"Features shape mismatch (Expected {model.in_features})!"
    
    # 5. OOF Saving Logic
    print("5. Verifying OOF Saving Logic...")
    val_preds = np.random.rand(50)
    val_embs = np.random.rand(50, 768)
    
    oof_df = val_df[['isic_id', 'target']].copy()
    oof_df['pred'] = val_preds
    
    save_path_csv = RESULTS_DIR / "verify_oof.csv"
    save_path_npy = RESULTS_DIR / "verify_oof_emb.npy"
    
    oof_df.to_csv(save_path_csv, index=False)
    np.save(save_path_npy, val_embs)
    
    assert save_path_csv.exists(), "OOF CSV not saved!"
    assert save_path_npy.exists(), "OOF NPY not saved!"
    
    # 6. Visual Verification (3 Pos, 3 Neg)
    print("6. Saving Sample Images (Visual Check)...")
    
    # Helper to save image
    def save_sample(idx, label, name):
        img_tensor, target = train_ds[idx]
        print(f"  Sample {name}: Tensor Range=[{img_tensor.min():.3f}, {img_tensor.max():.3f}] Mean={img_tensor.mean():.3f}")
        
        # Inverse Normalize (approximate)
        # Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1) * 255
        img = img.astype(np.uint8)
        
        print(f"  Sample {name}: Image Range=[{img.min()}, {img.max()}]")
        
        from PIL import Image
        Image.fromarray(img).save(RESULTS_DIR / f"verify_sample_{name}.png")
        print(f"  Saved {name} (Target: {target.item()})")

    # Find indices
    pos_indices = train_df[train_df['target'] == 1].index.tolist()
    neg_indices = train_df[train_df['target'] == 0].index.tolist()
    
    if len(pos_indices) >= 3:
        for i in range(3):
            save_sample(pos_indices[i], 1, f"pos_{i}")
    else:
        print("  Not enough positives in this small subset to save 3 samples.")
        
    if len(neg_indices) >= 3:
        for i in range(3):
            save_sample(neg_indices[i], 0, f"neg_{i}")

    # Clean up
    if save_path_csv.exists(): save_path_csv.unlink()
    if save_path_npy.exists(): save_path_npy.unlink()
    
    print("✅ Pipeline Verification Passed!")

if __name__ == "__main__":
    verify_pipeline()
