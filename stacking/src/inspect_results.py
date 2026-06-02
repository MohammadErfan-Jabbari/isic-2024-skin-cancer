import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path('./last_run/results')

def inspect_results(fold=0, model_name='eva02_small_patch14_336.mim_in22k_ft_in1k'):
    print(f"Inspecting Results for Fold {fold}, Model: {model_name}")
    print("="*60)
    
    # 1. Config
    config_path = RESULTS_DIR / f"config_{model_name}_fold{fold}.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Config Found: {config_path.name}")
        print(f"   Debug Mode: {config.get('debug')}")
        print(f"   Batch Size: {config.get('batch_size')}")
        print(f"   LR: {config.get('lr')}")
    else:
        print(f"❌ Config Missing: {config_path.name}")

    # 2. Training Log
    log_path = RESULTS_DIR / f"log_{model_name}_fold{fold}.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        print(f"\n✅ Training Log Found: {log_path.name} ({len(log_df)} epochs)")
        print(log_df.to_string(index=False))
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss')
        plt.plot(log_df['epoch'], log_df['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(log_df['epoch'], log_df['val_auc'], label='Val AUC')
        plt.legend()
        plt.title('AUC')
        plt.savefig(RESULTS_DIR / f"plot_{model_name}_fold{fold}.png")
        print(f"   Saved Plot: plot_{model_name}_fold{fold}.png")
    else:
        print(f"\n❌ Training Log Missing: {log_path.name}")

    # 3. OOF Predictions
    oof_path = RESULTS_DIR / f"oof_{model_name}_fold{fold}.csv"
    if oof_path.exists():
        oof_df = pd.read_csv(oof_path)
        print(f"\n✅ OOF CSV Found: {oof_path.name} ({len(oof_df)} rows)")
        print(f"   Columns: {list(oof_df.columns)}")
        print(f"   Target Mean: {oof_df['target'].mean():.4f}")
        print(f"   Pred Mean: {oof_df['pred'].mean():.4f}")
        
        # Check for NaNs
        if oof_df.isnull().any().any():
            print("   ⚠️  WARNING: NaNs found in OOF CSV!")
        else:
            print("   No NaNs in OOF CSV.")
    else:
        print(f"\n❌ OOF CSV Missing: {oof_path.name}")

    # 4. Embeddings
    emb_path = RESULTS_DIR / f"oof_emb_{model_name}_fold{fold}.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
        print(f"\n✅ Embeddings Found: {emb_path.name}")
        print(f"   Shape: {emb.shape}")
        print(f"   Range: [{emb.min():.3f}, {emb.max():.3f}]")
        print(f"   Mean: {emb.mean():.3f}")
        
        if np.isnan(emb).any():
            print("   ⚠️  WARNING: NaNs found in Embeddings!")
    else:
        print(f"\n❌ Embeddings Missing: {emb_path.name}")

    # 5. Model Weights
    model_path = RESULTS_DIR / f"{model_name}_fold{fold}.pth"
    if model_path.exists():
        print(f"\n✅ Model Weights Found: {model_path.name}")
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"   Keys: {len(state_dict)} keys loaded.")
    else:
        print(f"\n❌ Model Weights Missing: {model_path.name}")
        
    print("="*60)

if __name__ == "__main__":
    inspect_results()
