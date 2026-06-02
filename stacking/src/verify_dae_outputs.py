import numpy as np
import torch
from pathlib import Path

RESULTS_DIR = Path('./last_run/results')

def verify_dae():
    print("--- Verifying DAE Outputs ---")
    
    # 1. Check Files Exist
    files = [
        'dae_model.pth',
        'dae_encoder.pth',
        'dae_latent_train.npy',
        'dae_latent_test.npy',
        'dae_preprocessor.pkl'
    ]
    
    for f in files:
        path = RESULTS_DIR / f
        if not path.exists():
            print(f"❌ Missing: {f}")
            return
        print(f"✅ Found: {f}")
        
    # 2. Check Latent Shapes
    train_latent = np.load(RESULTS_DIR / 'dae_latent_train.npy')
    test_latent = np.load(RESULTS_DIR / 'dae_latent_test.npy')
    
    print(f"Train Latent Shape: {train_latent.shape}")
    print(f"Test Latent Shape: {test_latent.shape}")
    
    # Expected: (N_samples, 64)
    if train_latent.shape[1] != 64:
        print(f"❌ Incorrect Latent Dim: {train_latent.shape[1]} (Expected 64)")
    else:
        print("✅ Latent Dim Correct (64)")
        
    # 3. Check Values (Not NaN, not all zeros)
    if np.isnan(train_latent).any():
        print("❌ NaNs found in Train Latent!")
    elif np.all(train_latent == 0):
        print("❌ Train Latent is all zeros!")
    else:
        print(f"✅ Train Latent Stats: Mean={train_latent.mean():.4f}, Std={train_latent.std():.4f}")
        
    if np.isnan(test_latent).any():
        print("❌ NaNs found in Test Latent!")
    else:
        print(f"✅ Test Latent Stats: Mean={test_latent.mean():.4f}, Std={test_latent.std():.4f}")

if __name__ == "__main__":
    verify_dae()
