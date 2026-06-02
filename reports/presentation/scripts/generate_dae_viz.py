
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from pathlib import Path

# Config
DATA_DIR = Path('./data')
LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
OUTPUT_DIR = Path('public/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Generating DAE Latent Visualization...")
    
    # 1. Load Data
    print("Loading Metadata...")
    df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
    
    print("Loading Latent Vectors...")
    dae_latent = np.load(RESULTS_DIR / 'dae_latent_train.npy')
    
    # Slice to match (train set only)
    if len(dae_latent) > len(df):
        dae_latent = dae_latent[:len(df)]
        
    print(f"Data Shape: {dae_latent.shape}")
    
    # 2. Stratified Subsample
    # UMAP on 400k points is slow. Let's take 10k points.
    # We want to see if classes separate.
    # Take all malignant (only ~400) and random benigns.
    N_SAMPLES = 5000
    
    pos_idx = np.where(df['target'] == 1)[0]
    neg_idx = np.where(df['target'] == 0)[0]
    
    print(f"Positives: {len(pos_idx)}")
    
    # Sample balanced-ish? Or realistic?
    # Let's visualize with more benigns to show "needle in haystack" or clustering?
    # Usually balanced sampling is better for visualization of separability.
    
    n_pos = len(pos_idx)
    n_neg = N_SAMPLES - n_pos
    if n_neg > len(neg_idx): n_neg = len(neg_idx)
    
    chosen_neg = np.random.choice(neg_idx, n_neg, replace=False)
    
    subset_idx = np.concatenate([pos_idx, chosen_neg])
    # Shuffle
    np.random.shuffle(subset_idx)
    
    X_sample = dae_latent[subset_idx]
    y_sample = df['target'].iloc[subset_idx].values
    
    print(f"Running UMAP on {len(X_sample)} samples...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_sample)
    
    # 3. Plot
    plt.figure(figsize=(12, 10))
    
    # Plot benigns first
    mask_neg = y_sample == 0
    plt.scatter(embedding[mask_neg, 0], embedding[mask_neg, 1], 
                c='#95a5a6', alpha=0.3, s=10, label='Benign (Subsampled)')
    
    # Plot malignants on top
    mask_pos = y_sample == 1
    plt.scatter(embedding[mask_pos, 0], embedding[mask_pos, 1], 
                c='#e74c3c', alpha=0.8, s=25, label='Malignant (All)')
    
    plt.title('DAE Latent Space Projection (UMAP)', fontsize=16, fontweight='bold')
    plt.legend()
    plt.axis('off') # Hide axis for cleaner look
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dae_latent_viz.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {OUTPUT_DIR / 'dae_latent_viz.png'}")

if __name__ == "__main__":
    main()
