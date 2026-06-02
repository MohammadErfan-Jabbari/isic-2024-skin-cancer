import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Config
LAST_RUN_DIR = Path('./last_run')
SUBMISSION_DIR = LAST_RUN_DIR / 'submissions'

def blend_submissions():
    print("--- Blending Submissions ---")
    
    # Load Base Submissions
    # We need the 5 fold submissions.
    # We can generate them if they don't exist, but we should have them or can generate them easily.
    # Actually, we generated 'submission_fold0.csv' and 'submission_fold1.csv'.
    # We need fold 2, 3, 4.
    # Let's assume the user will run inference for them or we can just run inference for all folds individually now.
    # Wait, `inference_stacking.py` can generate them.
    
    # Let's check what we have.
    # We have fold 0 and 1. We need 2, 3, 4.
    # I will add a step to generate them first in the script if missing? 
    # No, better to keep this script simple.
    
    # Let's assume we have `submission_fold{i}.csv` for i in 0..4
    # I will run the generation commands for 2, 3, 4 in the background.
    
    dfs = []
    for i in range(5):
        p = SUBMISSION_DIR / f'submission_fold{i}.csv'
        if not p.exists():
            print(f"❌ Missing: {p}")
            return
        dfs.append(pd.read_csv(p))
        
    # Check alignment
    base_ids = dfs[0]['isic_id']
    for i in range(1, 5):
        if not dfs[i]['isic_id'].equals(base_ids):
            print("❌ ID Mismatch!")
            return
            
    # Extract predictions
    preds = np.array([df['target'].values for df in dfs]) # Shape: (5, N_test)
    
    # Strategy 1: Top-2 (Fold 4 + Fold 3)
    # Fold 4: 0.990, Fold 3: 0.972
    print("\nGenerating 'Top-2 Blend' (Fold 4 + Fold 3)...")
    # Weights: 0.5, 0.5
    w_top2 = np.array([0, 0, 0, 0.5, 0.5])
    pred_top2 = np.average(preds, axis=0, weights=w_top2)
    
    sub_top2 = pd.DataFrame({'isic_id': base_ids, 'target': pred_top2})
    sub_top2.to_csv(SUBMISSION_DIR / 'submission_blend_top2.csv', index=False)
    print(f"✅ Saved: submission_blend_top2.csv")
    
    # Strategy 2: Weighted (Favor Fold 4)
    # Weights: Fold 4=0.5, Fold 3=0.2, Others=0.1
    print("\nGenerating 'Weighted Blend' (Favor Fold 4)...")
    w_weighted = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    pred_weighted = np.average(preds, axis=0, weights=w_weighted)
    
    sub_weighted = pd.DataFrame({'isic_id': base_ids, 'target': pred_weighted})
    sub_weighted.to_csv(SUBMISSION_DIR / 'submission_blend_weighted.csv', index=False)
    print(f"✅ Saved: submission_blend_weighted.csv")
    
    # Strategy 3: Rank Blending (Optional, often more robust)
    # Convert to ranks then average
    print("\nGenerating 'Rank Blend' (Top-2)...")
    ranks = np.array([df['target'].rank(pct=True).values for df in dfs])
    rank_top2 = np.average(ranks, axis=0, weights=w_top2)
    
    sub_rank = pd.DataFrame({'isic_id': base_ids, 'target': rank_top2})
    sub_rank.to_csv(SUBMISSION_DIR / 'submission_blend_rank_top2.csv', index=False)
    print(f"✅ Saved: submission_blend_rank_top2.csv")

if __name__ == "__main__":
    blend_submissions()
