import pandas as pd
import numpy as np
from pathlib import Path
import sys

def compare_submissions(file1, file2):
    print(f"Comparing:\n  A: {file1}\n  B: {file2}\n")
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Merge on isic_id
    merged = df1.merge(df2, on='isic_id', suffixes=('_A', '_B'))
    
    print(f"Common samples: {len(merged)}")
    if len(merged) == 0:
        print("❌ No common samples found!")
        return
        
    pred_A = merged['target_A']
    pred_B = merged['target_B']
    
    # Correlation
    pearson = pred_A.corr(pred_B, method='pearson')
    spearman = pred_A.corr(pred_B, method='spearman')
    
    # Difference
    diff = np.abs(pred_A - pred_B)
    mean_diff = diff.mean()
    max_diff = diff.max()
    
    print(f"Pearson Correlation:  {pearson:.4f}")
    print(f"Spearman Correlation: {spearman:.4f}")
    print(f"Mean Abs Difference:  {mean_diff:.4f}")
    print(f"Max Abs Difference:   {max_diff:.4f}")
    
    print("\nTop 5 Disagreements:")
    merged['diff'] = diff
    print(merged.sort_values('diff', ascending=False).head(5)[['isic_id', 'target_A', 'target_B', 'diff']])

if __name__ == "__main__":
    base_dir = Path('./results')
    
    # New submission (Ensemble)
    new_sub = base_dir / 'dual_hybrid_v2/submission.csv'
    
    # Reference submission (Best previous)
    # Using rank_avg as it's usually the best
    ref_sub = base_dir / 'kfold_v2s_features_advanced_20251111_150340/submission_kfold_rank_avg.csv'
    
    if not new_sub.exists():
        print(f"File not found: {new_sub}")
        sys.exit(1)
    if not ref_sub.exists():
        print(f"File not found: {ref_sub}")
        sys.exit(1)
        
    compare_submissions(new_sub, ref_sub)
