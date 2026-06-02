
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path('./last_run/results')

def get_id_to_fold_mapping():
    """Create a mapping from isic_id to fold number using vision OOF files."""
    id_to_fold = {}
    for fold in range(5):
        # Using EdgeNeXt files as they are reliably present based on `ls` output earlier
        path = RESULTS_DIR / f"oof_edgenext_base_fold{fold}.csv"
        if path.exists():
            df = pd.read_csv(path)
            for isic_id in df['isic_id']:
                id_to_fold[isic_id] = fold
    return id_to_fold

def analyze_golden_split():
    # Load stacking predictions
    stacking_df = pd.read_csv(RESULTS_DIR / 'oof_stacking.csv')
    
    # Get fold mapping
    id_to_fold = get_id_to_fold_mapping()
    
    # Map folds to stacking df
    stacking_df['fold'] = stacking_df['isic_id'].map(id_to_fold)
    
    print(f"Total rows: {len(stacking_df)}")
    print(f"Rows with fold info: {stacking_df['fold'].notna().sum()}")
    
    # Calculate AUC per fold
    print("\nPer-Fold Performance (Validation Score):")
    print(f"{'Val Fold':<10} | {'Train Folds':<15} | {'Ensemble AUC':<10}")
    print("-" * 45)
    
    scores = {}
    
    for fold in range(5):
        fold_data = stacking_df[stacking_df['fold'] == fold]
        if len(fold_data) > 0:
            auc = roc_auc_score(fold_data['target'], fold_data['ensemble_pred'])
            train_set = "{" + ",".join([str(f) for f in range(5) if f != fold]) + "}"
            print(f"{fold:<10} | {train_set:<15} | {auc:.4f}")
            scores[fold] = auc
        else:
            print(f"{fold:<10} | N/A             | N/A")
            
    best_fold = max(scores, key=scores.get)
    print(f"\nBest Validation Fold: {best_fold} (Train on 0,1,2,3?)" if best_fold == 4 else f"\nBest Validation Fold: {best_fold}")

if __name__ == "__main__":
    analyze_golden_split()
