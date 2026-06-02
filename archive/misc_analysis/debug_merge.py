import pandas as pd
import numpy as np

METADATA_PATH = 'DeepLearning/Kaggle/data/new-train-metadata.csv'
EVA_OOF_PATH = 'DeepLearning/Kaggle/results/eva02_exp_v1/oof_predictions_fold1.csv'

def verify_merge():
    print("Loading metadata...")
    meta_df = pd.read_csv(METADATA_PATH, usecols=['isic_id', 'target'])
    print(f"Metadata shape: {meta_df.shape}")
    
    print("Loading OOF...")
    oof_df = pd.read_csv(EVA_OOF_PATH)
    # Clean prediction
    if oof_df['prediction'].dtype == 'object':
        oof_df['prediction'] = oof_df['prediction'].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
    oof_df['prediction'] = pd.to_numeric(oof_df['prediction'])
    
    print(f"OOF shape: {oof_df.shape}")
    
    # Merge
    merged = meta_df.merge(oof_df[['isic_id', 'prediction', 'target']], on='isic_id', how='inner', suffixes=('_meta', '_oof'))
    print(f"Merged shape: {merged.shape}")
    
    # Check target consistency
    mismatch = merged[merged['target_meta'] != merged['target_oof']]
    if len(mismatch) > 0:
        print(f"CRITICAL: {len(mismatch)} rows have mismatched targets!")
        print(mismatch.head())
    else:
        print("Targets match perfectly.")
        
    # Check AUC on merged data
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(merged['target_meta'], merged['prediction'])
    print(f"Merged AUC (using metadata target): {auc:.5f}")

if __name__ == "__main__":
    verify_merge()
