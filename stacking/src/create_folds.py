import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# Config
DATA_DIR = Path('./data')
OUTPUT_DIR = Path('./last_run/data')
N_FOLDS = 5
SEED = 42

def create_folds():
    print(f"Loading metadata from {DATA_DIR}...")
    df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv')
    
    # Create 'fold' column
    df['fold'] = -1
    
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # We split based on patient_id to avoid data leakage (same patient in train and val)
    # We stratify based on target to ensure class balance
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'], df['patient_id'])):
        df.loc[val_idx, 'fold'] = fold
        
    # Verify folds
    print("\nFold Distribution:")
    print(df['fold'].value_counts().sort_index())
    
    print("\nTarget Distribution per Fold:")
    for f in range(N_FOLDS):
        n_pos = df[(df['fold'] == f) & (df['target'] == 1)].shape[0]
        n_total = df[df['fold'] == f].shape[0]
        print(f"Fold {f}: {n_pos} positives ({n_pos/n_total:.5f} rate)")
        
    # Save
    output_path = OUTPUT_DIR / 'folds.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved folds to {output_path}")

if __name__ == "__main__":
    create_folds()
