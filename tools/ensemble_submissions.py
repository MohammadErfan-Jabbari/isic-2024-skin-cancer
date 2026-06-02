import pandas as pd
from pathlib import Path
import glob

def ensemble_folds(results_dir):
    print(f"Ensembling submissions in {results_dir}...")
    
    # Find all fold submissions
    files = sorted(glob.glob(str(results_dir / 'submission_fold_*.csv')))
    print(f"Found {len(files)} submission files:")
    for f in files:
        print(f"  - {Path(f).name}")
        
    if len(files) == 0:
        print("❌ No submission files found!")
        return
    
    # Load and sum
    dfs = [pd.read_csv(f) for f in files]
    
    # Verify IDs match
    base_ids = dfs[0]['isic_id']
    for i, df in enumerate(dfs[1:]):
        if not df['isic_id'].equals(base_ids):
            raise ValueError(f"ID mismatch in file {files[i+1]}")
            
    # Average
    avg_preds = sum(df['target'] for df in dfs) / len(dfs)
    
    submission = pd.DataFrame({
        'isic_id': base_ids,
        'target': avg_preds
    })
    
    output_path = results_dir / 'submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"\n✅ Saved ensemble to {output_path}")
    print(submission.head())

if __name__ == "__main__":
    results_dir = Path('./results/dual_hybrid_v2')
    ensemble_folds(results_dir)
