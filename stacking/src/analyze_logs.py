import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path('./last_run/results')

def analyze_logs():
    log_files = sorted(list(RESULTS_DIR.glob('log_*.csv')))
    
    print(f"{'Model':<40} | {'Fold':<5} | {'Best AUC':<10} | {'Final AUC':<10} | {'Best Epoch':<10}")
    print("-" * 85)
    
    all_aucs = []
    
    for log_file in log_files:
        # Filename format: log_{model_name}_fold{fold}.csv
        stem = log_file.stem
        fold = stem.split('_fold')[-1]
        model_name = stem.replace('log_', '').replace(f"_fold{fold}", "")
        
        df = pd.read_csv(log_file)
        
        best_epoch_idx = df['val_auc'].idxmax()
        best_auc = df['val_auc'].max()
        final_auc = df['val_auc'].iloc[-1]
        best_epoch = df['epoch'].iloc[best_epoch_idx]
        
        print(f"{model_name:<40} | {fold:<5} | {best_auc:.4f}     | {final_auc:.4f}      | {best_epoch:<10}")
        all_aucs.append(best_auc)

    print("-" * 85)
    print(f"Mean Best AUC: {sum(all_aucs)/len(all_aucs):.4f}")

if __name__ == "__main__":
    analyze_logs()
