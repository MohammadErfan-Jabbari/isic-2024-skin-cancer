import pickle
import numpy as np
from pathlib import Path

def analyze_training_results(model_name, result_dir):
    print(f"\nAnalysis for {model_name}:")
    aucs = []
    for fold in range(1, 6):
        p = Path(result_dir) / f"training_results_fold{fold}.pkl"
        if p.exists():
            with open(p, 'rb') as f:
                data = pickle.load(f)
                # Get best val_auc (or val_ema_auc if available)
                if 'val_ema_auc' in data:
                    best_auc = max(data['val_ema_auc'])
                else:
                    best_auc = max(data['val_auc'])
                aucs.append(best_auc)
                print(f"  Fold {fold} Best AUC: {best_auc:.5f}")
    
    if aucs:
        print(f"  Mean AUC: {np.mean(aucs):.5f}")

analyze_training_results("Eva02", "DeepLearning/Kaggle/results/eva02_exp_v1")
analyze_training_results("EdgeNeXt", "DeepLearning/Kaggle/results/edgenext_exp_v1")
