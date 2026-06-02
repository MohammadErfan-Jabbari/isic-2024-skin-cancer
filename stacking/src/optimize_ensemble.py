import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from pathlib import Path

# Config
LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
OOF_PATH = RESULTS_DIR / 'oof_stacking.csv'

def optimize_ensemble():
    print("--- Optimizing Ensemble Weights ---")
    
    # 1. Load OOF
    print(f"Loading OOF from {OOF_PATH}...")
    df = pd.read_csv(OOF_PATH)
    y = df['target'].values
    xgb_p = df['xgb_pred'].values
    mlp_p = df['mlp_pred'].values
    
    # 2. Define Objective Function (Negative AUC)
    def objective(weights):
        # weights[0] = xgb_weight
        # mlp_weight = 1 - weights[0]
        w_xgb = weights[0]
        w_mlp = 1 - w_xgb
        
        # Blend
        blend = w_xgb * xgb_p + w_mlp * mlp_p
        
        # AUC
        auc = roc_auc_score(y, blend)
        return -auc
    
    # 3. Optimize
    # Initial guess: 0.5
    init_guess = [0.5]
    # Bounds: 0 to 1
    bounds = [(0.0, 1.0)]
    
    result = minimize(objective, init_guess, method='Nelder-Mead', bounds=bounds) # Nelder-Mead ignores bounds, but let's try SLSQP if needed. 
    # Actually Nelder-Mead is robust. Let's just clip inside objective or use SLSQP.
    # Let's use SLSQP for bounds support.
    result = minimize(objective, init_guess, method='SLSQP', bounds=bounds)
    
    best_w_xgb = result.x[0]
    best_w_mlp = 1 - best_w_xgb
    best_auc = -result.fun
    
    print(f"\n✅ Optimization Complete!")
    print(f"Best Weights: XGB={best_w_xgb:.4f}, MLP={best_w_mlp:.4f}")
    print(f"Optimized AUC: {best_auc:.5f}")
    print(f"Baseline (0.5/0.5) AUC: {roc_auc_score(y, (xgb_p + mlp_p)/2):.5f}")
    print(f"XGB Only AUC: {roc_auc_score(y, xgb_p):.5f}")
    print(f"MLP Only AUC: {roc_auc_score(y, mlp_p):.5f}")
    
    return best_w_xgb

if __name__ == "__main__":
    optimize_ensemble()
