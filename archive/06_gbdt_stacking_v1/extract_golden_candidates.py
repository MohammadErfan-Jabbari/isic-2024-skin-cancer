import pandas as pd
from pathlib import Path

LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
SUBMISSION_DIR = LAST_RUN_DIR / 'submissions'
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

def extract_golden_candidates():
    print("Loading Debug Predictions...")
    df = pd.read_csv(RESULTS_DIR / 'golden_predictions_debug.csv')
    
    # Note: 'xgb_no_fe' and 'mlp_no_fe' in the debug file are actually 
    # the models trained in the LATEST run, which used Feature Engineering.
    # So these are effectively 'xgb_fe' and 'mlp_fe'.
    
    # 1. Golden XGB + MLP (With FE)
    # This is the direct competitor to the 0.990 Baseline (which was XGB+MLP No FE)
    print("Generating Golden XGB + MLP (With FE)...")
    pred_xgb_mlp_fe = (df['xgb_no_fe'] + df['mlp_no_fe']) / 2
    sub_fe = pd.DataFrame({'isic_id': df['isic_id'], 'target': pred_xgb_mlp_fe})
    sub_fe.to_csv(SUBMISSION_DIR / 'submission_golden_xgb_mlp_fe.csv', index=False)
    
    # 2. CatBoost Only (No FE)
    # To confirm if CatBoost is indeed the weak link
    print("Generating CatBoost Only (No FE)...")
    sub_cat = df[['isic_id', 'cat_no_fe']].rename(columns={'cat_no_fe': 'target'})
    sub_cat.to_csv(SUBMISSION_DIR / 'submission_golden_catboost_only.csv', index=False)
    
    print("✅ Generated candidates:")
    print(f"  1. {SUBMISSION_DIR / 'submission_golden_xgb_mlp_fe.csv'} (The Hope)")
    print(f"  2. {SUBMISSION_DIR / 'submission_golden_catboost_only.csv'} (The Check)")

if __name__ == "__main__":
    extract_golden_candidates()
