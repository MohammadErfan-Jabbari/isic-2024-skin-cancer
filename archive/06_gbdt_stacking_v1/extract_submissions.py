import pandas as pd
from pathlib import Path

LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
SUBMISSION_DIR = LAST_RUN_DIR / 'submissions'
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

def extract_submissions():
    print("Loading Debug Predictions...")
    df = pd.read_csv(RESULTS_DIR / 'golden_predictions_debug.csv')
    
    # 1. CatBoost No FE
    print("Generating CatBoost (No FE)...")
    sub_cat_no_fe = df[['isic_id', 'cat_no_fe']].rename(columns={'cat_no_fe': 'target'})
    sub_cat_no_fe.to_csv(SUBMISSION_DIR / 'submission_golden_catboost_no_fe.csv', index=False)
    
    # 2. CatBoost With FE
    print("Generating CatBoost (With FE)...")
    sub_cat_fe = df[['isic_id', 'cat_fe']].rename(columns={'cat_fe': 'target'})
    sub_cat_fe.to_csv(SUBMISSION_DIR / 'submission_golden_catboost_fe.csv', index=False)
    
    # 3. Golden "No FE" Ensemble (XGB + MLP + CatBoost No FE)
    # This tests the hypothesis: "Clean Data (Golden) + Clean Features (No FE) is best"
    print("Generating Golden 'No FE' Ensemble...")
    pred_no_fe = (df['xgb_no_fe'] + df['mlp_no_fe'] + df['cat_no_fe']) / 3
    sub_no_fe = pd.DataFrame({'isic_id': df['isic_id'], 'target': pred_no_fe})
    sub_no_fe.to_csv(SUBMISSION_DIR / 'submission_golden_no_fe_ensemble.csv', index=False)
    
    print("✅ Generated 3 additional submissions in submissions/")

if __name__ == "__main__":
    extract_submissions()
