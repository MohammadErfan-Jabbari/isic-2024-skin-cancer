import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

def check_oof_auc(path, name):
    try:
        df = pd.read_csv(path)
        # Clean prediction column
        if df['prediction'].dtype == 'object':
             df['prediction'] = df['prediction'].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        df['prediction'] = pd.to_numeric(df['prediction'])
        
        auc = roc_auc_score(df['target'], df['prediction'])
        print(f"{name} AUC: {auc:.5f}")
    except Exception as e:
        print(f"{name} Error: {e}")

check_oof_auc('DeepLearning/Kaggle/results/eva02_exp_v1/oof_predictions_fold1.csv', 'Eva02 Fold 1')
check_oof_auc('DeepLearning/Kaggle/results/edgenext_exp_v1/oof_predictions_fold1.csv', 'EdgeNeXt Fold 1')
