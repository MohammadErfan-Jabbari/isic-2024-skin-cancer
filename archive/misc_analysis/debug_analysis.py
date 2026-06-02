"""Debug analysis script to understand why stacking is underperforming."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from pathlib import Path

def score_pauc(y_true, y_pred, min_tpr=0.80):
    """Calculates pAUC above a minimum TPR threshold."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        mask = tpr >= min_tpr
        if mask.sum() < 2: return 0.0
        return auc(fpr[mask], tpr[mask])
    except:
        return 0.0

print("=" * 70)
print("INVESTIGATION: Why is Stacking Underperforming?")
print("=" * 70)

# 1. Load Stacking OOF
print("\n=== 1. STACKING OOF ANALYSIS ===")
oof = pd.read_csv('results/stacking_final_v1/stacking_oof.csv')
print(f'Total samples: {len(oof)}')
print(f'Positive (target=1): {oof["target"].sum()}')
print(f'Negative (target=0): {(oof["target"]==0).sum()}')

print('\n--- Prediction Distributions ---')
print('Stack Pred:')
print(oof['stack_pred'].describe())

print('\nEVA02 Pred (After Rank Norm in training):')
print(oof['eva02_pred'].describe())

print('\nEdgeNeXt Pred (After Rank Norm in training):')
print(oof['edgenext_pred'].describe())

# 2. AUC Scores
print('\n=== 2. AUC SCORES (on Training OOFs) ===')
print(f'Stack AUC: {roc_auc_score(oof["target"], oof["stack_pred"]):.5f}')
print(f'EVA02 AUC: {roc_auc_score(oof["target"], oof["eva02_pred"]):.5f}')
print(f'EdgeNeXt AUC: {roc_auc_score(oof["target"], oof["edgenext_pred"]):.5f}')

oof['avg_vision'] = (oof['eva02_pred'] + oof['edgenext_pred']) / 2
print(f'Avg Vision AUC: {roc_auc_score(oof["target"], oof["avg_vision"]):.5f}')

# 3. pAUC Scores (Competition Metric)
print('\n=== 3. pAUC SCORES (Competition Metric, TPR > 0.8) ===')
print(f'Stack pAUC: {score_pauc(oof["target"], oof["stack_pred"]):.5f}')
print(f'EVA02 pAUC: {score_pauc(oof["target"], oof["eva02_pred"]):.5f}')
print(f'EdgeNeXt pAUC: {score_pauc(oof["target"], oof["edgenext_pred"]):.5f}')
print(f'Avg Vision pAUC: {score_pauc(oof["target"], oof["avg_vision"]):.5f}')

# 4. Load RAW OOF predictions (before rank normalization)
print('\n=== 4. RAW VISION PREDICTIONS (Before Rank Norm) ===')
eva_oofs = []
edge_oofs = []
for fold in range(1, 6):
    eva_df = pd.read_csv(f'results/gen-train-run-eva-v2/oof_fold{fold}.csv')
    edge_df = pd.read_csv(f'results/gen-train-run-edgenext-v2/oof_fold{fold}.csv')
    
    # Clean prediction column
    if eva_df['pred'].dtype == object:
        eva_df['pred'] = eva_df['pred'].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)
    if edge_df['pred'].dtype == object:
        edge_df['pred'] = edge_df['pred'].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)
    
    eva_oofs.append(eva_df[['isic_id', 'target', 'pred']])
    edge_oofs.append(edge_df[['isic_id', 'target', 'pred']])

eva_all = pd.concat(eva_oofs)
edge_all = pd.concat(edge_oofs)

print(f'EVA02 RAW pred range: {eva_all["pred"].min():.6f} - {eva_all["pred"].max():.6f}')
print(f'EdgeNeXt RAW pred range: {edge_all["pred"].min():.6f} - {edge_all["pred"].max():.6f}')

print(f'\nEVA02 RAW AUC: {roc_auc_score(eva_all["target"], eva_all["pred"]):.5f}')
print(f'EdgeNeXt RAW AUC: {roc_auc_score(edge_all["target"], edge_all["pred"]):.5f}')

print(f'\nEVA02 RAW pAUC: {score_pauc(eva_all["target"], eva_all["pred"]):.5f}')
print(f'EdgeNeXt RAW pAUC: {score_pauc(edge_all["target"], edge_all["pred"]):.5f}')

# 5. Key Insight: Correlation Analysis
print('\n=== 5. CORRELATION ANALYSIS ===')
print(f'Correlation EVA02 vs EdgeNeXt (Rank Normed): {oof["eva02_pred"].corr(oof["edgenext_pred"]):.4f}')
print(f'Correlation EVA02 RAW vs EdgeNeXt RAW: {eva_all.merge(edge_all, on="isic_id", suffixes=("_eva", "_edge"))["pred_eva"].corr(eva_all.merge(edge_all, on="isic_id", suffixes=("_eva", "_edge"))["pred_edge"]):.4f}')

# 6. Check the Submission File
print('\n=== 6. SUBMISSION FILE ANALYSIS ===')
sub = pd.read_csv('results/stacking_final_v1/submission_file/submission_stacking_v1.csv')
print(f'Submission samples: {len(sub)}')
print(f'Prediction range: {sub["target"].min():.6f} - {sub["target"].max():.6f}')
print(f'Prediction mean: {sub["target"].mean():.6f}')
print(f'Prediction std: {sub["target"].std():.6f}')
print('\nSubmission head:')
print(sub.head(10))

# 7. CRITICAL: Check if predictions are all high
print('\n=== 7. CRITICAL CHECK: Are all submissions ~0.999? ===')
print(f'Predictions > 0.99: {(sub["target"] > 0.99).sum()} / {len(sub)} ({100*(sub["target"] > 0.99).sum()/len(sub):.1f}%)')
print(f'Predictions > 0.98: {(sub["target"] > 0.98).sum()} / {len(sub)} ({100*(sub["target"] > 0.98).sum()/len(sub):.1f}%)')
print(f'Predictions < 0.50: {(sub["target"] < 0.50).sum()} / {len(sub)} ({100*(sub["target"] < 0.50).sum()/len(sub):.1f}%)')

print('\n' + '=' * 70)
print('DIAGNOSIS SUMMARY')
print('=' * 70)
