#!/usr/bin/env python3
"""
17_1b_analyze_stacking_results.py - Comprehensive Stacking Analysis
====================================================================

Extensive analysis and visualizations for the stacking GBDT model:
1. OOF Prediction Distribution Analysis
2. Feature Importance Analysis (LightGBM + XGBoost comparison)
3. Error Analysis (False Positives/Negatives)
4. Calibration Curves
5. ROC/pAUC Analysis
6. Vision Model Contribution Analysis
7. Patient-Level Analysis
8. Threshold Optimization

Author: Data Science Pipeline
Date: 2025-11-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report,
    average_precision_score, roc_auc_score
)
from sklearn.calibration import calibration_curve
import warnings

warnings.filterwarnings('ignore')

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

SCRIPT_DIR = Path(__file__).parent


def load_results(results_dir):
    """Load all results from training run."""
    results = {}
    
    # Load OOF predictions
    oof_path = results_dir / 'oof_predictions.csv'
    if oof_path.exists():
        results['oof'] = pd.read_csv(oof_path)
        print(f"✅ Loaded OOF predictions: {len(results['oof']):,} samples")
    
    # Load metrics
    metrics_path = results_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            results['metrics'] = json.load(f)
        print(f"✅ Loaded metrics")
    
    # Load feature importance
    lgbm_imp_path = results_dir / 'lgbm_feature_importance.csv'
    if lgbm_imp_path.exists():
        results['lgbm_importance'] = pd.read_csv(lgbm_imp_path)
        print(f"✅ Loaded LightGBM feature importance")
    
    xgb_imp_path = results_dir / 'xgb_feature_importance.csv'
    if xgb_imp_path.exists():
        results['xgb_importance'] = pd.read_csv(xgb_imp_path)
        print(f"✅ Loaded XGBoost feature importance")
    
    # Load config
    config_path = results_dir / 'training_config.json'
    if config_path.exists():
        with open(config_path) as f:
            results['config'] = json.load(f)
        print(f"✅ Loaded training config")
    
    # Load error analysis
    fp_path = results_dir / 'top_100_false_positives.csv'
    fn_path = results_dir / 'top_100_false_negatives.csv'
    if fp_path.exists():
        results['false_positives'] = pd.read_csv(fp_path)
    if fn_path.exists():
        results['false_negatives'] = pd.read_csv(fn_path)
    
    return results


def score_pauc(y_true, y_pred, min_tpr=0.80):
    """Calculate partial AUC above min_tpr threshold."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    mask = tpr >= min_tpr
    if mask.sum() < 2:
        return 0.0
    return auc(fpr[mask], tpr[mask])


# ===========================
# VISUALIZATION FUNCTIONS
# ===========================

def plot_prediction_distributions(oof_df, save_dir):
    """Plot prediction distributions for positive vs negative samples."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    pred_cols = ['lgbm_pred', 'xgb_pred', 'ensemble_pred']
    pred_cols = [c for c in pred_cols if c in oof_df.columns]
    
    for idx, col in enumerate(pred_cols):
        ax = axes[0, idx]
        
        # Histogram by class
        pos = oof_df[oof_df['target'] == 1][col]
        neg = oof_df[oof_df['target'] == 0][col]
        
        ax.hist(neg, bins=100, alpha=0.7, label=f'Benign (n={len(neg):,})', density=True, color='blue')
        ax.hist(pos, bins=100, alpha=0.7, label=f'Malignant (n={len(pos):,})', density=True, color='red')
        
        ax.set_xlabel('Prediction Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{col.replace("_pred", "").upper()} Prediction Distribution')
        ax.legend()
        ax.set_xlim(0, 1)
        
        # Add separation metrics
        auc_score = roc_curve(oof_df['target'], oof_df[col])[0]
        threshold = 0.5
        tn = ((oof_df[col] < threshold) & (oof_df['target'] == 0)).sum()
        tp = ((oof_df[col] >= threshold) & (oof_df['target'] == 1)).sum()
        
    # Log-scale distributions (better visualization for imbalanced data)
    for idx, col in enumerate(pred_cols):
        ax = axes[1, idx]
        
        pos = oof_df[oof_df['target'] == 1][col]
        neg = oof_df[oof_df['target'] == 0][col]
        
        # Use log-scale y-axis
        ax.hist(neg, bins=100, alpha=0.7, label='Benign', density=True, color='blue')
        ax.hist(pos, bins=100, alpha=0.7, label='Malignant', density=True, color='red')
        ax.set_yscale('log')
        
        ax.set_xlabel('Prediction Score')
        ax.set_ylabel('Density (log scale)')
        ax.set_title(f'{col.replace("_pred", "").upper()} Distribution (Log Scale)')
        ax.legend()
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'prediction_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved: prediction_distributions.png")


def plot_roc_curves(oof_df, save_dir):
    """Plot ROC curves with pAUC region highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    pred_cols = ['lgbm_pred', 'xgb_pred', 'ensemble_pred']
    pred_cols = [c for c in pred_cols if c in oof_df.columns]
    colors = ['blue', 'green', 'red']
    
    # Full ROC curve
    ax = axes[0]
    for col, color in zip(pred_cols, colors):
        fpr, tpr, _ = roc_curve(oof_df['target'], oof_df[col])
        auc_score = auc(fpr, tpr)
        pauc_score = score_pauc(oof_df['target'], oof_df[col])
        
        label = f"{col.replace('_pred', '').upper()}: AUC={auc_score:.4f}, pAUC={pauc_score:.4f}"
        ax.plot(fpr, tpr, color=color, lw=2, label=label)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Full ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # pAUC region (TPR >= 0.80)
    ax = axes[1]
    for col, color in zip(pred_cols, colors):
        fpr, tpr, _ = roc_curve(oof_df['target'], oof_df[col])
        
        # Mask for pAUC region
        mask = tpr >= 0.80
        pauc_score = score_pauc(oof_df['target'], oof_df[col])
        
        ax.plot(fpr, tpr, color=color, lw=2, alpha=0.3)
        ax.plot(fpr[mask], tpr[mask], color=color, lw=3, 
                label=f"{col.replace('_pred', '').upper()}: pAUC={pauc_score:.4f}")
    
    # Highlight pAUC region
    ax.axhspan(0.80, 1.0, alpha=0.1, color='green', label='pAUC Region (TPR ≥ 0.80)')
    ax.axhline(y=0.80, color='green', linestyle='--', lw=1)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('pAUC Region (TPR ≥ 0.80) - Competition Metric')
    ax.legend(loc='lower right')
    ax.set_ylim(0.75, 1.01)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved: roc_curves.png")


def plot_precision_recall(oof_df, save_dir):
    """Plot Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    pred_cols = ['lgbm_pred', 'xgb_pred', 'ensemble_pred']
    pred_cols = [c for c in pred_cols if c in oof_df.columns]
    colors = ['blue', 'green', 'red']
    
    # PR curves
    ax = axes[0]
    for col, color in zip(pred_cols, colors):
        precision, recall, _ = precision_recall_curve(oof_df['target'], oof_df[col])
        ap = average_precision_score(oof_df['target'], oof_df[col])
        
        ax.plot(recall, precision, color=color, lw=2, 
                label=f"{col.replace('_pred', '').upper()}: AP={ap:.4f}")
    
    # Baseline (random)
    baseline = oof_df['target'].mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.4f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # F1 Score vs Threshold
    ax = axes[1]
    for col, color in zip(pred_cols, colors):
        thresholds = np.linspace(0.001, 0.999, 100)
        f1_scores = []
        
        for thresh in thresholds:
            pred_binary = (oof_df[col] >= thresh).astype(int)
            tp = ((pred_binary == 1) & (oof_df['target'] == 1)).sum()
            fp = ((pred_binary == 1) & (oof_df['target'] == 0)).sum()
            fn = ((pred_binary == 0) & (oof_df['target'] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        ax.plot(thresholds, f1_scores, color=color, lw=2,
                label=f"{col.replace('_pred', '').upper()}: Best F1={best_f1:.4f} @ {best_thresh:.3f}")
        ax.axvline(x=best_thresh, color=color, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Threshold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'precision_recall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved: precision_recall.png")


def plot_calibration_curves(oof_df, save_dir):
    """Plot calibration curves to assess probability reliability."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    pred_cols = ['lgbm_pred', 'xgb_pred', 'ensemble_pred']
    pred_cols = [c for c in pred_cols if c in oof_df.columns]
    colors = ['blue', 'green', 'red']
    
    # Calibration curves
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')
    
    for col, color in zip(pred_cols, colors):
        prob_true, prob_pred = calibration_curve(oof_df['target'], oof_df[col], n_bins=20, strategy='uniform')
        ax.plot(prob_pred, prob_true, 's-', color=color, lw=2, markersize=5,
                label=f"{col.replace('_pred', '').upper()}")
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Calibration histogram
    ax = axes[1]
    for col, color in zip(pred_cols, colors):
        ax.hist(oof_df[col], bins=50, alpha=0.4, color=color, 
                label=f"{col.replace('_pred', '').upper()}")
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Histogram')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'calibration_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved: calibration_curves.png")


def plot_feature_importance(results, save_dir):
    """Plot feature importance comparison between LightGBM and XGBoost."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 14))
    
    # LightGBM importance
    if 'lgbm_importance' in results:
        ax = axes[0]
        imp = results['lgbm_importance'].head(30).copy()
        imp = imp.sort_values('importance', ascending=True)
        
        colors = ['red' if 'pred' in f.lower() or 'vision' in f.lower() else 'steelblue' 
                  for f in imp['feature']]
        
        ax.barh(imp['feature'], imp['importance'], color=colors)
        ax.set_xlabel('Importance (Gain)')
        ax.set_title('LightGBM Top 30 Features\n(Red = Vision Model Features)')
        ax.grid(True, alpha=0.3, axis='x')
    
    # XGBoost importance
    if 'xgb_importance' in results:
        ax = axes[1]
        imp = results['xgb_importance'].head(30).copy()
        imp = imp.sort_values('importance', ascending=True)
        
        colors = ['red' if 'pred' in f.lower() or 'vision' in f.lower() else 'steelblue' 
                  for f in imp['feature']]
        
        ax.barh(imp['feature'], imp['importance'], color=colors)
        ax.set_xlabel('Importance (Gain)')
        ax.set_title('XGBoost Top 30 Features\n(Red = Vision Model Features)')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved: feature_importance.png")
    
    # Feature importance comparison
    if 'lgbm_importance' in results and 'xgb_importance' in results:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        lgbm_imp = results['lgbm_importance'].set_index('feature')['importance']
        xgb_imp = results['xgb_importance'].set_index('feature')['importance']
        
        # Normalize
        lgbm_imp = lgbm_imp / lgbm_imp.max()
        xgb_imp = xgb_imp / xgb_imp.max()
        
        # Get common features
        common = set(lgbm_imp.index) & set(xgb_imp.index)
        
        comparison = pd.DataFrame({
            'LightGBM': lgbm_imp.loc[list(common)],
            'XGBoost': xgb_imp.loc[list(common)]
        })
        
        # Plot scatter
        ax.scatter(comparison['LightGBM'], comparison['XGBoost'], alpha=0.5, s=50)
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Equal Importance')
        
        # Highlight top features
        top_features = comparison.mean(axis=1).nlargest(10).index
        for feat in top_features:
            ax.annotate(feat, (comparison.loc[feat, 'LightGBM'], comparison.loc[feat, 'XGBoost']),
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('LightGBM Normalized Importance')
        ax.set_ylabel('XGBoost Normalized Importance')
        ax.set_title('Feature Importance: LightGBM vs XGBoost')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'feature_importance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("📊 Saved: feature_importance_comparison.png")


def plot_vision_model_analysis(oof_df, save_dir):
    """Analyze contribution of vision models."""
    if 'eva02_pred' not in oof_df.columns or 'edgenext_pred' not in oof_df.columns:
        print("⚠️ Vision predictions not found in OOF data")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Vision model correlation
    ax = axes[0, 0]
    pos = oof_df[oof_df['target'] == 1]
    neg = oof_df[oof_df['target'] == 0]
    
    ax.scatter(neg['eva02_pred'], neg['edgenext_pred'], alpha=0.1, s=5, c='blue', label='Benign')
    ax.scatter(pos['eva02_pred'], pos['edgenext_pred'], alpha=0.5, s=20, c='red', label='Malignant')
    
    corr = oof_df['eva02_pred'].corr(oof_df['edgenext_pred'])
    ax.set_xlabel('EVA02 Prediction (Z-Score)')
    ax.set_ylabel('EdgeNeXt Prediction (Z-Score)')
    ax.set_title(f'Vision Model Correlation (r={corr:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Vision vs Ensemble
    ax = axes[0, 1]
    # Calculate mean vision
    oof_df['mean_vision'] = (oof_df['eva02_pred'] + oof_df['edgenext_pred']) / 2
    
    ax.scatter(neg['ensemble_pred'], (neg['eva02_pred'] + neg['edgenext_pred'])/2, 
               alpha=0.1, s=5, c='blue', label='Benign')
    ax.scatter(pos['ensemble_pred'], (pos['eva02_pred'] + pos['edgenext_pred'])/2,
               alpha=0.5, s=20, c='red', label='Malignant')
    
    ax.set_xlabel('Ensemble Prediction')
    ax.set_ylabel('Mean Vision Prediction (Z-Score)')
    ax.set_title('Ensemble vs Vision Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Vision prediction distributions by class
    ax = axes[0, 2]
    for col, color, label in [('eva02_pred', 'blue', 'EVA02'), ('edgenext_pred', 'green', 'EdgeNeXt')]:
        pos_vals = oof_df[oof_df['target'] == 1][col]
        neg_vals = oof_df[oof_df['target'] == 0][col]
        
        # Show difference
        ax.boxplot([neg_vals, pos_vals], positions=[0 if col == 'eva02_pred' else 2, 
                                                     1 if col == 'eva02_pred' else 3])
    
    ax.set_xticks([0.5, 2.5])
    ax.set_xticklabels(['EVA02\n(Benign/Malignant)', 'EdgeNeXt\n(Benign/Malignant)'])
    ax.set_ylabel('Prediction (Z-Score)')
    ax.set_title('Vision Predictions by Class')
    ax.grid(True, alpha=0.3)
    
    # Vision model individual performance
    ax = axes[1, 0]
    models = ['eva02_pred', 'edgenext_pred']
    aucs = []
    paucs = []
    
    for model in models:
        # Need to convert z-scores back to probability-like for ROC
        # Using sigmoid transformation
        probs = 1 / (1 + np.exp(-oof_df[model]))
        auc_val = roc_auc_score(oof_df['target'], probs)
        pauc_val = score_pauc(oof_df['target'], probs)
        aucs.append(auc_val)
        paucs.append(pauc_val)
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, aucs, width, label='AUC', color='steelblue')
    ax.bar(x + width/2, paucs, width, label='pAUC', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Vision Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(['EVA02', 'EdgeNeXt'])
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Scatter: Vision disagreement cases
    ax = axes[1, 1]
    oof_df['vision_diff'] = oof_df['eva02_pred'] - oof_df['edgenext_pred']
    
    # High disagreement cases
    high_disagree = oof_df[abs(oof_df['vision_diff']) > oof_df['vision_diff'].std() * 2]
    
    ax.hist(oof_df[oof_df['target'] == 0]['vision_diff'], bins=50, alpha=0.7, 
            label='Benign', density=True, color='blue')
    ax.hist(oof_df[oof_df['target'] == 1]['vision_diff'], bins=50, alpha=0.7,
            label='Malignant', density=True, color='red')
    
    ax.axvline(x=0, color='black', linestyle='--', lw=1)
    ax.set_xlabel('EVA02 - EdgeNeXt (Z-Score)')
    ax.set_ylabel('Density')
    ax.set_title('Vision Model Disagreement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Vision contribution to ensemble improvement
    ax = axes[1, 2]
    
    # Compare GBDT-only vs GBDT+Vision ensemble
    # We can show correlation of vision with target
    eva_corr = oof_df['eva02_pred'].corr(oof_df['target'])
    edge_corr = oof_df['edgenext_pred'].corr(oof_df['target'])
    lgbm_corr = oof_df['lgbm_pred'].corr(oof_df['target'])
    ensemble_corr = oof_df['ensemble_pred'].corr(oof_df['target'])
    
    correlations = [eva_corr, edge_corr, lgbm_corr, ensemble_corr]
    labels = ['EVA02', 'EdgeNeXt', 'LightGBM', 'Ensemble']
    colors = ['orange', 'green', 'blue', 'red']
    
    ax.bar(labels, correlations, color=colors)
    ax.set_ylabel('Correlation with Target')
    ax.set_title('Component Correlation with Target')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(correlations):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'vision_model_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved: vision_model_analysis.png")


def plot_error_analysis(oof_df, results, save_dir):
    """Analyze false positives and false negatives."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    pred_col = 'ensemble_pred'
    
    # FP and FN prediction distributions
    ax = axes[0, 0]
    
    # Use optimal threshold based on F1
    thresholds = np.linspace(0.001, 0.5, 100)
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        pred_binary = (oof_df[pred_col] >= thresh).astype(int)
        tp = ((pred_binary == 1) & (oof_df['target'] == 1)).sum()
        fp = ((pred_binary == 1) & (oof_df['target'] == 0)).sum()
        fn = ((pred_binary == 0) & (oof_df['target'] == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    # Classify at optimal threshold
    oof_df['predicted'] = (oof_df[pred_col] >= best_thresh).astype(int)
    
    tp_df = oof_df[(oof_df['predicted'] == 1) & (oof_df['target'] == 1)]
    tn_df = oof_df[(oof_df['predicted'] == 0) & (oof_df['target'] == 0)]
    fp_df = oof_df[(oof_df['predicted'] == 1) & (oof_df['target'] == 0)]
    fn_df = oof_df[(oof_df['predicted'] == 0) & (oof_df['target'] == 1)]
    
    ax.hist(fp_df[pred_col], bins=30, alpha=0.7, label=f'False Positives (n={len(fp_df):,})', color='orange')
    ax.hist(fn_df[pred_col], bins=30, alpha=0.7, label=f'False Negatives (n={len(fn_df):,})', color='purple')
    
    ax.axvline(x=best_thresh, color='red', linestyle='--', label=f'Threshold={best_thresh:.3f}')
    ax.set_xlabel('Ensemble Prediction')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Prediction Distribution (Optimal F1={best_f1:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Confusion matrix
    ax = axes[0, 1]
    cm = confusion_matrix(oof_df['target'], oof_df['predicted'])
    
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                xticklabels=['Predicted Benign', 'Predicted Malignant'],
                yticklabels=['Actual Benign', 'Actual Malignant'])
    ax.set_title(f'Confusion Matrix (Threshold={best_thresh:.3f})')
    
    # Sensitivity/Specificity vs Threshold
    ax = axes[0, 2]
    thresholds_grid = np.linspace(0.001, 0.999, 100)
    sensitivities = []
    specificities = []
    
    for thresh in thresholds_grid:
        pred_binary = (oof_df[pred_col] >= thresh).astype(int)
        tp_count = ((pred_binary == 1) & (oof_df['target'] == 1)).sum()
        tn_count = ((pred_binary == 0) & (oof_df['target'] == 0)).sum()
        fp_count = ((pred_binary == 1) & (oof_df['target'] == 0)).sum()
        fn_count = ((pred_binary == 0) & (oof_df['target'] == 1)).sum()
        
        sensitivity = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        specificity = tn_count / (tn_count + fp_count) if (tn_count + fp_count) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    ax.plot(thresholds_grid, sensitivities, 'r-', lw=2, label='Sensitivity (TPR)')
    ax.plot(thresholds_grid, specificities, 'b-', lw=2, label='Specificity (TNR)')
    ax.axhline(y=0.80, color='green', linestyle='--', alpha=0.7, label='Min TPR=0.80 (pAUC)')
    ax.axvline(x=best_thresh, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Rate')
    ax.set_title('Sensitivity/Specificity vs Threshold')
    ax.legend(loc='center left')
    ax.grid(True, alpha=0.3)
    
    # FP/FN by vision prediction
    if 'eva02_pred' in oof_df.columns:
        ax = axes[1, 0]
        
        ax.scatter(tp_df['eva02_pred'], tp_df['edgenext_pred'], alpha=0.3, s=10, c='green', label='TP')
        ax.scatter(tn_df['eva02_pred'], tn_df['edgenext_pred'], alpha=0.05, s=5, c='blue', label='TN')
        ax.scatter(fp_df['eva02_pred'], fp_df['edgenext_pred'], alpha=0.7, s=30, c='orange', label='FP')
        ax.scatter(fn_df['eva02_pred'], fn_df['edgenext_pred'], alpha=0.9, s=50, c='red', marker='x', label='FN')
        
        ax.set_xlabel('EVA02 Prediction (Z-Score)')
        ax.set_ylabel('EdgeNeXt Prediction (Z-Score)')
        ax.set_title('Error Cases in Vision Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Error rate by prediction confidence
    ax = axes[1, 1]
    oof_df['confidence'] = abs(oof_df[pred_col] - 0.5) * 2  # 0 to 1
    
    confidence_bins = np.linspace(0, 1, 11)
    error_rates = []
    counts = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (oof_df['confidence'] >= confidence_bins[i]) & (oof_df['confidence'] < confidence_bins[i+1])
        subset = oof_df[mask]
        
        if len(subset) > 0:
            errors = ((subset['predicted'] != subset['target']).sum()) / len(subset)
            error_rates.append(errors)
            counts.append(len(subset))
        else:
            error_rates.append(0)
            counts.append(0)
    
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    
    ax2 = ax.twinx()
    ax.bar(bin_centers, counts, width=0.08, alpha=0.3, color='blue', label='Sample Count')
    ax2.plot(bin_centers, error_rates, 'ro-', lw=2, markersize=8, label='Error Rate')
    
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Sample Count', color='blue')
    ax2.set_ylabel('Error Rate', color='red')
    ax.set_title('Error Rate by Confidence')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Patient-level error analysis
    ax = axes[1, 2]
    
    if 'patient_id' in oof_df.columns:
        patient_stats = oof_df.groupby('patient_id').agg({
            'target': 'sum',
            'predicted': 'sum',
            'isic_id': 'count'
        }).rename(columns={'isic_id': 'lesion_count', 'target': 'true_malignant', 'predicted': 'pred_malignant'})
        
        patient_stats['has_error'] = patient_stats['true_malignant'] != patient_stats['pred_malignant']
        
        # Error rate by number of lesions
        patient_stats_reset = patient_stats.reset_index()
        error_by_lesions = patient_stats_reset.groupby('lesion_count').agg({
            'has_error': 'mean',
            'patient_id': 'count'
        }).rename(columns={'patient_id': 'patient_count'})
        
        # Limit to reasonable lesion counts
        error_by_lesions = error_by_lesions[error_by_lesions.index <= 50]
        
        ax.bar(error_by_lesions.index, error_by_lesions['has_error'], alpha=0.7)
        ax.set_xlabel('Lesions per Patient')
        ax.set_ylabel('Patient Error Rate')
        ax.set_title('Patient-Level Errors by Lesion Count')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved: error_analysis.png")


def plot_patient_analysis(oof_df, save_dir):
    """Analyze predictions at patient level."""
    if 'patient_id' not in oof_df.columns:
        print("⚠️ patient_id not found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Patient lesion count distribution
    ax = axes[0, 0]
    patient_counts = oof_df.groupby('patient_id')['isic_id'].count()
    
    ax.hist(patient_counts, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(x=patient_counts.median(), color='red', linestyle='--', 
               label=f'Median={patient_counts.median():.0f}')
    ax.axvline(x=patient_counts.mean(), color='orange', linestyle='--',
               label=f'Mean={patient_counts.mean():.1f}')
    
    ax.set_xlabel('Lesions per Patient')
    ax.set_ylabel('Number of Patients')
    ax.set_title(f'Patient Lesion Count Distribution\n({len(patient_counts):,} patients)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Malignant rate by lesion count
    ax = axes[0, 1]
    patient_stats = oof_df.groupby('patient_id').agg({
        'target': ['sum', 'count'],
        'ensemble_pred': 'max'
    })
    patient_stats.columns = ['malignant_count', 'total_count', 'max_pred']
    patient_stats['malignant_rate'] = patient_stats['malignant_count'] / patient_stats['total_count']
    
    # Bin by lesion count
    bins = [1, 2, 5, 10, 20, 50, 1000]
    patient_stats['lesion_bin'] = pd.cut(patient_stats['total_count'], bins=bins)
    
    malignant_by_bin = patient_stats.groupby('lesion_bin')['malignant_rate'].mean()
    
    ax.bar(range(len(malignant_by_bin)), malignant_by_bin.values, alpha=0.7, color='coral')
    ax.set_xticks(range(len(malignant_by_bin)))
    ax.set_xticklabels([str(b) for b in malignant_by_bin.index], rotation=45, ha='right')
    ax.set_xlabel('Lesions per Patient')
    ax.set_ylabel('Mean Malignant Rate')
    ax.set_title('Malignant Rate by Patient Lesion Count')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Max prediction vs actual malignant
    ax = axes[0, 2]
    has_malignant = patient_stats['malignant_count'] > 0
    
    ax.hist(patient_stats[~has_malignant]['max_pred'], bins=50, alpha=0.7, 
            label=f'No Malignant (n={sum(~has_malignant):,})', density=True, color='blue')
    ax.hist(patient_stats[has_malignant]['max_pred'], bins=50, alpha=0.7,
            label=f'Has Malignant (n={sum(has_malignant):,})', density=True, color='red')
    
    ax.set_xlabel('Max Prediction Score')
    ax.set_ylabel('Density')
    ax.set_title('Max Prediction by Patient Status')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Prediction variance within patient
    ax = axes[1, 0]
    patient_pred_var = oof_df.groupby('patient_id')['ensemble_pred'].std()
    patient_has_malignant = oof_df.groupby('patient_id')['target'].max() > 0
    
    ax.hist(patient_pred_var[~patient_has_malignant], bins=50, alpha=0.7,
            label='No Malignant', density=True, color='blue')
    ax.hist(patient_pred_var[patient_has_malignant], bins=50, alpha=0.7,
            label='Has Malignant', density=True, color='red')
    
    ax.set_xlabel('Within-Patient Prediction Std Dev')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Variance within Patient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rank of malignant lesion within patient
    ax = axes[1, 1]
    
    # For patients with malignant lesions, what's the rank of the malignant lesion?
    malignant_only = oof_df[oof_df['target'] == 1].copy()
    
    def get_rank_within_patient(row, df):
        patient_df = df[df['patient_id'] == row['patient_id']]
        sorted_preds = patient_df.sort_values('ensemble_pred', ascending=False)
        rank = sorted_preds.index.get_loc(row.name) + 1 if row.name in sorted_preds.index else -1
        return rank
    
    # Simpler approach: group and rank
    oof_df['pred_rank'] = oof_df.groupby('patient_id')['ensemble_pred'].rank(ascending=False)
    
    malignant_ranks = oof_df[oof_df['target'] == 1]['pred_rank']
    
    ax.hist(malignant_ranks, bins=50, alpha=0.7, color='coral', edgecolor='white')
    ax.axvline(x=1, color='green', linestyle='--', lw=2, label='Rank 1 (Best)')
    
    pct_rank1 = (malignant_ranks == 1).mean() * 100
    ax.set_xlabel('Rank within Patient (1 = Highest Pred)')
    ax.set_ylabel('Count')
    ax.set_title(f'Malignant Lesion Rank ({pct_rank1:.1f}% are Rank 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative sensitivity by top-K
    ax = axes[1, 2]
    
    oof_df['is_top_k'] = oof_df['pred_rank'] <= oof_df.groupby('patient_id')['pred_rank'].transform(lambda x: range(1, len(x)+1))
    
    k_values = range(1, 11)
    sensitivities = []
    
    for k in k_values:
        top_k = oof_df[oof_df['pred_rank'] <= k]
        detected = top_k[top_k['target'] == 1]['patient_id'].nunique()
        total_malignant_patients = oof_df[oof_df['target'] == 1]['patient_id'].nunique()
        sensitivity = detected / total_malignant_patients if total_malignant_patients > 0 else 0
        sensitivities.append(sensitivity)
    
    ax.plot(k_values, sensitivities, 'bo-', lw=2, markersize=8)
    ax.axhline(y=0.80, color='red', linestyle='--', label='80% Sensitivity')
    ax.axhline(y=0.95, color='green', linestyle='--', label='95% Sensitivity')
    
    ax.set_xlabel('Top-K Lesions per Patient')
    ax.set_ylabel('Patient-Level Sensitivity')
    ax.set_title('Sensitivity by Top-K Selection')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'patient_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved: patient_analysis.png")


def plot_fold_analysis(results, save_dir):
    """Analyze performance across folds."""
    if 'metrics' not in results:
        return
    
    metrics = results['metrics']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Fold scores comparison
    ax = axes[0]
    x = np.arange(5)
    width = 0.35
    
    lgbm_scores = metrics.get('lgbm', {}).get('fold_scores', [])
    xgb_scores = metrics.get('xgb', {}).get('fold_scores', [])
    
    if lgbm_scores:
        ax.bar(x - width/2, lgbm_scores, width, label='LightGBM', color='steelblue')
    if xgb_scores:
        ax.bar(x + width/2, xgb_scores, width, label='XGBoost', color='coral')
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('AUC Score')
    ax.set_title('Fold-wise AUC Scores')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean line
    if lgbm_scores:
        ax.axhline(y=np.mean(lgbm_scores), color='steelblue', linestyle='--', alpha=0.7)
    if xgb_scores:
        ax.axhline(y=np.mean(xgb_scores), color='coral', linestyle='--', alpha=0.7)
    
    # Overall comparison
    ax = axes[1]
    
    models = []
    aucs = []
    paucs = []
    
    for model in ['lgbm', 'xgb', 'ensemble']:
        if model in metrics:
            models.append(model.upper())
            aucs.append(metrics[model]['auc'])
            paucs.append(metrics[model]['pauc'])
    
    x = np.arange(len(models))
    ax.bar(x - width/2, aucs, width, label='AUC', color='steelblue')
    ax.bar(x + width/2, paucs, width, label='pAUC', color='coral')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Overall Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (a, p) in enumerate(zip(aucs, paucs)):
        ax.text(i - width/2, a + 0.005, f'{a:.4f}', ha='center', fontsize=9)
        ax.text(i + width/2, p + 0.005, f'{p:.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Saved: fold_analysis.png")


def generate_summary_report(results, save_dir):
    """Generate a text summary report."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("STACKING GBDT MODEL - ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Config
    if 'config' in results:
        config = results['config']
        report_lines.append("## TRAINING CONFIGURATION")
        report_lines.append(f"   Timestamp: {config.get('timestamp', 'N/A')}")
        report_lines.append(f"   Samples: {config.get('n_samples', 'N/A'):,}")
        report_lines.append(f"   Features: {config.get('n_features', 'N/A')}")
        report_lines.append(f"   Noise Std: {config.get('noise_std', 'N/A')}")
        report_lines.append(f"   Leakage Excluded: {config.get('leakage_features_excluded', [])}")
        report_lines.append("")
    
    # Metrics
    if 'metrics' in results:
        metrics = results['metrics']
        report_lines.append("## MODEL PERFORMANCE")
        
        for model in ['lgbm', 'xgb', 'ensemble']:
            if model in metrics:
                m = metrics[model]
                report_lines.append(f"   {model.upper()}:")
                report_lines.append(f"      AUC:  {m['auc']:.5f}")
                report_lines.append(f"      pAUC: {m['pauc']:.5f}")
                if 'fold_scores' in m:
                    report_lines.append(f"      Fold Scores: {[f'{s:.4f}' for s in m['fold_scores']]}")
        report_lines.append("")
    
    # Top features
    if 'lgbm_importance' in results:
        report_lines.append("## TOP 20 FEATURES (LightGBM)")
        for i, row in results['lgbm_importance'].head(20).iterrows():
            report_lines.append(f"   {i+1:2d}. {row['feature']}: {row['importance']:.1f}")
        report_lines.append("")
    
    # Vision features
    if 'lgbm_importance' in results:
        vision_feats = results['lgbm_importance'][
            results['lgbm_importance']['feature'].str.contains('pred|vision', case=False)
        ]
        if len(vision_feats) > 0:
            report_lines.append("## VISION MODEL FEATURES")
            for i, row in vision_feats.iterrows():
                report_lines.append(f"   {row['feature']}: {row['importance']:.1f}")
            report_lines.append("")
    
    # OOF stats
    if 'oof' in results:
        oof = results['oof']
        report_lines.append("## OOF PREDICTION STATISTICS")
        report_lines.append(f"   Samples: {len(oof):,}")
        report_lines.append(f"   Positive Rate: {oof['target'].mean():.4f}")
        
        if 'ensemble_pred' in oof.columns:
            report_lines.append(f"   Ensemble Pred Mean: {oof['ensemble_pred'].mean():.4f}")
            report_lines.append(f"   Ensemble Pred Std:  {oof['ensemble_pred'].std():.4f}")
            report_lines.append(f"   Ensemble Pred Min:  {oof['ensemble_pred'].min():.4f}")
            report_lines.append(f"   Ensemble Pred Max:  {oof['ensemble_pred'].max():.4f}")
        report_lines.append("")
    
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)
    
    report_text = "\n".join(report_lines)
    
    with open(save_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n📄 Report saved to: {save_dir / 'analysis_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Stacking GBDT Results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to results directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: results_dir/analysis)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("📊 STACKING GBDT RESULTS ANALYSIS")
    print("=" * 70)
    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}")
    print()
    
    # Load results
    print("📂 Loading Results...")
    results = load_results(results_dir)
    
    if 'oof' not in results:
        print("❌ OOF predictions not found. Cannot proceed.")
        return
    
    oof_df = results['oof']
    
    # Generate all visualizations
    print("\n🎨 Generating Visualizations...")
    
    print("\n1/8 Prediction Distributions...")
    plot_prediction_distributions(oof_df, output_dir)
    
    print("\n2/8 ROC Curves...")
    plot_roc_curves(oof_df, output_dir)
    
    print("\n3/8 Precision-Recall...")
    plot_precision_recall(oof_df, output_dir)
    
    print("\n4/8 Calibration Curves...")
    plot_calibration_curves(oof_df, output_dir)
    
    print("\n5/8 Feature Importance...")
    plot_feature_importance(results, output_dir)
    
    print("\n6/8 Vision Model Analysis...")
    plot_vision_model_analysis(oof_df, output_dir)
    
    print("\n7/8 Error Analysis...")
    plot_error_analysis(oof_df, results, output_dir)
    
    print("\n8/8 Patient Analysis...")
    plot_patient_analysis(oof_df, output_dir)
    
    print("\n9/8 Fold Analysis...")
    plot_fold_analysis(results, output_dir)
    
    # Generate summary report
    print("\n📝 Generating Summary Report...")
    generate_summary_report(results, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📁 All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"   - {f.name}")


if __name__ == '__main__':
    main()
