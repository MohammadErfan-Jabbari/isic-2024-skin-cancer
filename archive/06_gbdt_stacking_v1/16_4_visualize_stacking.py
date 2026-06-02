import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from pathlib import Path

# Configuration
RESULTS_DIR = Path('DeepLearning/Kaggle/results/stacking_final_v1')
OOF_FILE = RESULTS_DIR / 'stacking_oof.csv'
VIS_DIR = RESULTS_DIR / 'visualizations'
VIS_DIR.mkdir(parents=True, exist_ok=True)

def score_pauc(y_true, y_pred, min_tpr=0.80):
    """Calculates pAUC above a minimum TPR threshold."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        mask = tpr >= min_tpr
        if mask.sum() < 2: return 0.0
        return auc(fpr[mask], tpr[mask])
    except:
        return 0.0

def plot_roc_comparison(df):
    plt.figure(figsize=(10, 8))
    
    models = {
        'EVA02 (Vision)': 'eva02_pred',
        'EdgeNeXt (Vision)': 'edgenext_pred',
        'Stacking Ensemble': 'stack_pred'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for (name, col), color in zip(models.items(), colors):
        if col not in df.columns: continue
        
        fpr, tpr, _ = roc_curve(df['target'], df[col])
        roc_auc = auc(fpr, tpr)
        p_auc = score_pauc(df['target'], df[col])
        
        plt.plot(fpr, tpr, color=color, lw=2, 
                 label=f'{name} (AUC = {roc_auc:.4f}, pAUC = {p_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison: Vision vs. Stacking')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(VIS_DIR / 'roc_comparison.png', dpi=300)
    print(f"Saved ROC comparison to {VIS_DIR / 'roc_comparison.png'}")

def plot_pauc_zoom(df):
    """Plots the ROC curve zoomed in on the critical region (TPR > 0.8)."""
    plt.figure(figsize=(10, 8))
    
    models = {
        'EVA02 (Vision)': 'eva02_pred',
        'EdgeNeXt (Vision)': 'edgenext_pred',
        'Stacking Ensemble': 'stack_pred'
    }
    
    for name, col in models.items():
        if col not in df.columns: continue
        
        fpr, tpr, _ = roc_curve(df['target'], df[col])
        
        # Filter for TPR > 0.8
        mask = tpr >= 0.8
        plt.plot(fpr[mask], tpr[mask], lw=2, label=name)
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Critical Region (TPR > 0.8) Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(VIS_DIR / 'pauc_zoom.png', dpi=300)
    print(f"Saved pAUC zoom to {VIS_DIR / 'pauc_zoom.png'}")

def plot_calibration(df):
    plt.figure(figsize=(10, 8))
    
    models = {
        'EVA02': 'eva02_pred',
        'Stacking': 'stack_pred'
    }
    
    for name, col in models.items():
        if col not in df.columns: continue
        
        prob_true, prob_pred = calibration_curve(df['target'], df[col], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)
        
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(VIS_DIR / 'calibration_plot.png', dpi=300)
    print(f"Saved calibration plot to {VIS_DIR / 'calibration_plot.png'}")

def plot_score_distribution(df):
    plt.figure(figsize=(12, 6))
    
    # Log scale for better visibility of low probabilities
    sns.histplot(data=df, x='stack_pred', hue='target', bins=50, log_scale=(False, True), common_norm=False)
    
    plt.title('Distribution of Stacking Predictions (Log Scale)')
    plt.xlabel('Predicted Probability')
    plt.savefig(VIS_DIR / 'score_distribution.png', dpi=300)
    print(f"Saved distribution plot to {VIS_DIR / 'score_distribution.png'}")

def plot_feature_importance(results_dir):
    fi_path = results_dir / 'feature_importance.csv'
    if not fi_path.exists():
        print("⚠️ Feature importance file not found.")
        return
        
    df = pd.read_csv(fi_path)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(data=df.head(20), x='importance', y='feature', hue='feature', palette='viridis', legend=False)
    plt.title('Top 20 Features in Stacking Model')
    plt.xlabel('Importance (Split)')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'feature_importance.png')
    plt.close()
    print(f"✅ Saved Feature Importance Plot to {VIS_DIR / 'feature_importance.png'}")

def main():
    print("📊 Generating Visualizations...")
    if not OOF_FILE.exists():
        print(f"❌ Error: {OOF_FILE} not found.")
        return
        
    df = pd.read_csv(OOF_FILE)
    
    # Ensure predictions are floats
    for col in ['eva02_pred', 'edgenext_pred', 'stack_pred']:
        if col in df.columns and df[col].dtype == object:
             df[col] = df[col].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)

    plot_roc_comparison(df)
    plot_pauc_zoom(df)
    plot_calibration(df)
    plot_score_distribution(df)
    plot_feature_importance(RESULTS_DIR)
    
    print("✅ Visualization Complete.")

if __name__ == '__main__':
    main()
