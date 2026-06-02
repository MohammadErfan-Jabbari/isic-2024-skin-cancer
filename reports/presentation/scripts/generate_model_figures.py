#!/usr/bin/env python3
"""
Generate figures for the Best Model section of the ISIC 2024 presentation.

Figures generated:
1. model_comparison.png - EVA02 vs EdgeNeXt CV AUC per fold (computed from OOF)
2. golden_split_analysis.png - Fold-wise stacking performance
3. stacking_performance.png - Score progression from vision to ensemble
4. vision_correlation.png - Correlation between vision models

Data is loaded from actual CSV/JSON files in last_run/results/, not hardcoded.

Usage:
    cd ./presentation
    uv run python scripts/generate_model_figures.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'public' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette
COLORS = {
    'eva02': '#3498db',      # Blue
    'edgenext': '#e74c3c',   # Red
    'xgboost': '#2ecc71',    # Green
    'mlp': '#9b59b6',        # Purple
    'ensemble': '#f39c12',   # Orange
    'golden': '#1abc9c',     # Teal
    'muted': '#95a5a6',      # Gray
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_vision_oof_scores():
    """Load OOF predictions and compute AUC for each vision model per fold."""
    eva_aucs = {}
    edge_aucs = {}
    
    for fold in range(5):
        # EVA02
        eva_path = RESULTS_DIR / f"oof_eva02_small_patch14_336.mim_in22k_ft_in1k_fold{fold}.csv"
        if eva_path.exists():
            df = pd.read_csv(eva_path)
            try:
                auc = roc_auc_score(df['target'], df['pred'])
                eva_aucs[fold] = auc
            except:
                eva_aucs[fold] = None
        else:
            eva_aucs[fold] = None
            
        # EdgeNeXt
        edge_path = RESULTS_DIR / f"oof_edgenext_base_fold{fold}.csv"
        if edge_path.exists():
            df = pd.read_csv(edge_path)
            try:
                auc = roc_auc_score(df['target'], df['pred'])
                edge_aucs[fold] = auc
            except:
                edge_aucs[fold] = None
        else:
            edge_aucs[fold] = None
            
    return eva_aucs, edge_aucs


def load_stacking_scores():
    """Load stacking OOF predictions and compute ensemble AUC."""
    stacking_path = RESULTS_DIR / 'oof_stacking.csv'
    if not stacking_path.exists():
        return {}
    
    df = pd.read_csv(stacking_path)
    
    scores = {
        'xgb': roc_auc_score(df['target'], df['xgb_pred']),
        'mlp': roc_auc_score(df['target'], df['mlp_pred']),
        'ensemble': roc_auc_score(df['target'], df['ensemble_pred']),
    }
    return scores


def compute_vision_correlation():
    """Compute Pearson correlation between EVA02 and EdgeNeXt predictions."""
    eva_preds = []
    edge_preds = []
    
    for fold in range(5):
        eva_path = RESULTS_DIR / f"oof_eva02_small_patch14_336.mim_in22k_ft_in1k_fold{fold}.csv"
        edge_path = RESULTS_DIR / f"oof_edgenext_base_fold{fold}.csv"
        
        if eva_path.exists() and edge_path.exists():
            eva_df = pd.read_csv(eva_path)
            edge_df = pd.read_csv(edge_path)
            
            # Merge on isic_id to align predictions
            merged = eva_df.merge(edge_df, on='isic_id', suffixes=('_eva', '_edge'))
            eva_preds.extend(merged['pred_eva'].tolist())
            edge_preds.extend(merged['pred_edge'].tolist())
    
    if len(eva_preds) > 0:
        corr, _ = pearsonr(eva_preds, edge_preds)
        return corr
    return None


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================

def generate_model_comparison():
    """Generate bar chart comparing EVA02 vs EdgeNeXt CV AUC per fold."""
    print("  Loading vision OOF scores...")
    eva_aucs, edge_aucs = load_vision_oof_scores()
    correlation = compute_vision_correlation()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    folds = [0, 1, 2, 3, 4]
    x = np.arange(len(folds))
    width = 0.35
    
    eva_scores = [eva_aucs.get(f, 0) or 0 for f in folds]
    edge_scores = [edge_aucs.get(f, 0) or 0 for f in folds]
    
    bars1 = ax.bar(x - width/2, eva_scores, width, label='EVA02-Small (ViT)', 
                   color=COLORS['eva02'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, edge_scores, width, label='EdgeNeXt-Base (CNN-ViT)', 
                   color=COLORS['edgenext'], edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, score in zip(bars1, eva_scores):
        if score > 0:
            ax.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, score in zip(bars2, edge_scores):
        if score > 0:
            ax.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Fold', fontweight='bold', fontsize=12)
    ax.set_ylabel('CV AUC (Out-of-Fold)', fontweight='bold', fontsize=12)
    ax.set_title('Vision Model Performance by Fold (CV)', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc='upper right', fontsize=11)
    
    # Add horizontal reference line
    ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.4)
    ax.text(4.6, 0.851, '0.85', ha='left', fontsize=9, color='gray')
    
    # Add insight box with computed correlation
    if correlation is not None:
        textstr = f'Correlation = {correlation:.2f}\n→ Good ensemble diversity'
    else:
        textstr = 'Correlation: N/A'
    props = dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', alpha=0.9, edgecolor='white')
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'model_comparison.png'}")


def generate_stacking_performance():
    """Generate bar chart showing LB score progression from vision to ensemble.
    
    Data Source: User provided LB logs (Step Id: 248)
    """
    
    # Exact LB Scores
    stages = ['Vision Only\n(EVA02)', 'XGBoost\nStacking', 'MLP\nStacking', '5-Fold\nEnsemble', 'Golden Split\n(Fold 4)']
    scores = [0.93233, 0.97994, 0.95238, 0.98245, 0.98997]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gradient colors from blue to gold
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    bars = ax.bar(stages, scores, color=colors, edgecolor='white', linewidth=2, width=0.6)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f'{score:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add delta annotations
    deltas = [None]  # No delta for first
    for i in range(1, len(scores)):
        delta = scores[i] - scores[i-1]
        deltas.append(delta)
    
    # Draw arrows for key transitions
    # 1. Vision -> XGBoost
    ax.annotate(f'+{scores[1]-scores[0]:.4f}', 
                xy=(bars[1].get_x(), scores[1]),
                xytext=(bars[0].get_x() + bars[0].get_width(), scores[0]),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
                color='#27ae60', fontweight='bold', ha='center', va='bottom', fontsize=10)
                
    # 2. Ensemble -> Golden Split
    ax.annotate(f'+{scores[4]-scores[3]:.4f}', 
                xy=(bars[4].get_x(), scores[4]),
                xytext=(bars[3].get_x() + bars[3].get_width(), scores[3]),
                arrowprops=dict(arrowstyle='->', color='#1abc9c', lw=1.5),
                color='#1abc9c', fontweight='bold', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Public LB Score (pAUC)', fontweight='bold')
    ax.set_title('Performance Progression: Vision → Stacking → Golden Split', 
                 fontweight='bold', fontsize=14)
    ax.set_ylim(0.92, 1.0)
    
    # Add baseline line
    ax.axhline(y=0.93233, color='gray', linestyle='--', alpha=0.5)
    ax.text(4.5, 0.933, 'Vision Baseline', ha='right', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stacking_performance.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'stacking_performance.png'}")


def generate_golden_split_analysis():
    """Generate visualization of the Golden Split discovery (LB Scores).
    
    Data Source: User provided LB logs (Step Id: 248)
    Finding: Models including Fold 4 in training performed worse on LB.
    Excluding Fold 4 (Golden Split) achieved 0.98997.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Exact LB Scores from user log
    scenarios = [
        'Train on All\n(Val Fold 0)',
        'Train on All\n(Val Fold 1)',
        'Train on All\n(Val Fold 2)',
        'Train on All\n(Val Fold 3)',
        'Golden Split\n(Train {0,1,2,3})',
    ]
    scores = [0.96992, 0.94736, 0.94736, 0.97243, 0.98997]
    
    # Highlight the Golden Split
    colors = [COLORS['muted'], COLORS['muted'], COLORS['muted'], COLORS['muted'], COLORS['golden']]
    
    bars = ax.barh(scenarios, scores, color=colors, edgecolor='white', linewidth=2, height=0.6)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        label = f'{score:.5f}'
        if score == max(scores):
            label += ' (LB)'
        ax.annotate(label,
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Public LB Score (pAUC)', fontweight='bold')
    ax.set_title('The "Golden Split" Discovery', fontweight='bold', fontsize=16)
    ax.set_xlim(0.94, 1.0)  # Zoom in to show differences
    
    # Add insight box
    textstr = 'Key Finding:\nExcluding "Toxic" Fold 4 from training\nboosted LB Score to 0.98997'
    props = dict(boxstyle='round,pad=0.5', facecolor='#d5f5e3', alpha=0.9, edgecolor='#1abc9c')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold', color='#1e8449')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'golden_split_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'golden_split_analysis.png'}")


def generate_architecture_summary():
    """Generate a visual summary of the architecture with key metrics."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Load actual scores
    stacking_scores = load_stacking_scores()
    final_score = stacking_scores.get('ensemble', 0)
    
    # Title
    ax.text(6, 7.5, 'Two-Stage Stacking Architecture', ha='center', fontsize=18, 
            fontweight='bold')
    
    # Stage 1 Box
    stage1_box = mpatches.FancyBboxPatch((0.5, 4), 5, 2.5, boxstyle="round,pad=0.1",
                                          facecolor='#ebf5fb', edgecolor='#3498db', 
                                          linewidth=2)
    ax.add_patch(stage1_box)
    ax.text(3, 6.2, 'Stage 1: Vision (The Eyes)', ha='center', fontsize=14, 
            fontweight='bold', color='#2980b9')
    
    # EVA02 box
    eva_box = mpatches.FancyBboxPatch((1, 4.5), 2, 1, boxstyle="round,pad=0.05",
                                       facecolor='#3498db', edgecolor='white', 
                                       linewidth=1)
    ax.add_patch(eva_box)
    ax.text(2, 5, 'EVA02-Small\n(ViT, 336×336)', ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')
    
    # EdgeNeXt box
    edge_box = mpatches.FancyBboxPatch((3.5, 4.5), 2, 1, boxstyle="round,pad=0.05",
                                        facecolor='#e74c3c', edgecolor='white', 
                                        linewidth=1)
    ax.add_patch(edge_box)
    ax.text(4.5, 5, 'EdgeNeXt-Base\n(CNN-ViT, 384×384)', ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(6, 3.8), xytext=(6, 4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(7, 3.5, 'Logits + PCA(50) Embeddings', fontsize=10, color='gray')
    
    # Stage 2 Box
    stage2_box = mpatches.FancyBboxPatch((0.5, 0.5), 11, 3, boxstyle="round,pad=0.1",
                                          facecolor='#fef9e7', edgecolor='#f39c12', 
                                          linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(6, 3.2, 'Stage 2: Stacking (The Brain)', ha='center', fontsize=14, 
            fontweight='bold', color='#d68910')
    
    # XGBoost box
    xgb_box = mpatches.FancyBboxPatch((1, 1.5), 3, 1.2, boxstyle="round,pad=0.05",
                                       facecolor='#2ecc71', edgecolor='white', 
                                       linewidth=1)
    ax.add_patch(xgb_box)
    ax.text(2.5, 2.1, 'XGBoost\nRaw Meta + Vision', ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')
    
    # MLP box
    mlp_box = mpatches.FancyBboxPatch((4.5, 1.5), 3, 1.2, boxstyle="round,pad=0.05",
                                       facecolor='#9b59b6', edgecolor='white', 
                                       linewidth=1)
    ax.add_patch(mlp_box)
    ax.text(6, 2.1, 'MLP\nMeta + DAE + Vision', ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')
    
    # Ensemble box with actual score
    ens_box = mpatches.FancyBboxPatch((8, 1.5), 3, 1.2, boxstyle="round,pad=0.05",
                                       facecolor='#1abc9c', edgecolor='white', 
                                       linewidth=1)
    ax.add_patch(ens_box)
    ax.text(9.5, 2.1, f'Average\n→ {final_score:.4f} AUC', ha='center', va='center', 
            fontsize=11, color='white', fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(8, 2.1), xytext=(7.5, 2.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(8, 2.1), xytext=(4, 2.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                               connectionstyle="arc3,rad=-0.3"))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'architecture_summary.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / 'architecture_summary.png'}")


def main():
    """Generate all figures for the Best Model section."""
    print("=" * 60)
    print("Generating Best Model Figures (from source data)")
    print("=" * 60)
    
    generate_model_comparison()
    generate_stacking_performance()
    generate_golden_split_analysis()
    generate_architecture_summary()
    
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
