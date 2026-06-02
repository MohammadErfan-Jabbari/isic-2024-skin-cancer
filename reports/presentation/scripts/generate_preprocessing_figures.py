"""
Generate presentation figures for Preprocessing & Augmentation section.
Creates professional, dark-themed plots for Slidev presentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
from PIL import Image
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'metadata_investigation/results'
FIGURES_DIR = BASE_DIR / 'presentation/public/figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Dark theme styling
plt.style.use('dark_background')
COLORS = {
    'primary': '#60A5FA',    # Blue
    'secondary': '#34D399',  # Green
    'accent': '#F472B6',     # Pink
    'warning': '#FBBF24',    # Yellow
    'text': '#E5E7EB',       # Light gray
    'bg': '#1F2937',         # Dark gray
}

def set_style():
    """Set consistent matplotlib style for all plots."""
    plt.rcParams.update({
        'figure.facecolor': '#0F172A',
        'axes.facecolor': '#1E293B',
        'axes.edgecolor': '#475569',
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': '#334155',
        'grid.alpha': 0.5,
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })

# ============================================
# PLOT 1: Augmentation Example Grid
# ============================================
def generate_augmentation_grid():
    """Create 3x3 grid showing augmentation transforms on a real lesion."""
    print("Generating augmentation grid...")
    
    # Load a malignant sample from HDF5
    train_meta = pd.read_csv(DATA_DIR / 'new-train-metadata.csv')
    malignant_ids = train_meta[train_meta['target'] == 1]['isic_id'].values
    
    with h5py.File(DATA_DIR / 'train-image-384.hdf5', 'r') as f:
        # Find first available malignant image
        for img_id in malignant_ids[:20]:
            if img_id in f:
                img_array = f[img_id][:]
                break
    
    # Define transforms to show
    transforms = [
        ('Original', None),
        ('Horizontal Flip', A.HorizontalFlip(p=1.0)),
        ('Vertical Flip', A.VerticalFlip(p=1.0)),
        ('Rotate 90°', A.RandomRotate90(p=1.0)),
        ('Color Jitter', A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0)),
        ('Gaussian Blur', A.GaussianBlur(blur_limit=(7, 7), p=1.0)),
        ('Motion Blur', A.MotionBlur(blur_limit=9, p=1.0)),
        ('Gaussian Noise', A.GaussNoise(std_range=(0.05, 0.1), p=1.0)),
        ('Transpose', A.Transpose(p=1.0)),
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.patch.set_facecolor('#0F172A')
    
    for idx, (ax, (name, transform)) in enumerate(zip(axes.flat, transforms)):
        if transform is None:
            img_transformed = img_array
        else:
            result = transform(image=img_array)
            img_transformed = result['image']
        
        ax.imshow(img_transformed)
        ax.set_title(name, fontsize=14, fontweight='bold', color=COLORS['text'], pad=10)
        ax.axis('off')
        
        # Add subtle border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#475569')
            spine.set_linewidth(2)
    
    plt.suptitle('Data Augmentation Examples', fontsize=18, fontweight='bold', 
                 color=COLORS['primary'], y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = FIGURES_DIR / 'augmentation_grid.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    print(f"  Saved: {save_path}")

# ============================================
# PLOT 2: Top Feature Correlations
# ============================================
def generate_correlation_plot():
    """Create horizontal bar chart of top feature correlations with target."""
    print("Generating correlation plot...")
    
    # Load correlation data (first column is feature name as index)
    corr_df = pd.read_csv(RESULTS_DIR / 'phase1_target_correlations.csv', index_col=0)
    corr_df = corr_df.reset_index().rename(columns={'index': 'feature'})
    
    # Get top 15 by absolute correlation
    corr_df['abs_corr'] = corr_df['correlation'].abs()
    top_features = corr_df.nlargest(15, 'abs_corr').sort_values('correlation')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0F172A')
    
    colors = [COLORS['secondary'] if x > 0 else COLORS['accent'] for x in top_features['correlation']]
    
    bars = ax.barh(top_features['feature'], top_features['correlation'], color=colors, edgecolor='none', height=0.7)
    
    ax.axvline(x=0, color='#475569', linewidth=1, linestyle='-')
    ax.set_xlabel('Correlation with Target', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Features by Target Correlation', fontsize=16, fontweight='bold', 
                 color=COLORS['primary'], pad=20)
    
    # Clean up feature names for display
    ax.set_yticklabels([f.replace('tbp_lv_', '').replace('_', ' ') for f in top_features['feature']])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, top_features['correlation']):
        x_pos = val + (0.002 if val > 0 else -0.002)
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha=ha, fontsize=9, color=COLORS['text'])
    
    plt.tight_layout()
    
    save_path = FIGURES_DIR / 'feature_correlations.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    print(f"  Saved: {save_path}")

# ============================================
# PLOT 3: Train/Test Distribution Shift
# ============================================
def generate_distribution_shift_plot():
    """Create side-by-side histograms for features with distribution shift."""
    print("Generating distribution shift plot...")
    
    # Load data
    train_meta = pd.read_csv(DATA_DIR / 'new-train-metadata.csv')
    test_meta = pd.read_csv(DATA_DIR / 'students-test-metadata.csv')
    
    # Features with significant shift (from your analysis)
    shift_features = ['tbp_lv_H', 'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_B']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#0F172A')
    
    for ax, feat in zip(axes.flat, shift_features):
        train_vals = train_meta[feat].dropna()
        test_vals = test_meta[feat].dropna()
        
        # Plot histograms
        ax.hist(train_vals, bins=50, alpha=0.7, color=COLORS['primary'], label='Train', density=True)
        ax.hist(test_vals, bins=50, alpha=0.7, color=COLORS['accent'], label='Test', density=True)
        
        # Add mean lines
        ax.axvline(train_vals.mean(), color=COLORS['primary'], linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(test_vals.mean(), color=COLORS['accent'], linestyle='--', linewidth=2, alpha=0.8)
        
        # Calculate shift in std units
        shift_std = (test_vals.mean() - train_vals.mean()) / train_vals.std()
        
        ax.set_title(f'{feat.replace("tbp_lv_", "")}\n(shift: {shift_std:.2f}σ)', 
                    fontsize=12, fontweight='bold', color=COLORS['text'])
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Train/Test Distribution Shift', fontsize=16, fontweight='bold', 
                 color=COLORS['warning'], y=1.02)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / 'distribution_shift.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    print(f"  Saved: {save_path}")

# ============================================
# PLOT 4: Missing Values (Lower Priority)
# ============================================
def generate_missing_values_plot():
    """Create bar chart of features with missing values."""
    print("Generating missing values plot...")
    
    missing_df = pd.read_csv(RESULTS_DIR / 'phase1_missing_values.csv')
    missing_df = missing_df[missing_df['missing_pct'] > 0].nlargest(15, 'missing_pct')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0F172A')
    
    bars = ax.barh(missing_df['column'], missing_df['missing_pct'] * 100, 
                   color=COLORS['warning'], edgecolor='none', height=0.6)
    
    ax.set_xlabel('Missing %', fontsize=12, fontweight='bold')
    ax.set_title('Features with Missing Values', fontsize=16, fontweight='bold', 
                 color=COLORS['warning'], pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, missing_df['missing_pct'] * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                va='center', fontsize=9, color=COLORS['text'])
    
    plt.tight_layout()
    
    save_path = FIGURES_DIR / 'missing_values.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    print(f"  Saved: {save_path}")

# ============================================
# PLOT 5: Age Distribution (Lower Priority)
# ============================================
def generate_age_distribution_plot():
    """Create age distribution by target class."""
    print("Generating age distribution plot...")
    
    train_meta = pd.read_csv(DATA_DIR / 'new-train-metadata.csv')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0F172A')
    
    # Plot by target
    benign = train_meta[train_meta['target'] == 0]['age_approx'].dropna()
    malignant = train_meta[train_meta['target'] == 1]['age_approx'].dropna()
    
    ax.hist(benign, bins=30, alpha=0.7, color=COLORS['secondary'], label=f'Benign (mean: {benign.mean():.1f})', density=True)
    ax.hist(malignant, bins=30, alpha=0.7, color=COLORS['accent'], label=f'Malignant (mean: {malignant.mean():.1f})', density=True)
    
    ax.axvline(benign.mean(), color=COLORS['secondary'], linestyle='--', linewidth=2)
    ax.axvline(malignant.mean(), color=COLORS['accent'], linestyle='--', linewidth=2)
    
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Age Distribution by Target', fontsize=16, fontweight='bold', 
                 color=COLORS['primary'], pad=20)
    ax.legend(loc='upper right', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    save_path = FIGURES_DIR / 'age_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    print(f"  Saved: {save_path}")

# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    set_style()
    
    print("=" * 50)
    print("Generating Presentation Figures")
    print("=" * 50)
    
    generate_augmentation_grid()
    generate_correlation_plot()
    generate_distribution_shift_plot()
    generate_missing_values_plot()
    generate_age_distribution_plot()
    
    print("=" * 50)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 50)
