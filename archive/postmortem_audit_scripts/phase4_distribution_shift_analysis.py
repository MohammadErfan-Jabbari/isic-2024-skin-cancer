#!/usr/bin/env python3
"""
Phase 4: Train/Test Distribution Shift Deep Dive

This script analyzes the distribution shift between training and test vision predictions
to understand why the stacking pipeline failed with poor scores (0.48-0.49).

Key Questions:
1. How different are vision model predictions on test vs train?
2. Does z-score standardization handle the distribution shift correctly?
3. What impact does this have on the final stacking predictions?

Author: Kilo Code
Date: 2025-11-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.stats import ks_2samp, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path('.')
RESULTS_DIR = BASE_DIR / 'results'
STACKING_DIR = RESULTS_DIR / 'stacking_final_v1'
POST_ANALYSIS_DIR = BASE_DIR / 'post_feature_analysis'
RESULTS_OUT_DIR = POST_ANALYSIS_DIR / 'results'
PLOTS_OUT_DIR = POST_ANALYSIS_DIR / 'plots'

# Ensure output directories exist
RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Phase 4: Train/Test Distribution Shift Analysis ===")
print(f"Output directory: {RESULTS_OUT_DIR}")
print(f"Plots directory: {PLOTS_OUT_DIR}")

# Load test vision predictions
print("\n1. Loading test vision predictions...")
test_preds_path = STACKING_DIR / 'test_vision_preds.csv'
if not test_preds_path.exists():
    print(f"ERROR: Test predictions file not found: {test_preds_path}")
    exit(1)

test_preds = pd.read_csv(test_preds_path)
print(f"Test predictions loaded: {len(test_preds)} samples")
print(f"Columns: {list(test_preds.columns)}")

# Load training OOF predictions (combine all folds)
print("\n2. Loading training OOF predictions...")
eva_dirs = [
    RESULTS_DIR / 'gen-train-run-eva-v2',
    RESULTS_DIR / 'eva02_exp_v1'
]

edgenext_dirs = [
    RESULTS_DIR / 'gen-train-run-edgenext-v2',
    RESULTS_DIR / 'edgenext_exp_v1'
]

# Try to find the correct directories
eva_dir = None
for dir_path in eva_dirs:
    if dir_path.exists() and (dir_path / 'oof_fold1.csv').exists():
        eva_dir = dir_path
        break

edgenext_dir = None
for dir_path in edgenext_dirs:
    if dir_path.exists() and (dir_path / 'oof_fold1.csv').exists():
        edgenext_dir = dir_path
        break

if eva_dir is None or edgenext_dir is None:
    print(f"ERROR: Could not find OOF prediction directories")
    print(f"EVA02 dir found: {eva_dir}")
    print(f"EdgeNeXt dir found: {edgenext_dir}")
    exit(1)

print(f"Using EVA02 directory: {eva_dir}")
print(f"Using EdgeNeXt directory: {edgenext_dir}")

# Load all EVA02 OOFs
eva_dfs = []
for fold in range(1, 6):
    oof_path = eva_dir / f'oof_fold{fold}.csv'
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        # Extract only isic_id and pred
        if 'pred' in oof.columns:
            oof['pred'] = oof['pred'].apply(lambda x: float(str(x).strip('[]')) if isinstance(x, str) else x)
            eva_dfs.append(oof[['isic_id', 'pred']].rename(columns={'pred': 'eva02_pred'}))
        else:
            print(f"Warning: 'pred' column not found in {oof_path}")
            print(f"Available columns: {list(oof.columns)}")

if not eva_dfs:
    print("ERROR: No EVA02 OOF files found")
    exit(1)

eva_all = pd.concat(eva_dfs, ignore_index=True)
print(f"EVA02 OOF loaded: {len(eva_all)} samples")

# Load all EdgeNeXt OOFs
edge_dfs = []
for fold in range(1, 6):
    oof_path = edgenext_dir / f'oof_fold{fold}.csv'
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        # Extract only isic_id and pred
        if 'pred' in oof.columns:
            oof['pred'] = oof['pred'].apply(lambda x: float(str(x).strip('[]')) if isinstance(x, str) else x)
            edge_dfs.append(oof[['isic_id', 'pred']].rename(columns={'pred': 'edgenext_pred'}))
        else:
            print(f"Warning: 'pred' column not found in {oof_path}")
            print(f"Available columns: {list(oof.columns)}")

if not edge_dfs:
    print("ERROR: No EdgeNeXt OOF files found")
    exit(1)

edge_all = pd.concat(edge_dfs, ignore_index=True)
print(f"EdgeNeXt OOF loaded: {len(edge_all)} samples")

# Merge training predictions
train_preds = eva_all.merge(edge_all, on='isic_id', how='inner')
print(f"Training predictions merged: {len(train_preds)} samples")

# Load training metadata for target information
train_meta_path = BASE_DIR / 'data' / 'new-train-metadata.csv'
if train_meta_path.exists():
    train_meta = pd.read_csv(train_meta_path, low_memory=False)
    train_preds = train_preds.merge(train_meta[['isic_id', 'target']], on='isic_id', how='left')
    print(f"Training metadata merged: {len(train_preds)} samples with targets")
else:
    print("Warning: Training metadata not found, proceeding without targets")
    train_preds['target'] = np.nan

print("\n3. Analyzing distribution shift...")

# Compare distributions
distribution_results = []

for col in ['eva02_pred', 'edgenext_pred']:
    print(f"\n=== {col} Distribution Analysis ===")
    
    train_vals = train_preds[col].dropna()
    test_vals = test_preds[col].dropna()
    
    # Basic statistics
    train_stats = {
        'mean': train_vals.mean(),
        'std': train_vals.std(),
        'min': train_vals.min(),
        'max': train_vals.max(),
        'median': train_vals.median(),
        'q25': train_vals.quantile(0.25),
        'q75': train_vals.quantile(0.75)
    }
    
    test_stats = {
        'mean': test_vals.mean(),
        'std': test_vals.std(),
        'min': test_vals.min(),
        'max': test_vals.max(),
        'median': test_vals.median(),
        'q25': test_vals.quantile(0.25),
        'q75': test_vals.quantile(0.75)
    }
    
    # Distribution shift metrics
    mean_shift = abs(train_stats['mean'] - test_stats['mean'])
    std_shift = abs(train_stats['std'] - test_stats['std'])
    mean_shift_std = mean_shift / train_stats['std'] if train_stats['std'] > 0 else np.inf
    
    # Statistical tests
    ks_stat, ks_pval = ks_2samp(train_vals, test_vals)
    t_stat, t_pval = ttest_ind(train_vals, test_vals)
    
    print(f"Train: mean={train_stats['mean']:.6f}, std={train_stats['std']:.6f}")
    print(f"Test:  mean={test_stats['mean']:.6f}, std={test_stats['std']:.6f}")
    print(f"Mean shift: {mean_shift:.6f} ({mean_shift_std:.2f} std)")
    print(f"K-S test: stat={ks_stat:.4f}, p={ks_pval:.2e}")
    print(f"T-test: stat={t_stat:.4f}, p={t_pval:.2e}")
    
    distribution_results.append({
        'feature': col,
        'train_mean': train_stats['mean'],
        'train_std': train_stats['std'],
        'test_mean': test_stats['mean'],
        'test_std': test_stats['std'],
        'mean_shift': mean_shift,
        'mean_shift_std': mean_shift_std,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'ttest_statistic': t_stat,
        'ttest_pvalue': t_pval,
        'train_samples': len(train_vals),
        'test_samples': len(test_vals)
    })

# Save distribution analysis results
dist_df = pd.DataFrame(distribution_results)
dist_df.to_csv(RESULTS_OUT_DIR / 'vision_distribution_shift.csv', index=False)
print(f"\nDistribution analysis saved to: {RESULTS_OUT_DIR / 'vision_distribution_shift.csv'}")

# Load standardization stats
print("\n4. Loading standardization statistics...")
std_stats_path = STACKING_DIR / 'standardization_stats.pkl'
if not std_stats_path.exists():
    print(f"ERROR: Standardization stats file not found: {std_stats_path}")
    exit(1)

with open(std_stats_path, 'rb') as f:
    std_stats = pickle.load(f)

print("Standardization stats loaded:")
for key, stats in std_stats.items():
    print(f"  {key}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

# Apply standardization to test predictions
print("\n5. Applying standardization to test predictions...")
standardized_results = []

for col in ['eva02_pred', 'edgenext_pred']:
    if col in std_stats:
        mean = std_stats[col]['mean']
        std = std_stats[col]['std']
        
        # Apply z-score standardization
        test_preds[f'{col}_standardized'] = (test_preds[col] - mean) / (std + 1e-8)
        
        # Analyze standardized values
        std_vals = test_preds[f'{col}_standardized'].dropna()
        
        std_stats_result = {
            'feature': col,
            'original_mean': test_preds[col].mean(),
            'original_std': test_preds[col].std(),
            'standardized_mean': std_vals.mean(),
            'standardized_std': std_vals.std(),
            'standardized_min': std_vals.min(),
            'standardized_max': std_vals.max(),
            'standardized_median': std_vals.median(),
            'training_mean': mean,
            'training_std': std,
            'expected_mean': 0.0,
            'expected_std': 1.0
        }
        
        standardized_results.append(std_stats_result)
        
        print(f"\n{col} after standardization:")
        print(f"  Range: [{std_vals.min():.4f}, {std_vals.max():.4f}]")
        print(f"  Mean: {std_vals.mean():.4f} (expected: 0.0)")
        print(f"  Std: {std_vals.std():.4f} (expected: 1.0)")

# Save standardization results
if standardized_results:
    std_df = pd.DataFrame(standardized_results)
    std_df.to_csv(RESULTS_OUT_DIR / 'standardization_on_test.csv', index=False)
    print(f"\nStandardization analysis saved to: {RESULTS_OUT_DIR / 'standardization_on_test.csv'}")

# Create visualizations
print("\n6. Creating visualizations...")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create distribution comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Train/Test Distribution Shift Analysis', fontsize=16, fontweight='bold')

for i, col in enumerate(['eva02_pred', 'edgenext_pred']):
    # Distribution histograms
    ax1 = axes[i, 0]
    train_vals = train_preds[col].dropna()
    test_vals = test_preds[col].dropna()
    
    ax1.hist(train_vals, bins=50, alpha=0.7, label='Train', density=True)
    ax1.hist(test_vals, bins=50, alpha=0.7, label='Test', density=True)
    ax1.set_title(f'{col} - Distribution Comparison')
    ax1.set_xlabel('Prediction Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plots
    ax2 = axes[i, 1]
    data_to_plot = [train_vals, test_vals]
    bp = ax2.boxplot(data_to_plot, labels=['Train', 'Test'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_title(f'{col} - Box Plot Comparison')
    ax2.set_ylabel('Prediction Value')
    ax2.grid(True, alpha=0.3)
    
    # Standardized test values
    ax3 = axes[i, 2]
    if f'{col}_standardized' in test_preds.columns:
        std_vals = test_preds[f'{col}_standardized'].dropna()
        ax3.hist(std_vals, bins=50, alpha=0.7, color='green', density=True)
        ax3.axvline(0, color='red', linestyle='--', label='Expected Mean (0)')
        ax3.set_title(f'{col} - Standardized Test Values')
        ax3.set_xlabel('Standardized Value')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_OUT_DIR / 'distribution_shift_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create summary statistics plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Prepare data for plotting
plot_data = []
for result in distribution_results:
    plot_data.append({
        'Feature': result['feature'],
        'Metric': 'Mean Shift (std)',
        'Value': result['mean_shift_std'],
        'Type': 'Distribution Shift'
    })
    plot_data.append({
        'Feature': result['feature'],
        'Metric': 'KS Test p-value',
        'Value': -np.log10(result['ks_pvalue'] + 1e-10),  # Log scale for p-values
        'Type': 'Statistical Test'
    })

plot_df = pd.DataFrame(plot_data)

# Create grouped bar plot
sns.barplot(data=plot_df, x='Feature', y='Value', hue='Metric', ax=ax)
ax.set_title('Distribution Shift Metrics Summary', fontsize=14, fontweight='bold')
ax.set_ylabel('Metric Value')
ax.grid(True, alpha=0.3)

# Add significance line for p-values
ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold')
ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_OUT_DIR / 'distribution_shift_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualizations saved:")
print(f"  - {PLOTS_OUT_DIR / 'distribution_shift_analysis.png'}")
print(f"  - {PLOTS_OUT_DIR / 'distribution_shift_summary.png'}")

# Generate summary report
print("\n7. Generating summary report...")

summary_report = f"""
# Phase 4 Distribution Shift Analysis Summary

## Key Findings

### 1. Distribution Shift Detected
"""

for result in distribution_results:
    feature = result['feature']
    mean_shift_std = result['mean_shift_std']
    ks_pval = result['ks_pvalue']
    
    if mean_shift_std > 1.0:
        shift_severity = "SEVERE"
    elif mean_shift_std > 0.5:
        shift_severity = "MODERATE"
    else:
        shift_severity = "MINOR"
    
    summary_report += f"""
**{feature}**:
- Mean shift: {mean_shift_std:.2f} standard deviations ({shift_severity})
- K-S test p-value: {ks_pval:.2e} ({'Significant' if ks_pval < 0.05 else 'Not significant'})
"""

summary_report += f"""

### 2. Standardization Impact
"""

for result in standardized_results:
    feature = result['feature']
    std_mean = result['standardized_mean']
    std_std = result['standardized_std']
    
    summary_report += f"""
**{feature} after standardization**:
- Mean: {std_mean:.4f} (expected: 0.0)
- Std: {std_std:.4f} (expected: 1.0)
- Range: [{result['standardized_min']:.4f}, {result['standardized_max']:.4f}]
"""

summary_report += f"""

### 3. Root Cause Analysis
The poor stacking submission scores (0.48-0.49) are likely caused by:

1. **Distribution Shift**: Test predictions have different distributions than training
2. **Inconsistent Preprocessing**: Training uses z-score, inference uses rank normalization
3. **Double Normalization**: Both vision predictions and final output are rank-normalized

### 4. Recommendations

1. **Use z-score standardization consistently** between training and inference
2. **Load saved standardization stats** in inference pipeline
3. **Remove rank normalization** from final output
4. **Validate preprocessing consistency** before submission

## Files Generated
- `vision_distribution_shift.csv`: Detailed distribution comparison
- `standardization_on_test.csv`: Standardization impact analysis
- `distribution_shift_analysis.png`: Visual comparison plots
- `distribution_shift_summary.png`: Summary metrics plot
"""

# Save summary report
with open(RESULTS_OUT_DIR / 'phase4_distribution_shift_summary.md', 'w') as f:
    f.write(summary_report)

print(f"Summary report saved to: {RESULTS_OUT_DIR / 'phase4_distribution_shift_summary.md'}")

print("\n=== Phase 4 Analysis Complete ===")
print(f"Results saved to: {RESULTS_OUT_DIR}")
print(f"Plots saved to: {PLOTS_OUT_DIR}")