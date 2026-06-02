#!/usr/bin/env python3
"""
Phase 3: Engineered Features Investigation
Saves all outputs to proper directories in metadata_investigation folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path('.')
METADATA_DIR = BASE_DIR / 'metadata_investigation'
RESULTS_DIR = METADATA_DIR / 'results'
PLOTS_DIR = METADATA_DIR / 'plots'

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

print("=== PHASE 3: ENGINEERED FEATURES INVESTIGATION ===")
print(f"Saving results to: {RESULTS_DIR}")
print(f"Saving plots to: {PLOTS_DIR}")

# Load the processed data with engineered features
processed_file = BASE_DIR / 'results/stacking_final_v1/analysis/processed_data_sample.csv'
print(f"\nLoading processed data from: {processed_file}")

try:
    df = pd.read_csv(processed_file)
    print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Initialize results dictionary
phase3_results = {
    'patient_relative_features': {},
    'patient_lesion_count': {},
    'lof_analysis': {},
    'vision_predictions': {}
}

print("\n" + "="*60)
print("3.1 PATIENT-RELATIVE FEATURES ANALYSIS")
print("="*60)

# Define the base features for patient-relative analysis
RELATIVE_FEATURE_COLS = [
    'tbp_lv_areaMM2', 'tbp_lv_deltaB', 'clin_size_long_diam_mm',
    'tbp_lv_minorAxisMM', 'tbp_lv_eccentricity', 'tbp_lv_norm_color',
    'tbp_lv_radial_color_std_max', 'tbp_lv_color_std_mean',
    'eva02_pred', 'edgenext_pred'
]

# Analyze patient-relative features
patient_relative_results = []

for base_col in RELATIVE_FEATURE_COLS:
    derived_cols = [f'{base_col}_ratio_mean', f'{base_col}_diff_mean', 
                    f'{base_col}_zscore', f'{base_col}_ratio_max', f'{base_col}_ratio_min']
    
    print(f"\n--- Analyzing derived features from {base_col} ---")
    
    for col in derived_cols:
        if col in df.columns:
            # Basic statistics
            stats = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'missing_pct': float(df[col].isnull().mean() * 100)
            }
            
            # Correlation with target (only if we have both classes)
            if len(df['target'].unique()) > 1:
                try:
                    corr, pval = pearsonr(df[col].dropna(), df.loc[df[col].notna(), 'target'])
                    stats['correlation'] = float(corr)
                    stats['pvalue'] = float(pval)
                except:
                    stats['correlation'] = np.nan
                    stats['pvalue'] = np.nan
            else:
                stats['correlation'] = np.nan
                stats['pvalue'] = np.nan
            
            patient_relative_results.append({
                'base_feature': base_col,
                'derived_feature': col,
                **stats
            })
            
            print(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, corr={stats['correlation']:.4f}")

# Save patient-relative features results
patient_relative_df = pd.DataFrame(patient_relative_results)
patient_relative_df.to_csv(RESULTS_DIR / 'phase3_patient_relative_features.csv', index=False)
print(f"✅ Saved patient-relative features results to {RESULTS_DIR / 'phase3_patient_relative_features.csv'}")

phase3_results['patient_relative_features'] = {
    'n_features_analyzed': len(patient_relative_results),
    'features_with_correlation': len([r for r in patient_relative_results if not np.isnan(r.get('correlation', np.nan))])
}

print("\n" + "="*60)
print("3.2 PATIENT LESION COUNT ANALYSIS")
print("="*60)

# Analyze patient lesion count distribution
if 'patient_lesion_count' in df.columns:
    lesion_count_stats = df['patient_lesion_count'].describe()
    print(f"\nPatient Lesion Count Statistics:")
    print(lesion_count_stats)
    
    # Additional statistics
    single_lesion = (df['patient_lesion_count'] == 1).sum()
    many_lesions = (df['patient_lesion_count'] > 100).sum()
    
    print(f"\nPatients with only 1 lesion: {single_lesion} ({single_lesion/len(df)*100:.1f}%)")
    print(f"Patients with >100 lesions: {many_lesions} ({many_lesions/len(df)*100:.1f}%)")
    
    # Target rate by lesion count bins
    count_bins = [1, 2, 5, 10, 20, 50, 100, 500, 2000]
    df['count_bin'] = pd.cut(df['patient_lesion_count'], bins=count_bins, include_lowest=True)
    target_by_count = df.groupby('count_bin', observed=True)['target'].agg(['mean', 'sum', 'count'])
    
    print(f"\nTarget rate by patient lesion count:")
    print(target_by_count)
    
    # Convert target_by_count to JSON-serializable format
    target_by_count_serializable = {}
    for idx, row in target_by_count.iterrows():
        bin_str = f"{idx.left:.0f}-{idx.right:.0f}"
        target_by_count_serializable[bin_str] = {
            'mean': float(row['mean']),
            'sum': int(row['sum']),
            'count': int(row['count'])
        }
    
    # Save lesion count analysis
    lesion_count_results = {
        'statistics': lesion_count_stats.to_dict(),
        'single_lesion_pct': float(single_lesion/len(df)*100),
        'many_lesions_pct': float(many_lesions/len(df)*100),
        'target_by_count': target_by_count_serializable
    }
    
    with open(RESULTS_DIR / 'phase3_patient_lesion_count.json', 'w') as f:
        json.dump(lesion_count_results, f, indent=2, default=str)
    
    print(f"✅ Saved patient lesion count analysis to {RESULTS_DIR / 'phase3_patient_lesion_count.json'}")
    
    # Plot lesion count distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(df['patient_lesion_count'], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Patient Lesion Count')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Patient Lesion Counts')
    axes[0].set_yscale('log')
    
    # Target rate by count bin
    bin_centers = [interval.mid for interval in target_by_count.index if interval is not pd.NaT]
    target_rates = [target_by_count.loc[interval, 'mean'] for interval in target_by_count.index if interval is not pd.NaT]
    
    axes[1].bar(range(len(bin_centers)), target_rates)
    axes[1].set_xlabel('Lesion Count Bin')
    axes[1].set_ylabel('Target Rate')
    axes[1].set_title('Target Rate by Patient Lesion Count')
    axes[1].set_xticks(range(len(bin_centers)))
    axes[1].set_xticklabels([f'{int(center)}' for center in bin_centers], rotation=45)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'phase3_patient_lesion_count_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved plot to {PLOTS_DIR / 'phase3_patient_lesion_count_analysis.png'}")
    
    phase3_results['patient_lesion_count'] = lesion_count_results

else:
    print("❌ 'patient_lesion_count' column not found in data")

print("\n" + "="*60)
print("3.3 LOCAL OUTLIER FACTOR (LOF) ANALYSIS")
print("="*60)

# Analyze LOF distribution
if 'patient_lof' in df.columns:
    lof_stats = df['patient_lof'].describe()
    print(f"\nLOF Statistics:")
    print(lof_stats)
    
    # Check for default values
    default_lof = (df['patient_lof'] == -1).sum()
    valid_lof = (df['patient_lof'] != -1).sum()
    
    print(f"\nLOF = -1 (default/small group): {default_lof} ({default_lof/len(df)*100:.1f}%)")
    print(f"Valid LOF scores: {valid_lof} ({valid_lof/len(df)*100:.1f}%)")
    
    # LOF by target (if we have both classes)
    lof_by_target_serializable = None
    if len(df['target'].unique()) > 1:
        lof_by_target = df.groupby('target')['patient_lof'].describe()
        print(f"\nLOF statistics by target:")
        print(lof_by_target)
        
        # Convert to JSON-serializable format
        lof_by_target_serializable = {}
        for target_val, row in lof_by_target.iterrows():
            lof_by_target_serializable[f'target_{int(target_val)}'] = {
                'count': float(row['count']),
                'mean': float(row['mean']),
                'std': float(row['std']),
                'min': float(row['min']),
                '25%': float(row['25%']),
                '50%': float(row['50%']),
                '75%': float(row['75%']),
                'max': float(row['max'])
            }
    else:
        print(f"\nCannot analyze LOF by target (only one class present)")
    
    # Save LOF analysis
    lof_results = {
        'statistics': lof_stats.to_dict(),
        'default_pct': float(default_lof/len(df)*100),
        'valid_pct': float(valid_lof/len(df)*100),
        'lof_by_target': lof_by_target_serializable
    }
    
    with open(RESULTS_DIR / 'phase3_lof_analysis.json', 'w') as f:
        json.dump(lof_results, f, indent=2, default=str)
    
    print(f"✅ Saved LOF analysis to {RESULTS_DIR / 'phase3_lof_analysis.json'}")
    
    # Plot LOF distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall distribution
    valid_lof_data = df[df['patient_lof'] != -1]['patient_lof']
    axes[0].hist(valid_lof_data, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('LOF Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Valid LOF Scores')
    
    # LOF by target (if possible)
    if lof_by_target_serializable is not None:
        for target in df['target'].unique():
            subset = df[df['target'] == target]
            valid_subset = subset[subset['patient_lof'] != -1]['patient_lof']
            if len(valid_subset) > 0:
                axes[1].hist(valid_subset, bins=30, alpha=0.5, label=f'Target {target}', density=True)
        axes[1].set_xlabel('LOF Score')
        axes[1].set_ylabel('Density')
        axes[1].set_title('LOF Distribution by Target')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'Single class present\nCannot show target comparison', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('LOF Distribution by Target')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'phase3_lof_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved plot to {PLOTS_DIR / 'phase3_lof_distribution.png'}")
    
    phase3_results['lof_analysis'] = lof_results

else:
    print("❌ 'patient_lof' column not found in data")

print("\n" + "="*60)
print("3.4 VISION PREDICTIONS ANALYSIS")
print("="*60)

# Analyze vision model predictions
vision_cols = ['eva02_pred', 'edgenext_pred']
vision_results = []

for col in vision_cols:
    if col in df.columns:
        stats = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'missing_pct': float(df[col].isnull().mean() * 100)
        }
        
        # AUC if we have both classes
        if len(df['target'].unique()) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(df['target'], df[col])
                stats['auc'] = float(auc)
            except:
                stats['auc'] = np.nan
        else:
            stats['auc'] = np.nan
        
        vision_results.append({
            'model': col,
            **stats
        })
        
        print(f"\n{col}:")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  AUC: {stats['auc']:.4f}" if not np.isnan(stats['auc']) else "  AUC: N/A")

# Correlation between models
if all(col in df.columns for col in vision_cols):
    model_corr = df['eva02_pred'].corr(df['edgenext_pred'])
    print(f"\nCorrelation between EVA02 and EdgeNeXt: {model_corr:.4f}")
    
    # Cases where models disagree
    df['vision_disagreement'] = abs(df['eva02_pred'] - df['edgenext_pred'])
    high_disagreement = df.nlargest(100, 'vision_disagreement')
    
    if len(df['target'].unique()) > 1:
        disagreement_target_dist = high_disagreement['target'].value_counts().to_dict()
        print(f"\nHigh disagreement cases - target distribution:")
        print(disagreement_target_dist)
    else:
        disagreement_target_dist = None
    
    vision_results.append({
        'model': 'correlation',
        'eva02_edgenext_correlation': float(model_corr),
        'high_disagreement_target_distribution': disagreement_target_dist
    })

# Save vision predictions analysis
vision_df = pd.DataFrame(vision_results)
vision_df.to_csv(RESULTS_DIR / 'phase3_vision_predictions.csv', index=False)
print(f"✅ Saved vision predictions analysis to {RESULTS_DIR / 'phase3_vision_predictions.csv'}")

# Plot vision predictions
if len(vision_cols) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Individual model distributions
    for i, col in enumerate(vision_cols):
        if col in df.columns:
            axes[0, i].hist(df[col], bins=50, alpha=0.7, edgecolor='black')
            axes[0, i].set_xlabel(col)
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].set_title(f'{col} Distribution')
    
    # Correlation plot
    if all(col in df.columns for col in vision_cols):
        axes[1, 0].scatter(df['eva02_pred'], df['edgenext_pred'], alpha=0.5)
        axes[1, 0].set_xlabel('EVA02 Prediction')
        axes[1, 0].set_ylabel('EdgeNeXt Prediction')
        axes[1, 0].set_title(f'Model Correlation: {model_corr:.4f}')
        
        # Disagreement analysis
        axes[1, 1].hist(df['vision_disagreement'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Vision Model Disagreement')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Model Disagreement')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'phase3_vision_predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved plot to {PLOTS_DIR / 'phase3_vision_predictions_analysis.png'}")

phase3_results['vision_predictions'] = {
    'models_analyzed': len([col for col in vision_cols if col in df.columns]),
    'correlation': float(model_corr) if all(col in df.columns for col in vision_cols) else None
}

print("\n" + "="*60)
print("PHASE 3 SUMMARY")
print("="*60)

print(f"\n✅ Patient-Relative Features: {phase3_results['patient_relative_features']['n_features_analyzed']} features analyzed")
print(f"✅ Patient Lesion Count: Range {df['patient_lesion_count'].min()}-{df['patient_lesion_count'].max()}")
print(f"✅ LOF Analysis: {phase3_results['lof_analysis']['valid_pct']:.1f}% valid scores")
print(f"✅ Vision Predictions: {phase3_results['vision_predictions']['models_analyzed']} models analyzed")

# Save overall Phase 3 results
with open(RESULTS_DIR / 'phase3_complete_results.json', 'w') as f:
    json.dump(phase3_results, f, indent=2, default=str)

print(f"\n✅ Complete Phase 3 results saved to {RESULTS_DIR / 'phase3_complete_results.json'}")
print(f"\n🎯 Phase 3 Analysis Complete!")
print(f"📁 All results saved to: {RESULTS_DIR}")
print(f"📊 All plots saved to: {PLOTS_DIR}")