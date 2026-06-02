#!/usr/bin/env python3
"""
Phase 3: Engineered Features Investigation
Analyzes patient-relative features, lesion counts, LOF, and vision predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / 'data'
RESULTS_DIR = SCRIPT_DIR.parent.parent / 'results' / 'stacking_final_v1'
PLOTS_DIR = SCRIPT_DIR.parent.parent / 'plots'

def load_data():
    """Load the processed data with engineered features"""
    print("📂 Loading processed data with engineered features...")
    
    # Load the sample processed data
    processed_sample = pd.read_csv(RESULTS_DIR / 'analysis' / 'processed_data_sample.csv')
    
    # Load the full OOF data for more comprehensive analysis
    stacking_oof = pd.read_csv(RESULTS_DIR / 'stacking_oof.csv')
    
    print(f"✅ Loaded sample: {len(processed_sample)} rows")
    print(f"✅ Loaded OOF data: {len(stacking_oof)} rows")
    
    return processed_sample, stacking_oof

def analyze_patient_relative_features(df):
    """Analyze patient-relative features (3.1)"""
    print("\n" + "="*60)
    print("3.1 PATIENT-RELATIVE FEATURES ANALYSIS")
    print("="*60)
    
    # Define the relative feature columns from the stacking script
    RELATIVE_FEATURE_COLS = [
        'tbp_lv_areaMM2', 'tbp_lv_deltaB', 'clin_size_long_diam_mm',
        'tbp_lv_minorAxisMM', 'tbp_lv_eccentricity', 'tbp_lv_norm_color',
        'tbp_lv_radial_color_std_max', 'tbp_lv_color_std_mean',
        'eva02_pred', 'edgenext_pred'
    ]
    
    results = {}
    
    for base_col in RELATIVE_FEATURE_COLS[:3]:  # Sample a few key features
        print(f"\n--- Analyzing derived features from {base_col} ---")
        
        # Find all derived columns for this base feature
        derived_cols = [f'{base_col}_ratio_mean', f'{base_col}_diff_mean', 
                       f'{base_col}_zscore', f'{base_col}_ratio_max', f'{base_col}_ratio_min']
        
        derived_cols = [col for col in derived_cols if col in df.columns]
        
        feature_results = {}
        for col in derived_cols:
            corr = df[col].corr(df['target'])
            mean_val = df[col].mean()
            std_val = df[col].std()
            feature_results[col] = {
                'correlation': corr,
                'mean': mean_val,
                'std': std_val
            }
            print(f"  {col}: corr={corr:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")
        
        results[base_col] = feature_results
    
    # Create visualization for key relative features
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot z-scores for key features
    key_features = ['tbp_lv_areaMM2_zscore', 'tbp_lv_deltaB_zscore', 'clin_size_long_diam_mm_zscore']
    
    for i, feat in enumerate(key_features):
        if feat in df.columns:
            # Distribution by target
            benign = df[df['target']==0][feat].dropna()
            malignant = df[df['target']==1][feat].dropna()
            
            axes[i].hist(benign, bins=50, alpha=0.5, label='Benign', density=True)
            axes[i].hist(malignant, bins=50, alpha=0.5, label='Malignant', density=True)
            axes[i].set_xlabel(f'{feat} Z-Score')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{feat} Distribution by Target')
            axes[i].legend()
            
            # Correlation with target
            corr = df[feat].corr(df['target'])
            axes[i].text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                        transform=axes[i].transAxes, verticalalignment='top')
    
    # Plot patient lesion count distribution
    if 'patient_lesion_count' in df.columns:
        axes[3].hist(df['patient_lesion_count'], bins=50, alpha=0.7, edgecolor='black')
        axes[3].set_xlabel('Patient Lesion Count')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Distribution of Lesions per Patient')
        axes[3].set_yscale('log')
        
        # Add statistics
        mean_count = df['patient_lesion_count'].mean()
        median_count = df['patient_lesion_count'].median()
        axes[3].axvline(mean_count, color='red', linestyle='--', label=f'Mean: {mean_count:.1f}')
        axes[3].axvline(median_count, color='orange', linestyle='--', label=f'Median: {median_count:.1f}')
        axes[3].legend()
    
    # Plot LOF distribution
    if 'patient_lof' in df.columns:
        lof_data = df['patient_lof'].replace(-1, np.nan).dropna()
        axes[4].hist(lof_data, bins=50, alpha=0.7, edgecolor='black')
        axes[4].set_xlabel('Local Outlier Factor')
        axes[4].set_ylabel('Frequency')
        axes[4].set_title('Distribution of LOF Scores')
        
        # LOF by target
        lof_benign = df[df['target']==0]['patient_lof'].replace(-1, np.nan).dropna()
        lof_malignant = df[df['target']==1]['patient_lof'].replace(-1, np.nan).dropna()
        
        if len(lof_malignant) > 0:
            axes[5].hist(lof_benign, bins=30, alpha=0.5, label='Benign', density=True)
            axes[5].hist(lof_malignant, bins=30, alpha=0.5, label='Malignant', density=True)
            axes[5].set_xlabel('Local Outlier Factor')
            axes[5].set_ylabel('Density')
            axes[5].set_title('LOF Distribution by Target')
            axes[5].legend()
        else:
            axes[5].text(0.5, 0.5, 'No malignant cases\nwith LOF data', 
                        transform=axes[5].transAxes, ha='center', va='center')
            axes[5].set_title('LOF Distribution by Target')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'phase3_patient_relative_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def analyze_patient_lesion_count(df):
    """Analyze patient lesion count (3.2)"""
    print("\n" + "="*60)
    print("3.2 PATIENT LESION COUNT ANALYSIS")
    print("="*60)
    
    if 'patient_lesion_count' not in df.columns:
        print("⚠️ patient_lesion_count column not found")
        return {}
    
    # Basic statistics
    patient_counts = df['patient_lesion_count']
    print(f"Patient lesion count statistics:")
    print(f"  Mean: {patient_counts.mean():.2f}")
    print(f"  Median: {patient_counts.median():.2f}")
    print(f"  Min: {patient_counts.min()}")
    print(f"  Max: {patient_counts.max()}")
    print(f"  Std: {patient_counts.std():.2f}")
    
    # Distribution analysis
    single_lesion = (patient_counts == 1).sum()
    many_lesions = (patient_counts > 100).sum()
    
    print(f"\nDistribution insights:")
    print(f"  Patients with only 1 lesion: {single_lesion}")
    print(f"  Patients with >100 lesions: {many_lesions}")
    
    # Target rate by lesion count
    count_bins = [1, 2, 5, 10, 20, 50, 100, 500, 2000]
    df['count_bin'] = pd.cut(df['patient_lesion_count'], bins=count_bins, include_lowest=True)
    target_by_count = df.groupby('count_bin', observed=True)['target'].agg(['mean', 'sum', 'count'])
    
    print(f"\nTarget rate by patient lesion count:")
    print(target_by_count)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    axes[0].hist(patient_counts, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Patient Lesion Count')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Lesions per Patient')
    axes[0].set_yscale('log')
    
    # Target rate by count bin
    if len(target_by_count) > 0:
        bin_centers = [interval.mid for interval in target_by_count.index if interval is not pd.NaT]
        target_rates = target_by_count['mean'].values
        axes[1].bar(range(len(bin_centers)), target_rates)
        axes[1].set_xticks(range(len(bin_centers)))
        axes[1].set_xticklabels([f'{int(x)}' for x in bin_centers], rotation=45)
        axes[1].set_xlabel('Patient Lesion Count')
        axes[1].set_ylabel('Target Rate')
        axes[1].set_title('Target Rate by Patient Lesion Count')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'phase3_patient_lesion_count.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mean': patient_counts.mean(),
        'median': patient_counts.median(),
        'single_lesion_patients': single_lesion,
        'many_lesion_patients': many_lesions,
        'target_by_count': target_by_count.to_dict()
    }

def analyze_lof(df):
    """Analyze Local Outlier Factor (3.3)"""
    print("\n" + "="*60)
    print("3.3 LOCAL OUTLIER FACTOR ANALYSIS")
    print("="*60)
    
    if 'patient_lof' not in df.columns:
        print("⚠️ patient_lof column not found")
        return {}
    
    lof_col = 'patient_lof'
    
    # Basic statistics
    lof_data = df[lof_col]
    default_count = (lof_data == -1).sum()
    valid_count = (lof_data != -1).sum()
    
    print(f"LOF statistics:")
    print(f"  Default values (-1): {default_count} ({default_count/len(df)*100:.1f}%)")
    print(f"  Valid LOF scores: {valid_count} ({valid_count/len(df)*100:.1f}%)")
    
    if valid_count > 0:
        valid_lof = lof_data[lof_data != -1]
        print(f"  Valid LOF range: [{valid_lof.min():.3f}, {valid_lof.max():.3f}]")
        print(f"  Valid LOF mean: {valid_lof.mean():.3f}")
        print(f"  Valid LOF std: {valid_lof.std():.3f}")
        
        # LOF by target
        lof_by_target = df.groupby('target')[lof_col].agg(['count', 'mean', 'std'])
        print(f"\nLOF statistics by target:")
        print(lof_by_target)
        
        # Correlation with target (excluding defaults)
        valid_mask = df[lof_col] != -1
        if valid_mask.sum() > 0:
            corr = df.loc[valid_mask, lof_col].corr(df.loc[valid_mask, 'target'])
            print(f"\nCorrelation with target (valid LOF only): {corr:.4f}")
    
    return {
        'default_count': default_count,
        'valid_count': valid_count,
        'lof_by_target': lof_by_target.to_dict() if valid_count > 0 else {}
    }

def analyze_vision_predictions(df):
    """Analyze vision predictions (3.4)"""
    print("\n" + "="*60)
    print("3.4 VISION PREDICTIONS ANALYSIS")
    print("="*60)
    
    vision_cols = ['eva02_pred', 'edgenext_pred']
    available_cols = [col for col in vision_cols if col in df.columns]
    
    if not available_cols:
        print("⚠️ No vision prediction columns found")
        return {}
    
    results = {}
    
    for col in available_cols:
        print(f"\n--- {col} Analysis ---")
        print(f"  Range: [{df[col].min():.6f}, {df[col].max():.6f}]")
        print(f"  Mean: {df[col].mean():.6f}")
        print(f"  Median: {df[col].median():.6f}")
        print(f"  Std: {df[col].std():.6f}")
        
        # AUC calculation
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(df['target'], df[col])
            print(f"  AUC: {auc:.4f}")
        except:
            auc = None
            print(f"  AUC: Could not calculate")
        
        # Calibration check
        mean_pred = df[col].mean()
        actual_rate = df['target'].mean()
        print(f"  Mean prediction: {mean_pred:.6f}")
        print(f"  Actual positive rate: {actual_rate:.6f}")
        print(f"  Calibration gap: {abs(mean_pred - actual_rate):.6f}")
        
        results[col] = {
            'range': (df[col].min(), df[col].max()),
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'auc': auc,
            'calibration_gap': abs(mean_pred - actual_rate)
        }
    
    # Correlation between models
    if len(available_cols) == 2:
        corr = df['eva02_pred'].corr(df['edgenext_pred'])
        print(f"\nCorrelation between EVA02 and EdgeNeXt: {corr:.4f}")
        results['model_correlation'] = corr
        
        # Cases where models disagree
        df['vision_disagreement'] = abs(df['eva02_pred'] - df['edgenext_pred'])
        high_disagreement = df.nlargest(100, 'vision_disagreement')
        print(f"\nHigh disagreement cases - target distribution:")
        print(high_disagreement['target'].value_counts())
        
        results['high_disagreement_target_rate'] = high_disagreement['target'].mean()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution plots
    for i, col in enumerate(available_cols):
        # Overall distribution
        axes[0, i].hist(df[col], bins=50, alpha=0.7, edgecolor='black')
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].set_title(f'{col} Distribution')
        
        # Distribution by target
        benign = df[df['target']==0][col].dropna()
        malignant = df[df['target']==1][col].dropna()
        
        axes[1, i].hist(benign, bins=30, alpha=0.5, label='Benign', density=True)
        axes[1, i].hist(malignant, bins=30, alpha=0.5, label='Malignant', density=True)
        axes[1, i].set_xlabel(col)
        axes[1, i].set_ylabel('Density')
        axes[1, i].set_title(f'{col} by Target')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'phase3_vision_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def main():
    """Main analysis function"""
    print("🚀 Starting Phase 3: Engineered Features Investigation")
    
    # Create plots directory if it doesn't exist
    PLOTS_DIR.mkdir(exist_ok=True)
    
    # Load data
    processed_sample, stacking_oof = load_data()
    
    # Use the larger dataset for analysis
    df = stacking_oof if len(stacking_oof) > len(processed_sample) else processed_sample
    
    # Run all analyses
    relative_results = analyze_patient_relative_features(df)
    count_results = analyze_patient_lesion_count(df)
    lof_results = analyze_lof(df)
    vision_results = analyze_vision_predictions(df)
    
    # Save results
    results = {
        'patient_relative_features': relative_results,
        'patient_lesion_count': count_results,
        'lof_analysis': lof_results,
        'vision_predictions': vision_results
    }
    
    import json
    with open(RESULTS_DIR / 'analysis' / 'phase3_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Phase 3 analysis complete!")
    print(f"📊 Results saved to: {RESULTS_DIR / 'analysis' / 'phase3_results.json'}")
    print(f"📈 Plots saved to: {PLOTS_DIR / 'phase3_*.png'}")

if __name__ == "__main__":
    main()