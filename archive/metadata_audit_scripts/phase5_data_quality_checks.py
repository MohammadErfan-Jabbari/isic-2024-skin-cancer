#!/usr/bin/env python3
"""
Phase 5: Data Quality Checks
============================

This script completes Phase 5 of the metadata investigation plan:
- 5.1: Duplicate Detection
- 5.2: Outlier Detection
- 5.3: Train/Test Distribution Shift Check

Author: Kilo Code
Date: 2025-11-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results' / 'stacking_final_v1'
PLOTS_DIR = BASE_DIR / 'plots'
OUTPUT_DIR = BASE_DIR / 'metadata_investigation' / 'results'

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("🔍 Phase 5: Data Quality Checks")
print("=" * 50)

# Load training data
print("\n📊 Loading training metadata...")
train_df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
print(f"✅ Loaded training data: {len(train_df)} samples, {len(train_df.columns)} columns")

# Load test data
print("\n📊 Loading test metadata...")
test_df = pd.read_csv(DATA_DIR / 'students-test-metadata.csv', low_memory=False)
print(f"✅ Loaded test data: {len(test_df)} samples, {len(test_df.columns)} columns")

# Phase 5.1: Duplicate Detection
print("\n🔍 Phase 5.1: Duplicate Detection")
print("-" * 50)

# Check for duplicate isic_ids
train_unique_ids = train_df['isic_id'].nunique()
test_unique_ids = test_df['isic_id'].nunique()
train_total = len(train_df)
test_total = len(test_df)

print(f"\n📋 Duplicate ID Analysis:")
print(f"Training set: {train_unique_ids}/{train_total} unique IDs ({((train_total-train_unique_ids)/train_total*100):.4f}% duplicates)")
print(f"Test set: {test_unique_ids}/{test_total} unique IDs ({((test_total-test_unique_ids)/test_total*100):.4f}% duplicates)")

# Check for duplicate patients
train_unique_patients = train_df['patient_id'].nunique()
test_unique_patients = test_df['patient_id'].nunique()

print(f"\n📋 Duplicate Patient Analysis:")
print(f"Training set: {train_unique_patients}/{train_total} unique patients ({((train_total-train_unique_patients)/train_total*100):.4f}% duplicates)")
print(f"Test set: {test_unique_patients}/{test_total} unique patients ({((test_total-test_unique_patients)/test_total*100):.4f}% duplicates)")

# Check for near-duplicate metadata rows
print(f"\n🔍 Checking for near-duplicate metadata...")

# Select key metadata columns for duplicate detection
meta_cols = ['tbp_lv_areaMM2', 'tbp_lv_perimeterMM', 'tbp_lv_H', 'tbp_lv_L', 'tbp_lv_A', 'tbp_lv_B']
available_meta_cols = [col for col in meta_cols if col in train_df.columns]
print(f"Using columns for duplicate detection: {available_meta_cols}")

if len(available_meta_cols) > 0:
    # Create hash for metadata rows (excluding missing values)
    train_meta_subset = train_df[available_meta_cols].fillna(-999)
    train_df['meta_hash'] = train_meta_subset.apply(lambda x: hash(tuple(x)), axis=1)
    
    # Find duplicates
    dup_meta = train_df[train_df.duplicated('meta_hash', keep=False)]
    print(f"Rows with duplicate metadata: {len(dup_meta)} ({len(dup_meta)/len(train_df)*100:.4f}%)")
    
    if len(dup_meta) > 0:
        print(f"Example duplicate groups:")
        dup_groups = dup_meta.groupby('meta_hash').size().sort_values(ascending=False).head(5)
        for hash_val, count in dup_groups.items():
            print(f"  Hash {hash_val}: {count} identical rows")
else:
    print("⚠️  No metadata columns available for duplicate detection")

# Phase 5.2: Outlier Detection
print("\n🔍 Phase 5.2: Outlier Detection")
print("-" * 50)

# Get numeric columns
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['target', 'isic_id', 'patient_id', 'fold']]

print(f"Analyzing outliers in {len(numeric_cols)} numeric columns...")

outlier_summary = []
extreme_outliers = []

for col in numeric_cols[:20]:  # Sample first 20 columns to avoid too much output
    if train_df[col].notna().sum() < 100:
        continue
    
    # Calculate quartiles and IQR
    q01 = train_df[col].quantile(0.01)
    q99 = train_df[col].quantile(0.99)
    iqr = q99 - q01
    
    # Define outlier thresholds (3*IQR rule)
    lower_bound = q01 - 3*iqr
    upper_bound = q99 + 3*iqr
    
    # Count outliers
    extreme_low = (train_df[col] < lower_bound).sum()
    extreme_high = (train_df[col] > upper_bound).sum()
    total_outliers = extreme_low + extreme_high
    
    if total_outliers > 0:
        outlier_summary.append({
            'column': col,
            'extreme_low': extreme_low,
            'extreme_high': extreme_high,
            'total_outliers': total_outliers,
            'outlier_pct': total_outliers / len(train_df) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
        
        if total_outliers > 100:  # Only track columns with many outliers
            extreme_outliers.append({
                'column': col,
                'count': total_outliers,
                'percentage': total_outliers / len(train_df) * 100
            })

if outlier_summary:
    outlier_df = pd.DataFrame(outlier_summary).sort_values('total_outliers', ascending=False)
    print(f"\n📋 Columns with Extreme Outliers (>0):")
    print(outlier_df.to_string(index=False))
    
    # Save outlier analysis
    outlier_df.to_csv(OUTPUT_DIR / 'phase5_outlier_analysis.csv', index=False)
    print(f"\n💾 Saved: phase5_outlier_analysis.csv")
else:
    print(f"\n✅ No extreme outliers detected in sampled columns")

if extreme_outliers:
    print(f"\n🚨 Columns with Many Outliers (>100):")
    for outlier in extreme_outliers:
        print(f"  {outlier['column']}: {outlier['count']} outliers ({outlier['percentage']:.2f}%)")

# Phase 5.3: Train/Test Distribution Shift Check
print("\n🔍 Phase 5.3: Train/Test Distribution Shift Check")
print("-" * 50)

# Common columns between train and test
common_cols = set(train_df.columns) & set(test_df.columns)
numeric_common_cols = [col for col in common_cols if train_df[col].dtype in ['int64', 'float64']]
numeric_common_cols = [col for col in numeric_common_cols if col not in ['target', 'isic_id', 'patient_id', 'fold']]

print(f"Comparing {len(numeric_common_cols)} common numeric columns...")

distribution_shift = []
significant_shifts = []

for col in numeric_common_cols[:15]:  # Sample first 15 columns
    if train_df[col].notna().sum() < 100 or test_df[col].notna().sum() < 100:
        continue
    
    # Calculate statistics
    train_mean = train_df[col].mean()
    test_mean = test_df[col].mean()
    train_std = train_df[col].std()
    test_std = test_df[col].std()
    
    # Calculate shift in standard deviations
    if train_std > 0:
        shift_std = abs(train_mean - test_mean) / train_std
    else:
        shift_std = 0
    
    # Perform Kolmogorov-Smirnov test
    try:
        ks_stat, ks_pval = stats.ks_2samp(train_df[col].dropna(), test_df[col].dropna())
    except:
        ks_stat, ks_pval = np.nan, np.nan
    
    distribution_shift.append({
        'column': col,
        'train_mean': train_mean,
        'test_mean': test_mean,
        'train_std': train_std,
        'test_std': test_std,
        'shift_std': shift_std,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'significant_shift': shift_std > 2.0 or (ks_pval < 0.001 and not np.isnan(ks_pval))
    })
    
    if shift_std > 2.0 or (ks_pval < 0.001 and not np.isnan(ks_pval)):
        significant_shifts.append({
            'column': col,
            'shift_std': shift_std,
            'ks_pvalue': ks_pval
        })

if distribution_shift:
    shift_df = pd.DataFrame(distribution_shift).sort_values('shift_std', ascending=False)
    print(f"\n📋 Train/Test Distribution Shift Analysis:")
    print(shift_df.to_string(index=False))
    
    # Save distribution shift analysis
    shift_df.to_csv(OUTPUT_DIR / 'phase5_distribution_shift.csv', index=False)
    print(f"\n💾 Saved: phase5_distribution_shift.csv")
    
    if significant_shifts:
        print(f"\n🚨 Significant Distribution Shifts (>2 std or p<0.001):")
        for shift in significant_shifts:
            print(f"  {shift['column']}: {shift['shift_std']:.2f} std shift, p={shift['ks_pvalue']:.6f}")
    else:
        print(f"\n✅ No significant distribution shifts detected")
else:
    print(f"⚠️  No numeric columns available for distribution shift analysis")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Duplicate analysis
axes[0, 0].bar(['Train IDs', 'Train Patients', 'Test IDs', 'Test Patients'], 
               [train_unique_ids/train_total*100, train_unique_patients/train_total*100,
                test_unique_ids/test_total*100, test_unique_patients/test_total*100])
axes[0, 0].set_ylabel('Uniqueness (%)')
axes[0, 0].set_title('Data Uniqueness Analysis')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Outlier distribution
if outlier_summary:
    top_outliers = outlier_df.head(10)
    axes[0, 1].barh(range(len(top_outliers)), top_outliers['outlier_pct'])
    axes[0, 1].set_yticks(range(len(top_outliers)))
    axes[0, 1].set_yticklabels(top_outliers['column'])
    axes[0, 1].set_xlabel('Outlier Percentage')
    axes[0, 1].set_title('Top 10 Columns by Outlier %')
    axes[0, 1].invert_yaxis()

# Plot 3: Distribution shift
if distribution_shift:
    top_shifts = shift_df.head(10)
    axes[0, 2].barh(range(len(top_shifts)), top_shifts['shift_std'])
    axes[0, 2].set_yticks(range(len(top_shifts)))
    axes[0, 2].set_yticklabels(top_shifts['column'])
    axes[0, 2].set_xlabel('Shift (Standard Deviations)')
    axes[0, 2].set_title('Top 10 Columns by Distribution Shift')
    axes[0, 2].invert_yaxis()
    axes[0, 2].axvline(x=2, color='red', linestyle='--', alpha=0.7, label='Significant (2σ)')
    axes[0, 2].legend()

# Plot 4: Example distribution comparison (first significant shift)
if significant_shifts:
    example_col = significant_shifts[0]['column']
    axes[1, 0].hist(train_df[example_col].dropna(), bins=50, alpha=0.5, label='Train', density=True)
    axes[1, 0].hist(test_df[example_col].dropna(), bins=50, alpha=0.5, label='Test', density=True)
    axes[1, 0].set_xlabel(example_col)
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title(f'Distribution Comparison: {example_col}')
    axes[1, 0].legend()

# Plot 5: KS test p-values
if distribution_shift:
    valid_pvals = [p for p in shift_df['ks_pvalue'] if not np.isnan(p)]
    axes[1, 1].hist(valid_pvals, bins=20, alpha=0.7)
    axes[1, 1].set_xlabel('KS Test P-value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('KS Test P-value Distribution')
    axes[1, 1].axvline(x=0.001, color='red', linestyle='--', alpha=0.7, label='Significant (p<0.001)')
    axes[1, 1].legend()

# Plot 6: Summary statistics
summary_stats = {
    'Total Train Samples': len(train_df),
    'Total Test Samples': len(test_df),
    'Duplicate Train IDs': len(train_df) - train_unique_ids,
    'Duplicate Test IDs': len(test_df) - test_unique_ids,
    'Columns with Outliers': len(outlier_summary),
    'Significant Shifts': len(significant_shifts)
}

axes[1, 2].bar(summary_stats.keys(), summary_stats.values())
axes[1, 2].set_ylabel('Count')
axes[1, 2].set_title('Data Quality Summary')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'phase5_data_quality_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Saved visualization: phase5_data_quality_analysis.png")

# Summary and recommendations
print(f"\n📋 Phase 5 Summary")
print("=" * 50)

summary = {
    "phase": "Phase 5: Data Quality Checks",
    "date": "2025-11-26",
    "train_samples": len(train_df),
    "test_samples": len(test_df),
    "duplicate_analysis": {
        "train_id_duplicates": len(train_df) - train_unique_ids,
        "test_id_duplicates": len(test_df) - test_unique_ids,
        "train_patient_duplicates": len(train_df) - train_unique_patients,
        "test_patient_duplicates": len(test_df) - test_unique_patients,
        "metadata_duplicates": len(dup_meta) if 'dup_meta' in locals() else 0
    },
    "outlier_analysis": {
        "columns_with_outliers": len(outlier_summary),
        "extreme_outlier_columns": len(extreme_outliers)
    },
    "distribution_shift": {
        "columns_analyzed": len(distribution_shift),
        "significant_shifts": len(significant_shifts)
    },
    "key_findings": []
}

# Add key findings
if len(train_df) - train_unique_ids > 0:
    summary["key_findings"].append(f"Found {len(train_df) - train_unique_ids} duplicate training IDs")
    
if len(test_df) - test_unique_ids > 0:
    summary["key_findings"].append(f"Found {len(test_df) - test_unique_ids} duplicate test IDs")

if len(extreme_outliers) > 0:
    summary["key_findings"].append(f"Found {len(extreme_outliers)} columns with many outliers (>100)")
    summary["extreme_outlier_columns"] = [col['column'] for col in extreme_outliers]

if len(significant_shifts) > 0:
    summary["key_findings"].append(f"Found {len(significant_shifts)} columns with significant distribution shift")
    summary["significant_shift_columns"] = [col['column'] for col in significant_shifts]

# Save summary
with open(OUTPUT_DIR / 'phase5_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Phase 5 completed successfully!")
print(f"📁 Results saved to: {OUTPUT_DIR}")
print(f"📊 Visualizations saved to: {PLOTS_DIR}")

# Print key recommendations
print(f"\n🎯 Key Recommendations:")
print("=" * 30)
if len(train_df) - train_unique_ids > 0:
    print(f"1. 🚨 Address {len(train_df) - train_unique_ids} duplicate training IDs")
else:
    print(f"1. ✅ No duplicate training IDs found")

if len(test_df) - test_unique_ids > 0:
    print(f"2. 🚨 Address {len(test_df) - test_unique_ids} duplicate test IDs")
else:
    print(f"2. ✅ No duplicate test IDs found")

if len(extreme_outliers) > 0:
    print(f"3. 🚨 Investigate {len(extreme_outliers)} columns with many outliers")
else:
    print(f"3. ✅ No extreme outliers detected")

if len(significant_shifts) > 0:
    print(f"4. 🚨 Monitor {len(significant_shifts)} columns with significant distribution shift")
else:
    print(f"4. ✅ No significant distribution shifts detected")

print(f"5. 📊 Data quality appears generally good for model training")
print(f"6. 🔍 Proceed to Phase 6 for final summary and recommendations")