#!/usr/bin/env python3
"""
Phase 1: Raw Data Profiling for ISIC 2024 Metadata Investigation
Executes all three sub-phases: Column Inventory, Missing Value Analysis, and Target Correlation Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pointbiserialr
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = Path('./data')
RESULTS_DIR = Path('./metadata_investigation/results')
PLOTS_DIR = Path('./metadata_investigation/plots')

# Create directories if they don't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("🔍 ISIC 2024 Metadata Investigation - Phase 1: Raw Data Profiling")
print("=" * 80)

# Load data
print("📁 Loading training metadata...")
df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
print(f"✅ Data loaded successfully: {len(df):,} rows × {len(df.columns)} columns")

# =============================================================================
# PHASE 1.1: LOAD AND INVENTORY ALL COLUMNS
# =============================================================================
print("\n" + "="*80)
print("📊 PHASE 1.1: COLUMN INVENTORY")
print("="*80)

# Basic inventory
print(f"Total samples: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Target distribution:")
target_counts = df['target'].value_counts().sort_index()
print(target_counts)
print(f"Target rate: {df['target'].mean():.6f} ({df['target'].mean()*100:.4f}%)")

# Column types
print("\n=== Column Types ===")
dtype_counts = df.dtypes.value_counts()
print(dtype_counts)

# List all columns by type
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
object_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns ({len(numeric_cols)}):")
for i, col in enumerate(numeric_cols):
    print(f"  {i+1:2d}. {col}")

print(f"\nObject columns ({len(object_cols)}):")
for i, col in enumerate(object_cols):
    print(f"  {i+1:2d}. {col}")

# Save column inventory
column_inventory = pd.DataFrame({
    'column': df.columns,
    'dtype': df.dtypes,
    'numeric': df.columns.isin(numeric_cols),
    'object': df.columns.isin(object_cols)
})
column_inventory.to_csv(RESULTS_DIR / 'phase1_column_inventory.csv', index=False)
print(f"\n💾 Column inventory saved to: {RESULTS_DIR / 'phase1_column_inventory.csv'}")

# =============================================================================
# PHASE 1.2: MISSING VALUE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("🔍 PHASE 1.2: MISSING VALUE ANALYSIS")
print("="*80)

# Missing value summary
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'column': missing.index,
    'missing_count': missing.values,
    'missing_pct': missing_pct.values,
    'dtype': df.dtypes.values
}).sort_values('missing_pct', ascending=False)

print("=== Missing Value Summary ===")
print("Top 20 columns with most missing values:")
print(missing_df.head(20).to_string(index=False))

# Identify columns with significant missing values
high_missing = missing_df[missing_df['missing_pct'] > 50]
moderate_missing = missing_df[(missing_df['missing_pct'] > 10) & (missing_df['missing_pct'] <= 50)]
low_missing = missing_df[(missing_df['missing_pct'] > 0) & (missing_df['missing_pct'] <= 10)]

print(f"\n=== Missing Value Categories ===")
print(f"High Missing (>50%): {len(high_missing)} columns")
if len(high_missing) > 0:
    print(f"  {high_missing['column'].tolist()}")

print(f"Moderate Missing (10-50%): {len(moderate_missing)} columns")
if len(moderate_missing) > 0:
    print(f"  {moderate_missing['column'].tolist()}")

print(f"Low Missing (0-10%): {len(low_missing)} columns")
if len(low_missing) > 0:
    print(f"  {low_missing['column'].tolist()}")

# Critical missing value analysis for specific columns
print(f"\n=== Critical Missing Value Analysis ===")

# Check mel_thick_mm specifically (expected ~99.8% missing)
if 'mel_thick_mm' in df.columns:
    mel_thick_missing = df['mel_thick_mm'].isna().sum()
    mel_thick_pct = mel_thick_missing / len(df) * 100
    print(f"mel_thick_mm: {mel_thick_missing:,} missing ({mel_thick_pct:.2f}%)")
    
    # Check target distribution for non-missing mel_thick_mm
    non_missing_mel = df[df['mel_thick_mm'].notna()]
    if len(non_missing_mel) > 0:
        mel_target_rate = non_missing_mel['target'].mean()
        print(f"  Non-missing mel_thick_mm target rate: {mel_target_rate:.4f} ({mel_target_rate*100:.2f}%)")
        print(f"  ⚠️  CRITICAL: This suggests {'LEAKAGE' if mel_target_rate > 0.8 else 'potential leakage'}")

# Check other potentially problematic columns
problematic_cols = ['age_approx', 'sex', 'anatom_site_general', 'tbp_lv_dnn_lesion_confidence', 'tbp_lv_nevi_confidence']
for col in problematic_cols:
    if col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100
        print(f"{col}: {missing_count:,} missing ({missing_pct:.2f}%)")

# Save missing value analysis
missing_df.to_csv(RESULTS_DIR / 'phase1_missing_values.csv', index=False)
print(f"\n💾 Missing value analysis saved to: {RESULTS_DIR / 'phase1_missing_values.csv'}")

# Create missing value visualization
plt.figure(figsize=(12, 8))
missing_top20 = missing_df.head(20)
plt.barh(range(len(missing_top20)), missing_top20['missing_pct'])
plt.yticks(range(len(missing_top20)), missing_top20['column'])
plt.xlabel('Missing Percentage (%)')
plt.title('Top 20 Columns by Missing Value Percentage')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'phase1_missing_values_top20.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 Missing values plot saved to: {PLOTS_DIR / 'phase1_missing_values_top20.png'}")

# =============================================================================
# PHASE 1.3: TARGET CORRELATION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("📈 PHASE 1.3: TARGET CORRELATION ANALYSIS")
print("="*80)

# Calculate correlations for numeric columns
correlations = {}
for col in numeric_cols:
    if col in ['target', 'isic_id', 'patient_id']:
        continue
    
    valid_mask = df[col].notna()
    n_valid = valid_mask.sum()
    
    if n_valid > 100:  # Need enough samples for meaningful correlation
        try:
            corr, pval = pointbiserialr(df.loc[valid_mask, 'target'], df.loc[valid_mask, col])
            correlations[col] = {
                'correlation': corr, 
                'pvalue': pval, 
                'n_valid': n_valid,
                'missing_pct': (1 - n_valid/len(df)) * 100
            }
        except Exception as e:
            print(f"⚠️  Error calculating correlation for {col}: {e}")

# Create correlation dataframe
if correlations:
    corr_df = pd.DataFrame(correlations).T
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    print("=== Top 20 Positive Correlations with Target ===")
    print(corr_df.head(20)[['correlation', 'pvalue', 'n_valid', 'missing_pct']].to_string())
    
    print("\n=== Top 20 Negative Correlations with Target ===")
    print(corr_df.tail(20)[['correlation', 'pvalue', 'n_valid', 'missing_pct']].to_string())
    
    # Identify strongest correlations
    strong_positive = corr_df[corr_df['correlation'] > 0.1]
    strong_negative = corr_df[corr_df['correlation'] < -0.1]
    
    print(f"\n=== Strong Correlations (|r| > 0.1) ===")
    print(f"Strong Positive (>0.1): {len(strong_positive)} features")
    if len(strong_positive) > 0:
        print(strong_positive[['correlation', 'pvalue']].to_string())
    
    print(f"Strong Negative (<-0.1): {len(strong_negative)} features")
    if len(strong_negative) > 0:
        print(strong_negative[['correlation', 'pvalue']].to_string())
    
    # Save correlation analysis
    corr_df.to_csv(RESULTS_DIR / 'phase1_target_correlations.csv')
    print(f"\n💾 Target correlations saved to: {RESULTS_DIR / 'phase1_target_correlations.csv'}")
    
    # Create correlation visualization
    plt.figure(figsize=(10, 12))
    top_30 = corr_df.head(30)
    colors = ['red' if x < 0 else 'blue' for x in top_30['correlation']]
    plt.barh(range(len(top_30)), top_30['correlation'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_30)), top_30.index)
    plt.xlabel('Correlation with Target')
    plt.title('Top 30 Features by Correlation with Target')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'phase1_correlations_top30.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Correlations plot saved to: {PLOTS_DIR / 'phase1_correlations_top30.png'}")

# =============================================================================
# SUMMARY AND KEY FINDINGS
# =============================================================================
print("\n" + "="*80)
print("📋 PHASE 1 SUMMARY & KEY FINDINGS")
print("="*80)

print(f"✅ Dataset Overview:")
print(f"   • Total samples: {len(df):,}")
print(f"   • Total features: {len(df.columns)}")
print(f"   • Numeric features: {len(numeric_cols)}")
print(f"   • Categorical features: {len(object_cols)}")
print(f"   • Target rate: {df['target'].mean():.6f} ({df['target'].mean()*100:.4f}%)")

print(f"\n🔍 Missing Value Insights:")
print(f"   • Columns with >50% missing: {len(high_missing)}")
print(f"   • Columns with 10-50% missing: {len(moderate_missing)}")
print(f"   • Columns with 0-10% missing: {len(low_missing)}")

if 'mel_thick_mm' in df.columns:
    mel_missing_pct = df['mel_thick_mm'].isna().sum() / len(df) * 100
    non_missing_mel = df[df['mel_thick_mm'].notna()]
    if len(non_missing_mel) > 0:
        mel_target_rate = non_missing_mel['target'].mean()
        print(f"   • ⚠️  CRITICAL - mel_thick_mm: {100-mel_missing_pct:.2f}% non-missing, {mel_target_rate*100:.1f}% target=1")
        print(f"     This suggests {'STRONG DATA LEAKAGE' if mel_target_rate > 0.9 else 'potential leakage'}")

if correlations:
    print(f"\n📈 Correlation Insights:")
    print(f"   • Features with strong positive correlation (>0.1): {len(strong_positive)}")
    print(f"   • Features with strong negative correlation (<-0.1): {len(strong_negative)}")
    
    if len(strong_positive) > 0:
        top_positive = strong_positive.head(3)
        print(f"   • Top 3 positive correlations:")
        for idx, row in top_positive.iterrows():
            print(f"     - {idx}: r={row['correlation']:.4f}")

print(f"\n💾 Results saved to:")
print(f"   • {RESULTS_DIR / 'phase1_column_inventory.csv'}")
print(f"   • {RESULTS_DIR / 'phase1_missing_values.csv'}")
print(f"   • {RESULTS_DIR / 'phase1_target_correlations.csv'}")
print(f"   • {PLOTS_DIR / 'phase1_missing_values_top20.png'}")
print(f"   • {PLOTS_DIR / 'phase1_correlations_top30.png'}")

print("\n🎯 Phase 1 Complete! Ready for Phase 2 deep dive analysis.")
print("="*80)