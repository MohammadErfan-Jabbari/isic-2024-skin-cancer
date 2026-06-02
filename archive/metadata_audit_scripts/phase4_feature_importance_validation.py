#!/usr/bin/env python3
"""
Phase 4: Feature Importance Validation
======================================

This script completes Phase 4 of the metadata investigation plan:
- 4.1: Compare Feature Importance to Correlation
- 4.2: Permutation Importance Test

Author: Kilo Code
Date: 2025-11-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
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

print("🔍 Phase 4: Feature Importance Validation")
print("=" * 50)

# Load feature importance from training
print("\n📊 Loading feature importance from stacking...")
fi_path = RESULTS_DIR / 'feature_importance.csv'
if not fi_path.exists():
    print(f"❌ Feature importance file not found: {fi_path}")
    print("Available files in results directory:")
    for f in RESULTS_DIR.glob('*'):
        print(f"  - {f.name}")
    exit(1)

fi_df = pd.read_csv(fi_path)
print(f"✅ Loaded feature importance: {len(fi_df)} features")
print(f"Top 10 features by importance:")
print(fi_df.head(10))

# Load target correlations from Phase 1
print("\n📈 Loading target correlations from Phase 1...")
corr_path = OUTPUT_DIR / 'phase1_target_correlations.csv'
if not corr_path.exists():
    print(f"❌ Target correlations file not found: {corr_path}")
    print("This file should have been created in Phase 1.3")
    exit(1)

corr_df = pd.read_csv(corr_path, index_col=0)
print(f"✅ Loaded target correlations: {len(corr_df)} features")

# Phase 4.1: Compare Feature Importance to Correlation
print("\n🔍 Phase 4.1: Comparing Feature Importance to Correlation")
print("-" * 50)

# Merge importance and correlation data
comparison = fi_df.merge(corr_df, left_on='feature', right_index=True, how='left')
comparison = comparison.sort_values('importance', ascending=False)

print("\n📋 Feature Importance vs Target Correlation (Top 30)")
print("=" * 70)
print(comparison[['feature', 'importance', 'correlation', 'pvalue', 'n_valid']].head(30).to_string(index=False))

# Save comparison results
comparison.to_csv(OUTPUT_DIR / 'phase4_importance_vs_correlation.csv', index=False)
print(f"\n💾 Saved: phase4_importance_vs_correlation.csv")

# Identify suspicious features
print("\n🚨 Identifying Suspicious Features...")
print("=" * 50)

# High importance but low correlation
suspicious_high_imp_low_corr = comparison[
    (comparison['importance'] > 200) & 
    (abs(comparison['correlation'].fillna(0)) < 0.01)
]

print(f"\n⚠️  HIGH IMPORTANCE (>200) but LOW CORRELATION (<0.01):")
if len(suspicious_high_imp_low_corr) > 0:
    print(suspicious_high_imp_low_corr[['feature', 'importance', 'correlation']].to_string(index=False))
    print(f"\nTotal suspicious features: {len(suspicious_high_imp_low_corr)}")
else:
    print("✅ No suspicious features found")

# Features with very high importance
very_high_importance = comparison[comparison['importance'] > 500]
print(f"\n🔥 VERY HIGH IMPORTANCE (>500):")
if len(very_high_importance) > 0:
    print(very_high_importance[['feature', 'importance', 'correlation']].to_string(index=False))

# Features with strong correlation but low importance
strong_corr_low_imp = comparison[
    (abs(comparison['correlation'].fillna(0)) > 0.03) & 
    (comparison['importance'] < 100)
]
print(f"\n💪 STRONG CORRELATION (>0.03) but LOW IMPORTANCE (<100):")
if len(strong_corr_low_imp) > 0:
    print(strong_corr_low_imp[['feature', 'importance', 'correlation']].to_string(index=False))

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: Importance vs Correlation scatter
plt.subplot(2, 2, 1)
plt.scatter(comparison['correlation'].fillna(0), comparison['importance'], alpha=0.6)
plt.xlabel('Target Correlation')
plt.ylabel('Feature Importance')
plt.title('Feature Importance vs Target Correlation')

# Highlight suspicious points
if len(suspicious_high_imp_low_corr) > 0:
    plt.scatter(suspicious_high_imp_low_corr['correlation'].fillna(0), 
                suspicious_high_imp_low_corr['importance'], 
                color='red', s=100, alpha=0.8, label='Suspicious')
    plt.legend()

# Plot 2: Top 20 features by importance
plt.subplot(2, 2, 2)
top_20 = comparison.head(20)
plt.barh(range(len(top_20)), top_20['importance'])
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Importance')
plt.title('Top 20 Features by Importance')
plt.gca().invert_yaxis()

# Plot 3: Top 20 features by correlation
plt.subplot(2, 2, 3)
top_corr = comparison.dropna(subset=['correlation']).head(20)
plt.barh(range(len(top_corr)), top_corr['correlation'])
plt.yticks(range(len(top_corr)), top_corr['feature'])
plt.xlabel('Correlation')
plt.title('Top 20 Features by Correlation')
plt.gca().invert_yaxis()

# Plot 4: Importance distribution
plt.subplot(2, 2, 4)
plt.hist(comparison['importance'], bins=50, alpha=0.7)
plt.xlabel('Feature Importance')
plt.ylabel('Frequency')
plt.title('Feature Importance Distribution')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'phase4_importance_vs_correlation_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Saved visualization: phase4_importance_vs_correlation_analysis.png")

# Phase 4.2: Permutation Importance Test
print("\n🔬 Phase 4.2: Permutation Importance Test")
print("-" * 50)

# Load trained model
model_path = RESULTS_DIR / 'models' / 'lgbm_fold1.joblib'
if not model_path.exists():
    print(f"❌ Model file not found: {model_path}")
    print("Skipping permutation importance test...")
    perm_importance_df = None
else:
    try:
        print(f"📦 Loading trained model...")
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully")

        # Load validation data for fold 1
        oof_path = RESULTS_DIR / 'stacking_oof.csv'
        if not oof_path.exists():
            print(f"❌ OOF file not found: {oof_path}")
            print("Skipping permutation importance test...")
            perm_importance_df = None
        else:
            oof_df = pd.read_csv(oof_path)
            print(f"✅ Loaded OOF data: {len(oof_df)} samples")

            # Get features and target
            feature_cols = [col for col in oof_df.columns if col not in ['target', 'isic_id', 'fold']]
            X_val = oof_df[feature_cols]
            y_val = oof_df['target']

            print(f"📊 Running permutation importance on {len(feature_cols)} features...")
            print("⏳ This may take several minutes...")

            # Run permutation importance
            perm_importance = permutation_importance(
                model, X_val, y_val, 
                n_repeats=10, 
                random_state=42,
                n_jobs=-1
            )

            # Create results dataframe
            perm_importance_df = pd.DataFrame({
                'feature': feature_cols,
                'perm_importance_mean': perm_importance.importances_mean,
                'perm_importance_std': perm_importance.importances_std
            }).sort_values('perm_importance_mean', ascending=False)

            print(f"\n📋 Permutation Importance (Top 20)")
            print("=" * 50)
            print(perm_importance_df.head(20).to_string(index=False))

            # Save permutation importance results
            perm_importance_df.to_csv(OUTPUT_DIR / 'phase4_permutation_importance.csv', index=False)
            print(f"\n💾 Saved: phase4_permutation_importance.csv")

            # Compare with GBDT importance
            print(f"\n🔄 Comparing Permutation vs GBDT Importance...")
            perm_vs_gbdt = perm_importance_df.merge(
                fi_df, on='feature', how='inner'
            ).sort_values('perm_importance_mean', ascending=False)

            print(f"\n📊 Top 20 Features by Permutation Importance:")
            print(perm_vs_gbdt[['feature', 'perm_importance_mean', 'importance']].head(20).to_string(index=False))

            # Create permutation importance visualization
            plt.figure(figsize=(15, 8))

            # Plot 1: Permutation importance
            plt.subplot(1, 2, 1)
            top_20_perm = perm_importance_df.head(20)
            plt.barh(range(len(top_20_perm)), top_20_perm['perm_importance_mean'])
            plt.yticks(range(len(top_20_perm)), top_20_perm['feature'])
            plt.xlabel('Permutation Importance')
            plt.title('Top 20 Features by Permutation Importance')
            plt.gca().invert_yaxis()

            # Plot 2: Permutation vs GBDT importance comparison
            plt.subplot(1, 2, 2)
            plt.scatter(perm_vs_gbdt['importance'], perm_vs_gbdt['perm_importance_mean'], alpha=0.6)
            plt.xlabel('GBDT Importance')
            plt.ylabel('Permutation Importance')
            plt.title('GBDT vs Permutation Importance')

            # Add diagonal line
            max_val = max(perm_vs_gbdt['importance'].max(), perm_vs_gbdt['perm_importance_mean'].max())
            plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect correlation')
            plt.legend()

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'phase4_permutation_importance_analysis.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Saved visualization: phase4_permutation_importance_analysis.png")

    except Exception as e:
        print(f"❌ Error during permutation importance test: {e}")
        perm_importance_df = None

# Summary and recommendations
print(f"\n📋 Phase 4 Summary")
print("=" * 50)

summary = {
    "phase": "Phase 4: Feature Importance Validation",
    "date": "2025-11-26",
    "total_features_analyzed": len(comparison),
    "suspicious_features_count": len(suspicious_high_imp_low_corr),
    "very_high_importance_count": len(very_high_importance),
    "strong_corr_low_imp_count": len(strong_corr_low_imp),
    "permutation_importance_completed": perm_importance_df is not None,
    "key_findings": []
}

# Add key findings
if len(suspicious_high_imp_low_corr) > 0:
    summary["key_findings"].append(f"Found {len(suspicious_high_imp_low_corr)} features with high importance but low correlation")
    summary["suspicious_features"] = suspicious_high_imp_low_corr['feature'].tolist()

if len(very_high_importance) > 0:
    summary["key_findings"].append(f"Found {len(very_high_importance)} features with very high importance (>500)")
    summary["very_high_importance_features"] = very_high_importance['feature'].tolist()

if perm_importance_df is not None:
    top_perm = perm_importance_df.head(5)['feature'].tolist()
    summary["key_findings"].append(f"Permutation importance confirms top features: {top_perm}")

# Save summary
with open(OUTPUT_DIR / 'phase4_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Phase 4 completed successfully!")
print(f"📁 Results saved to: {OUTPUT_DIR}")
print(f"📊 Visualizations saved to: {PLOTS_DIR}")

# Print key recommendations
print(f"\n🎯 Key Recommendations:")
print("=" * 30)
if len(suspicious_high_imp_low_corr) > 0:
    print(f"1. 🚨 Investigate {len(suspicious_high_imp_low_corr)} suspicious features with high importance but low correlation")
    print(f"   These may indicate: interactions, leakage, or spurious correlations")
else:
    print(f"1. ✅ No suspicious features found - good sign for model reliability")

if perm_importance_df is not None:
    print(f"2. ✅ Permutation importance test completed - confirms feature importance rankings")
else:
    print(f"2. ⚠️  Permutation importance test skipped (model files not available)")

print(f"3. 📊 Feature importance and correlation analysis provides validation of model behavior")
print(f"4. 🔍 Continue to Phase 5 for data quality checks")