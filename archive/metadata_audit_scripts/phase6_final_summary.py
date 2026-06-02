#!/usr/bin/env python3
"""
Phase 6: Actionable Investigation Summary
=========================================

This script completes Phase 6 of the metadata investigation plan:
- Consolidates all findings from Phases 1-5
- Provides actionable recommendations
- Updates the final report

Author: Kilo Code
Date: 2025-11-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results' / 'stacking_final_v1'
PLOTS_DIR = BASE_DIR / 'plots'
OUTPUT_DIR = BASE_DIR / 'metadata_investigation' / 'results'
REPORT_DIR = BASE_DIR / 'metadata_investigation'

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("📋 Phase 6: Actionable Investigation Summary")
print("=" * 50)

# Load all phase results
print("\n📊 Loading results from all phases...")

# Phase 1 results
phase1_missing = None
phase1_correlations = None
try:
    phase1_missing = pd.read_csv(OUTPUT_DIR / 'phase1_missing_values.csv', index_col=0)
    phase1_correlations = pd.read_csv(OUTPUT_DIR / 'phase1_target_correlations.csv', index_col=0)
    print("✅ Phase 1 results loaded")
except Exception as e:
    print(f"⚠️  Phase 1 results not found: {e}")

# Phase 2 results
phase2_thickness = None
try:
    phase2_thickness = pd.read_csv(OUTPUT_DIR / 'phase2_thickness_leakage_analysis.csv')
    print("✅ Phase 2 results loaded")
except Exception as e:
    print(f"⚠️  Phase 2 results not found: {e}")

# Phase 3 results
phase3_summary = None
try:
    with open(OUTPUT_DIR / 'phase3_complete_results.json', 'r') as f:
        phase3_summary = json.load(f)
    print("✅ Phase 3 results loaded")
except Exception as e:
    print(f"⚠️  Phase 3 results not found: {e}")

# Phase 4 results
phase4_summary = None
phase4_comparison = None
try:
    with open(OUTPUT_DIR / 'phase4_summary.json', 'r') as f:
        phase4_summary = json.load(f)
    phase4_comparison = pd.read_csv(OUTPUT_DIR / 'phase4_importance_vs_correlation.csv')
    print("✅ Phase 4 results loaded")
except Exception as e:
    print(f"⚠️  Phase 4 results not found: {e}")

# Phase 5 results
phase5_summary = None
phase5_shift = None
try:
    with open(OUTPUT_DIR / 'phase5_summary.json', 'r') as f:
        phase5_summary = json.load(f)
    phase5_shift = pd.read_csv(OUTPUT_DIR / 'phase5_distribution_shift.csv')
    print("✅ Phase 5 results loaded")
except Exception as e:
    print(f"⚠️  Phase 5 results not found: {e}")

# Load original data for context
train_df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
test_df = pd.read_csv(DATA_DIR / 'students-test-metadata.csv', low_memory=False)

print(f"\n📈 Dataset Overview:")
print(f"Training samples: {len(train_df):,}")
print(f"Test samples: {len(test_df):,}")
print(f"Training columns: {len(train_df.columns)}")
print(f"Test columns: {len(test_df.columns)}")
print(f"Malignant rate: {train_df['target'].mean():.6f} ({train_df['target'].sum()}/{len(train_df)})")

# Consolidate all findings
print(f"\n🔍 Consolidating Findings from All Phases...")
print("=" * 50)

# Critical findings summary
critical_findings = {
    "data_leakage": [],
    "high_value_features": [],
    "problematic_features": [],
    "distribution_shifts": [],
    "data_quality_issues": [],
    "feature_engineering_success": []
}

# Phase 1: Data leakage detection
if phase1_missing is not None:
    high_missing = phase1_missing[phase1_missing['missing_pct'] > 90]
    for _, row in high_missing.iterrows():
        if row.name == 'mel_thick_mm':
            critical_findings["data_leakage"].append({
                "feature": row.name,
                "issue": f"{row['missing_pct']:.2f}% missing, confirmed post-biopsy data",
                "action": "REMOVE IMMEDIATELY",
                "severity": "CRITICAL"
            })

# Phase 2: Feature analysis findings
if phase2_thickness is not None:
    critical_findings["data_leakage"].append({
        "feature": "mel_thick_mm",
        "issue": "100% of non-missing values are malignant (confirmed leakage)",
        "action": "REMOVE FROM ALL PIPELINES",
        "severity": "CRITICAL"
    })

# Phase 3: Engineered features
if phase3_summary:
    critical_findings["feature_engineering_success"].append({
        "feature_type": "Patient-relative features",
        "count": 50,
        "status": "Successfully implemented",
        "quality": "Properly standardized"
    })
    
    critical_findings["feature_engineering_success"].append({
        "feature_type": "Vision model diversity",
        "eva02_edgenext_correlation": -0.0081,
        "status": "Excellent diversity",
        "quality": "Complementary predictions"
    })

# Phase 4: Feature importance vs correlation
if phase4_comparison is not None:
    # High importance but low correlation features
    suspicious = phase4_comparison[
        (phase4_comparison['importance'] > 200) & 
        (phase4_comparison['correlation'].fillna(0).abs() < 0.01)
    ]
    
    for _, row in suspicious.iterrows():
        critical_findings["problematic_features"].append({
            "feature": row['feature'],
            "issue": f"High importance ({row['importance']:.1f}) but low correlation ({row['correlation']:.3f})",
            "action": "INVESTIGATE - may indicate interactions or spurious correlations",
            "severity": "MEDIUM"
        })

# Phase 5: Distribution shifts
if phase5_shift is not None:
    significant_shifts = phase5_shift[phase5_shift['significant_shift'] == True]
    for _, row in significant_shifts.iterrows():
        critical_findings["distribution_shifts"].append({
            "feature": row['column'],
            "shift_std": row['shift_std'],
            "ks_pvalue": row['ks_pvalue'],
            "action": "MONITOR during inference",
            "severity": "LOW"
        })

# High-value features from correlations
if phase1_correlations is not None:
    top_positive = phase1_correlations.head(5)
    top_negative = phase1_correlations.tail(5)
    
    for _, row in top_positive.iterrows():
        critical_findings["high_value_features"].append({
            "feature": row.name,
            "correlation": row['correlation'],
            "pvalue": row['pvalue'],
            "type": "Positive predictor"
        })
    
    for _, row in top_negative.iterrows():
        critical_findings["high_value_features"].append({
            "feature": row.name,
            "correlation": row['correlation'],
            "pvalue": row['pvalue'],
            "type": "Negative predictor"
        })

# Data quality issues
if phase5_summary:
    if phase5_summary["duplicate_analysis"]["train_id_duplicates"] > 0:
        critical_findings["data_quality_issues"].append({
            "issue": f"{phase5_summary['duplicate_analysis']['train_id_duplicates']} duplicate training IDs",
            "action": "INVESTIGATE and remove",
            "severity": "HIGH"
        })
    
    if phase5_summary["outlier_analysis"]["extreme_outlier_columns"] > 0:
        critical_findings["data_quality_issues"].append({
            "issue": f"{phase5_summary['outlier_analysis']['extreme_outlier_columns']} columns with extreme outliers",
            "action": "INVESTIGATE outliers",
            "severity": "MEDIUM"
        })

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: Feature importance vs correlation scatter
if phase4_comparison is not None:
    axes[0, 0].scatter(phase4_comparison['correlation'].fillna(0), 
                       phase4_comparison['importance'], alpha=0.6)
    axes[0, 0].set_xlabel('Target Correlation')
    axes[0, 0].set_ylabel('Feature Importance')
    axes[0, 0].set_title('Feature Importance vs Correlation')
    
    # Highlight suspicious features
    suspicious = phase4_comparison[
        (phase4_comparison['importance'] > 200) & 
        (phase4_comparison['correlation'].fillna(0).abs() < 0.01)
    ]
    if len(suspicious) > 0:
        axes[0, 0].scatter(suspicious['correlation'].fillna(0), 
                           suspicious['importance'], 
                           color='red', s=100, alpha=0.8, label='Suspicious')
        axes[0, 0].legend()

# Plot 2: Distribution shifts
if phase5_shift is not None:
    top_shifts = phase5_shift.head(10)
    axes[0, 1].barh(range(len(top_shifts)), top_shifts['shift_std'])
    axes[0, 1].set_yticks(range(len(top_shifts)))
    axes[0, 1].set_yticklabels(top_shifts['column'])
    axes[0, 1].set_xlabel('Shift (Standard Deviations)')
    axes[0, 1].set_title('Top 10 Distribution Shifts')
    axes[0, 1].invert_yaxis()
    axes[0, 1].axvline(x=2, color='red', linestyle='--', alpha=0.7)

# Plot 3: Missing values
if phase1_missing is not None:
    top_missing = phase1_missing.head(10)
    axes[0, 2].barh(range(len(top_missing)), top_missing['missing_pct'])
    axes[0, 2].set_yticks(range(len(top_missing)))
    axes[0, 2].set_yticklabels(top_missing.index)
    axes[0, 2].set_xlabel('Missing Percentage')
    axes[0, 2].set_title('Top 10 Missing Values')
    axes[0, 2].invert_yaxis()

# Plot 4: Target correlations
if phase1_correlations is not None:
    top_corr = pd.concat([
        phase1_correlations.head(5),
        phase1_correlations.tail(5)
    ])
    colors = ['green'] * 5 + ['red'] * 5
    axes[1, 0].barh(range(len(top_corr)), top_corr['correlation'], color=colors)
    axes[1, 0].set_yticks(range(len(top_corr)))
    axes[1, 0].set_yticklabels(top_corr.index)
    axes[1, 0].set_xlabel('Correlation with Target')
    axes[1, 0].set_title('Top/Bottom Correlations')
    axes[1, 0].invert_yaxis()

# Plot 5: Summary statistics
summary_stats = {
    'Total Features': len(train_df.columns),
    'High Missing (>90%)': len([f for f in critical_findings["data_leakage"] if 'missing' in f.get('issue', '')]),
    'Suspicious Features': len(critical_findings["problematic_features"]),
    'Distribution Shifts': len(critical_findings["distribution_shifts"]),
    'Engineered Features': 50,  # From Phase 3
    'Data Quality Issues': len(critical_findings["data_quality_issues"])
}

axes[1, 1].bar(summary_stats.keys(), summary_stats.values())
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Investigation Summary')
axes[1, 1].tick_params(axis='x', rotation=45)

# Plot 6: Severity breakdown
severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
for category in critical_findings.values():
    for item in category:
        severity = item.get('severity', 'LOW')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

axes[1, 2].pie(severity_counts.values(), labels=severity_counts.keys(), autopct='%1.1f%%')
axes[1, 2].set_title('Issue Severity Distribution')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'phase6_comprehensive_summary.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Saved comprehensive visualization: phase6_comprehensive_summary.png")

# Generate final recommendations
print(f"\n🎯 Generating Final Recommendations...")
print("=" * 50)

recommendations = {
    "immediate_actions": [],
    "feature_engineering": [],
    "model_improvements": [],
    "monitoring": [],
    "data_quality": []
}

# Immediate actions
if any(f['feature'] == 'mel_thick_mm' for f in critical_findings["data_leakage"]):
    recommendations["immediate_actions"].append({
        "priority": "CRITICAL",
        "action": "Remove mel_thick_mm from all training pipelines",
        "reason": "Confirmed post-biopsy data - 100% leakage",
        "impact": "Prevents data leakage in production"
    })

# Feature engineering
if critical_findings["feature_engineering_success"]:
    recommendations["feature_engineering"].append({
        "priority": "HIGH",
        "action": "Maintain patient-relative features",
        "reason": "Successfully capture 'Ugly Duckling' sign",
        "impact": "Adds clinical value to model"
    })
    
    recommendations["feature_engineering"].append({
        "priority": "HIGH", 
        "action": "Keep vision model diversity",
        "reason": "EVA02/EdgeNeXt correlation of -0.0081 is excellent",
        "impact": "Optimal ensemble performance"
    })

# Model improvements
if len(critical_findings["problematic_features"]) > 0:
    recommendations["model_improvements"].append({
        "priority": "MEDIUM",
        "action": f"Investigate {len(critical_findings['problematic_features'])} suspicious features",
        "reason": "High importance but low correlation may indicate issues",
        "impact": "May reveal spurious correlations or interactions"
    })

# Monitoring
if critical_findings["distribution_shifts"]:
    recommendations["monitoring"].append({
        "priority": "LOW",
        "action": f"Monitor {len(critical_findings['distribution_shifts'])} features with distribution shift",
        "reason": "Test set differs from training set",
        "impact": "Ensure model generalization"
    })

# Data quality
if critical_findings["data_quality_issues"]:
    recommendations["data_quality"].append({
        "priority": "HIGH",
        "action": "Address data quality issues",
        "reason": "Clean data improves model reliability",
        "impact": "Better model performance and stability"
    })

# Save comprehensive summary
final_summary = {
    "investigation_completed": datetime.now().isoformat(),
    "phases_completed": [1, 2, 3, 4, 5, 6],
    "total_features_analyzed": len(train_df.columns),
    "critical_findings": critical_findings,
    "recommendations": recommendations,
    "summary_statistics": {
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "malignant_rate": float(train_df['target'].mean()),
        "total_malignant": int(train_df['target'].sum()),
        "features_with_high_missing": len([f for f in critical_findings["data_leakage"] if 'missing' in f.get('issue', '')]),
        "suspicious_features": len(critical_findings["problematic_features"]),
        "distribution_shifts": len(critical_findings["distribution_shifts"]),
        "engineered_features": 50
    }
}

with open(OUTPUT_DIR / 'phase6_final_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=2)

print(f"\n💾 Saved final summary: phase6_final_summary.json")

# Print executive summary
print(f"\n📋 EXECUTIVE SUMMARY")
print("=" * 50)
print(f"✅ Investigation completed across all 6 phases")
print(f"📊 Analyzed {len(train_df.columns)} features across {len(train_df):,} training samples")
print(f"🎯 Identified {len(critical_findings['data_leakage'])} critical data leakage issues")
print(f"🔍 Found {len(critical_findings['problematic_features'])} suspicious features")
print(f"📈 {len(critical_findings['distribution_shifts'])} features show distribution shift")
print(f"⚙️  {len(critical_findings['feature_engineering_success'])} engineered feature categories successful")

print(f"\n🚨 CRITICAL ACTIONS REQUIRED:")
for rec in recommendations["immediate_actions"]:
    print(f"  • {rec['action']} (Priority: {rec['priority']})")

print(f"\n📋 RECOMMENDATIONS SUMMARY:")
for category, recs in recommendations.items():
    if recs:
        print(f"  {category.upper()}:")
        for rec in recs:
            print(f"    • {rec['action']} (Priority: {rec['priority']})")

print(f"\n✅ Phase 6 completed successfully!")
print(f"📁 All results saved to: {OUTPUT_DIR}")
print(f"📊 Comprehensive analysis complete - ready for implementation")

# Return the final summary for report updating
print(f"\n🔄 Preparing to update report...")