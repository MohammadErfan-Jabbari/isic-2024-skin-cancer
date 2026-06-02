#!/usr/bin/env python3
"""Compare OOF predictions across different models."""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("OOF PREDICTION DISTRIBUTIONS (Training Data)")
print("="*60)

# Load EVA02 OOFs
eva_oofs = []
for i in range(1, 6):
    oof = pd.read_csv(f'results/gen-train-run-eva-v2/oof_fold{i}.csv')
    oof['fold'] = i
    eva_oofs.append(oof)
eva_all = pd.concat(eva_oofs, ignore_index=True)

# Load EdgeNeXt OOFs
edgenext_oofs = []
for i in range(1, 6):
    oof = pd.read_csv(f'results/gen-train-run-edgenext-v2/oof_fold{i}.csv')
    oof['fold'] = i
    edgenext_oofs.append(oof)
edgenext_all = pd.concat(edgenext_oofs, ignore_index=True)

# Load KFold V2S OOF
try:
    kfold_oof = pd.read_csv('results/kfold_v2s_features_advanced_20251111_150340/individual_fold_predictions.csv')
    has_kfold = True
except:
    has_kfold = False

print()
print("EVA02 OOF (All 5 Folds):")
print(f"  Count:    {len(eva_all):,}")
print(f"  Mean:     {eva_all.pred.mean():.6f}")
print(f"  Std:      {eva_all.pred.std():.6f}")
print(f"  Min:      {eva_all.pred.min():.6f}")
print(f"  Max:      {eva_all.pred.max():.6f}")
print(f"  Median:   {eva_all.pred.median():.6f}")

print()
print("EdgeNeXt OOF (All 5 Folds):")
print(f"  Count:    {len(edgenext_all):,}")
print(f"  Mean:     {edgenext_all.pred.mean():.6f}")
print(f"  Std:      {edgenext_all.pred.std():.6f}")
print(f"  Min:      {edgenext_all.pred.min():.6f}")
print(f"  Max:      {edgenext_all.pred.max():.6f}")
print(f"  Median:   {edgenext_all.pred.median():.6f}")

if has_kfold:
    print()
    print("KFold V2S (EfficientNetV2-S + Metadata):")
    # Check what columns exist
    print(f"  Columns: {list(kfold_oof.columns)}")
    
    # Find prediction columns
    pred_cols = [c for c in kfold_oof.columns if 'pred' in c.lower()]
    if pred_cols:
        for col in pred_cols[:3]:  # Show first 3
            print(f"  {col}:")
            print(f"    Mean:   {kfold_oof[col].mean():.6f}")
            print(f"    Std:    {kfold_oof[col].std():.6f}")
            print(f"    Min:    {kfold_oof[col].min():.6f}")
            print(f"    Max:    {kfold_oof[col].max():.6f}")

# Now compare to test predictions
print()
print("="*60)
print("TEST PREDICTION DISTRIBUTIONS")
print("="*60)

# Stacking v2 submission
stacking = pd.read_csv('results/stacking_v2_20251126_153448/submissions/submission.csv')
print()
print("Stacking v2 (current):")
print(f"  Mean:   {stacking.target.mean():.6f}")
print(f"  Std:    {stacking.target.std():.6f}")
print(f"  Min:    {stacking.target.min():.6f}")
print(f"  Max:    {stacking.target.max():.6f}")

# Best kfold submission
best = pd.read_csv('results/kfold_v2s_features_advanced_20251111_150340/submission_kfold_median_all.csv')
print()
print("KFold V2S Best (0.16741):")
print(f"  Mean:   {best.target.mean():.6f}")
print(f"  Std:    {best.target.std():.6f}")
print(f"  Min:    {best.target.min():.6f}")
print(f"  Max:    {best.target.max():.6f}")

# Key insight
print()
print("="*60)
print("KEY INSIGHT")
print("="*60)
print()
print("The RATIO of test mean to OOF mean reveals distribution shift:")
print(f"  EVA02:     OOF mean={eva_all.pred.mean():.4f}")
print(f"  EdgeNeXt:  OOF mean={edgenext_all.pred.mean():.4f}")
print(f"  Stacking:  Test mean={stacking.target.mean():.4f}")
print(f"  KFold:     Test mean={best.target.mean():.4f}")
