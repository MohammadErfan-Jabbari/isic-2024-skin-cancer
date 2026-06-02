#!/usr/bin/env python
"""Verify the saved scaler is correct."""
import pickle
from pathlib import Path

scaler_path = Path('results/test_scaler_debug/scaler_fold1.pkl')

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print('='*70)
print('SCALER VERIFICATION')
print('='*70)
print(f'✓ Scaler file exists: {scaler_path}')
print(f'  Type: {type(scaler).__name__}')
print(f'  Module: {type(scaler).__module__}')
print(f'  Has transform: {hasattr(scaler, "transform")}')
print(f'  Has fit_transform: {hasattr(scaler, "fit_transform")}')
print(f'  Has get_scale (GradScaler): {hasattr(scaler, "get_scale")}')
print()

if hasattr(scaler, 'transform'):
    print('✓ SUCCESS! This is a proper StandardScaler!')
    print('  The scaler fix is working correctly.')
    if hasattr(scaler, 'mean_'):
        print(f'  Mean shape: {scaler.mean_.shape}')
        print(f'  Scale shape: {scaler.scale_.shape}')
else:
    print('✗ ERROR! This is NOT a StandardScaler!')
    if hasattr(scaler, 'get_scale'):
        print('  This appears to be a GradScaler (wrong type)')

print('='*70)
