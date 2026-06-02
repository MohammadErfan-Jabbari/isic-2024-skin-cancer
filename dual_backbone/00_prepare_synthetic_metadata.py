"""
18_0: Prepare Synthetic Metadata
================================
Assigns realistic metadata to synthetic malignant images by randomly sampling
from the 343 real malignant cases.

This approach:
1. Preserves the real malignant metadata distribution
2. Avoids fake/identical metadata for all synthetic samples
3. Provides regularization noise for the fusion head

Usage:
    uv run python 18_0_prepare_synthetic_metadata.py
    
Output:
    generative/data/synthetic_malignant_metadata_enriched.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# ===========================
# CONFIGURATION
# ===========================

DATA_DIR = Path('DeepLearning/Kaggle/data')
SYNTH_DIR = Path('DeepLearning/Kaggle/generative/data')
SEED = 42

def main():
    print("="*70)
    print("SYNTHETIC METADATA PREPARATION")
    print("="*70)
    
    np.random.seed(SEED)
    
    # 1. Load real training metadata
    print("\n[1/5] Loading real training metadata...")
    real_df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
    print(f"  Total samples: {len(real_df):,}")
    print(f"  Malignant samples: {real_df['target'].sum()}")
    print(f"  Benign samples: {(real_df['target'] == 0).sum():,}")
    
    # 2. Extract real malignant samples
    print("\n[2/5] Extracting real malignant metadata...")
    real_malignant = real_df[real_df['target'] == 1].reset_index(drop=True)
    print(f"  Real malignant count: {len(real_malignant)}")
    
    # 3. Load current synthetic metadata
    print("\n[3/5] Loading synthetic metadata...")
    synth_df = pd.read_csv(SYNTH_DIR / 'synthetic_malignant_metadata.csv')
    print(f"  Synthetic count: {len(synth_df)}")
    
    # Show current problem - all identical
    print("\n  Current issue - all metadata identical:")
    print(f"    age_approx unique values: {synth_df['age_approx'].nunique()}")
    print(f"    tbp_lv_areaMM2 unique values: {synth_df['tbp_lv_areaMM2'].nunique()}")
    print(f"    sex unique values: {synth_df['sex'].nunique()}")
    
    # 4. Random sampling from real malignant
    print("\n[4/5] Assigning random metadata from real malignant cases...")
    
    n_synth = len(synth_df)
    
    # Sample indices with replacement
    sampled_indices = np.random.choice(
        len(real_malignant),
        size=n_synth,
        replace=True  # 343 real -> 6000 synthetic requires replacement
    )
    
    # Columns to copy from real malignant (exclude identifiers)
    exclude_cols = ['isic_id', 'patient_id', 'target', 'lesion_id', 
                    'attribution', 'copyright_license', 'image_type']
    
    metadata_cols = [col for col in real_malignant.columns if col not in exclude_cols]
    print(f"  Copying {len(metadata_cols)} metadata columns")
    
    # Create enriched synthetic dataframe
    synth_enriched = synth_df[['isic_id']].copy()  # Keep synthetic isic_id
    synth_enriched['target'] = 1  # All synthetic are malignant
    
    # Create unique patient IDs for synthetic samples
    synth_enriched['patient_id'] = [f'synthetic_pat_{i:05d}' for i in range(n_synth)]
    
    # Copy metadata from sampled real malignant cases
    for col in metadata_cols:
        synth_enriched[col] = real_malignant.iloc[sampled_indices][col].values
    
    # Set synthetic-specific identifiers
    synth_enriched['attribution'] = 'synthetic'
    synth_enriched['copyright_license'] = 'cc-0'
    synth_enriched['image_type'] = 'dermoscopic'
    
    # 5. Save enriched metadata
    print("\n[5/5] Saving enriched synthetic metadata...")
    output_path = SYNTH_DIR / 'synthetic_malignant_metadata_enriched.csv'
    synth_enriched.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    # Verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    print(f"\n  Enriched synthetic count: {len(synth_enriched)}")
    print(f"  Columns: {len(synth_enriched.columns)}")
    
    print("\n  Metadata diversity (should now be varied):")
    print(f"    age_approx unique values: {synth_enriched['age_approx'].nunique()}")
    print(f"    tbp_lv_areaMM2 unique values: {synth_enriched['tbp_lv_areaMM2'].nunique():.0f}")
    print(f"    sex unique values: {synth_enriched['sex'].nunique()}")
    print(f"    anatom_site_general unique values: {synth_enriched['anatom_site_general'].nunique()}")
    
    # Compare distributions
    print("\n  Distribution comparison (real malignant vs synthetic):")
    
    comparison_cols = ['age_approx', 'tbp_lv_areaMM2', 'clin_size_long_diam_mm']
    for col in comparison_cols:
        real_mean = real_malignant[col].mean()
        synth_mean = synth_enriched[col].mean()
        print(f"    {col}:")
        print(f"      Real malignant mean: {real_mean:.2f}")
        print(f"      Synthetic mean:      {synth_mean:.2f}")
    
    # Save stats
    stats = {
        'timestamp': datetime.now().isoformat(),
        'seed': SEED,
        'n_real_malignant': len(real_malignant),
        'n_synthetic': n_synth,
        'n_metadata_cols': len(metadata_cols),
        'sampling_method': 'random_with_replacement',
        'output_file': str(output_path)
    }
    
    stats_path = SYNTH_DIR / 'synthetic_metadata_enrichment_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to: {stats_path}")
    
    print("\n" + "="*70)
    print("✅ SYNTHETIC METADATA PREPARATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
