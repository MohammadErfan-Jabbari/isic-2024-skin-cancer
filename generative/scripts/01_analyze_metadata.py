"""
Step 1A: Analyze Training Metadata
Identify positive (malignant) samples and understand dataset structure.
"""

import pandas as pd
import sys
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = DATA_DIR / "new-train-metadata.csv"

print("=" * 80)
print("STEP 1A: ANALYZE TRAINING METADATA")
print("=" * 80)

# Read metadata
print("\nReading metadata from:", METADATA_PATH)
df = pd.read_csv(METADATA_PATH, usecols=['target', 'isic_id', 'patient_id'])

# Display target distribution
print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)
target_counts = df['target'].value_counts().sort_index()
print(target_counts)

total_samples = len(df)
benign_count = target_counts.get(0, 0)
malignant_count = target_counts.get(1, 0)

print(f"\nTotal samples: {total_samples}")
print(f"Benign (target=0): {benign_count}")
print(f"Malignant (target=1): {malignant_count}")
print(f"Class Imbalance Ratio (Benign:Malignant): {benign_count}:{malignant_count} ≈ 1:{benign_count/malignant_count:.1f}")

# Analyze positive samples
print("\n" + "=" * 80)
print("POSITIVE SAMPLES ANALYSIS (target=1)")
print("=" * 80)
positive_samples = df[df['target'] == 1]
print(f"Total positive samples: {len(positive_samples)}")
print(f"Unique patients with malignant lesions: {positive_samples['patient_id'].nunique()}")

print(f"\nFirst 15 positive samples:")
print(positive_samples.head(15).to_string(index=False))

# Check patient overlap
print("\n" + "=" * 80)
print("PATIENT DISTRIBUTION")
print("=" * 80)
print(f"Total unique patients: {df['patient_id'].nunique()}")
patients_with_malignant = positive_samples['patient_id'].unique()
print(f"Patients with malignant lesions: {len(patients_with_malignant)}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY FOR STEP 1")
print("=" * 80)
print(f"✓ Metadata file successfully loaded")
print(f"✓ Total positive samples to extract: {len(positive_samples)}")
print(f"✓ Target resolution for SD training: 128×128 pixels")
print(f"✓ Label column: 'target' (0=benign, 1=malignant)")
print(f"✓ Image ID column: 'isic_id'")
print(f"✓ Patient grouping column: 'patient_id'")
print("\n✓ Step 1A complete. Ready to proceed with Step 1B (extraction).")
print("=" * 80)
