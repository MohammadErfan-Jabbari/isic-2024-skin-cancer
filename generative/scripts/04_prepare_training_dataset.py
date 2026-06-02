#!/usr/bin/env python3
"""
Step 2.3: Prepare Training Dataset for Stable Diffusion Fine-tuning
Organizes 343 extracted images + metadata into diffusers-compatible format.
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
SOURCE_DIR = Path("DeepLearning/Kaggle/generative/data/train_images_128")
TARGET_DIR = Path("DeepLearning/Kaggle/generative/data/training_dataset")
SOURCE_METADATA = SOURCE_DIR / "metadata.jsonl"

print("="*70)
print("📦 STEP 2.3: PREPARE TRAINING DATASET FOR DIFFUSERS")
print("="*70)

# Verify source
print(f"\n🔍 Checking source directory...")
if not SOURCE_DIR.exists():
    print(f"❌ Source directory not found: {SOURCE_DIR}")
    exit(1)

source_images = list(SOURCE_DIR.glob("ISIC_*.png"))
print(f"✅ Found {len(source_images)} images in source")

if not SOURCE_METADATA.exists():
    print(f"❌ Source metadata not found: {SOURCE_METADATA}")
    exit(1)
print(f"✅ Found metadata.jsonl in source")

# Create target directory
print(f"\n📂 Creating target directory: {TARGET_DIR}")
TARGET_DIR.mkdir(parents=True, exist_ok=True)
print(f"✅ Target directory ready")

# Copy all images
print(f"\n🖼️  Copying {len(source_images)} images...")
for img_path in tqdm(sorted(source_images), desc="Copying images"):
    target_path = TARGET_DIR / img_path.name
    shutil.copy2(img_path, target_path)
print(f"✅ All images copied")

# Copy metadata.jsonl
print(f"\n📝 Copying metadata.jsonl...")
target_metadata = TARGET_DIR / "metadata.jsonl"
shutil.copy2(SOURCE_METADATA, target_metadata)
print(f"✅ Metadata copied")

# Verify
print(f"\n✔️  VERIFICATION:")
target_images = list(TARGET_DIR.glob("ISIC_*.png"))
print(f"   Images in target: {len(target_images)}")

with open(target_metadata) as f:
    metadata_lines = len(f.readlines())
print(f"   Metadata entries: {metadata_lines}")

# Show sample
print(f"\n📋 Sample metadata entry:")
with open(target_metadata) as f:
    sample = json.loads(f.readline())
    for key, value in sample.items():
        print(f"   {key}: {value}")

# Final structure
print(f"\n" + "="*70)
print(f"✅ TRAINING DATASET READY")
print(f"="*70)
print(f"\n📂 Directory structure:")
print(f"   {TARGET_DIR}/")
print(f"   ├── {len(target_images)} ISIC_*.png files")
print(f"   └── metadata.jsonl (343 entries)")
print(f"\nTotal size: {sum(p.stat().st_size for p in target_images) / 1e6:.1f} MB")
print(f"\n✨ Ready for Phase 2 fine-tuning!")
print(f"="*70)
