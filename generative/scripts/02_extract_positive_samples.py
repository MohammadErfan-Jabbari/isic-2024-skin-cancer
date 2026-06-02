"""
Step 1B: Extract Positive (Malignant) Samples from HDF5
Resize to 128x128 for Stable Diffusion fine-tuning.
"""

import h5py
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
from io import BytesIO

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = DATA_DIR / "new-train-metadata.csv"
HDF5_PATH = DATA_DIR / "train-image.hdf5"
OUTPUT_DIR = PROJECT_ROOT / "generative" / "data" / "train_images_128"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
TARGET_SIZE = (128, 128)
RESAMPLE_METHOD = Image.Resampling.LANCZOS

print("=" * 80)
print("STEP 1B: EXTRACT & PREPROCESS POSITIVE SAMPLES")
print("=" * 80)

# Step 1: Load metadata and filter positive samples
print("\n[1/4] Loading metadata and filtering positive samples...")
df = pd.read_csv(METADATA_PATH, usecols=['isic_id', 'target'])
positive_samples = df[df['target'] == 1].copy()
positive_ids = set(positive_samples['isic_id'].values)

print(f"✓ Total positive samples to extract: {len(positive_ids)}")

# Step 2: Open HDF5 and extract images
print("\n[2/4] Extracting images from HDF5...")
extracted_count = 0
failed_count = 0
failed_ids = []
image_mapping = {}  # Track original indices

with h5py.File(HDF5_PATH, 'r') as hdf5_file:
    all_keys = list(hdf5_file.keys())
    
    for idx, key in enumerate(tqdm(all_keys, desc="Processing HDF5")):
        if key in positive_ids:
            try:
                # Read JPEG-encoded bytes from HDF5
                jpeg_bytes = hdf5_file[key][()]
                
                # Decode JPEG to PIL Image
                img = Image.open(BytesIO(jpeg_bytes))
                
                # Convert to RGB if not already
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get original size for tracking
                original_shape = img.size
                
                # Resize to 128x128
                img_resized = img.resize(TARGET_SIZE, resample=RESAMPLE_METHOD)
                
                # Save as PNG
                output_path = OUTPUT_DIR / f"{key}.png"
                img_resized.save(output_path, format='PNG')
                
                # Track in mapping
                image_mapping[key] = {
                    'original_size': original_shape,
                    'resized_shape': img_resized.size,
                    'output_file': str(output_path.name)
                }
                
                extracted_count += 1
            
            except Exception as e:
                print(f"\n✗ Failed to process {key}: {e}")
                failed_count += 1
                failed_ids.append(key)

print(f"\n✓ Successfully extracted: {extracted_count} images")
if failed_count > 0:
    print(f"✗ Failed: {failed_count} images")
    print(f"  Failed IDs: {failed_ids}")

# Step 3: Save metadata mapping
print("\n[3/4] Saving metadata mapping...")
metadata_output = OUTPUT_DIR / "extraction_metadata.json"
with open(metadata_output, 'w') as f:
    json.dump({
        'total_extracted': extracted_count,
        'total_failed': failed_count,
        'target_size': TARGET_SIZE,
        'resample_method': 'LANCZOS',
        'images': image_mapping
    }, f, indent=2)

print(f"✓ Metadata saved to: {metadata_output}")

# Step 4: Create JSONL caption file
print("\n[4/4] Creating caption file for Stable Diffusion training...")
caption_templates = [
    "A dermoscopic image of a malignant skin lesion",
    "A close-up photo of a melanoma",
    "A skin lesion with irregular borders and asymmetry",
    "A malignant melanoma with varied color",
    "A dermoscopic image of a dysplastic nevus",
    "A close-up photo of a malignant lesion",
    "A skin lesion image with asymmetric features",
    "A dermoscopic view of a melanoma",
]

jsonl_path = OUTPUT_DIR / "metadata.jsonl"
with open(jsonl_path, 'w') as jsonl_file:
    for idx, (image_id, metadata) in enumerate(image_mapping.items()):
        # Cycle through caption templates
        caption = caption_templates[idx % len(caption_templates)]
        
        # Create JSONL entry
        entry = {
            'file_name': metadata['output_file'],
            'text': caption,
            'isic_id': image_id
        }
        jsonl_file.write(json.dumps(entry) + '\n')

print(f"✓ Caption file created: {jsonl_path}")

# Print Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Output Directory: {OUTPUT_DIR}")
print(f"Total Images Extracted: {extracted_count}")
print(f"Image Size: {TARGET_SIZE}")
print(f"Resample Method: LANCZOS")
print(f"Metadata File: extraction_metadata.json")
print(f"Caption File: metadata.jsonl")
print(f"\n✓ Step 1B complete! Ready for Step 1C (Validation)")
print("=" * 80)
