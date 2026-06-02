
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
INPUT_DIR = Path('DeepLearning/Kaggle/generative/data/synthetic_malignant_filtered')
OUTPUT_DIR = Path('DeepLearning/Kaggle/generative/data')
TRAIN_META_PATH = Path('DeepLearning/Kaggle/data/new-train-metadata.csv')

# Features to fill (must match training scripts)
NUMERICAL_FEATURES = [
    'tbp_lv_H', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM',
    'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_Hext',
    'clin_size_long_diam_mm', 'tbp_lv_radial_color_std_max',
    'tbp_lv_B', 'tbp_lv_color_std_mean', 'tbp_lv_Aext',
    'tbp_lv_stdLExt', 'tbp_lv_norm_color', 'tbp_lv_A',
    'age_approx'
]

CATEGORICAL_FEATURES = [
    'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple'
]

def get_average_malignant_profile(train_meta_path):
    """Calculates the average metadata profile for malignant cases."""
    print("Calculating average malignant profile...")
    df = pd.read_csv(train_meta_path, low_memory=False)
    
    # Filter for malignant cases
    mal_df = df[df['target'] == 1].copy()
    print(f"  Found {len(mal_df)} malignant cases.")
    
    profile = {}
    
    # Numerical: Mean
    for col in NUMERICAL_FEATURES:
        if col in mal_df.columns:
            profile[col] = mal_df[col].mean()
        else:
            profile[col] = 0.0
            
    # Categorical: Mode
    for col in CATEGORICAL_FEATURES:
        if col in mal_df.columns:
            profile[col] = mal_df[col].mode()[0]
        else:
            profile[col] = 'missing'
            
    return profile

def main():
    print("🚀 Starting Phase 5: Packing Synthetic Data")
    
    # 1. Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_files = sorted(list(INPUT_DIR.glob('*.png')) + list(INPUT_DIR.glob('*.jpg')))
    print(f"Found {len(image_files)} images in {INPUT_DIR}")
    
    if len(image_files) == 0:
        print("❌ No images found! Aborting.")
        return

    # 2. Create Metadata
    profile = get_average_malignant_profile(TRAIN_META_PATH)
    
    metadata_rows = []
    for i, img_path in enumerate(image_files):
        row = profile.copy()
        row['isic_id'] = img_path.stem
        row['target'] = 1
        row['patient_id'] = f'synthetic_pat_{i:05d}' # Unique patient ID to prevent leakage grouping
        row['attribution'] = 'synthetic'
        row['copyright_license'] = 'cc-0'
        row['image_type'] = 'dermoscopic'
        metadata_rows.append(row)
        
    meta_df = pd.DataFrame(metadata_rows)
    meta_csv_path = OUTPUT_DIR / 'synthetic_malignant_metadata.csv'
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"✅ Saved metadata to {meta_csv_path}")
    
    # 3. Create HDF5 Files
    sizes = [224, 384]
    
    for size in sizes:
        h5_path = OUTPUT_DIR / f'synthetic_malignant_{size}.hdf5'
        print(f"\n📦 Packing {size}x{size} images to {h5_path}...")
        
        with h5py.File(h5_path, 'w') as f:
            for img_path in tqdm(image_files):
                img_id = img_path.stem
                
                # Open and Resize
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    # High-quality upscale
                    img_resized = img.resize((size, size), resample=Image.LANCZOS)
                    img_array = np.array(img_resized, dtype=np.uint8)
                
                # Save to HDF5
                f.create_dataset(img_id, data=img_array, compression='lzf')
                
        print(f"✅ Finished {size}x{size}")

    print("\n🎉 Phase 5 Complete!")
    print(f"   - Metadata: {meta_csv_path}")
    print(f"   - HDF5 (224): {OUTPUT_DIR / 'synthetic_malignant_224.hdf5'}")
    print(f"   - HDF5 (384): {OUTPUT_DIR / 'synthetic_malignant_384.hdf5'}")

if __name__ == '__main__':
    main()
