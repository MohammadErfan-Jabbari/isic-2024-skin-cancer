"""
preprocess_hdf5_v2.py
---------------------
Upscales ISIC 2024 HDF5 datasets to 384x384 using High-Quality Lanczos Resampling.

This is an alternative preprocessing pipeline for higher-resolution training.
Original images (67-139px) → 384x384 using LANCZOS (preserves edges better than BILINEAR)

Usage: uv run python preprocess_hdf5_v2.py
"""
import h5py
import numpy as np
from PIL import Image
import io
from tqdm import tqdm
from pathlib import Path
import os
import time

# ================= CONFIGURATION =================
DATA_DIR = Path('data')
INPUT_TRAIN = DATA_DIR / 'train-image.hdf5'
INPUT_TEST = DATA_DIR / 'test-image.hdf5'

# New high-res files
OUTPUT_TRAIN = DATA_DIR / 'train-image-384.hdf5'
OUTPUT_TEST = DATA_DIR / 'test-image-384.hdf5'

# Target size for EVA-02 / EfficientNetV2-M
TARGET_SIZE = (384, 384)
# =================================================


def preprocess_dataset(input_path, output_path, target_size=(384, 384)):
    """
    Preprocess HDF5 file: decode JPEGs, upsample to 384x384, save with compression.
    
    Args:
        input_path: Path to original HDF5 with compressed JPEG images
        output_path: Path to save preprocessed HDF5 with upscaled images
        target_size: Target image size (height, width)
    """
    if not input_path.exists():
        print(f"⚠️  Input file not found: {input_path}")
        return

    print(f"\n{'='*70}")
    print(f"Processing: {input_path.name}")
    print(f"Output: {output_path.name}")
    print(f"Target Resolution: {target_size}")
    print(f"Resampling Method: LANCZOS (High Quality)")
    print(f"{'='*70}\n")
    
    # Open files
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # Get all image IDs
        image_ids = list(f_in.keys())
        total = len(image_ids)
        
        print(f"Total images: {total:,}\n")
        
        # Storage statistics
        original_size = 0
        upscaled_size = 0
        
        # Process each image
        start_time = time.time()
        
        for img_id in tqdm(image_ids, desc="Upscaling to 384x384"):
            # 1. Load compressed JPEG bytes
            img_bytes = f_in[img_id][()]
            original_size += len(img_bytes)
            
            # 2. Decode to PIL Image
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # 3. High-Quality Upscaling (LANCZOS)
            # LANCZOS preserves edge sharpness better than BILINEAR for upsampling
            img_resized = img.resize(target_size, resample=Image.Resampling.LANCZOS)
            
            # 4. Convert to numpy array (uint8 to save space)
            img_array = np.array(img_resized, dtype=np.uint8)
            
            # 5. Save with compression (LZF is fast and decent compression)
            f_out.create_dataset(
                img_id,
                data=img_array,
                compression='lzf',
                shuffle=True  # Improves compression ratio
            )
            
            upscaled_size += img_array.nbytes
        
        elapsed = time.time() - start_time
        
        # Final statistics
        print(f"\n{'='*70}")
        print(f"PREPROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"Average rate: {total/elapsed:.1f} images/sec")
        print(f"Original size: {original_size/1e9:.2f} GB (compressed JPEG bytes)")
        print(f"Upscaled size: {upscaled_size/1e9:.2f} GB (uncompressed arrays)")
        print(f"Output file: {output_path}")
        print(f"Output file size: {os.path.getsize(output_path) / 1e9:.2f} GB (with LZF compression)")
        print(f"{'='*70}\n")


def verify_preprocessed_file(file_path, num_samples=10):
    """
    Verify that preprocessed file is valid and show sample verification.
    
    Args:
        file_path: Path to preprocessed HDF5 file
        num_samples: Number of samples to verify
    """
    print(f"\n{'='*70}")
    print(f"VERIFYING: {file_path.name}")
    print(f"{'='*70}\n")
    
    with h5py.File(file_path, 'r') as f:
        image_ids = list(f.keys())
        total = len(image_ids)
        
        print(f"✓ Total images: {total:,}")
        
        # Check random samples
        import random
        samples = random.sample(image_ids, min(num_samples, total))
        
        print(f"✓ Checking {len(samples)} random samples...\n")
        
        all_valid = True
        for img_id in samples:
            img_array = f[img_id][:]
            
            # Verify shape
            if img_array.shape != (384, 384, 3):
                print(f"  ✗ {img_id}: Wrong shape: {img_array.shape}")
                all_valid = False
                continue
            
            # Verify dtype
            if img_array.dtype != np.uint8:
                print(f"  ✗ {img_id}: Wrong dtype: {img_array.dtype}")
                all_valid = False
                continue
            
            # Verify value range
            if not (img_array.min() >= 0 and img_array.max() <= 255):
                print(f"  ✗ {img_id}: Invalid value range: [{img_array.min()}, {img_array.max()}]")
                all_valid = False
                continue
            
            print(f"  ✓ {img_id}: shape={img_array.shape}, dtype={img_array.dtype}, "
                  f"range=[{img_array.min()}, {img_array.max()}]")
        
        print(f"\n{'='*70}")
        if all_valid:
            print("VERIFICATION PASSED ✓")
        else:
            print("VERIFICATION FAILED ✗")
        print(f"{'='*70}\n")


def estimate_preprocessing_time(input_path, sample_size=1000):
    """
    Estimate total preprocessing time by timing a sample.
    
    Args:
        input_path: Path to original HDF5 file
        sample_size: Number of images to use for estimation
    """
    print(f"\n{'='*70}")
    print(f"ESTIMATING PREPROCESSING TIME")
    print(f"{'='*70}\n")
    
    with h5py.File(input_path, 'r') as f:
        image_ids = list(f.keys())
        total_images = len(image_ids)
        
        print(f"Total images: {total_images:,}")
        print(f"Testing with {sample_size} samples (384x384 LANCZOS)...\n")
        
        # Process sample
        start = time.time()
        for img_id in tqdm(image_ids[:sample_size], desc="Testing"):
            img_bytes = f[img_id][()]
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img = img.resize((384, 384), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.uint8)
        
        elapsed = time.time() - start
        rate = sample_size / elapsed
        
        # Estimate total time
        estimated_total = total_images / rate
        
        print(f"\n{'='*70}")
        print(f"ESTIMATION RESULTS")
        print(f"{'='*70}")
        print(f"Processing rate: {rate:.1f} images/sec")
        print(f"Estimated total time: {estimated_total/60:.1f} minutes ({estimated_total/3600:.2f} hours)")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HDF5 IMAGE PREPROCESSING PIPELINE (V2 - 384x384 HIGH-RES)")
    print("="*70)
    
    # Check if input files exist
    if not INPUT_TRAIN.exists():
        print(f"❌ Error: {INPUT_TRAIN} not found!")
        exit(1)
    
    if not INPUT_TEST.exists():
        print(f"❌ Error: {INPUT_TEST} not found!")
        exit(1)
    
    # Step 1: Estimate time
    print("\n[STEP 1] Estimating preprocessing time...")
    estimate_preprocessing_time(INPUT_TRAIN, sample_size=1000)
    
    user_input = input("Continue with preprocessing? (yes/no): ").strip().lower()
    if user_input != 'yes':
        print("Preprocessing cancelled.")
        exit(0)
    
    # Step 2: Preprocess training data
    print("\n[STEP 2] Preprocessing training images to 384x384...")
    preprocess_dataset(INPUT_TRAIN, OUTPUT_TRAIN, target_size=TARGET_SIZE)
    
    # Step 3: Preprocess test data
    print("\n[STEP 3] Preprocessing test images to 384x384...")
    preprocess_dataset(INPUT_TEST, OUTPUT_TEST, target_size=TARGET_SIZE)
    
    # Step 4: Verify outputs
    print("\n[STEP 4] Verifying preprocessed files...")
    verify_preprocessed_file(OUTPUT_TRAIN, num_samples=10)
    verify_preprocessed_file(OUTPUT_TEST, num_samples=10)
    
    # Final summary
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE COMPLETE! ✓")
    print("="*70)
    print("\nHigh-resolution preprocessed files created:")
    print(f"  • Training: {OUTPUT_TRAIN}")
    print(f"  • Test: {OUTPUT_TEST}")
    print("\nNext steps:")
    print("1. Update your Dataset class to load from *-384.hdf5 files")
    print("2. Adjust model input_size and batch_size for 384x384 images")
    print("3. Consider using larger models (EfficientNetV2-M, EVA-02, ViT-B)")
    print("4. Monitor GPU memory (384x384 uses ~2.8x more than 224x224)")
    print("="*70 + "\n")
