"""
Preprocess HDF5 images for faster training.

This script:
1. Loads compressed JPEG images from original HDF5 files
2. Decodes and resizes to 224x224
3. Saves as uncompressed numpy arrays for fast loading
4. Preserves all image IDs for metadata matching

Run once before training to speed up all future training runs.
"""

import h5py
import numpy as np
from PIL import Image
import io
from tqdm import tqdm
from pathlib import Path
import time

def preprocess_hdf5(input_path, output_path, target_size=(224, 224)):
    """
    Preprocess HDF5 file: decode JPEGs, resize, save as arrays.
    
    Args:
        input_path: Path to original HDF5 with compressed images
        output_path: Path to save preprocessed HDF5
        target_size: Target image size (height, width)
    """
    print(f"\n{'='*70}")
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    print(f"Target size: {target_size}")
    print(f"{'='*70}\n")
    
    # Open input and output files
    input_file = h5py.File(input_path, 'r')
    output_file = h5py.File(output_path, 'w')
    
    # Get all image IDs
    image_ids = list(input_file.keys())
    total_images = len(image_ids)
    
    print(f"Total images to process: {total_images:,}\n")
    
    # Storage statistics
    original_size = 0
    preprocessed_size = 0
    
    # Process each image
    start_time = time.time()
    
    for idx, img_id in enumerate(tqdm(image_ids, desc="Preprocessing")):
        # Load compressed JPEG bytes
        img_bytes = input_file[img_id][:]
        original_size += len(img_bytes)
        
        # Decode JPEG
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert('RGB')
        
        # Resize to target size
        img = img.resize(target_size, Image.BILINEAR)
        
        # Convert to numpy array (uint8 to save space)
        img_array = np.array(img, dtype=np.uint8)
        
        # Save to output HDF5 with light compression
        # lzf is fast compression, good balance between speed and size
        output_file.create_dataset(
            img_id, 
            data=img_array, 
            compression='lzf',
            shuffle=True  # Improves compression
        )
        
        preprocessed_size += img_array.nbytes
        
        # Progress update every 10,000 images
        if (idx + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            eta = (total_images - idx - 1) / rate
            print(f"\n  Processed: {idx+1:,}/{total_images:,} images")
            print(f"  Rate: {rate:.1f} images/sec")
            print(f"  ETA: {eta/60:.1f} minutes")
    
    # Close files
    input_file.close()
    output_file.close()
    
    # Final statistics
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average rate: {total_images/elapsed:.1f} images/sec")
    print(f"Original size: {original_size/1e9:.2f} GB")
    print(f"Preprocessed size: {preprocessed_size/1e9:.2f} GB (uncompressed)")
    print(f"Output file: {output_path}")
    print(f"{'='*70}\n")


def verify_preprocessed_file(file_path, num_samples=10):
    """
    Verify that preprocessed file is valid and show sample images.
    
    Args:
        file_path: Path to preprocessed HDF5 file
        num_samples: Number of samples to verify
    """
    print(f"\n{'='*70}")
    print(f"VERIFYING: {file_path}")
    print(f"{'='*70}\n")
    
    with h5py.File(file_path, 'r') as f:
        image_ids = list(f.keys())
        total = len(image_ids)
        
        print(f"✓ Total images: {total:,}")
        
        # Check random samples
        import random
        samples = random.sample(image_ids, min(num_samples, total))
        
        print(f"✓ Checking {len(samples)} random samples...\n")
        
        for img_id in samples:
            img_array = f[img_id][:]
            
            # Verify shape
            assert img_array.shape == (224, 224, 3), f"Wrong shape: {img_array.shape}"
            
            # Verify dtype
            assert img_array.dtype == np.uint8, f"Wrong dtype: {img_array.dtype}"
            
            # Verify value range
            assert img_array.min() >= 0 and img_array.max() <= 255
            
            print(f"  ✓ {img_id}: shape={img_array.shape}, dtype={img_array.dtype}, "
                  f"range=[{img_array.min()}, {img_array.max()}]")
        
        print(f"\n{'='*70}")
        print("VERIFICATION PASSED ✓")
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
        print(f"Testing with {sample_size} samples...\n")
        
        # Process sample
        start = time.time()
        for img_id in tqdm(image_ids[:sample_size], desc="Testing"):
            img_bytes = f[img_id][:]
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img = img.resize((224, 224), Image.BILINEAR)
            img_array = np.array(img, dtype=np.uint8)
        
        elapsed = time.time() - start
        rate = sample_size / elapsed
        
        # Estimate total time
        estimated_total = total_images / rate
        
        print(f"\n{'='*70}")
        print(f"ESTIMATION RESULTS")
        print(f"{'='*70}")
        print(f"Processing rate: {rate:.1f} images/sec")
        print(f"Estimated total time: {estimated_total/60:.1f} minutes ({estimated_total/3600:.1f} hours)")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    # Configuration
    data_dir = Path('data')
    
    input_train = data_dir / 'train-image.hdf5'
    input_test = data_dir / 'test-image.hdf5'
    
    output_train = data_dir / 'train-image-preprocessed.hdf5'
    output_test = data_dir / 'test-image-preprocessed.hdf5'
    
    target_size = (224, 224)
    
    # Check if input files exist
    if not input_train.exists():
        print(f"❌ Error: {input_train} not found!")
        exit(1)
    
    if not input_test.exists():
        print(f"❌ Error: {input_test} not found!")
        exit(1)
    
    print("\n" + "="*70)
    print("HDF5 IMAGE PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Estimate time (optional - comment out to skip)
    print("\n[STEP 1] Estimating preprocessing time...")
    estimate_preprocessing_time(input_train, sample_size=1000)
    
    user_input = input("Continue with preprocessing? (yes/no): ").strip().lower()
    if user_input != 'yes':
        print("Preprocessing cancelled.")
        exit(0)
    
    # Step 2: Preprocess training data
    print("\n[STEP 2] Preprocessing training images...")
    preprocess_hdf5(input_train, output_train, target_size)
    
    # Step 3: Preprocess test data
    print("\n[STEP 3] Preprocessing test images...")
    preprocess_hdf5(input_test, output_test, target_size)
    
    # Step 4: Verify outputs
    print("\n[STEP 4] Verifying preprocessed files...")
    verify_preprocessed_file(output_train, num_samples=10)
    verify_preprocessed_file(output_test, num_samples=10)
    
    # Final summary
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE COMPLETE! ✓")
    print("="*70)
    print("\nPreprocessed files created:")
    print(f"  • Training: {output_train}")
    print(f"  • Test: {output_test}")
    print("\nYou can now use these files for much faster training!")
    print("Update your Dataset class to load from these preprocessed files.")
    print("="*70 + "\n")
