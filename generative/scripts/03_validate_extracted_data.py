#!/usr/bin/env python3
"""
Step 1C: Validation Script for Extracted Malignant Samples
Validates all 343 extracted images and metadata for Phase 2 (Stable Diffusion training).
"""

import json
import sys
from pathlib import Path
from PIL import Image
from collections import defaultdict

# Configuration
DATA_DIR = Path("DeepLearning/Kaggle/generative/data/train_images_128")
METADATA_FILE = DATA_DIR / "metadata.jsonl"
EXTRACTION_METADATA_FILE = DATA_DIR / "extraction_metadata.json"
TARGET_SIZE = (128, 128)
TARGET_MODE = "RGB"
EXPECTED_IMAGE_COUNT = 343


def validate_directory_structure():
    """Check if data directory and key files exist."""
    print("\n" + "="*60)
    print("1️⃣  VALIDATING DIRECTORY STRUCTURE")
    print("="*60)
    
    checks = {
        "Data directory exists": DATA_DIR.exists(),
        "Metadata JSONL exists": METADATA_FILE.exists(),
        "Extraction metadata exists": EXTRACTION_METADATA_FILE.exists(),
    }
    
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")
        if not result:
            return False
    
    return True


def validate_extraction_metadata():
    """Validate extraction_metadata.json structure."""
    print("\n" + "="*60)
    print("2️⃣  VALIDATING EXTRACTION METADATA")
    print("="*60)
    
    try:
        with open(EXTRACTION_METADATA_FILE) as f:
            meta = json.load(f)
        
        # Check required keys
        required_keys = ["total_extracted", "target_size", "resample_method", "images"]
        for key in required_keys:
            status = "✅" if key in meta else "❌"
            print(f"{status} Key '{key}' present: {key in meta}")
            if key not in meta:
                return False
        
        # Validate values
        print(f"\n📊 Metadata Summary:")
        print(f"   Total extracted: {meta['total_extracted']}")
        print(f"   Target size: {meta['target_size']}")
        print(f"   Resample method: {meta['resample_method']}")
        print(f"   Image count in metadata: {len(meta['images'])}")
        
        if meta['total_extracted'] != EXPECTED_IMAGE_COUNT:
            print(f"❌ Expected {EXPECTED_IMAGE_COUNT} images, got {meta['total_extracted']}")
            return False
        
        if meta['target_size'] != list(TARGET_SIZE):
            print(f"❌ Expected target size {TARGET_SIZE}, got {meta['target_size']}")
            return False
        
        print("✅ All extraction metadata valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def validate_image_files():
    """Validate all PNG image files."""
    print("\n" + "="*60)
    print("3️⃣  VALIDATING IMAGE FILES")
    print("="*60)
    
    image_files = list(DATA_DIR.glob("ISIC_*.png"))
    print(f"Found {len(image_files)} PNG files")
    
    if len(image_files) != EXPECTED_IMAGE_COUNT:
        print(f"❌ Expected {EXPECTED_IMAGE_COUNT} images, got {len(image_files)}")
        return False
    
    corrupted = []
    size_mismatch = []
    mode_mismatch = []
    file_sizes = []
    
    print("\nValidating each image...")
    for i, img_path in enumerate(sorted(image_files)):
        try:
            img = Image.open(img_path)
            file_size = img_path.stat().st_size
            file_sizes.append(file_size)
            
            # Check dimensions
            if img.size != TARGET_SIZE:
                size_mismatch.append((img_path.name, img.size))
            
            # Check color mode
            if img.mode != TARGET_MODE:
                mode_mismatch.append((img_path.name, img.mode))
            
            # Verify image is loadable
            img.verify()
            
        except Exception as e:
            corrupted.append((img_path.name, str(e)))
    
    # Report results
    print(f"\n✅ Successfully loaded: {len(image_files) - len(corrupted)}/{len(image_files)}")
    
    if corrupted:
        print(f"\n❌ Corrupted images ({len(corrupted)}):")
        for name, error in corrupted:
            print(f"   - {name}: {error}")
        return False
    
    if size_mismatch:
        print(f"\n❌ Size mismatches ({len(size_mismatch)}):")
        for name, size in size_mismatch:
            print(f"   - {name}: {size} (expected {TARGET_SIZE})")
        return False
    
    if mode_mismatch:
        print(f"\n❌ Mode mismatches ({len(mode_mismatch)}):")
        for name, mode in mode_mismatch:
            print(f"   - {name}: {mode} (expected {TARGET_MODE})")
        return False
    
    # File size statistics
    print(f"\n📊 File Size Statistics:")
    print(f"   Min: {min(file_sizes) / 1024:.1f} KB")
    print(f"   Max: {max(file_sizes) / 1024:.1f} KB")
    print(f"   Mean: {sum(file_sizes) / len(file_sizes) / 1024:.1f} KB")
    print(f"   Total: {sum(file_sizes) / 1024 / 1024:.1f} MB")
    
    print("\n✅ All images valid")
    return True


def validate_metadata_jsonl():
    """Validate metadata.jsonl file."""
    print("\n" + "="*60)
    print("4️⃣  VALIDATING METADATA JSONL")
    print("="*60)
    
    required_fields = {"file_name", "text", "isic_id"}
    parse_errors = []
    missing_fields = []
    line_count = 0
    
    print("Parsing JSONL file...")
    try:
        with open(METADATA_FILE) as f:
            for line_num, line in enumerate(f, 1):
                line_count = line_num
                
                try:
                    entry = json.loads(line.strip())
                    
                    # Check required fields
                    missing = required_fields - set(entry.keys())
                    if missing:
                        missing_fields.append((line_num, missing))
                    
                except json.JSONDecodeError as e:
                    parse_errors.append((line_num, str(e)))
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Report results
    print(f"✅ Total lines: {line_count}")
    
    if parse_errors:
        print(f"\n❌ JSON parse errors ({len(parse_errors)}):")
        for line_num, error in parse_errors[:5]:  # Show first 5
            print(f"   - Line {line_num}: {error}")
        return False
    
    if missing_fields:
        print(f"\n❌ Missing fields ({len(missing_fields)}):")
        for line_num, missing in missing_fields[:5]:  # Show first 5
            print(f"   - Line {line_num}: missing {missing}")
        return False
    
    if line_count != EXPECTED_IMAGE_COUNT:
        print(f"❌ Expected {EXPECTED_IMAGE_COUNT} lines, got {line_count}")
        return False
    
    print("✅ All JSONL entries valid")
    return True


def validate_file_consistency():
    """Verify 1:1 mapping between images and metadata entries."""
    print("\n" + "="*60)
    print("5️⃣  VALIDATING FILE CONSISTENCY")
    print("="*60)
    
    # Get image files
    image_files = {img.stem for img in DATA_DIR.glob("ISIC_*.png")}
    
    # Get metadata entries
    metadata_ids = set()
    try:
        with open(METADATA_FILE) as f:
            for line in f:
                entry = json.loads(line.strip())
                file_name = entry.get("file_name", "")
                isic_id = file_name.replace(".png", "")
                metadata_ids.add(isic_id)
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return False
    
    # Check consistency
    orphaned_images = image_files - metadata_ids
    missing_images = metadata_ids - image_files
    
    if orphaned_images:
        print(f"❌ Orphaned images (in filesystem but not in metadata):")
        for img_id in list(orphaned_images)[:5]:
            print(f"   - {img_id}")
        if len(orphaned_images) > 5:
            print(f"   ... and {len(orphaned_images) - 5} more")
        return False
    
    if missing_images:
        print(f"❌ Missing images (in metadata but not in filesystem):")
        for img_id in list(missing_images)[:5]:
            print(f"   - {img_id}")
        if len(missing_images) > 5:
            print(f"   ... and {len(missing_images) - 5} more")
        return False
    
    print(f"✅ Perfect 1:1 mapping: {len(image_files)} images ↔ {len(metadata_ids)} metadata entries")
    return True


def generate_final_report():
    """Generate final validation report."""
    print("\n" + "="*60)
    print("📋 FINAL VALIDATION REPORT")
    print("="*60)
    
    checks = [
        ("Directory Structure", validate_directory_structure),
        ("Extraction Metadata", validate_extraction_metadata),
        ("Image Files", validate_image_files),
        ("Metadata JSONL", validate_metadata_jsonl),
        ("File Consistency", validate_file_consistency),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n❌ Unexpected error in {check_name}: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("🎯 SUMMARY")
    print("="*60)
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if all(results.values()):
        print("\n" + "="*60)
        print("🚀 ALL CHECKS PASSED!")
        print("="*60)
        print("\n✨ Step 1B extraction is complete and verified.")
        print("📦 Ready for Phase 2: Stable Diffusion fine-tuning")
        print(f"   - 343 malignant images at 128×128")
        print(f"   - Complete metadata with captions")
        print(f"   - All files validated and consistent")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("❌ VALIDATION FAILED")
        print("="*60)
        print(f"\n⚠️  {total - passed} check(s) failed. Please review above.")
        return False


if __name__ == "__main__":
    success = generate_final_report()
    sys.exit(0 if success else 1)
