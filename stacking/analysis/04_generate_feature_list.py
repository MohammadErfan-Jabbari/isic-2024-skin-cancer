import pandas as pd
from pathlib import Path

DATA_DIR = Path('./data')
OUTPUT_FILE = Path('./last_run/feature_list.txt')

def generate_list():
    print("Loading metadata...")
    train_df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', nrows=1)
    test_df = pd.read_csv(DATA_DIR / 'students-test-metadata.csv', nrows=1)
    
    # 1. Intersection
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols.intersection(test_cols)
    print(f"Common Columns: {len(common_cols)}")
    
    # 2. Exclude ID/Target/Leakage
    exclude = {
        'isic_id', 'patient_id', 'target', 'image_type', 'tbp_tile_type', 
        'attribution', 'copyright_license', 'lesion_id',
        'mel_thick_mm', 'mel_mitotic_index', 'tbp_lv_dnn_lesion_confidence'
    }
    # Also exclude iddx_* (Diagnosis Leakage)
    exclude.update([c for c in train_cols if c.startswith('iddx_')])
    
    final_features = [c for c in common_cols if c not in exclude]
    final_features.sort()
    
    print(f"Final Safe Features: {len(final_features)}")
    
    with open(OUTPUT_FILE, 'w') as f:
        for feat in final_features:
            f.write(f"{feat}\n")
            
    print(f"Saved to {OUTPUT_FILE}")
    print("List:")
    print(final_features)

if __name__ == "__main__":
    generate_list()
