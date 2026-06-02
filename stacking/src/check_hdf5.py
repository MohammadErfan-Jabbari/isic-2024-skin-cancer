import h5py
import io
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd

DATA_DIR = Path('./data')
HDF5_PATH = DATA_DIR / 'train-image-384.hdf5'
META_PATH = DATA_DIR / 'new-train-metadata.csv'

def check_hdf5():
    print(f"Checking {HDF5_PATH}...")
    
    # Load Metadata to get IDs
    df = pd.read_csv(META_PATH, usecols=['isic_id'])
    isic_ids = df['isic_id'].values
    
    print(f"Total IDs in Metadata: {len(isic_ids)}")
    
    with h5py.File(HDF5_PATH, 'r') as f:
        print(f"Total IDs in HDF5: {len(f.keys())}")
        
        # Check first 1000 and random 1000
        sample_ids = isic_ids[:1000].tolist() + np.random.choice(isic_ids, 1000).tolist()
        
        failures = 0
        for isic_id in tqdm(sample_ids):
            if isic_id not in f:
                print(f"Missing: {isic_id}")
                failures += 1
                continue
                
            try:
                img_bytes = f[isic_id][()]
                Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                # print(f"Corrupt: {isic_id} - {e}")
                failures += 1
                
        print(f"\nScanned {len(sample_ids)} images.")
        print(f"Failures: {failures}")
        print(f"Failure Rate: {failures/len(sample_ids)*100:.2f}%")

import numpy as np
if __name__ == "__main__":
    check_hdf5()
