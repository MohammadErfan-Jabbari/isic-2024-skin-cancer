import torch
from torch.utils.data import Dataset
import h5py
import pandas as pd
import numpy as np
import io
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

class ISICDataset(Dataset):
    def __init__(self, hdf5_path, metadata, transform=None, is_test=False):
        self.hdf5_path = hdf5_path
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        
        # Open HDF5 file (lazy loading)
        self.fp = None
        
    def _init_file(self):
        if self.fp is None:
            self.fp = h5py.File(self.hdf5_path, 'r')
            
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        self._init_file()
        
        row = self.metadata.iloc[idx]
        isic_id = row['isic_id']
        
        # Load Image
        try:
            # HDF5 structure: isic_id -> image_data
            data = self.fp[isic_id][()]
            
            # Check if it's already a numpy array (Preprocessed HDF5)
            if isinstance(data, np.ndarray) and data.ndim == 3:
                image = data
            else:
                # It's bytes (Original HDF5)
                image = Image.open(io.BytesIO(data))
                image = np.array(image)
        except Exception as e:
            print(f"Error loading {isic_id}: {e}")
            # Return black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            
        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default to tensor if no transform
            image = ToTensorV2()(image=image)['image']
            
        # Return
        if self.is_test:
            return image, row['isic_id']
        else:
            target = torch.tensor(row['target'], dtype=torch.float32)
            return image, target

# Verification Block
if __name__ == "__main__":
    print("Verifying ISICDataset...")
    
    # Config
    DATA_DIR = Path('./data')
    HDF5_PATH = DATA_DIR / 'train-image.hdf5'
    META_PATH = DATA_DIR / 'new-train-metadata.csv'
    
    # Load small metadata
    df = pd.read_csv(META_PATH, nrows=100)
    
    # Transforms
    transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])
    
    # Init Dataset
    ds = ISICDataset(HDF5_PATH, df, transform=transforms)
    print(f"Dataset Length: {len(ds)}")
    
    # Check Item
    img, target = ds[0]
    print(f"Image Shape: {img.shape}")
    print(f"Target: {target}")
    print(f"Target Type: {target.dtype}")
    
    # Assertions
    assert img.shape == (3, 224, 224), "Image shape mismatch!"
    assert isinstance(target, torch.Tensor), "Target is not a Tensor!"
    
    print("✅ Dataset Verification Passed!")
