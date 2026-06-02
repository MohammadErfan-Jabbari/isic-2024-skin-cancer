
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
MODEL_DIR = Path('DeepLearning/Kaggle/results/v2s_features_20251110_155122')
DATA_DIR = Path('DeepLearning/Kaggle/data')
BATCH_SIZE = 128
GPU_ID = 0

# ===========================
# FEATURE ENGINEERING (Copied from 10_1)
# ===========================

def engineer_features(df):
    df = df.copy()
    
    # AGE FEATURES
    df['age_group'] = pd.cut(df['age_approx'], 
                             bins=[0, 30, 50, 70, 100],
                             labels=['young', 'middle', 'senior', 'elderly'])
    df['age_risk'] = (df['age_approx'] > 50).astype(int)
    
    # SIZE FEATURES
    df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    df['size_category'] = pd.cut(df['lesion_size_mm'],
                                 bins=[0, 6, 10, 20, 100],
                                 labels=['small', 'medium', 'large', 'very_large'])
    df['large_lesion'] = (df['lesion_size_mm'] > 6).astype(int)
    
    # SHAPE FEATURES
    df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
    df['eccentricity'] = df['tbp_lv_minorAxisMM'] / (df['tbp_lv_areaMM2']**0.5 + 1e-6)
    
    # COLOR FEATURES
    df['color_variance'] = np.sqrt(
        df['tbp_lv_deltaB']**2 + 
        df['tbp_lv_radial_color_std_max']**2 +
        df['tbp_lv_color_std_mean']**2
    )
    df['color_uniformity'] = 1 / (df['tbp_lv_norm_color'] + 1e-6)
    df['darkness_score'] = df['tbp_lv_B'] / (df['tbp_lv_H'] + 1e-6)
    
    # ANATOMICAL FEATURES
    high_risk_sites = ['torso', 'upper extremity', 'posterior torso', 'anterior torso']
    df['high_risk_site'] = df['anatom_site_general'].isin(high_risk_sites).astype(int)
    
    site_risk_map = {
        'torso': 3, 'posterior torso': 3, 'anterior torso': 3,
        'upper extremity': 2, 'lower extremity': 2, 'head/neck': 2,
        'palms/soles': 1, 'oral/genital': 1
    }
    df['site_risk_score'] = df['anatom_site_general'].map(site_risk_map).fillna(0)
    
    # INTERACTION FEATURES
    df['age_size_risk'] = df['age_approx'] * df['lesion_size_mm']
    df['age_site_risk'] = df['age_approx'] * df['high_risk_site']
    df['color_size_risk'] = df['color_variance'] * df['lesion_size_mm']
    
    # ASYMMETRY SCORE
    df['asymmetry_score'] = (
        df['tbp_lv_norm_color'] +
        df['tbp_lv_radial_color_std_max'] +
        (1 / (df['shape_regularity'] + 1e-6))
    ) / 3
    
    # LOG TRANSFORMS
    df['log_area'] = np.log1p(df['tbp_lv_areaMM2'])
    df['log_perimeter'] = np.log1p(df['tbp_lv_perimeterMM'])
    df['log_size'] = np.log1p(df['lesion_size_mm'])
    
    # RATIOS
    df['h_to_b_ratio'] = df['tbp_lv_H'] / (df['tbp_lv_B'] + 1e-6)
    df['a_to_b_ratio'] = df['tbp_lv_A'] / (df['tbp_lv_B'] + 1e-6)
    
    return df

def preprocess_metadata_with_features(df, is_train=True, scaler=None, encoders=None):
    # 1. Engineer features FIRST
    df = engineer_features(df)
    
    # 2. Define all numerical features (original + engineered)
    NUMERICAL_FEATURES = [
        'tbp_lv_H', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM',
        'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_Hext',
        'clin_size_long_diam_mm', 'tbp_lv_radial_color_std_max',
        'tbp_lv_B', 'tbp_lv_color_std_mean', 'tbp_lv_Aext',
        'tbp_lv_stdLExt', 'tbp_lv_norm_color', 'tbp_lv_A',
        'age_approx',
        'lesion_size_mm', 'shape_regularity', 'eccentricity',
        'color_variance', 'color_uniformity', 'darkness_score',
        'site_risk_score', 'age_size_risk', 'age_site_risk',
        'color_size_risk', 'asymmetry_score',
        'log_area', 'log_perimeter', 'log_size',
        'h_to_b_ratio', 'a_to_b_ratio'
    ]
    
    CATEGORICAL_FEATURES = [
        'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple',
        'age_group', 'size_category', 'age_risk', 'large_lesion', 'high_risk_site'
    ]
    
    # 3. Handle missing values
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if is_train else 0)
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing')
    
    # 4. Scale numerical features
    if is_train:
        scaler = StandardScaler()
        df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])
    else:
        df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])
    
    # 5. One-hot encode categorical features
    if is_train:
        encoders = {}
        encoded_dfs = []
        for col in CATEGORICAL_FEATURES:
            encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
            encoders[col] = encoded.columns.tolist()
            encoded_dfs.append(encoded)
        result_df = pd.concat([df[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
    else:
        encoded_dfs = []
        for col in CATEGORICAL_FEATURES:
            encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
            for train_col in encoders[col]:
                if train_col not in encoded.columns:
                    encoded[train_col] = 0
            encoded = encoded[encoders[col]]
            encoded_dfs.append(encoded)
        result_df = pd.concat([df[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
    
    return result_df, scaler, encoders

# ===========================
# MODEL CLASSES
# ===========================

class MetadataProcessor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        return self.fc(x)

class EfficientNetV2Hybrid(nn.Module):
    def __init__(self, metadata_dim):
        super().__init__()
        self.efficientnet = efficientnet_v2_s(weights=None)
        self.efficientnet.classifier = nn.Identity()
        self.metadata_processor = MetadataProcessor(metadata_dim)
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, metadata):
        img_features = self.efficientnet(image)
        meta_features = self.metadata_processor(metadata)
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(combined)

class HybridDataset(Dataset):
    def __init__(self, hdf5_path, metadata_df, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hdf5_file = None
        
        with h5py.File(hdf5_path, 'r') as f:
            available_ids = set(f.keys())
        
        self.metadata = metadata_df[
            metadata_df['isic_id'].isin(available_ids)
        ].reset_index(drop=True)
        
        feature_cols = [col for col in self.metadata.columns 
                       if col not in ['isic_id', 'target']]
        self.metadata_features = self.metadata[feature_cols].values.astype(np.float32)
        
    def _ensure_hdf5_open(self):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        self._ensure_hdf5_open()
        row = self.metadata.iloc[idx]
        image_id = row['isic_id']
        img_array = self.hdf5_file[image_id][:]
        image = Image.fromarray(img_array)
        if self.transform:
            image = self.transform(image)
        metadata = torch.tensor(self.metadata_features[idx], dtype=torch.float32)
        label = row['target']
        return image, metadata, label

# ===========================
# MAIN ANALYSIS
# ===========================

def main():
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Preprocessors
    print(f"Loading preprocessors from {MODEL_DIR}...")
    with open(MODEL_DIR / 'preprocessors.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
    scaler = preprocessors['scaler']
    encoders = preprocessors['encoders']

    # 2. Load Metadata
    print("Loading metadata...")
    df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
    
    # 3. Select Samples
    # Take all positives and a random sample of negatives (e.g., 2000)
    pos_df = df[df['target'] == 1]
    neg_df = df[df['target'] == 0].sample(n=2000, random_state=42)
    
    print(f"Selected {len(pos_df)} positive samples and {len(neg_df)} negative samples.")
    
    sample_df = pd.concat([pos_df, neg_df]).reset_index(drop=True)
    
    # 4. Preprocess Metadata
    print("Preprocessing metadata...")
    processed_df, _, _ = preprocess_metadata_with_features(
        sample_df, is_train=False, scaler=scaler, encoders=encoders
    )
    processed_df['isic_id'] = sample_df['isic_id'].values
    processed_df['target'] = sample_df['target'].values
    
    metadata_dim = len(processed_df.columns) - 2
    print(f"Metadata dimension: {metadata_dim}")

    # 5. Load Model
    print("Loading model...")
    model = EfficientNetV2Hybrid(metadata_dim=metadata_dim).to(device)
    checkpoint = torch.load(MODEL_DIR / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 6. Create Dataset & Loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Ensure size matches model expectation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = HybridDataset(
        DATA_DIR / 'train-image-preprocessed.hdf5',
        processed_df,
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 7. Run Inference
    print("Running inference...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, metadata, labels in tqdm(loader):
            images = images.to(device)
            metadata = metadata.to(device)
            outputs = model(images, metadata)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 8. Analyze Results
    pos_probs = all_probs[all_labels == 1]
    neg_probs = all_probs[all_labels == 0]
    
    print("\n" + "="*40)
    print("CONFIDENCE ANALYSIS")
    print("="*40)
    
    def print_stats(name, data):
        print(f"\n{name} Samples (n={len(data)}):")
        print(f"  Mean:   {np.mean(data):.4f}")
        print(f"  Median: {np.median(data):.4f}")
        print(f"  Min:    {np.min(data):.4f}")
        print(f"  Max:    {np.max(data):.4f}")
        print(f"  25th %: {np.percentile(data, 25):.4f}")
        print(f"  75th %: {np.percentile(data, 75):.4f}")
        print(f"  90th %: {np.percentile(data, 90):.4f}")
        print(f"  99th %: {np.percentile(data, 99):.4f}")

    print_stats("POSITIVE (Malignant)", pos_probs)
    print_stats("NEGATIVE (Benign)", neg_probs)
    
    print("\n" + "="*40)
    print("INTERPRETATION")
    print("="*40)
    
    # Check where 0.29 falls
    pos_percentile = (pos_probs < 0.29).mean() * 100
    neg_percentile = (neg_probs < 0.29).mean() * 100
    
    print(f"A confidence of 0.29 is:")
    print(f"  - Higher than {pos_percentile:.1f}% of POSITIVE samples")
    print(f"  - Higher than {neg_percentile:.1f}% of NEGATIVE samples")

if __name__ == '__main__':
    main()
