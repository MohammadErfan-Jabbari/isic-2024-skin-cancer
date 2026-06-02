
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse
import shutil
import warnings

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================

# Features expected by the model (must match training script 10_1)
NUMERICAL_FEATURES = [
    'tbp_lv_H', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM',
    'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 'tbp_lv_Hext',
    'clin_size_long_diam_mm', 'tbp_lv_radial_color_std_max',
    'tbp_lv_B', 'tbp_lv_color_std_mean', 'tbp_lv_Aext',
    'tbp_lv_stdLExt', 'tbp_lv_norm_color', 'tbp_lv_A',
    'age_approx',
    # Engineered
    'lesion_size_mm', 'shape_regularity', 'eccentricity',
    'color_variance', 'color_uniformity', 'darkness_score',
    'site_risk_score', 'age_size_risk', 'age_site_risk',
    'color_size_risk', 'asymmetry_score',
    'log_area', 'log_perimeter', 'log_size',
    'h_to_b_ratio', 'a_to_b_ratio'
]

CATEGORICAL_FEATURES = [
    'sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location_simple',
    # Engineered
    'age_group', 'size_category', 'age_risk', 'large_lesion', 'high_risk_site'
]

# ===========================
# MODEL ARCHITECTURE (Matching 10_1)
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
        # Note: weights=None because we load state_dict
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

# ===========================
# FEATURE ENGINEERING
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

def get_average_malignant_profile(train_meta_path):
    """Calculates the average metadata profile for malignant cases."""
    print("Calculating average malignant profile...")
    df = pd.read_csv(train_meta_path, low_memory=False)
    
    # Filter for malignant cases
    mal_df = df[df['target'] == 1].copy()
    print(f"  Found {len(mal_df)} malignant cases.")
    
    # Engineer features first to get all columns
    mal_df = engineer_features(mal_df)
    
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

def prepare_dummy_metadata(profile, num_samples, scaler, encoders):
    """Creates a DataFrame of dummy metadata and preprocesses it."""
    # Create DataFrame with repeated profile
    data = {col: [val] * num_samples for col, val in profile.items()}
    df = pd.DataFrame(data)
    
    # Scale numerical
    df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])
    
    # Encode categorical
    encoded_dfs = []
    for col in CATEGORICAL_FEATURES:
        encoded = pd.get_dummies(df[col], prefix=col, dtype=float)
        # Align with encoder columns
        for train_col in encoders[col]:
            if train_col not in encoded.columns:
                encoded[train_col] = 0
        encoded = encoded[encoders[col]]
        encoded_dfs.append(encoded)
    
    result_df = pd.concat([df[NUMERICAL_FEATURES]] + encoded_dfs, axis=1)
    return torch.tensor(result_df.values, dtype=torch.float32)

# ===========================
# MAIN
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True, help='Path to model directory containing best_model.pth and preprocessors.pkl')
    parser.add_argument('--input-dir', type=str, default='generative/data/synthetic_images_128px', help='Directory with synthetic images')
    parser.add_argument('--output-dir', type=str, default='generative/data/synthetic_malignant_filtered', help='Output directory for filtered images')
    parser.add_argument('--threshold', type=float, default=0.15, help='Probability threshold (0.15 is avg for real malignant)')
    parser.add_argument('--topk', type=int, default=6000, help='Number of images to keep')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Preprocessors
    model_dir = Path(args.model_dir)
    with open(model_dir / 'preprocessors.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
    scaler = preprocessors['scaler']
    encoders = preprocessors['encoders']
    
    # 2. Get Average Malignant Profile
    profile = get_average_malignant_profile('data/new-train-metadata.csv')
    
    # 3. List Images
    input_dir = Path(args.input_dir)
    image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))
    print(f"Found {len(image_files)} synthetic images.")
    
    # 4. Prepare Dummy Metadata (Batch processing would be better for RAM, but 10k is small)
    print("Preparing dummy metadata...")
    dummy_meta_tensor = prepare_dummy_metadata(profile, len(image_files), scaler, encoders)
    metadata_dim = dummy_meta_tensor.shape[1]
    print(f"Metadata dimension: {metadata_dim}")
    
    # 5. Load Model
    print(f"Loading model from {model_dir}...")
    model = EfficientNetV2Hybrid(metadata_dim=metadata_dim).to(device)
    checkpoint = torch.load(model_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 6. Inference Loop
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), # Model expects 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    scores = []
    batch_size = 64
    
    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i:i+batch_size]
            batch_meta = dummy_meta_tensor[i:i+batch_size].to(device)
            
            batch_imgs = []
            for p in batch_files:
                img = Image.open(p).convert('RGB')
                batch_imgs.append(transform(img))
            
            batch_imgs = torch.stack(batch_imgs).to(device)
            
            outputs = model(batch_imgs, batch_meta)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            for path, prob in zip(batch_files, probs):
                scores.append({'path': path, 'prob': prob})
    
    # 7. Filter and Save
    df_scores = pd.DataFrame(scores)
    df_scores = df_scores.sort_values('prob', ascending=False)
    
    print("\nTop 5 scores:")
    print(df_scores.head())
    
    # Filter
    filtered = df_scores[df_scores['prob'] > args.threshold]
    if len(filtered) < args.topk:
        print(f"Warning: Only {len(filtered)} images passed threshold {args.threshold}. Taking top {args.topk} anyway.")
        filtered = df_scores.head(args.topk)
    else:
        filtered = filtered.head(args.topk)
    
    print(f"\nSelected {len(filtered)} images.")
    
    # Copy to output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying to {output_dir}...")
    for _, row in tqdm(filtered.iterrows(), total=len(filtered)):
        src = row['path']
        dst = output_dir / src.name
        shutil.copy(src, dst)
        
    # Save selection log
    filtered.to_csv(output_dir / 'selection_log.csv', index=False)
    print("Done.")

if __name__ == '__main__':
    main()
