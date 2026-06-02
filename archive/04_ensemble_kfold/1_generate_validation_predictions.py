
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import timm
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATASET AND MODEL DEFINITIONS ---

class HybridDataset(Dataset):
    def __init__(self, hdf5_path, metadata_df, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hdf5_file = None
        self.metadata = metadata_df
        feature_cols = [col for col in self.metadata.columns if col not in ['isic_id', 'target']]
        self.metadata_features = self.metadata[feature_cols].values.astype(np.float32)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        
        row = self.metadata.iloc[idx]
        image_id = row['isic_id']
        img_array = self.hdf5_file[image_id][:]
        image = Image.fromarray(img_array)
        
        if self.transform:
            image = self.transform(image)
        
        metadata = torch.tensor(self.metadata_features[idx], dtype=torch.float32)
        return image, metadata, image_id

class GenericHybridModel(nn.Module):
    def __init__(self, backbone_name, metadata_dim, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        num_backbone_features = self.backbone.num_features
        
        self.metadata_processor = nn.Sequential(
            nn.Linear(metadata_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_backbone_features + 64, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, image, metadata):
        img_features = self.backbone(image)
        meta_features = self.metadata_processor(metadata)
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(combined)

def get_model_and_metadata_dim(model_key, preprocessors):
    # This mapping connects your directory names to timm model names
    BACKBONE_MAP = {
        'efficientnetv2_hybrid': 'tf_efficientnetv2_s.in21k_ft_in1k',
        'resnet34_hybrid': 'resnet34.a1_in1k',
        'v2s_features': 'tf_efficientnetv2_s.in21k_ft_in1k',
        'convnext_features': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
        'vit_features': 'vit_small_patch16_224.augreg_in21k_ft_in1k',
        'kfold_advanced': 'tf_efficientnetv2_s.in21k_ft_in1k'
    }
    
    if model_key not in BACKBONE_MAP:
        return None, None

    # Calculate metadata dimension from loaded preprocessors
    metadata_dim = len(preprocessors['scaler'].mean_)
    for col, cats in preprocessors['encoders'].items():
        metadata_dim += len(cats)
        
    model = GenericHybridModel(BACKBONE_MAP[model_key], metadata_dim).to(device)
    return model, metadata_dim

def generate_predictions(model, loader, device):
    model.eval()
    all_ids = []
    all_preds = []
    with torch.no_grad():
        for images, metadata, img_ids in tqdm(loader, desc="Generating Predictions"):
            images, metadata = images.to(device), metadata.to(device)
            outputs = model(images, metadata)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_ids.extend(img_ids)
            all_preds.extend(probs)
    return pd.DataFrame({'isic_id': all_ids, 'target': all_preds})

def main():
    print("="*80)
    print("Step 1: Generate Predictions on Hold-Out Validation Set")
    print("="*80)
    
    # Define models to process
    results_base = Path('DeepLearning/Kaggle/results')
    model_registry = {
        'efficientnetv2_hybrid': {'pattern': 'efficientnet_v2_hybrid_*'},
        'resnet34_hybrid': {'pattern': 'resnet34_hybrid_*'},
        'v2s_features': {'pattern': 'v2s_features_*'},
        'convnext_features': {'pattern': 'convnext_features_*'},
        'vit_features': {'pattern': 'vit_features_*'},
        'kfold_advanced': {'pattern': 'kfold_v2s_features_advanced_*'}
    }

    try:
        full_train_df = pd.read_csv('DeepLearning/Kaggle/data/new-train-metadata.csv')
        _, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42, stratify=full_train_df['target'])
    except FileNotFoundError:
        print("❌ ERROR: Could not find 'DeepLearning/Kaggle/data/new-train-metadata.csv'. Exiting.")
        return

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for model_key, model_info in model_registry.items():
        print(f"\n--- Processing model: {model_key} ---")
        dirs = sorted(results_base.glob(model_info['pattern']))
        if not dirs:
            print("  No result directory found. Skipping.")
            continue
        
        model_dir = dirs[-1]
        preprocessors_path = model_dir / 'preprocessors.pkl'
        checkpoint_path = next(model_dir.glob('best_*.pth'), None)

        if not preprocessors_path.exists() or not checkpoint_path:
            print(f"  Missing preprocessors or checkpoint in {model_dir}. Skipping.")
            continue

        with open(preprocessors_path, 'rb') as f:
            preprocessors = pickle.load(f)

        model, metadata_dim = get_model_and_metadata_dim(model_key, preprocessors)
        if model is None:
            print(f"  Model key '{model_key}' not supported in this script. Skipping.")
            continue
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        val_dataset = HybridDataset(
            hdf5_path='DeepLearning/Kaggle/data/train-image-preprocessed.hdf5',
            metadata_df=val_df,
            transform=val_transform
        )
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)
        
        preds_df = generate_predictions(model, val_loader, device)
        
        output_path = model_dir / 'predictions_on_validation_set.csv'
        preds_df.to_csv(output_path, index=False)
        print(f"  ✓ Predictions saved to: {output_path}")

    print("\n" + "="*80)
    print("Prediction generation complete.")
    print("="*80)

if __name__ == '__main__':
    main()
