
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import timm
import cv2
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from PIL import Image
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Config
DATA_DIR = Path('./data')
LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
OUTPUT_DIR = Path('public/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_ID = 'ISIC_0096034'  # Malignant sample

class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.in_features = self.backbone.num_features
        self.head = nn.Linear(self.in_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

def get_image_from_hdf5(isic_id, img_size=336):
    hdf5_path = DATA_DIR / 'train-image.hdf5'
    with h5py.File(hdf5_path, 'r') as fp:
        if isic_id not in fp:
            print(f"❌ ID {isic_id} not found in HDF5")
            return None
        data = fp[isic_id][()]
        image = Image.open(io.BytesIO(data))
        image = np.array(image)
        
    return image

def preprocess_image(image, img_size):
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ])
    augmented = transform(image=image)
    tensor = augmented['image'].unsqueeze(0)  # Add batch dim
    
    # Also return normalized numpy image for visualization (H,W,C, float 0-1)
    viz_img = cv2.resize(image, (img_size, img_size))
    viz_img = np.float32(viz_img) / 255.0
    
    return tensor.to(DEVICE), viz_img

def reshape_transform_eva02(tensor):
    """Reshape transform for EVA02 ViT (Grid 24x24 for 336px input)."""
    # tensor: [batch, patches, channels] -> needs [batch, channels, height, width]
    # Remove class token if present
    target_len = 24 * 24
    if tensor.shape[1] == target_len + 1:
        tensor = tensor[:, 1:, :] 
    
    height = 24
    width = 24
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    
    # Bring channels to first dim: [B, H, W, C] -> [B, C, H, W]
    result = result.permute(0, 3, 1, 2)
    return result

def visualize_model(model_name, weights_path, target_layer_func, reshape_func=None, img_size=336, label="Model"):
    print(f"Loading {label} from {weights_path}...")
    model = ISICModel(model_name, pretrained=False)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False) # strict=False to be safe with head
    model.to(DEVICE)
    model.eval()
    
    # Target Layers
    target_layers = target_layer_func(model)
    
    # Load Image
    raw_img = get_image_from_hdf5(SAMPLE_ID)
    input_tensor, viz_img = preprocess_image(raw_img, img_size)
    
    # Run CAM
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_func)
    
    # We focus on target=1 (Malignant) - logits output has 1 channel
    # ClassifierOutputTarget(0) because num_classes=1 output
    # Wait, simple binary usually uses output[0]
    targets = [ClassifierOutputTarget(0)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay
    visualization = show_cam_on_image(viz_img, grayscale_cam, use_rgb=True)
    return visualization, viz_img

def main():
    # 1. EVA02 Visualization
    eva_path = RESULTS_DIR / 'eva02_small_patch14_336.mim_in22k_ft_in1k_fold4.pth' # Using Fold 4 as it had high score
    # EVA02 Target: Last block's norm1 is usually good for ViT
    def get_eva_layers(m): return [m.backbone.blocks[-1].norm1]
    
    try:
        eva_cam, original_img = visualize_model(
            'eva02_small_patch14_336.mim_in22k_ft_in1k', 
            eva_path, 
            get_eva_layers, 
            reshape_transform_eva02, 
            img_size=336, 
            label="EVA02"
        )
    except Exception as e:
        print(f"Error EVA02: {e}")
        eva_cam = None

    # 2. EdgeNeXt Visualization
    edge_path = RESULTS_DIR / 'edgenext_base_fold0.pth'
    # EdgeNeXt Target: Last stage
    def get_edge_layers(m): return [m.backbone.stages[-1].blocks[-1]]
    
    try:
        edge_cam, _ = visualize_model(
            'edgenext_base', 
            edge_path, 
            get_edge_layers, 
            None, # EdgeNeXt produces feature maps, no reshape needed usually (or it might)
            # EdgeNeXt uses Conv stages, so output is spatial [B, C, H, W]
            img_size=384, 
            label="EdgeNeXt"
        )
    except Exception as e:
        print(f"Error EdgeNeXt: {e}")
        edge_cam = None
        
    # 3. Create Figure
    if eva_cam is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(original_img)
        axes[0].set_title(f'Original Lesion\n({SAMPLE_ID})', fontweight='bold')
        axes[0].axis('off')
        
        # EVA02
        axes[1].imshow(eva_cam)
        axes[1].set_title('EVA02 Attention\n(Global Context)', fontweight='bold', color=COLORS['eva02'])
        axes[1].axis('off')
        
        # EdgeNeXt
        if edge_cam is not None:
            # Resize EdgeNeXt cam to 336 for display consistency
            edge_cam_resized = cv2.resize(edge_cam, (336, 336))
            axes[2].imshow(edge_cam_resized)
            axes[2].set_title('EdgeNeXt Attention\n(Local Texture)', fontweight='bold', color=COLORS['edgenext'])
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'vision_activation_maps.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR / 'vision_activation_maps.png'}")

# Define colors for plotting
COLORS = {
    'eva02': '#3498db',
    'edgenext': '#e74c3c'
}

if __name__ == "__main__":
    main()
