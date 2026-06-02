#!/usr/bin/env python3
"""
Step 6: Visualize Stable Diffusion Fine-Tuning Results
Visualize training progress through validation images and final model generations.
"""

import os
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from diffusers import StableDiffusionPipeline
import numpy as np
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "results_dir": "DeepLearning/Kaggle/generative/results",
    "validation_prompts": [
        "A dermoscopic image of a malignant skin lesion",
        "A close-up photo of a melanoma",
        "A skin lesion with irregular borders and asymmetry",
    ],
    "num_new_samples": 8,
    "grid_cols": 4,
}

# Set GPU (use GPU 2 as before)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

print("🚀 Visualizing SD Fine-Tuning Results")

results_path = Path(CONFIG["results_dir"])
if not results_path.exists():
    print(f"❌ Results directory not found: {results_path}")
    exit(1)

# ============================================================================
# 1. FIND ALL VALIDATION EPOCHS
# ============================================================================

print("\n📂 Scanning validation images...")
epoch_dirs = sorted([d for d in results_path.glob("epoch_*") if d.is_dir()])
if not epoch_dirs:
    print("❌ No validation epoch directories found!")
    exit(1)

print(f"✅ Found {len(epoch_dirs)} validation epochs: {[d.name for d in epoch_dirs[:5]]}{'...' if len(epoch_dirs)>5 else ''}")

# Select key epochs for comparison (first, middle, last)
key_epochs = [epoch_dirs[0], epoch_dirs[len(epoch_dirs)//2], epoch_dirs[-1]]
epoch_names = [d.name for d in key_epochs]
print(f"📊 Comparing epochs: {epoch_names}")

# ============================================================================
# 2. LOAD FINAL MODEL & GENERATE NEW SAMPLES
# ============================================================================

print("\n🤖 Loading final model for new generations...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        results_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()  # Memory efficient
    print("✅ Final model loaded!")
except Exception as e:
    print(f"⚠️  Could not load final model: {e}")
    pipe = None

# Generate new samples
new_images = []
if pipe:
    print("🎨 Generating new samples...")
    for prompt in tqdm(CONFIG["validation_prompts"] * 3):  # Repeat for more samples
        img = pipe(
            prompt, 
            num_inference_steps=30,
            guidance_scale=7.5,
            height=512,
            width=512,
        ).images[0]
        new_images.append(img)

# ============================================================================
# 3. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle("Stable Diffusion Fine-Tuning Progress\n(Malignant Skin Lesion Generation)", fontsize=20, y=0.95)

# Row 1-3: Epoch comparisons
for row, epoch_dir in enumerate(key_epochs, 1):
    epoch_path = epoch_dir
    val_images = list(epoch_path.glob("val_*.png"))[:4]  # First 4 images
    
    for col, img_path in enumerate(val_images):
        img = Image.open(img_path)
        axes[row-1, col].imshow(img)
        axes[row-1, col].set_title(f"{epoch_names[row-1]}\n{img_path.name}", fontsize=10)
        axes[row-1, col].axis('off')
    
    # Empty cells if fewer images
    for col in range(len(val_images), 4):
        axes[row-1, col].axis('off')

# Row 4: New generations from final model
if new_images:
    for col, img in enumerate(new_images[:4]):
        axes[3, col].imshow(img)
        axes[3, col].set_title("Final Model\n(New Generation)", fontsize=10)
        axes[3, col].axis('off')
else:
    for col in range(4):
        axes[3, col].text(0.5, 0.5, "No Final Model", ha='center', va='center', transform=axes[3, col].transAxes)
        axes[3, col].axis('off')

plt.tight_layout()
plt.savefig(results_path / "training_progress_overview.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Overview saved: {results_path / 'training_progress_overview.png'}")

# ============================================================================
# 4. DETAILED PROGRESSION GRID
# ============================================================================

print("\n📈 Creating detailed progression grid...")
all_epochs = epoch_dirs[:10]  # First 10 epochs for progression

fig2, axes2 = plt.subplots(len(all_epochs), 1, figsize=(12, 4*len(all_epochs)))
if len(all_epochs) == 1:
    axes2 = [axes2]

for i, epoch_dir in enumerate(all_epochs):
    epoch_path = epoch_dir
    val_images = sorted(list(epoch_path.glob("val_*.png")))[:3]
    
    for j, img_path in enumerate(val_images):
        img = Image.open(img_path)
        axes2[i].imshow(img)
        if j == 0:
            axes2[i].set_title(f"Epoch {epoch_dir.name.replace('epoch_', '')}", fontsize=12)
        axes2[i].axis('off')
    
    # Add spacer if fewer images
    if len(val_images) < 3:
        axes2[i].text(0.5, 0.5, " ", ha='center', va='center', transform=axes2[i].transAxes)

plt.tight_layout()
plt.savefig(results_path / "epoch_progression.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Progression saved: {results_path / 'epoch_progression.png'}")

# ============================================================================
# 5. SUMMARY STATISTICS
# ============================================================================

print("\n📊 SUMMARY:")
print(f"• Training Results Directory: {results_path}")
print(f"• Validation Epochs Found: {len(epoch_dirs)}")
print(f"• Key Epochs Compared: {', '.join(epoch_names)}")
print(f"• New Samples Generated: {len(new_images) if pipe else 0}")
print(f"• Final Model Load: {'✅ Success' if pipe else '❌ Failed'}")

if list(results_path.glob("checkpoint*")):
    print(f"• Checkpoints Available: {len(list(results_path.glob('checkpoint*')))}")

print("\n🎉 Analysis complete! Check the generated PNG files for visual progress.")
