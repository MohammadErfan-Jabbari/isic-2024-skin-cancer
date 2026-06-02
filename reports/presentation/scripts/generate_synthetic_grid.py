
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Paths
SOURCE_DIR = Path("./generative/data/synthetic_malignant_filtered")
OUTPUT_PATH = Path("public/figures/synthetic_grid.png")

# Config
GRID_SIZE = (4, 4) # 4x4 grid
NUM_IMAGES = GRID_SIZE[0] * GRID_SIZE[1]
IMG_SIZE = (128, 128)

def generate_grid():
    # Get all png files
    all_files = list(SOURCE_DIR.glob("*.png"))
    print(f"Found {len(all_files)} synthetic images.")
    
    # Sort to be deterministic or random? User wants "nice".
    # Let's shuffle with seed for reproducibility but diversity
    random.seed(42)
    selected_files = random.sample(all_files, NUM_IMAGES)
    
    # Create Figure
    fig, axes = plt.subplots(GRID_SIZE[0], GRID_SIZE[1], figsize=(10, 10))
    # Remove whitespace
    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
    
    for i, ax in enumerate(axes.flat):
        img_path = selected_files[i]
        img = Image.open(img_path).convert('RGB')
        
        ax.imshow(img)
        ax.axis('off')
        # Optional: Add border if needed, but clean is better
        
    # Save
    print(f"Saving grid to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
if __name__ == "__main__":
    generate_grid()
