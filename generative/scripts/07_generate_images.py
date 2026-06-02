#!/usr/bin/env python3
"""
Phase 3: Synthetic Data Production
Generate 10,000 synthetic images using the fine-tuned Stable Diffusion model.
"""

import os
import torch
import pandas as pd
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
import random

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================

CONFIG = {
    # Paths
    "model_path": "DeepLearning/Kaggle/generative/results",
    "output_dir": "DeepLearning/Kaggle/generative/data/synthetic_images_128px",

    # Generation Parameters
    "num_images_to_generate": 10000,
    "batch_size": 64,  # Adjust based on VRAM
    "image_resolution": 128, # Must match training resolution

    # Inference Parameters
    "inference_steps": 30,
    "guidance_scale": 7.5,

    # Prompt Pool for Diversity
    "prompt_pool": [
        "A dermoscopic image of a malignant skin lesion",
        "A close-up photograph of a melanoma",
        "A skin lesion with irregular borders and asymmetry",
        "Dermoscopy of a malignant nevus, showing variegation in color",
        "High-resolution dermoscopic photo of a cancerous skin mole",
    ]
}

def main():
    """Main function to run the image generation pipeline."""

    print("🚀 Starting Phase 3: Synthetic Data Production")
    
    # --- GPU Setup ---
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Aborting.")
        return
    
    # Use GPU 2, consistent with previous steps
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda:0")
    print(f"✅ Using device: {device} ({torch.cuda.get_device_name(0)})")

    # --- Directory Setup ---
    output_path = Path(CONFIG["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory prepared: {output_path}")

    # --- Model Loading ---
    print("\n🔄 Loading fine-tuned model...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            CONFIG["model_path"],
            torch_dtype=torch.float16,
            safety_checker=None
        )
        # Use a high-quality scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        pipe.enable_attention_slicing() # VRAM optimization
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # --- Generation Loop ---
    print(f"\n🎨 Starting generation of {CONFIG['num_images_to_generate']} images...")
    
    num_generated = 0
    metadata = []
    
    with tqdm(total=CONFIG["num_images_to_generate"], desc="Generating Images") as pbar:
        while num_generated < CONFIG["num_images_to_generate"]:
            
            current_batch_size = min(CONFIG["batch_size"], CONFIG["num_images_to_generate"] - num_generated)
            if current_batch_size <= 0:
                break

            # Randomly sample prompts for the batch
            prompts = [random.choice(CONFIG["prompt_pool"]) for _ in range(current_batch_size)]
            
            # Generate images
            with torch.autocast("cuda"):
                images = pipe(
                    prompt=prompts,
                    height=CONFIG["image_resolution"],
                    width=CONFIG["image_resolution"],
                    num_inference_steps=CONFIG["inference_steps"],
                    guidance_scale=CONFIG["guidance_scale"],
                ).images

            # Save images and record metadata
            for i, image in enumerate(images):
                file_name = f"synthetic_{num_generated + i:05d}.png"
                image.save(output_path / file_name)
                metadata.append({
                    "file_name": file_name,
                    "prompt": prompts[i]
                })

            num_generated += len(images)
            pbar.update(len(images))

    print(f"✅ Generation complete!")

    # --- Save Metadata ---
    metadata_df = pd.DataFrame(metadata)
    metadata_path = output_path / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"💾 Metadata saved to: {metadata_path}")

    # --- Final Report ---
    print("\n📊 Summary:")
    print(f"   - Total Images Generated: {num_generated}")
    print(f"   - Output Directory: {output_path}")
    print("🎉 Phase 3 complete!")


if __name__ == "__main__":
    main()
