#!/usr/bin/env python3
"""
Step 2.4: Fine-tune Stable Diffusion on Malignant Skin Lesion Images
Phase 2 of Synthetic Data Generation Pipeline

Architecture:
  - Base Model: ThisIsReal v7 (SD 1.5 based)
  - Custom VAE: vae-ft-mse-840000-ema-pruned
  - Resolution: 128x128 (fixed)
  - Training: Full fine-tuning (UNet only)
  - GPU: 2 & 3 (CUDA_VISIBLE_DEVICES=2,3)
"""

import os
import sys

# ============================================================================
# SET GPU BEFORE IMPORTING TORCH
# ============================================================================
# GPU IDs to use
GPU_IDS = [2, 3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))

import torch
import json
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

print("="*70)
print("📦 IMPORTING DIFFUSERS LIBRARIES...")
print("="*70)

try:
    from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL
    from diffusers.models import UNet2DConditionModel
    from diffusers.optimization import get_cosine_schedule_with_warmup
    from diffusers.training_utils import EMAModel, compute_snr
    from transformers import CLIPTextModel, CLIPTokenizer
    from datasets import load_dataset
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    import torch.nn.functional as F
    print("✅ All diffusers imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("🔧 CONFIGURATION")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model paths
    "model_path": "DeepLearning/Kaggle/generative/checkpoints/thisisreal_v7/thisisreal_v7.safetensors",
    "config_path": "DeepLearning/Kaggle/generative/checkpoints/v1-inference.yaml",
    "vae_path": "DeepLearning/Kaggle/generative/checkpoints/vae/vae-ft-mse-840000-ema-pruned.safetensors",
    
    # Dataset
    "dataset_dir": "DeepLearning/Kaggle/generative/data/training_dataset",
    "output_dir": "DeepLearning/Kaggle/generative/results",
    
    # Training parameters
    "resolution": 128,
    "batch_size": 8,
    "num_epochs": 50,
    "max_train_steps": None,  # Will be calculated
    "learning_rate": 1e-5,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "fp16",
    "weight_decay": 1e-2,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    
    # Optimization & stability
    "snr_gamma": 5.0,  # Min-SNR weighting
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 500,
    "use_ema": True,  # Exponential Moving Average
    "enable_xformers": True,
    "gradient_checkpointing": True,
    
    # Checkpointing
    "checkpointing_steps": None,  # Will be set based on epochs
    "checkpoint_start_epoch": 30,
    
    # Validation
    "validation_epochs": 5,
    "validation_prompts": [
        "A dermoscopic image of a malignant skin lesion",
        "A close-up photo of a melanoma",
        "A skin lesion with irregular borders and asymmetry",
    ],
    
    # GPU configuration
    "gpu_ids": [2, 3],
    "use_multi_gpu": False,  # Single GPU for now
    "primary_device": 0,  # First GPU in CUDA_VISIBLE_DEVICES
}

print(f"\n✅ GPU Configuration (Set before torch import):")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"   Using GPU(s): {GPU_IDS}")

# ============================================================================
# VALIDATION & VERIFICATION
# ============================================================================

print("\n" + "="*70)
print("🔍 VERIFYING PATHS & FILES")
print("="*70)

def verify_path(path_str, name):
    """Verify file/directory exists."""
    path = Path(path_str)
    if path.exists():
        if path.is_file():
            size = path.stat().st_size / 1e9
            print(f"✅ {name:<30} {size:.2f} GB")
        else:
            items = len(list(path.glob("*")))
            print(f"✅ {name:<30} ({items} items)")
        return True
    else:
        print(f"❌ {name:<30} NOT FOUND: {path}")
        return False

all_ok = True
all_ok &= verify_path(CONFIG["model_path"], "ThisIsReal Checkpoint")
all_ok &= verify_path(CONFIG["vae_path"], "Custom VAE")
all_ok &= verify_path(CONFIG["dataset_dir"], "Training Dataset")

if not all_ok:
    print("\n❌ Some files are missing!")
    sys.exit(1)

# Verify dataset contents
dataset_path = Path(CONFIG["dataset_dir"])
images = list(dataset_path.glob("ISIC_*.png"))
metadata_file = dataset_path / "metadata.jsonl"

print(f"\n📊 Dataset Verification:")
print(f"   Images: {len(images)}")
print(f"   Metadata: {metadata_file.exists()}")

if len(images) != 343 or not metadata_file.exists():
    print("❌ Dataset incomplete!")
    sys.exit(1)

print("\n✅ All files verified!")

# ============================================================================
# DEVICE SETUP
# ============================================================================

print("\n" + "="*70)
print("🖥️  DEVICE SETUP")
print("="*70)

if not torch.cuda.is_available():
    print("❌ CUDA not available!")
    sys.exit(1)

# Create device object AFTER torch is initialized
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")
print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# MAIN FUNCTION (STEP-BY-STEP MODULAR)
# ============================================================================

def load_models():
    """Step 1: Load ThisIsReal and custom VAE."""
    print("\n" + "="*70)
    print("📥 LOADING MODELS (COMPONENT-WISE)")
    print("="*70)
    
    # 1. Load Scheduler and Tokenizer from standard SD 1.5
    print(f"\n1️⃣  Loading Scheduler & Tokenizer (Standard SD 1.5)...")
    try:
        scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        print("✅ Scheduler & Tokenizer loaded")
    except Exception as e:
        print(f"❌ Error loading standard components: {e}")
        return None, None, None, None, None

    # 2. Load UNet from Checkpoint
    print(f"\n2️⃣  Loading UNet from ThisIsReal v7...")
    try:
        unet = UNet2DConditionModel.from_single_file(
            CONFIG["model_path"],
            original_config=CONFIG["config_path"],
            torch_dtype=torch.float32, # Load in fp32 for training stability with mixed precision
        )
        unet = unet.to(device, dtype=torch.float32)
        print("✅ UNet loaded successfully")
    except Exception as e:
        print(f"❌ Error loading UNet: {e}")
        return None, None, None, None, None

    # 3. Load Text Encoder (Standard SD 1.5)
    # Note: CLIPTextModel.from_single_file is not available. 
    # We use the standard SD 1.5 text encoder which is compatible with ThisIsReal.
    print(f"\n3️⃣  Loading Text Encoder (Standard SD 1.5)...")
    try:
        text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
        text_encoder = text_encoder.to(device, dtype=torch.float16)
        print("✅ Text Encoder loaded successfully")
    except Exception as e:
        print(f"❌ Error loading Text Encoder: {e}")
        return None, None, None, None, None
    
    # 4. Load Custom VAE
    print(f"\n4️⃣  Loading custom VAE from: {CONFIG['vae_path']}")
    try:
        vae = AutoencoderKL.from_single_file(
            CONFIG["vae_path"],
            torch_dtype=torch.float16,
        )
        vae = vae.to(device, dtype=torch.float16)
        print("✅ Custom VAE loaded successfully")
    except Exception as e:
        print(f"❌ Error loading custom VAE: {e}")
        return None, None, None, None, None
    
    print(f"\n5️⃣  Component Summary:")
    print(f"   ✅ VAE: {vae.__class__.__name__} (Device: {next(vae.parameters()).device})")
    print(f"   ✅ Text Encoder: {text_encoder.__class__.__name__} (Device: {next(text_encoder.parameters()).device})")
    print(f"   ✅ Tokenizer: {tokenizer.__class__.__name__}")
    print(f"   ✅ UNet: {unet.__class__.__name__} (Device: {next(unet.parameters()).device})")
    print(f"   ✅ Scheduler: {scheduler.__class__.__name__}")
    
    return vae, text_encoder, tokenizer, unet, scheduler


class SkinLesionDataset(Dataset):
    """
    Dataset for loading skin lesion images and captions.
    """
    def __init__(self, dataset_dir, tokenizer, size=128):
        self.dataset_dir = Path(dataset_dir)
        self.tokenizer = tokenizer
        self.size = size
        self.entries = []
        
        # Load metadata
        metadata_path = self.dataset_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, "r") as f:
            for line in f:
                self.entries.append(json.loads(line))
        
        # Define transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_path = self.dataset_dir / entry["file_name"]
        
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.transforms(image)
        except Exception as e:
            print(f"Warning: Error loading image {image_path}: {e}")
            # Return a dummy tensor in case of error to avoid crashing
            pixel_values = torch.zeros((3, self.size, self.size))

        # Tokenize caption
        input_ids = self.tokenizer(
            entry["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids
        }


        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids
        }


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, epoch):
    """Generate validation images."""
    print(f"\n🎨 Generating validation images for Epoch {epoch}...")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(42)
    images = []
    
    for prompt in CONFIG["validation_prompts"]:
        with torch.autocast("cuda"):
            image = pipeline(prompt, num_inference_steps=30, generator=generator).images[0]
        images.append((prompt, image))
    
    # Save images
    save_dir = Path(CONFIG["output_dir"]) / f"epoch_{epoch}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (prompt, img) in enumerate(images):
        img.save(save_dir / f"val_{i}_{prompt[:30].replace(' ', '_')}.png")
    
    print(f"✅ Saved validation images to {save_dir}")
    del pipeline
    torch.cuda.empty_cache()


def train_loop(vae, text_encoder, tokenizer, unet, scheduler):
    """Step 3: Main Training Loop."""
    print("\n" + "="*70)
    print("🏋️  STARTING TRAINING LOOP")
    print("="*70)
    
    # 1. Setup Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        mixed_precision=CONFIG["mixed_precision"],
        project_dir=CONFIG["output_dir"]
    )
    
    # 2. Setup Dataset & DataLoader
    dataset = SkinLesionDataset(
        dataset_dir=CONFIG["dataset_dir"],
        tokenizer=tokenizer,
        size=CONFIG["resolution"]
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    
    # 3. Prepare Models
    # Freeze VAE and Text Encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    
    # Enable gradient checkpointing for UNet
    if CONFIG["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    
    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=CONFIG["learning_rate"],
        betas=(CONFIG["adam_beta1"], CONFIG["adam_beta2"]),
        weight_decay=CONFIG["weight_decay"],
        eps=CONFIG["adam_epsilon"],
    )
    
    num_update_steps_per_epoch = len(train_dataloader) // CONFIG["gradient_accumulation_steps"]
    max_train_steps = CONFIG["num_epochs"] * num_update_steps_per_epoch
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=CONFIG["lr_warmup_steps"],
        num_training_steps=max_train_steps,
    )
    
    # 5. Prepare with Accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move frozen models to device
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)
    
    # 6. Training Loop
    global_step = 0
    
    print(f"✅ Training configuration:")
    print(f"   Num examples: {len(dataset)}")
    print(f"   Num epochs: {CONFIG['num_epochs']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Total optimization steps: {max_train_steps}")
    print(f"   Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
    
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    epoch_losses = []
    
    for epoch in range(CONFIG["num_epochs"]):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latents
                pixel_values = batch["pixel_values"].to(dtype=torch.float16)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                
                # Add noise to the latents
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Get the target for loss depending on the prediction type
                if scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")
                
                # Cast to float32 for stability and to avoid "Found dtype Half but expected Float" error
                model_pred = model_pred.float()
                target = target.float()
                
                # Compute Min-SNR-Gamma Loss
                if CONFIG["snr_gamma"] is not None:
                    snr = compute_snr(scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, CONFIG["snr_gamma"] * torch.ones_like(snr)], dim=1).min(dim=1)[0]
                    if scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    
                    # Ensure weights are also float32
                    mse_loss_weights = mse_loss_weights.float()
                    
                    loss = F.mse_loss(model_pred, target, reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                else:
                    loss = F.mse_loss(model_pred, target, reduction="mean")
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), CONFIG["max_grad_norm"])
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()
                progress_bar.set_postfix({"loss": train_loss / (step + 1)})
        
        avg_epoch_loss = train_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        # Validation & Checkpointing
        if (epoch + 1) % CONFIG["validation_epochs"] == 0:
            if accelerator.is_main_process:
                log_validation(vae, text_encoder, tokenizer, unet, CONFIG, accelerator, epoch + 1)
        
        if (epoch + 1) >= CONFIG["checkpoint_start_epoch"] and (epoch + 1) % 5 == 0:
             if accelerator.is_main_process:
                save_path = Path(CONFIG["output_dir"]) / f"checkpoint-epoch-{epoch+1}"
                accelerator.save_state(save_path)
                print(f"💾 Saved checkpoint to {save_path}")

    print("\n✅ Training complete!")
    
    # Save training history
    if accelerator.is_main_process:
        import pickle
        torch.save({"epoch_losses": epoch_losses}, Path(CONFIG["output_dir"]) / "training_history.pt")
        print(f"💾 Training history saved to {CONFIG['output_dir']}/training_history.pt")
    
    # Save final model
    if accelerator.is_main_process:
        final_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        final_pipeline.save_pretrained(CONFIG["output_dir"])
        print(f"🎉 Final model saved to {CONFIG['output_dir']}")


if __name__ == "__main__":
    
    print("\n" + "🚀 "*35)
    print("PHASE 2.4: TRAINING START")
    print("🚀 "*35)
    
    # Step 1: Load models
    vae, text_encoder, tokenizer, unet, scheduler = load_models()
    if vae is None:
        sys.exit(1)
    
    # Step 2: Start Training
    train_loop(vae, text_encoder, tokenizer, unet, scheduler)
