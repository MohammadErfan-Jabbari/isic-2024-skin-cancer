import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import argparse
import joblib
from feature_engineering import engineer_features, NEW_FEATURES

# Config
DATA_DIR = Path('./data')
LAST_RUN_DIR = Path('./last_run')
RESULTS_DIR = LAST_RUN_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature List (The 38 Safe Features)
SAFE_FEATURES = [
    'age_approx', 'sex', 'anatom_site_general',
    'clin_size_long_diam_mm', 'tbp_lv_nevi_confidence',
    'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext',
    'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext',
    'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean',
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
    'tbp_lv_eccentricity', 'tbp_lv_location', 'tbp_lv_location_simple',
    'tbp_lv_minorAxisMM', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
    'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max',
    'tbp_lv_stdL', 'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
    'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z'
] + NEW_FEATURES

class DAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

def get_preprocessor():
    # Define Numerical and Categorical columns based on SAFE_FEATURES
    # We need to check the types from the dataframe first, but we can hardcode known types
    # or infer them dynamically. Dynamic is safer.
    return None # Will be created in main

def train_dae(args):
    print("--- Step 1: Training Denoising Autoencoder (DAE) ---")
    
    # 1. Load Data
    print("Loading Data...")
    train_df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', low_memory=False)
    test_df = pd.read_csv(DATA_DIR / 'students-test-metadata.csv', low_memory=False)
    
    # Apply Feature Engineering
    print("Applying Feature Engineering...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape: {test_df.shape}")
    
    # 2. Feature Selection
    print(f"Selecting {len(SAFE_FEATURES)} Safe Features...")
    # Verify all features exist
    missing_train = [f for f in SAFE_FEATURES if f not in train_df.columns]
    missing_test = [f for f in SAFE_FEATURES if f not in test_df.columns]
    
    if missing_train:
        raise ValueError(f"Missing features in Train: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing features in Test: {missing_test}")
        
    X_train_raw = train_df[SAFE_FEATURES].copy()
    X_test_raw = test_df[SAFE_FEATURES].copy()
    
    # Combine for Global Scaling (Unsupervised)
    X_all = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)
    
    # 3. Preprocessing
    print("Preprocessing (Global Scaling)...")
    num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_all.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"  Numerical: {len(num_cols)}")
    print(f"  Categorical: {len(cat_cols)}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ]
    )
    
    X_all_processed = preprocessor.fit_transform(X_all)
    print(f"  Processed Shape: {X_all_processed.shape}")
    
    # Save Preprocessor for later use (if needed)
    joblib.dump(preprocessor, RESULTS_DIR / 'dae_preprocessor.pkl')
    
    # Split back
    n_train = len(train_df)
    X_train = X_all_processed[:n_train]
    X_test = X_all_processed[n_train:]
    
    # 4. Train DAE
    print("\nTraining DAE...")
    input_dim = X_train.shape[1]
    model = DAE(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Create Loader (Train on ALL data? Or just Train? Usually Train on ALL for DAE)
    # Let's train on ALL data since it's unsupervised and we want to learn the manifold of the test set too.
    dataset = TensorDataset(torch.tensor(X_all_processed, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for (bx,) in loader:
            bx = bx.to(DEVICE)
            
            # Add Noise (Denoising)
            noise = torch.randn_like(bx) * args.noise_level
            bx_noisy = bx + noise
            
            optimizer.zero_grad()
            recon, _ = model(bx_noisy)
            loss = criterion(recon, bx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f}")
        
    # 5. Extract Latent Features
    print("\nExtracting Latent Features...")
    model.eval()
    
    # Helper to extract in batches
    def extract(data_array):
        t = torch.tensor(data_array, dtype=torch.float32)
        ds = TensorDataset(t)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        latents = []
        with torch.no_grad():
            for (bx,) in dl:
                bx = bx.to(DEVICE)
                _, z = model(bx)
                latents.append(z.cpu().numpy())
        return np.vstack(latents)
    
    latent_train = extract(X_train)
    latent_test = extract(X_test)
    
    print(f"  Train Latent Shape: {latent_train.shape}")
    print(f"  Test Latent Shape: {latent_test.shape}")
    
    # 6. Save Results
    print("Saving Results...")
    torch.save(model.state_dict(), RESULTS_DIR / 'dae_model.pth')
    torch.save(model.encoder.state_dict(), RESULTS_DIR / 'dae_encoder.pth')
    np.save(RESULTS_DIR / 'dae_latent_train.npy', latent_train)
    np.save(RESULTS_DIR / 'dae_latent_test.npy', latent_test)
    
    print("✅ Step 1 Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--noise_level', type=float, default=0.1)
    args = parser.parse_args()
    
    train_dae(args)
