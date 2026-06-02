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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Config
DATA_DIR = Path('./data')
RESULTS_DIR = Path('./last_run/experiments')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. DAE Model
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

def run_experiment():
    print("Setting up Experiment B: Metadata Representation (Raw vs DAE)")
    
    # 1. Load Data
    df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv')
    # Filter leakage
    leakage_cols = ['mel_thick_mm', 'mel_mitotic_index']
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    
    # Select features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    target_col = 'target'
    
    if target_col in num_cols: num_cols.remove(target_col)
    # Remove ID cols and high cardinality
    ignore = ['isic_id', 'patient_id', 'image_type', 'tbp_tile_type', 'attribution', 'copyright_license', 'lesion_id']
    # Also drop iddx cols (diagnosis leakage) and MISSING IN TEST features
    ignore.extend([c for c in df.columns if c.startswith('iddx_')])
    ignore.append('tbp_lv_dnn_lesion_confidence')
    ignore.append('mel_thick_mm')
    ignore.append('mel_mitotic_index')
    
    num_cols = [c for c in num_cols if c not in ignore]
    cat_cols = [c for c in cat_cols if c not in ignore]
    
    print(f"Features: {len(num_cols)} Numerical, {len(cat_cols)} Categorical")
    
    # 2. Preprocessing
    # Numerical: Impute Median -> Standardize
    # Categorical: Impute Constant -> OneHot
    
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
    
    X = preprocessor.fit_transform(df)
    y = df[target_col].values
    
    print(f"Processed Shape: {X.shape}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Train DAE
    print("\nTraining DAE...")
    input_dim = X_train.shape[1]
    dae = DAE(input_dim).to(DEVICE)
    optimizer = optim.Adam(dae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    
    dataset = TensorDataset(X_train_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(10): # Quick training
        dae.train()
        total_loss = 0
        for (bx,) in loader:
            # Add noise
            noise = torch.randn_like(bx) * 0.1
            bx_noisy = bx + noise
            
            optimizer.zero_grad()
            recon, _ = dae(bx_noisy)
            loss = criterion(recon, bx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
        
    # Extract Latent Features
    dae.eval()
    with torch.no_grad():
        _, z_train = dae(X_train_tensor)
        _, z_val = dae(X_val_tensor)
        
    X_train_dae = z_train.cpu().numpy()
    X_val_dae = z_val.cpu().numpy()
    
    # 4. Compare XGBoost Performance
    print("\nComparing XGBoost Performance...")
    
    # A. Raw Features
    clf_raw = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric='auc', random_state=42)
    clf_raw.fit(X_train, y_train)
    preds_raw = clf_raw.predict_proba(X_val)[:, 1]
    auc_raw = roc_auc_score(y_val, preds_raw)
    
    # B. DAE Features
    clf_dae = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric='auc', random_state=42)
    clf_dae.fit(X_train_dae, y_train)
    preds_dae = clf_dae.predict_proba(X_val_dae)[:, 1]
    auc_dae = roc_auc_score(y_val, preds_dae)
    
    # C. Combined
    X_train_comb = np.hstack([X_train, X_train_dae])
    X_val_comb = np.hstack([X_val, X_val_dae])
    clf_comb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric='auc', random_state=42)
    clf_comb.fit(X_train_comb, y_train)
    preds_comb = clf_comb.predict_proba(X_val_comb)[:, 1]
    auc_comb = roc_auc_score(y_val, preds_comb)
    
    print("\nResults (XGBoost):")
    print(f"Raw Features: {auc_raw:.4f}")
    print(f"DAE Features: {auc_dae:.4f}")
    print(f"Combined:     {auc_comb:.4f}")
    
    # 5. Compare MLP Performance (User Request)
    print("\nComparing MLP Performance...")
    
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )
        def forward(self, x):
            return self.net(x)
            
    def train_mlp(X_tr, y_tr, X_v, y_v):
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
        X_v_t = torch.tensor(X_v, dtype=torch.float32).to(DEVICE)
        
        model = SimpleMLP(X_tr.shape[1]).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        ds = TensorDataset(X_tr_t, y_tr_t)
        dl = DataLoader(ds, batch_size=256, shuffle=True)
        
        for ep in range(5):
            model.train()
            for bx, by in dl:
                optimizer.zero_grad()
                out = model(bx).squeeze()
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds = model(X_v_t).squeeze().sigmoid().cpu().numpy()
        return roc_auc_score(y_v, preds)

    auc_mlp_raw = train_mlp(X_train, y_train, X_val, y_val)
    auc_mlp_dae = train_mlp(X_train_dae, y_train, X_val_dae, y_val)
    auc_mlp_comb = train_mlp(X_train_comb, y_train, X_val_comb, y_val)
    
    print("\nResults (MLP):")
    print(f"Raw Features: {auc_mlp_raw:.4f}")
    print(f"DAE Features: {auc_mlp_dae:.4f}")
    print(f"Combined:     {auc_mlp_comb:.4f}")

if __name__ == "__main__":
    run_experiment()
