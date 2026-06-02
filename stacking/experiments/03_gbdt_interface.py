import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.decomposition import PCA

# Config
DATA_DIR = Path('./data')
RESULTS_DIR = Path('./last_run/experiments')

def run_experiment():
    print("Setting up Experiment C: Vision-to-GBDT Interface")
    
    # 1. Load Data (Metadata only for target)
    df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv')
    y = df['target'].values
    
    # 2. Simulate Vision Outputs
    # We need to simulate:
    # A. Logits: Highly correlated with target (AUC ~0.8-0.9)
    # B. Embeddings: 768-dim vector containing the information for Logits + extra noise/features
    
    print("Simulating Vision Outputs...")
    n_samples = len(y)
    emb_dim = 768
    
    # Create synthetic embeddings
    # Positive samples have a specific pattern in the first few dimensions
    X_emb = np.random.randn(n_samples, emb_dim).astype(np.float32)
    
    # Add signal to first 10 dimensions for positive class
    pos_idx = np.where(y == 1)[0]
    X_emb[pos_idx, :10] += 0.5
    
    # Create Logits from Embeddings (Linear projection)
    # This simulates the "Head" of the vision model
    weights = np.random.randn(emb_dim)
    weights[:10] = 1.0 # High weight on signal dims
    weights[10:] = 0.01 # Noise
    
    logits = X_emb @ weights
    # Add some non-linearity/noise to make it imperfect
    logits += np.random.randn(n_samples) * 0.5
    
    # Check base AUC of logits
    base_auc = roc_auc_score(y, logits)
    print(f"Base Vision AUC (Logits): {base_auc:.4f}")
    
    # Split
    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 3. Compare GBDT Performance
    print("\nComparing GBDT Performance...")
    
    # A. Logits Only
    # Input: Logits (1 feature)
    X_train_logits = logits[train_idx].reshape(-1, 1)
    X_val_logits = logits[val_idx].reshape(-1, 1)
    
    clf_logits = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, eval_metric='auc', random_state=42)
    clf_logits.fit(X_train_logits, y_train)
    preds_logits = clf_logits.predict_proba(X_val_logits)[:, 1]
    auc_logits = roc_auc_score(y_val, preds_logits)
    
    # B. Embeddings Only (PCA reduced to avoid curse of dimensionality for GBDT?)
    # XGBoost can handle 768 features, but it's slow. Let's try raw first.
    # Actually, let's use PCA to 50 dims to simulate a "bottleneck" or efficient transfer.
    print("Computing PCA on embeddings...")
    pca = PCA(n_components=50)
    X_emb_pca = pca.fit_transform(X_emb)
    
    X_train_emb = X_emb_pca[train_idx]
    X_val_emb = X_emb_pca[val_idx]
    
    clf_emb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, eval_metric='auc', random_state=42)
    clf_emb.fit(X_train_emb, y_train)
    preds_emb = clf_emb.predict_proba(X_val_emb)[:, 1]
    auc_emb = roc_auc_score(y_val, preds_emb)
    
    # C. Hybrid (Logits + PCA Embeddings)
    X_train_hyb = np.hstack([X_train_logits, X_train_emb])
    X_val_hyb = np.hstack([X_val_logits, X_val_emb])
    
    clf_hyb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, eval_metric='auc', random_state=42)
    clf_hyb.fit(X_train_hyb, y_train)
    preds_hyb = clf_hyb.predict_proba(X_val_hyb)[:, 1]
    auc_hyb = roc_auc_score(y_val, preds_hyb)
    
    print("\nResults:")
    print(f"Logits Only:   {auc_logits:.4f}")
    print(f"Embeddings:    {auc_emb:.4f}")
    print(f"Hybrid:        {auc_hyb:.4f}")

if __name__ == "__main__":
    run_experiment()
