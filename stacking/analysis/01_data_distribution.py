import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
DATA_DIR = Path('./data')
RESULTS_DIR = Path('./last_run/analysis')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_data():
    print("Loading data...")
    train_meta = pd.read_csv(DATA_DIR / 'new-train-metadata.csv')
    print(f"Total samples: {len(train_meta)}")
    
    # 1. Class Imbalance
    target_counts = train_meta['target'].value_counts()
    print("\n1. Class Distribution:")
    print(target_counts)
    print(f"Positive Ratio: {target_counts[1] / len(train_meta):.5f}")
    
    # 2. Metadata Structure (for DAE)
    print("\n2. Metadata Structure:")
    numerical_cols = train_meta.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_meta.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numerical Columns ({len(numerical_cols)}):")
    print(numerical_cols[:10], "...")
    print(f"Categorical Columns ({len(categorical_cols)}):")
    print(categorical_cols[:10], "...")
    
    # Check for missing values in numerical columns (crucial for DAE)
    missing_num = train_meta[numerical_cols].isnull().mean().sort_values(ascending=False)
    print("\nTop Missing Numerical Columns:")
    print(missing_num.head(10))
    
    # 3. Easy vs Hard Samples (Proxy via metadata)
    # We don't have OOFs yet for this "clean" run, but we can check if certain metadata correlates strongly
    # For example, tbp_lv_dnn_lesion_confidence
    if 'tbp_lv_dnn_lesion_confidence' in train_meta.columns:
        print("\n3. DNN Confidence Analysis:")
        sns.histplot(data=train_meta, x='tbp_lv_dnn_lesion_confidence', hue='target', bins=50, kde=True)
        plt.title('DNN Confidence Distribution by Target')
        plt.savefig(RESULTS_DIR / 'dnn_confidence_dist.png')
        print(f"Saved plot to {RESULTS_DIR / 'dnn_confidence_dist.png'}")
        
        # Calculate "Hard" positives (low confidence)
        hard_positives = train_meta[(train_meta['target'] == 1) & (train_meta['tbp_lv_dnn_lesion_confidence'] > 90)] # High confidence = Benign usually? Wait, let's check correlation.
        # Usually high confidence means confident in prediction. If it's "lesion confidence", maybe it means "confidence it is a lesion"?
        # Let's check correlation
        corr = train_meta['tbp_lv_dnn_lesion_confidence'].corr(train_meta['target'])
        print(f"Correlation (DNN Conf vs Target): {corr:.4f}")
        
    # 4. Leakage Check
    leakage_cols = ['mel_thick_mm', 'mel_mitotic_index']
    print("\n4. Leakage Check:")
    for col in leakage_cols:
        if col in train_meta.columns:
            non_null = train_meta[col].notnull().sum()
            non_null_target = train_meta.loc[train_meta[col].notnull(), 'target'].mean()
            print(f"{col}: {non_null} non-nulls. Target mean for non-nulls: {non_null_target:.4f}")

if __name__ == "__main__":
    analyze_data()
