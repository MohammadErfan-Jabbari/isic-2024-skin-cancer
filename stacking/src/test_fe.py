import pandas as pd
import numpy as np
from pathlib import Path
from feature_engineering import engineer_features

DATA_DIR = Path('./data')

def verify_fe():
    print("--- Verifying Feature Engineering ---")
    
    # Load small sample
    print("Loading sample data...")
    df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', nrows=1000, low_memory=False)
    
    print(f"Original Shape: {df.shape}")
    
    # Apply FE
    print("Applying Feature Engineering...")
    df_eng = engineer_features(df)
    
    print(f"Engineered Shape: {df_eng.shape}")
    
    # Check new columns
    new_cols = set(df_eng.columns) - set(df.columns)
    print(f"New Features Added ({len(new_cols)}):")
    for c in sorted(list(new_cols)):
        print(f"  - {c}")
        
    # Check for NaNs/Infs in new columns
    print("\nChecking for NaNs/Infs in new features...")
    for col in new_cols:
        n_nans = df_eng[col].isna().sum()
        n_infs = np.isinf(df_eng[col]).sum() if pd.api.types.is_numeric_dtype(df_eng[col]) else 0
        
        if n_nans > 0 or n_infs > 0:
            print(f"  ⚠️ {col}: NaNs={n_nans}, Infs={n_infs}")
            # Show sample
            if n_nans > 0:
                print(f"    Sample NaNs:\n{df_eng[df_eng[col].isna()][col].head(3)}")
        else:
            pass # print(f"  ✅ {col}: OK")
            
    print("\nData Types:")
    print(df_eng[list(new_cols)].dtypes)
    
    print("\n✅ Verification Complete")

if __name__ == "__main__":
    verify_fe()
