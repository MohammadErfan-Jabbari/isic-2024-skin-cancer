import pandas as pd
from pathlib import Path

DATA_DIR = Path('./data')

def check_mismatch():
    print("Loading metadata...")
    train_df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv', nrows=1)
    test_df = pd.read_csv(DATA_DIR / 'students-test-metadata.csv', nrows=1)
    
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    print(f"\nTrain Columns: {len(train_cols)}")
    print(f"Test Columns: {len(test_cols)}")
    
    # 1. Missing in Test (Risk of Training on unavailable features)
    missing_in_test = train_cols - test_cols
    print("\n1. Columns in Train but MISSING in Test (CRITICAL):")
    for c in sorted(missing_in_test):
        print(f"  - {c}")
        
    # 2. Missing in Train (New features in Test?)
    missing_in_train = test_cols - train_cols
    print("\n2. Columns in Test but MISSING in Train:")
    for c in sorted(missing_in_train):
        print(f"  - {c}")

if __name__ == "__main__":
    check_mismatch()
