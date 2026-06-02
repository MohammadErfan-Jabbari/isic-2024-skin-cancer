import pandas as pd
from pathlib import Path

DATA_DIR = Path('./data')

def check_cardinality():
    df = pd.read_csv(DATA_DIR / 'new-train-metadata.csv')
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print("Cardinality Check:")
    for col in cat_cols:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count}")

if __name__ == "__main__":
    check_cardinality()
