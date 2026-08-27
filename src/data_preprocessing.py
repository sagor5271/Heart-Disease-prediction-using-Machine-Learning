"""
data_preprocessing.py
Cleans the raw heart disease dataset and saves a processed, model-ready
CSV file to data/processed/.

Usage
-----
    python src/data_preprocessing.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import CATEGORICAL_COLS, NUMERIC_COLS_FOR_OUTLIERS, detect_outliers_iqr, load_raw_data

RAW_PATH = os.path.join("data", "raw", "heart.csv")
PROCESSED_PATH = os.path.join("data", "processed", "heart_processed.csv")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and invalid zero entries."""
    df = df.drop_duplicates()

    # Cholesterol = 0 and RestingBP = 0 are physiologically impossible -> treat as missing
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)

    # Impute with the median (robust to skew/outliers)
    df["Cholesterol"] = df["Cholesterol"].fillna(df["Cholesterol"].median())
    df["RestingBP"] = df["RestingBP"].fillna(df["RestingBP"].median())

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns."""
    df_encoded = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers (IQR method) from RestingBP and Cholesterol."""
    df_clean = df.copy()
    for col in ["RestingBP", "Cholesterol"]:
        _, lower, upper = detect_outliers_iqr(df_clean, col)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean


def run_pipeline(raw_path: str = RAW_PATH, processed_path: str = PROCESSED_PATH) -> pd.DataFrame:
    df = load_raw_data(raw_path)
    print(f"Loaded raw data: {df.shape}")

    df = clean_data(df)
    print(f"After cleaning (duplicates removed, missing values imputed): {df.shape}")

    df = encode_categoricals(df)
    print("Categorical columns encoded:", CATEGORICAL_COLS)

    df = remove_outliers(df)
    print(f"After outlier removal: {df.shape}")

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to: {processed_path}")

    return df


if __name__ == "__main__":
    run_pipeline()
