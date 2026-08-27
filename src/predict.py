"""
predict.py
Loads the saved model and scaler to make predictions on new patient data.

Usage
-----
    python src/predict.py --input sample_input.csv
    python src/predict.py --input sample_input.csv --output predictions.csv

The input CSV must contain the same raw columns as heart.csv (before
encoding), e.g.:
    Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope
"""

import argparse
import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import CATEGORICAL_COLS

MODELS_DIR = "models"


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    return model, scaler, feature_columns


def preprocess_input(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Encode categorical columns and align column order with training data."""
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    df = df[feature_columns]
    return df


def predict(input_path: str, output_path: str = None) -> pd.DataFrame:
    model, scaler, feature_columns = load_artifacts()

    raw_df = pd.read_csv(input_path)
    processed_df = preprocess_input(raw_df, feature_columns)
    scaled_df = pd.DataFrame(scaler.transform(processed_df), columns=feature_columns, index=processed_df.index)

    predictions = model.predict(scaled_df)
    probabilities = model.predict_proba(scaled_df)[:, 1]

    result = raw_df.copy()
    result["Predicted_HeartDisease"] = predictions
    result["Probability"] = probabilities.round(4)

    print(result[["Predicted_HeartDisease", "Probability"]])

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict heart disease risk from patient data.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default=None, help="Path to save predictions CSV (optional)")
    args = parser.parse_args()

    predict(args.input, args.output)
