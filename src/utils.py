"""
utils.py
Shared helper functions used across the preprocessing, training, and
prediction scripts for the Heart Disease Prediction project.
"""

import pandas as pd

CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC_COLS_FOR_OUTLIERS = ["RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
TARGET_COL = "HeartDisease"


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw heart disease CSV file."""
    return pd.read_csv(path)


def detect_outliers_iqr(data: pd.DataFrame, col: str):
    """
    Detect outliers in a numeric column using the IQR (Interquartile Range) method.

    Returns
    -------
    outliers : pd.DataFrame
        Rows considered outliers for the given column.
    lower : float
        Lower bound of the acceptable range.
    upper : float
        Upper bound of the acceptable range.
    """
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower) | (data[col] > upper)]
    return outliers, lower, upper


def evaluate_predictions(y_true, y_pred) -> dict:
    """Compute accuracy, precision, recall, and F1-score for a set of predictions."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
    }
