"""Small helpers for working with customer CSV files and churn predictions."""

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import load_dataset, prepare_features


def predict_csv(model: Pipeline, csv_path: str | Path) -> pd.DataFrame:
    """Load an unlabeled customer CSV and append churn labels and probabilities."""
    customers = load_dataset(csv_path)
    features = prepare_features(customers)
    predictions = model.predict(features).astype(int)
    result = customers.copy()
    result["Predicted_Churn"] = pd.Series(predictions, index=result.index).map(
        {0: "No", 1: "Yes"}
    )
    if hasattr(model, "predict_proba"):
        result["Churn_Probability"] = model.predict_proba(features)[:, 1]
    return result