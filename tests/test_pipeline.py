"""Tests for the Telco churn preprocessing and inference pipeline."""

import numpy as np
import pandas as pd

from src.model import evaluate_model, train_model
from src.preprocessing import split_features_target, split_dataset
from src.utils import predict_csv


def _customer_rows(count: int, include_target: bool = True) -> pd.DataFrame:
    """Create a compact Telco-shaped dataset for deterministic pipeline tests."""
    rows = pd.DataFrame(
        {
            "customerID": [f"customer-{index}" for index in range(count)],
            "tenure": [index % 60 for index in range(count)],
            "MonthlyCharges": [35.0 + (index % 40) for index in range(count)],
            "TotalCharges": ["" if index % 11 == 0 else f"{index * 35.5:.2f}" for index in range(count)],
            "Contract": ["Month-to-month" if index % 2 else "One year" for index in range(count)],
            "InternetService": ["Fiber optic" if index % 3 else "DSL" for index in range(count)],
        }
    )
    if include_target:
        rows["Churn"] = ["Yes" if index % 3 == 0 else "No" for index in range(count)]
    return rows


def test_split_features_target_encodes_labels_and_removes_identifier() -> None:
    """Labeled data yields binary labels and excludes IDs from model features."""
    features, target = split_features_target(_customer_rows(12))

    assert "customerID" not in features.columns
    assert "Churn" not in features.columns
    assert set(target.unique()) == {0, 1}


def test_split_dataset_keeps_both_classes_in_train_and_test() -> None:
    """Stratified splitting retains both churn classes in each subset."""
    x_train, x_test, y_train, y_test = split_dataset(_customer_rows(40))

    assert len(x_train) == 32
    assert len(x_test) == 8
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_pipeline_scores_unseen_csv_with_unseen_categories(tmp_path) -> None:
    """A fitted pipeline predicts probabilities for a separate CSV with new categories."""
    training = _customer_rows(60)
    features, target = split_features_target(training)
    model = train_model(features, target, model_name="Logistic Regression")

    new_customers = pd.DataFrame(
        {
            "customerID": ["new-1", "new-2"],
            "tenure": [8, 55],
            "MonthlyCharges": [89.5, 42.0],
            "TotalCharges": ["9999.99", ""],
            "Contract": ["Two year", "Month-to-month"],
            "InternetService": ["Satellite", "DSL"],
        }
    )
    new_csv = tmp_path / "unseen_customers.csv"
    new_customers.to_csv(new_csv, index=False)

    predictions = predict_csv(model, new_csv)

    assert len(predictions) == 2
    assert predictions["Predicted_Churn"].isin(["Yes", "No"]).all()
    assert predictions["Churn_Probability"].between(0, 1).all()
    assert np.isfinite(predictions["Churn_Probability"]).all()


def test_evaluate_model_returns_expected_metrics() -> None:
    """Evaluation reports bounded classification metrics on held-out customers."""
    features, target = split_features_target(_customer_rows(60))
    model = train_model(features.iloc[:48], target.iloc[:48], model_name="Decision Tree")

    metrics = evaluate_model(model, features.iloc[48:], target.iloc[48:])

    assert {"accuracy", "precision", "recall", "f1", "roc_auc"}.issubset(metrics)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())