"""Data loading, feature preparation, and train/test splitting utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMNS = ("customerID",)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a Telco customer dataset from a CSV file."""
    return pd.read_csv(csv_path)


def prepare_features(data: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Remove the target and identifier fields from a customer feature table."""
    excluded = [target_column, *IDENTIFIER_COLUMNS]
    return data.drop(columns=excluded, errors="ignore").copy()


def encode_target(target: pd.Series) -> pd.Series:
    """Encode churn labels as 0 (No) and 1 (Yes), preserving numeric binary labels."""
    if target.isna().any():
        raise ValueError("Target column contains missing values.")

    values = set(target.unique())
    if values.issubset({0, 1, False, True}):
        return target.astype(int)

    normalized = target.astype(str).str.strip().str.lower()
    if set(normalized.unique()).issubset({"no", "yes"}):
        return normalized.map({"no": 0, "yes": 1}).astype(int)

    raise ValueError("Churn labels must be Yes/No or binary 0/1 values.")


def split_features_target(
    data: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features and encoded churn labels from a labeled dataset."""
    if target_column not in data.columns:
        raise ValueError(f"Required target column {target_column!r} is missing.")
    features = prepare_features(data, target_column=target_column)
    if features.empty or not len(features.columns):
        raise ValueError("No usable feature columns remain after removing identifiers and target.")
    return features, encode_target(data[target_column])


def split_dataset(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split labeled customer data while preserving the churn class ratio."""
    features, target = split_features_target(data, target_column=target_column)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def _add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Convert charges to numeric and add tenure-derived features when inputs exist."""
    result = data.copy()
    if "TotalCharges" in result.columns:
        result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
    if "tenure" in result.columns:
        result["tenure_group"] = pd.cut(
            result["tenure"],
            bins=[0, 12, 24, 48, np.inf],
            labels=["0-11", "12-23", "24-47", "48+"],
            right=False,
        ).astype(object)
    if "MonthlyCharges" in result.columns and "tenure" in result.columns:
        result["monthly_charge_per_tenure"] = (
            result["MonthlyCharges"] / result["tenure"].replace(0, 1)
        )
    return result


def build_preprocessor() -> Pipeline:
    """Build a fit-on-training-data transformer for numeric and categorical features."""
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(_add_engineered_features, validate=False)),
            ("encoding", column_transformer),
        ]
    )