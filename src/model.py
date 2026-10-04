"""Model construction, training, and evaluation for churn classification."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor


def _model_options(random_state: int) -> dict[str, object]:
    """Create fresh classifier instances for the supported model names."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_estimators=200
        ),
    }


def build_model_pipeline(
    model_name: str, training_features: pd.DataFrame, random_state: int = 42
) -> Pipeline:
    """Build an end-to-end preprocessing and classifier pipeline."""
    options = _model_options(random_state)
    if model_name not in options:
        supported = ", ".join(options)
        raise ValueError(f"Unknown model {model_name!r}. Choose from: {supported}.")
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", options[model_name]),
        ]
    )


def train_model(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    model_name: str = "Random Forest",
    random_state: int = 42,
) -> Pipeline:
    """Fit and return a complete churn prediction pipeline."""
    pipeline = build_model_pipeline(model_name, training_features, random_state)
    return pipeline.fit(training_features, training_target)


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    """Calculate classification metrics, including ROC-AUC when probabilities exist."""
    predictions = model.predict(features)
    scores: dict[str, float] = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }
    if hasattr(model, "predict_proba") and len(np.unique(target)) == 2:
        probabilities = model.predict_proba(features)[:, 1]
        scores["roc_auc"] = float(roc_auc_score(target, probabilities))
    return scores