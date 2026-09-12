"""
model.py

PROCESS stage of the IPO framework: build the full model pipeline
(preprocessing + KNN classifier), train it, and evaluate it.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from .config import N_NEIGHBORS, TEST_SIZE, RANDOM_STATE
from .preprocessing import build_preprocessor


def build_model_pipeline(n_neighbors: int = N_NEIGHBORS) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )


def split_train_test(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def evaluate_model(model: Pipeline, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_weighted": f1_score(y_test, predictions, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=model.classes_),
        "labels": list(model.classes_),
        "classification_report": classification_report(y_test, predictions),
        "predictions": predictions,
    }
