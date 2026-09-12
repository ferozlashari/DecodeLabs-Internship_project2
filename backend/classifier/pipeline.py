"""
pipeline.py

OUTPUT stage of the IPO framework: ties INPUT (data_loader) and
PROCESS (model) together into one runnable pipeline, and formats a
human-readable report of the results.
"""

from .config import TARGET_COLUMN
from .data_loader import load_dataset, dataset_summary
from .preprocessing import split_features_and_target
from .model import build_model_pipeline, split_train_test, evaluate_model


def format_report(results: dict) -> str:
    labels = results["labels"]
    cm = results["confusion_matrix"]

    lines = [
        "=" * 60,
        "PROJECT 2 — DATA CLASSIFICATION EVALUATION REPORT",
        "=" * 60,
        f"Accuracy:          {results['accuracy']:.4f}",
        f"F1 Score (weighted): {results['f1_weighted']:.4f}",
        "",
        "Confusion Matrix",
        "(rows = actual class, columns = predicted class)",
        "Labels: " + ", ".join(labels),
    ]
    for label, row in zip(labels, cm):
        lines.append(f"{label:>10}: {list(row)}")

    lines += [
        "",
        "Classification Report",
        results["classification_report"],
    ]
    return "\n".join(lines)


def run_pipeline(data_path=None, n_neighbors=None, test_size=None, verbose: bool = True) -> dict:
    df = load_dataset(data_path)
    if verbose:
        print(dataset_summary(df))

    X, y = split_features_and_target(df, TARGET_COLUMN)
    kwargs = {}
    if test_size is not None:
        kwargs["test_size"] = test_size
    X_train, X_test, y_train, y_test = split_train_test(X, y, **kwargs)

    model_kwargs = {}
    if n_neighbors is not None:
        model_kwargs["n_neighbors"] = n_neighbors
    model = build_model_pipeline(**model_kwargs)
    model.fit(X_train, y_train)

    results = evaluate_model(model, X_test, y_test)
    if verbose:
        print(format_report(results))

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results": results,
    }
