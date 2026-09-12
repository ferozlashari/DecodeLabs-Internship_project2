"""
main.py — CLI entry point for Project 2 (no frontend needed).

Run with:
    python main.py
    python main.py --k 9
"""

import argparse
import os

from classifier.pipeline import run_pipeline, format_report
from classifier.config import REPORT_OUTPUT_PATH, MODEL_OUTPUT_PATH, CONFUSION_MATRIX_PATH, N_NEIGHBORS


def parse_args():
    parser = argparse.ArgumentParser(description="OrderStatus classification pipeline")
    parser.add_argument("--k", type=int, default=N_NEIGHBORS, help="Number of neighbors for KNN")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving the confusion matrix image")
    return parser.parse_args()


def save_confusion_matrix_plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = results["confusion_matrix"]
    labels = results["labels"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — OrderStatus")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    outcome = run_pipeline(n_neighbors=args.k)
    results = outcome["results"]

    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(format_report(results))
    print(f"\nSaved report to {REPORT_OUTPUT_PATH}")

    if not args.no_plot:
        save_confusion_matrix_plot(results, CONFUSION_MATRIX_PATH)
        print(f"Saved confusion matrix plot to {CONFUSION_MATRIX_PATH}")

    import joblib
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(outcome["model"], MODEL_OUTPUT_PATH)
    print(f"Saved trained model to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
