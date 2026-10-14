"""Metrics, threshold selection and charts for the fraud models."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true, scores):
    """Return the score cut-off that gives the highest F1 on these labels."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall pair has no threshold.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def compute_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_true, scores), 4),
        "roc_auc": round(roc_auc_score(y_true, scores), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def format_table(results):
    """Plain-text comparison table, one row per model."""
    header = f"{'Model':<22}{'Precision':>10}{'Recall':>9}{'F1':>8}{'PR-AUC':>9}{'ROC-AUC':>9}{'Missed':>8}{'False alerts':>14}"
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        cm = m["confusion_matrix"]
        lines.append(
            f"{name:<22}{m['precision']:>10.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}"
            f"{m['pr_auc']:>9.4f}{m['roc_auc']:>9.4f}{cm['fn']:>8}{cm['fp']:>14}"
        )
    return "\n".join(lines)


def plot_pr_curves(y_true, scores_by_model, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, label=f"{name} (PR-AUC {ap:.3f})")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Precision-recall curves on the test set")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(metrics, model_name, out_path):
    cm = metrics["confusion_matrix"]
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(grid, cmap="Blues")
    for (i, j), value in np.ndenumerate(grid):
        ax.text(j, i, f"{value:,}", ha="center", va="center",
                color="white" if value > grid.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} (threshold {metrics['threshold']:.3f})")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
