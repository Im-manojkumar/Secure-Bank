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


def min_cost_threshold(y_true, scores, amounts, review_cost):
    """Return the score cut-off that costs the bank least on these transactions.

    Each alert costs review_cost (an analyst's time and a call to the customer),
    and each fraud that is not flagged costs its full amount. Flagging more
    catches more fraud but costs more reviews; this picks the balance.
    """
    y_true, scores, amounts = (np.asarray(v, dtype=float) for v in (y_true, scores, amounts))
    order = np.argsort(-scores, kind="mergesort")
    s, fraud_amount = scores[order], (amounts * y_true)[order]
    # Flag the k highest scores, for k = 1..n: missed fraud plus the cost of k reviews.
    cost = fraud_amount.sum() - np.cumsum(fraud_amount) + review_cost * np.arange(1, len(s) + 1)
    # A cut-off can only fall between different scores, so ties are flagged together.
    ends = np.flatnonzero(np.r_[s[:-1] != s[1:], True])
    best = ends[np.argmin(cost[ends])]
    if cost[best] >= fraud_amount.sum():
        # Flagging nothing is cheapest: put the cut-off above every score.
        return float(np.nextafter(s[0], np.inf))
    return float(s[best])


def alert_costs(y_true, y_pred, amounts, review_cost):
    """What a threshold's alerts cost: fraud missed, reviews done, and the total."""
    y_true, y_pred, amounts = (np.asarray(v, dtype=float) for v in (y_true, y_pred, amounts))
    missed = float((amounts * y_true * (1 - y_pred)).sum())
    caught = float((amounts * y_true * y_pred).sum())
    alerts = int(y_pred.sum())
    return {
        "review_cost": review_cost,
        "alerts": alerts,
        "fraud_caught_amount": round(caught, 2),
        "fraud_missed_amount": round(missed, 2),
        "total_cost": round(missed + review_cost * alerts, 2),
    }


def compute_metrics(y_true, scores, threshold, amounts=None, review_cost=None):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    extra = {} if amounts is None else {"cost": alert_costs(y_true, y_pred, amounts, review_cost)}
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_true, scores), 4),
        "roc_auc": round(roc_auc_score(y_true, scores), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    } | extra


def format_table(results):
    """Plain-text comparison table, one row per model."""
    with_cost = all("cost" in m for m in results.values())
    header = f"{'Model':<22}{'Precision':>10}{'Recall':>9}{'F1':>8}{'PR-AUC':>9}{'ROC-AUC':>9}{'Missed':>8}{'False alerts':>14}"
    if with_cost:
        header += f"{'Total cost':>14}"
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        cm = m["confusion_matrix"]
        line = (
            f"{name:<22}{m['precision']:>10.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}"
            f"{m['pr_auc']:>9.4f}{m['roc_auc']:>9.4f}{cm['fn']:>8}{cm['fp']:>14}"
        )
        if with_cost:
            line += f"{m['cost']['total_cost']:>14,.0f}"
        lines.append(line)
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
