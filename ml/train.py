"""Train and compare four fraud models, then save the best one.

Usage:
    python -m ml.train                         # uses data/raw/creditcard.csv
    python -m ml.train --data path/to/file.csv
    python -m ml.train --no-smote              # use class weights instead of SMOTE

Each model is a pipeline of StandardScaler -> SMOTE -> classifier. SMOTE
inside an imblearn pipeline runs only while fitting, so the validation and
test sets are never resampled.

The best model is chosen by PR-AUC on the validation set, and its alert
threshold is the one that maximises F1 on the validation set. The test set
is scored once at the end for the report.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ml.evaluate import (
    best_f1_threshold,
    compute_metrics,
    format_table,
    plot_confusion_matrix,
    plot_pr_curves,
)
from ml.preprocess import DEFAULT_DATA_PATH, FEATURES, load_data, split_data

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
RANDOM_STATE = 42


def build_models(use_smote=True):
    """Return the four classifiers to compare, keyed by display name."""
    # Without SMOTE, class weights make the rare fraud class count more.
    weight = None if use_smote else "balanced"
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=weight, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight=weight, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1,
            class_weight=weight, random_state=RANDOM_STATE
        ),
        # A linear SVM scales to ~170k training rows; a kernel SVM would take hours.
        "SVM (linear)": LinearSVC(
            C=0.1, class_weight=weight, max_iter=5000, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(classifier, use_smote=True):
    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", classifier))
    return Pipeline(steps)


def fraud_scores(pipeline, X):
    """Higher score = more suspicious. Uses probabilities when the model has them."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def train(data_path=DEFAULT_DATA_PATH, use_smote=True, out_dir=ARTIFACTS_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, RANDOM_STATE)
    print(f"Train {len(X_train):,} / validation {len(X_val):,} / test {len(X_test):,} rows")

    fitted, val_results, test_results, test_scores = {}, {}, {}, {}
    for name, clf in build_models(use_smote).items():
        start = time.time()
        pipeline = build_pipeline(clf, use_smote).fit(X_train, y_train)
        val_scores = fraud_scores(pipeline, X_val)
        threshold = best_f1_threshold(y_val, val_scores)
        val_results[name] = compute_metrics(y_val, val_scores, threshold)

        test_scores[name] = fraud_scores(pipeline, X_test)
        test_results[name] = compute_metrics(y_test, test_scores[name], threshold)
        fitted[name] = pipeline
        print(f"  {name}: trained in {time.time() - start:.1f}s, "
              f"validation PR-AUC {val_results[name]['pr_auc']:.4f}")

    best_name = max(val_results, key=lambda n: val_results[n]["pr_auc"])
    best = test_results[best_name]

    print("\nValidation set (used to pick the model and threshold)")
    print(format_table(val_results))
    print("\nTest set (final, untouched until now)")
    print(format_table(test_results))
    print(f"\nBest model: {best_name} (threshold {best['threshold']})")

    joblib.dump(fitted[best_name], out_dir / "model.joblib")
    metadata = {
        "model_name": best_name,
        "threshold": best["threshold"],
        "score_type": "probability" if hasattr(fitted[best_name], "predict_proba") else "decision_function",
        "features": FEATURES,
        "smote": use_smote,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "selection": "highest validation PR-AUC; threshold maximises validation F1",
    }
    (out_dir / "model_meta.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "metrics.json").write_text(
        json.dumps({"validation": val_results, "test": test_results, "best_model": best_name}, indent=2)
    )
    report = (
        "Validation set\n" + format_table(val_results)
        + "\n\nTest set\n" + format_table(test_results)
        + f"\n\nBest model: {best_name} (threshold {best['threshold']})\n"
    )
    (out_dir / "comparison.txt").write_text(report)
    plot_pr_curves(y_test, test_scores, out_dir / "pr_curves.png")
    plot_confusion_matrix(best, best_name, out_dir / "confusion_matrix.png")
    print(f"Saved model and reports to {out_dir}")

    return best_name, val_results, test_results


def main():
    parser = argparse.ArgumentParser(description="Train SecureBank fraud models.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to creditcard.csv")
    parser.add_argument("--no-smote", action="store_true", help="Use class weights instead of SMOTE")
    parser.add_argument("--out", default=ARTIFACTS_DIR, help="Folder for the model and reports")
    args = parser.parse_args()
    train(args.data, use_smote=not args.no_smote, out_dir=args.out)


if __name__ == "__main__":
    main()
