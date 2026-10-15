"""Train and compare five fraud models, then save the best one.

Usage:
    python -m ml.train                                    # data/raw/creditcard.csv
    python -m ml.train --data path/to/file.csv
    python -m ml.train --dataset sparkov --data data/raw/fraudTrain.csv --test-data data/raw/fraudTest.csv
    python -m ml.train --dataset paysim --data data/raw/paysim.csv --max-train-rows 1000000
    python -m ml.train --no-smote                         # use class weights instead of SMOTE
    python -m ml.train --random-split                     # old random split, for comparison

Each dataset format (see ml/datasets.py) gets its own model, saved under
ml/artifacts/<dataset>/. Each model is a pipeline of StandardScaler ->
SMOTE -> classifier. SMOTE inside an imblearn pipeline runs only while
fitting, so the validation and test sets are never resampled.

Transactions are split by time (see ml/preprocess.py): train on the
earliest, pick the model and threshold on the next period, and test on the
latest, the way a deployed model meets new data.

The best model is chosen by PR-AUC on the validation set. Its alert
threshold is the one that costs least on the validation set: each alert
costs a fixed review fee (--review-cost) and each missed fraud costs its
amount. --threshold-rule f1 picks the threshold with the best F1 instead.
The test set is scored once at the end for the report.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ml.datasets import DEFAULT_PROFILE, LABEL, PROFILES, get_profile
from ml.evaluate import (
    best_f1_threshold,
    min_cost_threshold,
    compute_metrics,
    format_table,
    plot_confusion_matrix,
    plot_pr_curves,
)
from ml.explain import baseline_values, fraud_scores
from ml.preprocess import DEFAULT_DATA_PATH, load_data, split_data, split_with_test_file

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
RANDOM_STATE = 42
# What one alert costs to review, in the dataset's currency: about 15 minutes of an
# analyst's time plus a call or message to the customer. An assumption, not a measured
# figure; a bank would use its own. Override with --review-cost.
REVIEW_COST = 10.0
THRESHOLD_RULES = ("cost", "f1")

__all__ = ["ARTIFACTS_DIR", "build_models", "build_pipeline", "fraud_scores", "train"]


def build_models(use_smote=True):
    """Return the five classifiers to compare, keyed by display name."""
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
        # A linear SVM scales to hundreds of thousands of rows; a kernel SVM would take hours.
        "SVM (linear)": LinearSVC(
            C=0.1, class_weight=weight, max_iter=5000, random_state=RANDOM_STATE
        ),
        # Boosted trees: each tree corrects the errors of the ones before it. Without the
        # L2 penalty, leaves that hold only SMOTE fraud rows get extreme values and
        # training can fall apart (validation PR-AUC 0.02 on PaySim); with it, 0.997.
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=False, class_weight=weight, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(classifier, use_smote=True):
    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", classifier))
    return Pipeline(steps)


def limit_training_rows(X, y, max_rows, random_state=RANDOM_STATE):
    """Keep every fraud row and a random sample of genuine rows, up to max_rows in total.

    Only the training split is cut down; validation and test keep their real mix.
    """
    if not max_rows or len(X) <= max_rows:
        return X, y
    fraud = y.index[y == 1]
    genuine = y.index[y == 0]
    keep_genuine = max(max_rows - len(fraud), 0)
    sampled = pd.Index(np.random.default_rng(random_state).choice(genuine, keep_genuine, replace=False))
    keep = fraud.append(sampled).sort_values()
    print(f"Training on {len(keep):,} of {len(X):,} training rows (all {len(fraud):,} fraud rows kept).")
    return X.loc[keep], y.loc[keep]


def add_reviewed_rows(profile, X_train, y_train, reviewed, held_out, history):
    """Append analyst-reviewed rows to the training set. Returns (X, y, counts).

    A reviewed row whose inputs match a validation or test row is dropped:
    training on it would let the model see the answers it is graded on.
    """
    fields = profile.field_names
    held = set(pd.util.hash_pandas_object(held_out[fields], index=False))
    overlap = pd.util.hash_pandas_object(reviewed[fields], index=False).isin(held).to_numpy()
    keep = reviewed[~overlap].reset_index(drop=True)
    counts = {"fraud": int((keep[LABEL] == 1).sum()), "legitimate": int((keep[LABEL] == 0).sum()),
              "dropped_as_validation_or_test": int(overlap.sum())}
    print(f"Adding {len(keep):,} analyst-reviewed rows to training ({counts['fraud']:,} fraud, "
          f"{counts['legitimate']:,} legitimate); {counts['dropped_as_validation_or_test']:,} skipped "
          "because they are validation or test rows.")
    if not len(keep):
        return X_train, y_train, counts
    X_extra = profile.features(keep, history=history)
    X_extra.index = pd.RangeIndex(len(keep)).map(lambda i: f"reviewed-{i}")
    y_extra = pd.Series(keep[LABEL].astype(int).to_numpy(), index=X_extra.index, name=y_train.name)
    return pd.concat([X_train, X_extra]), pd.concat([y_train, y_extra]), counts


def threshold_comparison(model_name, rule, chosen, other_rule, other):
    """Plain-text lines comparing the saved threshold with the other rule's, on the test set."""
    names = {"cost": "lowest cost", "f1": "best F1"}
    lines = [f"\nThreshold choice for {model_name} on the test set "
             f"(each alert costs {chosen['cost']['review_cost']:g} to review):"]
    for label, m in ((f"{names[rule]} (saved)", chosen), (names[other_rule], other)):
        c, cm = m["cost"], m["confusion_matrix"]
        lines.append(
            f"  {label:<20} threshold {m['threshold']:.4f}: {cm['tp']:,} fraud caught, {cm['fn']:,} missed, "
            f"{cm['fp']:,} false alerts; fraud missed {c['fraud_missed_amount']:,.0f} "
            f"+ reviews {c['review_cost'] * c['alerts']:,.0f} = total cost {c['total_cost']:,.0f}"
        )
    return "\n".join(lines)


def train(data_path=DEFAULT_DATA_PATH, use_smote=True, out_dir=None, dataset=DEFAULT_PROFILE,
          test_path=None, max_train_rows=None, by_time=True, threshold_rule="cost",
          review_cost=REVIEW_COST, reviewed=None):
    """Train, compare and save. Returns (best model name, validation results, test results).

    reviewed: cleaned rows with an analyst's verdict in LABEL (see
    services/feedback.py). They are added to the training set only, never
    to validation or test, and any that are also validation or test rows
    are dropped so the scores stay honest.
    """
    if threshold_rule not in THRESHOLD_RULES:
        raise ValueError(f"threshold_rule must be one of {', '.join(THRESHOLD_RULES)}")
    profile = get_profile(dataset)
    out_dir = Path(out_dir or ARTIFACTS_DIR / dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path, dataset)
    test_df = load_data(test_path, dataset) if test_path else None
    if test_path:
        X_train, X_val, X_test, y_train, y_val, y_test = split_with_test_file(
            df, test_df, dataset, RANDOM_STATE, by_time
        )
        held_out = pd.concat([df.loc[X_val.index], test_df])
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, dataset, RANDOM_STATE, by_time)
        held_out = df.loc[X_val.index.append(X_test.index)]
    print(f"Train {len(X_train):,} / validation {len(X_val):,} / test {len(X_test):,} rows")
    rows = {"train": len(X_train), "validation": len(X_val), "test": len(X_test)}
    X_train, y_train = limit_training_rows(X_train, y_train, max_train_rows)

    review_counts = None
    if reviewed is not None and len(reviewed):
        X_train, y_train, review_counts = add_reviewed_rows(profile, X_train, y_train, reviewed, held_out, df)
    del df, test_df, held_out

    val_amounts = X_val[profile.amount_feature].to_numpy()
    test_amounts = X_test[profile.amount_feature].to_numpy()

    fitted, val_results, test_results, val_scores, test_scores = {}, {}, {}, {}, {}
    for name, clf in build_models(use_smote).items():
        start = time.time()
        pipeline = build_pipeline(clf, use_smote).fit(X_train, y_train)
        val_scores[name] = fraud_scores(pipeline, X_val)
        if threshold_rule == "cost":
            threshold = min_cost_threshold(y_val, val_scores[name], val_amounts, review_cost)
        else:
            threshold = best_f1_threshold(y_val, val_scores[name])
        val_results[name] = compute_metrics(y_val, val_scores[name], threshold, val_amounts, review_cost)

        test_scores[name] = fraud_scores(pipeline, X_test)
        test_results[name] = compute_metrics(y_test, test_scores[name], threshold, test_amounts, review_cost)
        fitted[name] = pipeline
        print(f"  {name}: trained in {time.time() - start:.1f}s, "
              f"validation PR-AUC {val_results[name]['pr_auc']:.4f}")

    best_name = max(val_results, key=lambda n: val_results[n]["pr_auc"])
    best = test_results[best_name]
    # The other rule's threshold for the same model, to show what the choice changes.
    other_rule = "f1" if threshold_rule == "cost" else "cost"
    other_threshold = (best_f1_threshold(y_val, val_scores[best_name]) if other_rule == "f1"
                       else min_cost_threshold(y_val, val_scores[best_name], val_amounts, review_cost))
    other = compute_metrics(y_test, test_scores[best_name], other_threshold, test_amounts, review_cost)

    print("\nValidation set (used to pick the model and threshold)")
    print(format_table(val_results))
    print("\nTest set (final, untouched until now)")
    print(format_table(test_results))
    print(f"\nBest model: {best_name} (threshold {best['threshold']})")
    print(threshold_comparison(best_name, threshold_rule, best, other_rule, other))

    joblib.dump(fitted[best_name], out_dir / "model.joblib")
    metadata = {
        "dataset": dataset,
        "dataset_title": profile.title,
        "model_name": best_name,
        "threshold": best["threshold"],
        "score_type": "probability" if hasattr(fitted[best_name], "predict_proba") else "decision_function",
        "features": profile.feature_names,
        "smote": use_smote,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "time" if by_time else "random",
        "rows": rows | ({"train_used": len(X_train)} if len(X_train) != rows["train"] else {}),
        "reviewed_cases": review_counts,
        "threshold_rule": threshold_rule,
        "review_cost": review_cost,
        "selection": "highest validation PR-AUC; threshold "
                     + ("costs least on validation (missed fraud amount + review cost per alert)"
                        if threshold_rule == "cost" else "maximises validation F1"),
        "test_metrics": {k: best[k] for k in ("precision", "recall", "f1", "pr_auc")}
                        | {"total_cost": best["cost"]["total_cost"]},
        # A typical genuine transaction, the comparison point for explanations.
        "baseline": baseline_values(profile, X_train[y_train == 0]),
    }
    (out_dir / "model_meta.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "metrics.json").write_text(
        json.dumps({"split": metadata["split"], "threshold_rule": threshold_rule, "review_cost": review_cost,
                    "validation": val_results, "test": test_results, "best_model": best_name,
                    "other_threshold_rule": {"rule": other_rule, "test": other}}, indent=2)
    )
    report = (
        f"Dataset: {profile.title}\n"
        f"Split: {'by time (train on earliest, test on latest)' if by_time else 'random'}\n\n"
        "Validation set\n" + format_table(val_results)
        + "\n\nTest set\n" + format_table(test_results)
        + f"\n\nBest model: {best_name} (threshold {best['threshold']})\n"
        + threshold_comparison(best_name, threshold_rule, best, other_rule, other) + "\n"
    )
    (out_dir / "comparison.txt").write_text(report)
    plot_pr_curves(y_test, test_scores, out_dir / "pr_curves.png")
    plot_confusion_matrix(best, best_name, out_dir / "confusion_matrix.png")
    print(f"Saved model and reports to {out_dir}")

    return best_name, val_results, test_results


def main():
    parser = argparse.ArgumentParser(description="Train SecureBank fraud models.")
    parser.add_argument("--dataset", default=DEFAULT_PROFILE, choices=sorted(PROFILES),
                        help="Format of the data file (default: creditcard)")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="CSV to train on")
    parser.add_argument("--test-data", help="Separate CSV to test on (e.g. Sparkov's fraudTest.csv)")
    parser.add_argument("--max-train-rows", type=int,
                        help="Keep all fraud rows but sample genuine ones so training uses at most this many rows")
    parser.add_argument("--no-smote", action="store_true", help="Use class weights instead of SMOTE")
    parser.add_argument("--random-split", action="store_true",
                        help="Shuffle rows into the sets instead of splitting by time (overstates accuracy)")
    parser.add_argument("--threshold-rule", choices=THRESHOLD_RULES, default="cost",
                        help="How to pick the alert threshold: lowest cost (default) or best F1")
    parser.add_argument("--review-cost", type=float, default=REVIEW_COST,
                        help=f"Cost of reviewing one alert, in the dataset's currency (default {REVIEW_COST:g})")
    parser.add_argument("--out", help="Folder for the model and reports (default: ml/artifacts/<dataset>)")
    args = parser.parse_args()
    train(args.data, use_smote=not args.no_smote, out_dir=args.out, dataset=args.dataset,
          test_path=args.test_data, max_train_rows=args.max_train_rows, by_time=not args.random_split,
          threshold_rule=args.threshold_rule, review_cost=args.review_cost)


if __name__ == "__main__":
    main()
