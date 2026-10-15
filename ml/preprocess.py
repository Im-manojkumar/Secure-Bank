"""Load and split a fraud dataset for training.

Which columns a file needs and how they become model inputs is described by
a profile in ml/datasets.py. Splitting follows time by default: the model
learns from earlier transactions and is tested on later ones. Scaling and resampling are NOT done here: they
live inside the model pipelines in train.py so they are fitted on training
data only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.datasets import DEFAULT_PROFILE, LABEL, get_profile

# The Kaggle credit-card columns, kept for code and tests that use them directly.
TARGET = "Class"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_DATA_PATH = DATA_DIR / "creditcard.csv"


def load_data(path=DEFAULT_DATA_PATH, dataset=DEFAULT_PROFILE):
    """Read a CSV, check and clean its columns, and drop duplicate rows.

    Returns the cleaned rows with the fraud label in the LABEL column.
    """
    profile = get_profile(dataset)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the CSV in data/raw/ or pass --data with its location."
        )

    header = pd.read_csv(path, nrows=0).columns
    wanted = profile.required_columns + [profile.target]
    missing = [c for c in wanted if c not in header]
    if missing:
        raise ValueError(f"Dataset is missing columns for the {profile.title} format: {', '.join(missing)}")
    # Only the columns the model needs: the PaySim file has 6 million rows.
    raw = pd.read_csv(path, usecols=wanted)

    df, invalid = profile.clean(raw)
    unlabelled = int(df[LABEL].isna().sum())
    df = df.dropna(subset=[LABEL])
    df[LABEL] = df[LABEL].astype(int)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"{path.name}: read {len(raw):,} rows, dropped {invalid + unlabelled:,} invalid and "
          f"{before - len(df):,} duplicate rows; {len(df):,} remain with {int(df[LABEL].sum()):,} fraud cases.")
    return df


def _time_cuts(order, fractions):
    """Boolean masks that cut rows into consecutive periods by time.

    fractions are the cumulative shares where each period ends, e.g. (0.6, 0.8).
    Rows with the same time always land in the same period, so a period can be
    a little bigger or smaller than its share.
    """
    ordered = np.sort(order.to_numpy())
    edges = [ordered[min(int(f * len(ordered)), len(ordered) - 1)] for f in fractions]
    masks, previous = [], None
    for edge in edges + [None]:
        mask = pd.Series(True, index=order.index)
        if previous is not None:
            mask &= order >= previous
        if edge is not None:
            mask &= order < edge
        masks.append(mask)
        previous = edge
    return masks


def _check_has_fraud(**splits):
    for name, y in splits.items():
        if len(y) == 0 or y.sum() == 0:
            raise ValueError(
                f"The {name} period has no fraud cases, so models cannot be compared on it. "
                "Use a larger file, or --random-split."
            )


def split_data(df, dataset=DEFAULT_PROFILE, random_state=42, by_time=True):
    """Split into train, validation and test sets: 60%, 20%, 20%.

    By default the split follows time: the model learns from the earliest 60%
    of transactions, the validation set (which picks the model and its alert
    threshold) is the next 20%, and the test set is the latest 20%. This is how
    a deployed model is used: trained on the past, judged on what comes next.

    by_time=False gives the older random split, stratified on the label. It
    mixes days across the sets, so the test set holds transactions from the
    same days the model learned from, and its scores come out too optimistic.

    The test set is touched only once, for the final report.
    """
    profile = get_profile(dataset)
    X, y = profile.features(df), df[LABEL]
    if by_time:
        train, val, test = _time_cuts(profile.time_order(df), (0.6, 0.8))
        _check_has_fraud(training=y[train], validation=y[val], test=y[test])
        return X[train], X[val], X[test], y[train], y[val], y[test]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_with_test_file(df, test_df, dataset=DEFAULT_PROFILE, random_state=42, by_time=True):
    """Use a separate file as the test set, and split the rest 75% train, 25% validation.

    The Sparkov dataset ships this way: fraudTest.csv covers the months after
    fraudTrain.csv, so the test checks the model on later transactions. By
    default the validation set is the latest 25% of the training file, so
    the threshold is also chosen on transactions after the ones learned from.
    """
    profile = get_profile(dataset)
    X, y = profile.features(df), df[LABEL]
    if by_time:
        train, val = _time_cuts(profile.time_order(df), (0.75,))
        _check_has_fraud(training=y[train], validation=y[val], test=test_df[LABEL])
        X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=random_state
        )
    # The training file's payments count as each card's history when scoring the test file.
    return X_train, X_val, profile.features(test_df, history=df), y_train, y_val, test_df[LABEL]
