"""Load and split the credit-card fraud dataset.

The dataset has the columns Time, V1..V28, Amount and Class (1 = fraud).
Splitting is stratified so every split keeps the same small fraud rate.
Scaling and resampling are NOT done here: they live inside the model
pipelines in train.py so they are fitted on training data only.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Class"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard.csv"


def load_data(path=DEFAULT_DATA_PATH):
    """Read the CSV, check its columns and drop duplicate rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put creditcard.csv in data/raw/ "
            "or pass --data with its location."
        )

    df = pd.read_csv(path)
    validate_columns(df)

    missing = int(df[FEATURES + [TARGET]].isna().sum().sum())
    if missing:
        raise ValueError(f"Dataset has {missing} missing values; clean them before training.")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {before:,} rows, dropped {before - len(df):,} duplicates, "
          f"{int(df[TARGET].sum()):,} fraud cases remain.")
    return df


def validate_columns(df):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    labels = set(df[TARGET].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"{TARGET} must contain only 0 and 1, found {sorted(labels)}")


def split_data(df, random_state=42):
    """Split 60% train, 20% validation, 20% test, stratified on Class.

    The validation set picks the best model and its alert threshold.
    The test set is touched only once, for the final report.
    """
    X, y = df[FEATURES], df[TARGET]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
