"""Load and split a fraud dataset for training.

Which columns a file needs and how they become model inputs is described by
a profile in ml/datasets.py. Splitting is stratified so every split keeps
the same small fraud rate. Scaling and resampling are NOT done here: they
live inside the model pipelines in train.py so they are fitted on training
data only.
"""

from pathlib import Path

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


def split_data(df, dataset=DEFAULT_PROFILE, random_state=42):
    """Split 60% train, 20% validation, 20% test, stratified on the label.

    The validation set picks the best model and its alert threshold.
    The test set is touched only once, for the final report.
    """
    X, y = get_profile(dataset).features(df), df[LABEL]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_with_test_file(df, test_df, dataset=DEFAULT_PROFILE, random_state=42):
    """Use a separate file as the test set, and split the rest 75% train, 25% validation.

    The Sparkov dataset ships this way: fraudTest.csv covers the months after
    fraudTrain.csv, so the test checks the model on later transactions.
    """
    profile = get_profile(dataset)
    X, y = profile.features(df), df[LABEL]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    return X_train, X_val, profile.features(test_df), y_train, y_val, test_df[LABEL]
