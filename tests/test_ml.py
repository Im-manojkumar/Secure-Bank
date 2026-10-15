import json

import numpy as np
import pandas as pd
import pytest

from ml.predict import FraudModel
from ml.datasets import LABEL
from ml.preprocess import load_data, split_data, split_with_test_file
from ml.sample_data import make_paysim_sample, make_sample, make_sparkov_sample
from ml.evaluate import best_f1_threshold
from ml.train import train


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    make_sample(rows=6000, fraud_rate=0.02).to_csv(path, index=False)
    return path


def test_load_data_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    make_sample(rows=100).drop(columns=["V3"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="V3"):
        load_data(path)


def test_load_data_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/raw"):
        load_data(tmp_path / "nope.csv")


def test_load_data_drops_duplicates(tmp_path):
    df = make_sample(rows=200)
    path = tmp_path / "dupes.csv"
    pd.concat([df, df.iloc[:10]]).to_csv(path, index=False)
    assert len(load_data(path)) == 200


def test_split_follows_time(sample_csv):
    df = load_data(sample_csv)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    # Every training row happened before every validation row, and those before every test row.
    assert X_train["Time"].max() < X_val["Time"].min()
    assert X_val["Time"].max() < X_test["Time"].min()
    assert abs(len(X_train) / len(df) - 0.6) < 0.01
    assert abs(len(X_test) / len(df) - 0.2) < 0.01


def test_time_split_keeps_rows_with_the_same_time_together():
    df = make_paysim_sample(rows=3000, random_state=3)
    df["step"] = np.repeat(np.arange(1, 11), 300)  # ten hours, 300 rows each
    df["isFraud"] = np.tile([0] * 29 + [1], 100)
    df = df.rename(columns={"isFraud": LABEL})
    X_train, X_val, X_test, *_ = split_data(df, dataset="paysim")
    steps = [set(df.loc[X.index, "step"]) for X in (X_train, X_val, X_test)]
    assert steps == [set(range(1, 7)), {7, 8}, {9, 10}]


def test_time_split_refuses_a_period_without_fraud():
    df = make_sample(rows=1000, fraud_rate=0).rename(columns={"Class": LABEL})
    df.loc[:100, LABEL] = 1  # fraud only in the earliest rows
    with pytest.raises(ValueError, match="validation period has no fraud"):
        split_data(df)


def test_sparkov_validation_is_the_latest_part_of_the_training_file():
    df = make_sparkov_sample(rows=4000, fraud_rate=0.05).rename(columns={"is_fraud": LABEL})
    test_df = make_sparkov_sample(rows=500, fraud_rate=0.05, random_state=1).rename(columns={"is_fraud": LABEL})
    X_train, X_val, X_test, *_ = split_with_test_file(df, test_df, dataset="sparkov")
    when = pd.to_datetime(df["trans_date_trans_time"])
    assert when[X_train.index].max() < when[X_val.index].min()
    assert len(X_test) == 500


def test_random_split_is_stratified_and_disjoint(sample_csv):
    df = load_data(sample_csv)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, by_time=False)
    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)
    rate = df["label"].mean()
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - rate) < 0.01


def test_best_f1_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_f1_threshold(y, scores)
    assert 0.3 < t <= 0.8


def test_train_saves_model_that_scores_new_rows(sample_csv, tmp_path):
    best_name, val_results, test_results = train(sample_csv, out_dir=tmp_path)
    assert set(val_results) == {"Logistic Regression", "Decision Tree", "Random Forest", "SVM (linear)", "Gradient Boosting"}
    for name in ("model.joblib", "model_meta.json", "metrics.json", "pr_curves.png", "confusion_matrix.png"):
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "model_meta.json").read_text())["split"] == "time"
    # The fake data has a clear fraud pattern, so the chosen model should find most of it.
    assert test_results[best_name]["recall"] > 0.7

    model = FraudModel(tmp_path)
    rows = make_sample(rows=50, random_state=7)
    scores, flagged = model.score(rows)
    assert len(scores) == 50 and flagged.dtype == bool
    with pytest.raises(ValueError, match="Missing columns"):
        model.score(rows.drop(columns=["Amount"]))
