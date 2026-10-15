"""Learning from analysts: resolved cases become training rows for the next model."""

import json
import shutil

import pandas as pd

from ml.datasets import LABEL
from ml.preprocess import load_data
from ml.sample_data import make_sample
from ml.train import train
from models import AuditLog, db
from services.feedback import reviewed_cases


def resolve_cases(client, outcomes):
    cases = client.get("/api/alerts?per_page=10").get_json()["items"]
    for case, outcome in zip(cases, outcomes):
        response = client.patch(f"/api/cases/{case['id']}", json={"status": "Resolved", "outcome": outcome})
        assert response.status_code == 200
    return cases


def test_resolved_cases_become_labelled_rows(app, admin, imported):
    resolve_cases(admin, ["Confirmed fraud", "Legitimate"])
    with app.app_context():
        rows = reviewed_cases("creditcard")
        assert sorted(rows[LABEL]) == [0, 1]
        assert {"Time", "V1", "Amount"} <= set(rows.columns)
        assert len(reviewed_cases("sparkov")) == 0  # other formats are not mixed in


def test_reviewed_rows_join_training_but_never_validation_or_test(tmp_path):
    path = tmp_path / "data.csv"
    make_sample(rows=3000, fraud_rate=0.03).to_csv(path, index=False)
    df = load_data(path)
    new = make_sample(rows=20, fraud_rate=0.5, random_state=5).rename(columns={"Class": LABEL})
    latest = df.sort_values("Time").tail(5)  # test rows: the latest 20% by time
    reviewed = pd.concat([new, latest], ignore_index=True)

    train(path, out_dir=tmp_path / "model", reviewed=reviewed)
    meta = json.loads((tmp_path / "model" / "model_meta.json").read_text())
    counts = meta["reviewed_cases"]
    assert counts["dropped_as_validation_or_test"] == 5
    assert counts["fraud"] + counts["legitimate"] == 20
    assert meta["rows"]["train_used"] == meta["rows"]["train"] + 20


def test_retrain_command_keeps_the_old_model_and_logs_it(app, admin, imported, model_dir, tmp_path):
    resolve_cases(admin, ["Confirmed fraud", "Legitimate", "Confirmed fraud"])
    models = tmp_path / "models"
    shutil.copytree(model_dir / "creditcard", models / "creditcard")
    app.config["MODEL_DIR"] = models
    data = tmp_path / "base.csv"
    make_sample(rows=2000, fraud_rate=0.03, random_state=2).to_csv(data, index=False)
    old_stamp = json.loads((models / "creditcard" / "model_meta.json").read_text())["trained_at"]

    result = app.test_cli_runner().invoke(args=["retrain", "--dataset", "creditcard", "--data", str(data)])
    assert result.exit_code == 0, result.output
    assert "3 resolved cases" in result.output

    backup = models / "creditcard" / "previous" / old_stamp.replace(":", "-")
    assert (backup / "model.joblib").exists() and (backup / "model_meta.json").exists()
    meta = json.loads((models / "creditcard" / "model_meta.json").read_text())
    assert meta["reviewed_cases"]["fraud"] + meta["reviewed_cases"]["legitimate"] == 3
    with app.app_context():
        entry = db.session.query(AuditLog).filter_by(action="model_retrained").one()
        assert entry.details["reviewed_cases"] == 3


def test_retrain_rejects_an_unknown_format(app, tmp_path):
    data = tmp_path / "x.csv"
    data.write_text("a\n1\n")
    result = app.test_cli_runner().invoke(args=["retrain", "--dataset", "bank", "--data", str(data)])
    assert result.exit_code != 0 and "Unknown dataset" in result.output
