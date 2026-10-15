"""Spotting imports that no longer look like the model's training data."""

import numpy as np
import pandas as pd

from ml import drift
from ml.datasets import get_profile
from ml.sample_data import make_sample
from tests.conftest import csv_upload


def test_psi_is_near_zero_for_the_same_spread_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    profile = get_profile("creditcard")
    reference_rows = make_sample(rows=5000, random_state=1)
    reference = drift.build_reference(profile, profile.features(reference_rows), rng.random(5000))

    same = make_sample(rows=2000, random_state=2)
    result = drift.check(reference, profile.features(same), rng.random(2000))
    assert result["level"] == "stable" and drift.describe(result) == ""

    shifted = same.assign(Amount=same["Amount"] * 20)
    result = drift.check(reference, profile.features(shifted), rng.random(2000) ** 4)
    assert result["level"] == "major"
    assert result["most_shifted"][0]["input"] == "Amount"
    assert result["score_psi"] >= drift.MAJOR
    assert drift.describe(result).startswith("Warning:")


def test_drift_is_not_judged_on_a_handful_of_rows():
    profile = get_profile("creditcard")
    rows = profile.features(make_sample(rows=50))
    reference = drift.build_reference(profile, rows, np.zeros(50))
    assert drift.check(reference, rows, np.zeros(50)) is None


def test_drift_shows_on_the_import_and_the_dashboard(app, admin):
    normal = make_sample(rows=600, fraud_rate=0.03, random_state=21)
    source = admin.post("/api/transactions/import", data=csv_upload(normal, "normal.csv"),
                        content_type="multipart/form-data").get_json()["source"]
    assert source["drift"]["level"] == "stable" and "Warning" not in source["message"]
    assert admin.get("/api/dashboard/summary").get_json()["drift_warnings"] == []

    odd = make_sample(rows=600, fraud_rate=0.03, random_state=22)
    odd["Amount"] *= 50
    odd["V1"] += 4
    source = admin.post("/api/transactions/import", data=csv_upload(odd, "odd.csv"),
                        content_type="multipart/form-data").get_json()["source"]
    assert source["drift"]["level"] == "major"
    assert "look very different from the training data" in source["message"]

    warnings = admin.get("/api/dashboard/summary").get_json()["drift_warnings"]
    assert [(w["dataset"], w["source"], w["level"]) for w in warnings] == [("creditcard", "odd.csv", "major")]
    assert admin.get("/api/dashboard/summary?dataset=sparkov").get_json()["drift_warnings"] == []
