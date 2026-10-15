"""The Sparkov and PaySim formats, explanations, and the app's handling of several models."""

import pandas as pd
import pytest

from ml.datasets import LABEL, card_history, card_ids, detect_profile, get_profile
from ml.predict import FraudModel
from ml.sample_data import make_paysim_sample, make_sample, make_sparkov_sample
from models import DataSource, Transaction, db
from tests.conftest import csv_upload


# --- Formats ---------------------------------------------------------------------

@pytest.mark.parametrize("maker, key", [
    (make_sample, "creditcard"), (make_sparkov_sample, "sparkov"), (make_paysim_sample, "paysim"),
])
def test_formats_are_recognised_from_their_columns(maker, key):
    profile, missing = detect_profile(maker(rows=5).columns)
    assert profile.key == key and missing == []


def test_sparkov_cleaning_drops_bad_rows_and_personal_data():
    df = make_sparkov_sample(rows=20).astype(object)
    df.loc[0, "amt"] = -1
    df.loc[1, "category"] = "casino"
    df.loc[2, "trans_date_trans_time"] = "not a date"
    df.loc[3, "is_fraud"] = 3
    clean, invalid = get_profile("sparkov").clean(df)
    assert invalid == 4 and len(clean) == 16
    for private in ("first", "last", "street", "job", "cc_num", "zip", "gender"):
        assert private not in clean.columns
    assert clean["card_last4"].str.len().eq(4).all()
    assert not clean["merchant"].str.startswith("fraud_").any()


def test_sparkov_features_are_readable():
    profile = get_profile("sparkov")
    clean, _ = profile.clean(make_sparkov_sample(rows=50))
    X = profile.features(clean)
    assert list(X.columns) == profile.feature_names
    assert X["hour"].between(0, 23).all() and (X["age"] > 0).all() and (X["distance_km"] >= 0).all()
    # Exactly one merchant category is set per row.
    assert X.filter(like="cat_").sum(axis=1).eq(1).all()


def test_paysim_features_include_balance_mismatches():
    profile = get_profile("paysim")
    row = {"step": 30, "type": "TRANSFER", "amount": 500.0, "oldbalanceOrg": 500.0, "newbalanceOrig": 0.0,
           "oldbalanceDest": 0.0, "newbalanceDest": 0.0}
    X = profile.features(pd.DataFrame([row]))
    assert X.loc[0, "empties_account"] == 1
    assert X.loc[0, "dest_balance_error"] == 500.0
    assert X.loc[0, "hour"] == 6
    assert X.loc[0, "type_TRANSFER"] == 1


def test_card_history_counts_only_earlier_payments_of_the_same_card():
    df = pd.DataFrame({
        "card_id": ["a", "a", "a", "b", "a"],
        "trans_date_trans_time": ["2020-01-01 10:00:00", "2020-01-01 10:30:00", "2020-01-01 12:00:00",
                                  "2020-01-01 10:10:00", "2020-01-03 12:00:00"],
        "amt": [10.0, 20.0, 90.0, 5.0, 30.0],
    })
    h = card_history(df)
    assert h["card_txn_1h"].tolist() == [0, 1, 0, 0, 0]
    assert h["card_txn_24h"].tolist() == [0, 1, 2, 0, 0]
    assert h["hours_since_last"].tolist() == [720, 0.5, 1.5, 720, 48]
    assert h.loc[2, "amount_vs_card_avg"] == 6.0  # 90 against an average of 15
    assert h.loc[3, "amount_vs_card_avg"] == 1.0  # first payment of card b


def test_card_history_uses_earlier_payments_that_are_not_scored():
    row = pd.DataFrame({"card_id": ["a"], "trans_date_trans_time": ["2020-01-01 12:00:00"], "amt": [90.0]})
    past = pd.DataFrame({
        "card_id": ["a", "a", "z"],
        "trans_date_trans_time": ["2020-01-01 11:50:00", "2020-01-02 09:00:00", "2020-01-01 11:55:00"],
        "amt": [30.0, 999.0, 1.0],
    })
    h = card_history(row, past)
    # Card z and card a's later payment do not count.
    assert h.loc[0, "card_txn_1h"] == 1 and h.loc[0, "amount_vs_card_avg"] == 3.0
    assert abs(h.loc[0, "hours_since_last"] - 1 / 6) < 1e-9


def test_card_ids_are_stable_and_hide_the_number():
    ids = card_ids(pd.Series(["4111111111111111", " 4111111111111111", "5500000000000004"]))
    assert ids[0] == ids[1] != ids[2]
    assert len(ids[0]) == 16 and "4111" not in ids[0]


def test_each_format_trains_its_own_model(model_dir):
    for key in ("creditcard", "sparkov", "paysim"):
        model = FraudModel(model_dir / key)
        assert model.meta["dataset"] == key
        assert set(model.meta["baseline"]) == set(model.profile.feature_names)


def test_explanations_name_the_planted_pattern(model_dir):
    """The fake Sparkov fraud is large late-night purchases, so those inputs should top the reasons."""
    model = FraudModel(model_dir / "sparkov")
    clean, _ = model.profile.clean(make_sparkov_sample(rows=400, fraud_rate=0.5, random_state=9))
    fraud = clean[clean[LABEL] == 1]
    scores, flagged = model.score(fraud)
    reasons = model.explain(fraud[flagged])
    assert reasons and all(len(r) <= 3 for r in reasons)
    top = [r[0]["feature"] for r in reasons if r]
    assert top.count("Amount") + top.count("Time of day") + top.count("Merchant category") > len(top) / 2
    first = next(r for r in reasons if r)[0]
    assert set(first) == {"feature", "value", "typical", "effect"} and first["effect"] > 0


# --- App ----------------------------------------------------------------------

def import_csv(client, df, name):
    return client.post("/api/transactions/import", data=csv_upload(df, name), content_type="multipart/form-data")


@pytest.mark.parametrize("maker, key, currency", [
    (make_sparkov_sample, "sparkov", "USD"), (make_paysim_sample, "paysim", None),
])
def test_import_scores_each_format_with_its_model(app, admin, maker, key, currency):
    response = import_csv(admin, maker(rows=300, fraud_rate=0.1, random_state=4), f"{key}.csv")
    assert response.status_code == 201, response.get_json()
    source = response.get_json()["source"]
    assert source["dataset"] == key and source["rows_imported"] == 300

    items = admin.get(f"/api/transactions?dataset={key}&prediction=suspicious&per_page=100").get_json()["items"]
    assert items and all(t["dataset"] == key and t["currency"] == currency for t in items)
    detail = admin.get(f"/api/transactions/{items[0]['id']}").get_json()["transaction"]
    assert set(get_profile(key).field_names) <= set(detail["features"])
    assert "cc_num" not in detail["features"]  # card numbers are never stored
    assert detail["prediction"]["explanation"], "suspicious rows store why they were flagged"

    alerts = admin.get("/api/alerts?per_page=5").get_json()["items"]
    assert "explanation" in alerts[0]["transaction"]

    # Re-importing the same file adds nothing.
    again = import_csv(admin, maker(rows=300, fraud_rate=0.1, random_state=4), f"{key}.csv").get_json()["source"]
    assert again["rows_imported"] == 0


def test_import_compares_payments_with_the_card_history_already_stored(app, admin):
    from services.prediction import card_history as stored_history

    sample = make_sparkov_sample(rows=300, fraud_rate=0.05, random_state=11)
    sample["trans_date_trans_time"] = pd.to_datetime(sample["trans_date_trans_time"])
    sample = sample.sort_values("trans_date_trans_time")
    sample["trans_date_trans_time"] = sample["trans_date_trans_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    first, second = sample.iloc[:200], sample.iloc[200:]
    assert import_csv(admin, first, "first.csv").status_code == 201

    with app.app_context():
        cards = card_ids(second["cc_num"]).tolist()
        history = stored_history(cards)
        assert len(history) and set(history["card_id"]) <= set(cards)
        assert db.session.query(Transaction).filter(Transaction.card_id.is_(None)).count() == 0

    assert import_csv(admin, second, "second.csv").status_code == 201
    with app.app_context():
        assert db.session.query(Transaction).count() == 300


def test_low_risk_rows_store_no_explanation(app, admin):
    import_csv(admin, make_paysim_sample(rows=200, fraud_rate=0.05, random_state=8), "p.csv")
    items = admin.get("/api/transactions?prediction=low_risk&per_page=5").get_json()["items"]
    assert items and all(t["prediction"]["explanation"] is None for t in items)


def test_import_rejects_unknown_format(admin):
    df = pd.DataFrame({"date": ["2024-01-01"], "value": [5]})
    response = import_csv(admin, df, "other.csv")
    assert response.status_code == 400
    assert "do not match any supported format" in response.get_json()["error"]


def test_import_names_missing_columns_of_the_closest_format(admin):
    response = import_csv(admin, make_sparkov_sample(rows=5).drop(columns=["dob"]), "s.csv")
    assert response.status_code == 400
    assert "Sparkov" in response.get_json()["error"] and "dob" in response.get_json()["error"]


def test_import_without_a_trained_model_is_refused(app, admin, tmp_path):
    app.config["MODEL_DIR"] = tmp_path  # no models here
    app.extensions.pop("fraud_models", None)
    response = import_csv(admin, make_paysim_sample(rows=5), "p.csv")
    assert response.status_code == 503
    assert "python -m ml.train --dataset paysim" in response.get_json()["error"]


def test_models_endpoint_lists_formats_and_models(analyst):
    items = {m["key"]: m for m in analyst.get("/api/models").get_json()["items"]}
    assert set(items) == {"creditcard", "sparkov", "paysim"}
    assert items["sparkov"]["model"]["threshold"] is not None
    assert [f["name"] for f in items["paysim"]["fields"]][:2] == ["step", "type"]
    category = next(f for f in items["sparkov"]["fields"] if f["name"] == "category")
    assert {"value": "shopping_net", "label": "Shopping (online)"} in category["options"]


def test_predict_with_a_dataset_explains_the_score(analyst):
    row = {"dataset": "sparkov", "trans_date_trans_time": "2020-07-01T23:30:00", "category": "shopping_net",
           "amt": 950.0, "dob": "1960-05-01", "city_pop": 5000, "lat": 40.0, "long": -100.0,
           "merch_lat": 40.3, "merch_long": -100.2}
    body = analyst.post("/api/predict", json=row).get_json()
    assert body["dataset"] == "sparkov" and body["result"] in ("Suspicious", "Low risk")
    assert isinstance(body["explanation"], list)


def test_predict_validates_each_format(analyst):
    body = analyst.post("/api/predict", json={"dataset": "paysim", "step": 1, "type": "WIRE", "amount": -3}).get_json()
    details = body["details"]
    assert "type must be one of: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER" in details
    assert "amount cannot be negative" in details
    assert "oldbalanceOrg must be a number" in details
    assert analyst.post("/api/predict", json={"dataset": "nope"}).status_code == 400
    bad_date = {"dataset": "sparkov", "dob": "yesterday"}
    assert "dob must be a date" in analyst.post("/api/predict", json=bad_date).get_json()["details"]


def test_transaction_filter_rejects_unknown_dataset(admin):
    assert admin.get("/api/transactions?dataset=bank").status_code == 400


def test_dashboard_totals_flagged_amounts_per_currency(app, admin):
    import_csv(admin, make_sample(rows=500, fraud_rate=0.1, random_state=2), "c.csv")
    import_csv(admin, make_sparkov_sample(rows=500, fraud_rate=0.1, random_state=2), "s.csv")
    amounts = admin.get("/api/dashboard/summary").get_json()["predicted_fraud_amounts"]
    assert {a["currency"] for a in amounts} == {"EUR", "USD"}
    with app.app_context():
        expected = sum(
            t.amount for t in db.session.query(Transaction).join(DataSource).filter(DataSource.dataset == "sparkov")
            if t.prediction.predicted_label == 1
        )
    assert next(a["amount"] for a in amounts if a["currency"] == "USD") == pytest.approx(expected, abs=0.01)


def test_dashboard_and_alerts_filter_by_dataset(app, admin):
    import_csv(admin, make_sample(rows=500, fraud_rate=0.1, random_state=3), "c.csv")
    import_csv(admin, make_sparkov_sample(rows=500, fraud_rate=0.1, random_state=3), "s.csv")
    everything = admin.get("/api/dashboard/summary").get_json()
    sparkov = admin.get("/api/dashboard/summary?dataset=sparkov").get_json()
    card = admin.get("/api/dashboard/summary?dataset=creditcard&granularity=hour").get_json()

    assert sparkov["total_transactions"] == card["total_transactions"] == 500
    assert sparkov["predicted_fraud"] + card["predicted_fraud"] == everything["predicted_fraud"]
    assert [a["currency"] for a in sparkov["predicted_fraud_amounts"]] == ["USD"]
    assert sum(sparkov["cases_by_status"].values()) == sparkov["predicted_fraud"]
    assert sum(p["transactions"] for p in sparkov["trend"]) == 500
    assert all(len(p["period"]) == 16 for p in card["trend"]), "hourly buckets look like 2013-09-01 14:00"

    alerts = admin.get("/api/alerts?dataset=sparkov&per_page=100").get_json()
    assert alerts["total"] == sparkov["predicted_fraud"]
    assert all(a["transaction"]["dataset"] == "sparkov" for a in alerts["items"])

    assert admin.get("/api/dashboard/summary?dataset=bank").status_code == 400
    assert admin.get("/api/alerts?dataset=bank").status_code == 400
