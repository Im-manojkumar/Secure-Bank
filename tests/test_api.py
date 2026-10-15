from ml.preprocess import FEATURES
from ml.sample_data import make_sample
from models import FraudCase, Prediction, Transaction, db
from tests.conftest import csv_upload, login


# --- Authentication and roles -------------------------------------------------

def test_endpoints_require_login(client):
    for path in ("/api/transactions", "/api/alerts", "/api/dashboard/summary", "/api/auth/me"):
        response = client.get(path)
        assert response.status_code == 401
        assert response.get_json()["error"] == "Please log in first."


def test_login_rejects_wrong_password_with_generic_message(client):
    response = client.post("/api/auth/login", json={"email": "admin@test.local", "password": "nope"})
    assert response.status_code == 401
    unknown = client.post("/api/auth/login", json={"email": "who@test.local", "password": "nope"})
    assert unknown.get_json() == response.get_json()


def test_login_me_and_logout(client):
    login(client, "analyst")
    assert client.get("/api/auth/me").get_json()["user"]["role"] == "analyst"
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401


def test_analyst_cannot_import(analyst):
    df = make_sample(rows=10)
    response = analyst.post("/api/transactions/import", data=csv_upload(df), content_type="multipart/form-data")
    assert response.status_code == 403


# --- Import -------------------------------------------------------------------

def test_import_stores_transactions_predictions_and_cases(app, imported):
    assert imported["status"] == "completed"
    assert imported["rows_received"] == 1000
    assert imported["rows_imported"] == 1000
    with app.app_context():
        assert db.session.query(Transaction).count() == 1000
        assert db.session.query(Prediction).count() == 1000
        flagged = db.session.query(Prediction).filter_by(predicted_label=1).count()
        assert flagged > 0
        assert db.session.query(FraudCase).count() == flagged
        assert db.session.query(FraudCase).filter_by(status="Open").count() == flagged


def test_reimporting_same_file_skips_duplicates(app, admin, imported):
    df = make_sample(rows=1000, fraud_rate=0.05, random_state=3)
    response = admin.post("/api/transactions/import", data=csv_upload(df), content_type="multipart/form-data")
    source = response.get_json()["source"]
    assert source["rows_imported"] == 0
    assert source["rows_skipped"] == 1000
    with app.app_context():
        assert db.session.query(Transaction).count() == 1000


def test_import_skips_invalid_rows_and_reports_them(admin):
    df = make_sample(rows=20).astype(object)
    df.loc[0, "V3"] = "abc"
    df.loc[1, "Amount"] = -5
    df.loc[2, "Class"] = 7
    response = admin.post("/api/transactions/import", data=csv_upload(df), content_type="multipart/form-data")
    source = response.get_json()["source"]
    assert response.status_code == 201
    assert source["rows_imported"] == 17
    assert source["rows_skipped"] == 3


def test_import_rejects_missing_columns_and_records_failure(admin):
    df = make_sample(rows=5).drop(columns=["V7"])
    response = admin.post("/api/transactions/import", data=csv_upload(df), content_type="multipart/form-data")
    assert response.status_code == 400
    assert "V7" in response.get_json()["error"]
    sources = admin.get("/api/sources").get_json()["items"]
    assert sources[0]["status"] == "failed"


def test_import_rejects_non_csv_and_missing_file(admin):
    df = make_sample(rows=5)
    assert admin.post("/api/transactions/import", data=csv_upload(df, "data.xlsx"),
                      content_type="multipart/form-data").status_code == 400
    assert admin.post("/api/transactions/import", data={}, content_type="multipart/form-data").status_code == 400


def test_import_works_without_class_column(app, admin):
    df = make_sample(rows=30).drop(columns=["Class"])
    response = admin.post("/api/transactions/import", data=csv_upload(df), content_type="multipart/form-data")
    assert response.status_code == 201
    with app.app_context():
        assert db.session.query(Transaction).filter(Transaction.dataset_label.is_(None)).count() == 30


# --- Search -------------------------------------------------------------------

def test_list_transactions_paginates(admin, imported):
    body = admin.get("/api/transactions?per_page=10&page=2").get_json()
    assert body["total"] == 1000
    assert len(body["items"]) == 10
    assert body["page"] == 2
    assert admin.get("/api/transactions?per_page=5000").get_json()["per_page"] == 100


def test_list_transactions_filters(admin, imported):
    suspicious = admin.get("/api/transactions?prediction=suspicious&per_page=100").get_json()
    assert suspicious["total"] > 0
    assert all(t["prediction"]["predicted_label"] == 1 for t in suspicious["items"])

    big = admin.get("/api/transactions?min_amount=100&max_amount=200&per_page=100").get_json()
    assert all(100 <= t["amount"] <= 200 for t in big["items"])

    early = admin.get("/api/transactions?date_to=2013-09-01T12:00:00&per_page=100").get_json()
    assert all(t["timestamp"] <= "2013-09-01T12:00:00" for t in early["items"])

    open_cases = admin.get("/api/transactions?status=Open").get_json()
    assert open_cases["total"] == suspicious["total"]


def test_list_transactions_rejects_bad_filters(admin, imported):
    assert admin.get("/api/transactions?min_amount=lots").status_code == 400
    assert admin.get("/api/transactions?date_from=yesterday").status_code == 400
    assert admin.get("/api/transactions?prediction=maybe").status_code == 400


def test_get_transaction_includes_features(admin, imported):
    first = admin.get("/api/transactions?per_page=1").get_json()["items"][0]
    detail = admin.get(f"/api/transactions/{first['id']}").get_json()["transaction"]
    assert set(detail["features"]) == set(FEATURES)
    assert admin.get("/api/transactions/999999").status_code == 404


# --- Prediction ---------------------------------------------------------------

def test_predict_scores_a_transaction(analyst):
    row = make_sample(rows=1).iloc[0][FEATURES].to_dict()
    body = analyst.post("/api/predict", json=row).get_json()
    assert body["result"] in ("Suspicious", "Low risk")
    assert body["predicted_label"] in (0, 1)
    assert "risk_score" in body and "model_version" in body


def test_predict_validates_input(analyst):
    row = make_sample(rows=1).iloc[0][FEATURES].to_dict()
    row["V1"] = "high"
    del row["Amount"]
    response = analyst.post("/api/predict", json=row)
    assert response.status_code == 400
    assert "V1 must be a number" in response.get_json()["details"]
    assert "Amount must be a number" in response.get_json()["details"]
    assert analyst.post("/api/predict", data="not json").status_code == 400


# --- Alerts and cases ---------------------------------------------------------

def _first_alert(client):
    return client.get("/api/alerts?per_page=1").get_json()["items"][0]


def test_alerts_sorted_by_risk(admin, imported):
    items = admin.get("/api/alerts?per_page=100").get_json()["items"]
    scores = [a["transaction"]["risk_score"] for a in items]
    assert scores == sorted(scores, reverse=True)
    assert admin.get("/api/alerts?status=Closed").status_code == 400


def test_case_review_flow(analyst, imported):
    case = _first_alert(analyst)

    response = analyst.patch(f"/api/cases/{case['id']}", json={"status": "Under Review", "review_notes": "Called customer"})
    assert response.status_code == 200
    body = response.get_json()["case"]
    assert body["status"] == "Under Review"
    assert body["reviewed_by"] == "analyst@test.local"

    # Resolving needs an outcome.
    assert analyst.patch(f"/api/cases/{case['id']}", json={"status": "Resolved"}).status_code == 400
    resolved = analyst.patch(f"/api/cases/{case['id']}", json={"status": "Resolved", "outcome": "Legitimate"})
    assert resolved.get_json()["case"]["outcome"] == "Legitimate"

    # The model's prediction is untouched by the reviewer's outcome.
    transaction = analyst.get(f"/api/transactions/{case['transaction_id']}").get_json()["transaction"]
    assert transaction["prediction"]["predicted_label"] == 1

    # Reopening clears the outcome.
    reopened = analyst.patch(f"/api/cases/{case['id']}", json={"status": "Open"}).get_json()["case"]
    assert reopened["outcome"] is None


def test_case_update_validation(analyst, imported):
    case = _first_alert(analyst)
    url = f"/api/cases/{case['id']}"
    assert analyst.patch(url, json={"status": "Done"}).status_code == 400
    assert analyst.patch(url, json={"owner": "me"}).status_code == 400
    assert analyst.patch(url, json={}).status_code == 400
    assert analyst.patch(url, json={"outcome": "Legitimate"}).status_code == 400
    assert analyst.patch(url, json={"review_notes": 5}).status_code == 400
    assert analyst.patch("/api/cases/999999", json={"status": "Open"}).status_code == 404


# --- Dashboard ----------------------------------------------------------------

def test_dashboard_summary(admin, imported):
    body = admin.get("/api/dashboard/summary").get_json()
    assert body["total_transactions"] == 1000
    assert body["predicted_fraud"] == body["cases_by_status"]["Open"]
    assert body["pending_review"] == body["predicted_fraud"]
    assert sum(d["transactions"] for d in body["trend"]) == 1000
    hourly = admin.get("/api/dashboard/summary?granularity=hour").get_json()["trend"]
    assert len(hourly) > len(body["trend"])
    assert hourly[0]["period"].endswith(":00")
    assert sum(d["suspicious"] for d in hourly) == body["predicted_fraud"]


def test_dashboard_summary_empty_database(admin):
    body = admin.get("/api/dashboard/summary").get_json()
    assert body["total_transactions"] == 0
    assert body["predicted_fraud_rate"] == 0
    assert body["trend"] == []


def test_api_errors_are_json(admin):
    response = admin.get("/api/nothing-here")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_duplicate_check_ignores_number_formatting(app, admin):
    """The same row read as int in one file and float in another is still a duplicate."""
    df = make_sample(rows=10)
    as_int = df.assign(Time=df["Time"].astype(int))
    as_float = as_int.assign(Time=as_int["Time"].astype(float))
    admin.post("/api/transactions/import", data=csv_upload(as_float), content_type="multipart/form-data")
    response = admin.post("/api/transactions/import", data=csv_upload(as_int), content_type="multipart/form-data")
    assert response.get_json()["source"]["rows_imported"] == 0


def test_init_db_adds_columns_missing_from_an_older_database(tmp_path):
    import sqlalchemy
    from app import add_missing_columns, create_app
    from config import TestConfig

    class FileConfig(TestConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path / 'old.db'}"

    app = create_app(FileConfig)
    with app.app_context():
        db.create_all()
        with db.engine.begin() as conn:  # make it look like a database from before card_id existed
            conn.execute(sqlalchemy.text("DROP INDEX ix_transactions_card_id"))
            conn.execute(sqlalchemy.text("ALTER TABLE transactions DROP COLUMN card_id"))
        add_missing_columns()
        columns = {c["name"] for c in sqlalchemy.inspect(db.engine).get_columns("transactions")}
        assert "card_id" in columns
        add_missing_columns()  # running it again changes nothing
