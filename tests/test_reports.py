"""Saved reports and CSV downloads."""

import csv
import io

import pytest

from ml.sample_data import make_sample, make_sparkov_sample
from tests.conftest import csv_upload, login

SUSPICIOUS_SPARKOV = {"dataset": "sparkov", "prediction": "suspicious"}


@pytest.fixture
def two_formats(app):
    """Credit card and Sparkov rows, imported by the admin."""
    admin = app.test_client()
    login(admin, "admin")
    for df, name in ((make_sample(rows=400, fraud_rate=0.1, random_state=5), "card.csv"),
                     (make_sparkov_sample(rows=400, fraud_rate=0.1, random_state=5), "sparkov.csv")):
        response = admin.post("/api/transactions/import", data=csv_upload(df, name), content_type="multipart/form-data")
        assert response.status_code == 201, response.get_json()


def save(client, name, filters):
    return client.post("/api/reports", json={"name": name, "filters": filters})


def rows_of(response):
    return list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))


def test_saved_report_totals_match_the_search(analyst, two_formats):
    response = save(analyst, "Sparkov alerts", SUSPICIOUS_SPARKOV | {"min_amount": "", "colour": "red"})
    assert response.status_code == 201
    report = response.get_json()["report"]
    assert report["filters"] == SUSPICIOUS_SPARKOV, "empty and unknown filters are dropped"
    assert report["created_by"] == "analyst@test.local" and report["can_delete"]

    body = analyst.get(f"/api/reports/{report['id']}").get_json()
    summary = body["summary"]
    search = analyst.get("/api/transactions?dataset=sparkov&prediction=suspicious").get_json()
    assert summary["transactions"] == summary["suspicious"] == search["total"] > 0
    assert [d["dataset"] for d in summary["by_dataset"]] == ["sparkov"]
    assert summary["by_dataset"][0]["currency"] == "USD"
    assert summary["by_dataset"][0]["suspicious_amount"] == pytest.approx(summary["by_dataset"][0]["amount"])
    assert summary["cases_by_status"]["Open"] == search["total"]

    listed = analyst.get("/api/reports").get_json()["items"]
    assert [r["name"] for r in listed] == ["Sparkov alerts"]


def test_report_reflects_case_reviews(analyst, two_formats):
    report = save(analyst, "All alerts", {"prediction": "suspicious"}).get_json()["report"]
    case = analyst.get("/api/alerts?per_page=1").get_json()["items"][0]
    analyst.patch(f"/api/cases/{case['id']}", json={"status": "Resolved", "outcome": "Confirmed fraud"})
    summary = analyst.get(f"/api/reports/{report['id']}").get_json()["summary"]
    assert summary["cases_by_status"]["Resolved"] == 1
    assert summary["outcomes"] == {"Confirmed fraud": 1}


def test_report_with_no_filters_covers_everything(admin, two_formats):
    report = save(admin, "Everything", {}).get_json()["report"]
    summary = admin.get(f"/api/reports/{report['id']}").get_json()["summary"]
    assert summary["transactions"] == 800
    assert {d["dataset"] for d in summary["by_dataset"]} == {"creditcard", "sparkov"}


@pytest.mark.parametrize("payload, message", [
    ({"name": "  ", "filters": {}}, "Give the report a name."),
    ({"name": "x" * 121, "filters": {}}, "at most 120 characters"),
    ({"name": "Bad", "filters": {"min_amount": "lots"}}, "min_amount must be a number"),
    ({"name": "Bad", "filters": {"status": "Closed"}}, "status must be one of"),
    ({"name": "Bad", "filters": ["dataset"]}, "filters must be an object"),
])
def test_report_validation(analyst, payload, message):
    response = analyst.post("/api/reports", json=payload)
    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_report_names_are_unique(analyst):
    assert save(analyst, "Weekly", {}).status_code == 201
    response = save(analyst, "weekly", {})
    assert response.status_code == 409


def test_only_the_owner_or_an_admin_can_delete(app, two_formats):
    admin, analyst = app.test_client(), app.test_client()
    login(admin, "admin")
    login(analyst, "analyst")
    admins = save(admin, "Admin's report", {}).get_json()["report"]
    analysts = save(analyst, "Analyst's report", {}).get_json()["report"]

    assert analyst.get(f"/api/reports/{admins['id']}").get_json()["report"]["can_delete"] is False
    assert analyst.delete(f"/api/reports/{admins['id']}").status_code == 403
    assert analyst.delete(f"/api/reports/{analysts['id']}").status_code == 204
    assert admin.delete(f"/api/reports/{admins['id']}").status_code == 204
    assert admin.get("/api/reports").get_json()["items"] == []
    assert admin.get(f"/api/reports/{admins['id']}").status_code == 404


def test_report_downloads_as_csv(analyst, two_formats):
    report = save(analyst, "Sparkov alerts!", SUSPICIOUS_SPARKOV).get_json()["report"]
    response = analyst.get(f"/api/reports/{report['id']}/export")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert 'filename="securebank-sparkov-alerts.csv"' in response.headers["Content-Disposition"]
    rows = rows_of(response)
    total = analyst.get("/api/transactions?dataset=sparkov&prediction=suspicious").get_json()["total"]
    assert len(rows) == total
    first = rows[0]
    assert first["dataset"] == "sparkov" and first["currency"] == "USD" and first["prediction"] == "suspicious"
    assert first["case_status"] == "Open" and first["top_reason"]
    assert first["source_row"].startswith("sparkov.csv#row")
    assert "first" not in first and "cc_num" not in first, "no personal data in the download"


def test_search_downloads_as_csv(admin, two_formats):
    response = admin.get("/api/transactions/export?dataset=creditcard&max_amount=50")
    rows = rows_of(response)
    assert rows and all(r["dataset"] == "creditcard" and float(r["amount"]) <= 50 for r in rows)
    assert admin.get("/api/transactions/export?dataset=bank").status_code == 400


def test_csv_cells_cannot_run_as_formulas(admin):
    df = make_sample(rows=20, fraud_rate=0.1, random_state=7)
    admin.post("/api/transactions/import", data=csv_upload(df, "=HYPERLINK(evil).csv"),
               content_type="multipart/form-data")
    rows = rows_of(admin.get("/api/transactions/export"))
    assert rows and all(r["source_row"].startswith("'=HYPERLINK") for r in rows)


def test_downloads_are_limited_and_audited(app, admin, two_formats):
    app.config["EXPORT_MAX_ROWS"] = 100
    check = admin.get("/api/transactions/export?check=1").get_json()
    assert check == {"rows": 800, "limit": 100, "allowed": False}
    refused = admin.get("/api/transactions/export")
    assert refused.status_code == 400 and "800" in refused.get_json()["error"]

    assert admin.get("/api/transactions/export?dataset=sparkov&prediction=suspicious").status_code == 200
    report = save(admin, "Alerts", {"prediction": "suspicious"}).get_json()["report"]
    admin.delete(f"/api/reports/{report['id']}")
    actions = [e["action"] for e in admin.get("/api/audit").get_json()["items"]]
    assert actions[:3] == ["report_deleted", "report_saved", "report_exported"]
    exported = admin.get("/api/audit?action=report_exported").get_json()["items"]
    assert len(exported) == 1, "a refused or checked download is not logged as a download"
    assert exported[0]["details"]["filters"] == SUSPICIOUS_SPARKOV


def test_reports_need_a_login(client):
    assert client.get("/api/reports").status_code == 401
    assert client.post("/api/reports", json={"name": "x"}).status_code == 401
    assert client.get("/api/transactions/export").status_code == 401
