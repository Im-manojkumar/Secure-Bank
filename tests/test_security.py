import re

import pytest

from ml.sample_data import make_sample
from models import AuditLog, db
from services.audit import MAX_FAILED_LOGINS
from tests.conftest import csv_upload, login


def audit_actions(app):
    with app.app_context():
        return [a.action for a in db.session.query(AuditLog).order_by(AuditLog.id)]


def bad_login(client, email="admin@test.local"):
    return client.post("/api/auth/login", json={"email": email, "password": "wrong-password"})


# --- Audit log ----------------------------------------------------------------

def test_logins_and_logout_are_audited(app, client):
    bad_login(client)
    login(client, "admin")
    client.post("/api/auth/logout")
    assert audit_actions(app) == ["login_failed", "login_success", "logout"]
    with app.app_context():
        failed = db.session.query(AuditLog).filter_by(action="login_failed").one()
        assert failed.subject == "admin@test.local"
        assert failed.user_id is None
        assert failed.ip_address == "127.0.0.1"


def test_case_update_is_audited_with_before_and_after(app, analyst, imported):
    case = analyst.get("/api/alerts?per_page=1").get_json()["items"][0]
    analyst.patch(f"/api/cases/{case['id']}", json={"status": "Resolved", "outcome": "Confirmed fraud", "review_notes": "x"})
    with app.app_context():
        entry = db.session.query(AuditLog).filter_by(action="case_updated").one()
        assert entry.subject == f"case:{case['id']}"
        assert entry.user.email == "analyst@test.local"
        assert entry.details == {
            "before": {"status": "Open", "outcome": None},
            "after": {"status": "Resolved", "outcome": "Confirmed fraud"},
            "notes_changed": True,
        }


def test_rejected_case_update_is_not_audited(app, analyst, imported):
    case = analyst.get("/api/alerts?per_page=1").get_json()["items"][0]
    analyst.patch(f"/api/cases/{case['id']}", json={"status": "Nope"})
    assert "case_updated" not in audit_actions(app)


def test_imports_are_audited(app, admin):
    admin.post("/api/transactions/import", data=csv_upload(make_sample(rows=20)), content_type="multipart/form-data")
    admin.post("/api/transactions/import", data=csv_upload(make_sample(rows=5).drop(columns=["V1"])),
               content_type="multipart/form-data")
    actions = audit_actions(app)
    assert "import_completed" in actions and "import_failed" in actions
    with app.app_context():
        done = db.session.query(AuditLog).filter_by(action="import_completed").one()
        assert done.details["rows_imported"] == 20
        assert done.user.email == "admin@test.local"


def test_cli_create_user_is_audited(app):
    result = app.test_cli_runner().invoke(args=["create-user", "new@test.local", "--role", "analyst"],
                                          input="longpassword\nlongpassword\n")
    assert result.exit_code == 0, result.output
    with app.app_context():
        entry = db.session.query(AuditLog).filter_by(action="user_created").one()
        assert entry.subject == "new@test.local"
        assert entry.details == {"role": "analyst", "via": "cli"}


def test_audit_api_is_admin_only_and_filters(app, client):
    bad_login(client, "analyst@test.local")
    login(client, "analyst")
    assert client.get("/api/audit").status_code == 403
    client.post("/api/auth/logout")

    login(client, "admin")
    body = client.get("/api/audit").get_json()
    assert body["items"][0]["action"] == "login_success"  # newest first
    failed = client.get("/api/audit?action=login_failed").get_json()["items"]
    assert [e["subject"] for e in failed] == ["analyst@test.local"]
    assert client.get("/api/audit?subject=ANALYST").get_json()["total"] >= 2
    assert client.get("/api/audit?action=hack").status_code == 400


# --- Login lockout ------------------------------------------------------------

def test_login_blocked_after_repeated_failures(app, client):
    for _ in range(MAX_FAILED_LOGINS):
        assert bad_login(client).status_code == 401
    # Even the right password is refused while blocked.
    response = client.post("/api/auth/login", json={"email": "admin@test.local", "password": "password123"})
    assert response.status_code == 429
    assert "Too many failed attempts" in response.get_json()["error"]
    assert audit_actions(app)[-1] == "login_blocked"
    # Other accounts are not affected.
    login(client, "analyst")


def test_successful_login_resets_the_failure_count(client):
    for _ in range(MAX_FAILED_LOGINS - 1):
        bad_login(client)
    login(client, "admin")
    client.post("/api/auth/logout")
    for _ in range(MAX_FAILED_LOGINS - 1):
        assert bad_login(client).status_code == 401
    login(client, "admin")


# --- Cross-site requests and headers ------------------------------------------

def test_cross_site_api_writes_are_blocked(client):
    response = client.post("/api/auth/login", headers={"Origin": "https://evil.example"},
                           json={"email": "admin@test.local", "password": "password123"})
    assert response.status_code == 403
    assert response.get_json()["error"] == "Cross-site request blocked."


def test_same_site_api_writes_are_allowed(client):
    response = client.post("/api/auth/login", headers={"Origin": "http://localhost"},
                           json={"email": "admin@test.local", "password": "password123"})
    assert response.status_code == 200


def test_cross_site_reads_are_not_blocked_by_origin_check(admin):
    assert admin.get("/api/dashboard/summary", headers={"Origin": "https://evil.example"}).status_code == 200


def test_security_headers(client):
    for path in ("/login", "/api/health"):
        headers = client.get(path).headers
        assert "script-src 'self'" in headers["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
    assert client.get("/api/health").headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("path", ["/login", "/", "/transactions", "/transactions/1", "/alerts", "/cases/1",
                                  "/predict", "/import", "/audit", "/reports", "/reports/1"])
def test_pages_have_no_inline_scripts_or_styles(admin, path):
    html = admin.get(path).get_data(as_text=True) if path != "/login" else admin.application.test_client().get(path).get_data(as_text=True)
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "inline <script> would be blocked by the CSP"
    assert 'style="' not in html, "inline style attributes would be blocked by the CSP"


def test_audit_page_is_admin_only(client):
    login(client, "analyst")
    assert client.get("/audit").status_code == 403
    assert b'href="/audit"' not in client.get("/").data


def test_admin_sees_audit_page(admin):
    response = admin.get("/audit")
    assert response.status_code == 200
    assert b'href="/audit"' in response.data


def test_file_databases_use_write_ahead_logging(model_dir, tmp_path):
    from app import create_app
    from config import TestConfig

    class Config(TestConfig):
        MODEL_DIR = model_dir
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path / 'wal.db'}"

    with create_app(Config).app_context():
        with db.engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
