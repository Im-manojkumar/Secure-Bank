"""Production mode: required secrets, HTTPS-only cookies, the reverse proxy, health checks and backups."""

import sqlite3

import pytest
import sqlalchemy

import ml.datasets
from app import ConfigError, create_app
from config import TestConfig
from models import AuditLog, User, db
from tests.conftest import login

PRODUCTION_CARD_KEY = b"a-long-random-production-card-key-0123456789"
PRODUCTION_SECRET = "a-long-random-production-secret-0123456789"


def make_app(model_dir, **settings):
    config = type("Config", (TestConfig,), {"MODEL_DIR": model_dir, **settings})
    app = create_app(config)
    with app.app_context():
        db.create_all()
        user = User(email="admin@test.local", role="admin")
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()
    return app


@pytest.fixture
def production_key(monkeypatch):
    monkeypatch.setattr(ml.datasets, "CARD_ID_KEY", PRODUCTION_CARD_KEY)


# --- Production settings --------------------------------------------------------

def test_production_refuses_to_start_without_a_secret_key(model_dir, production_key):
    with pytest.raises(ConfigError, match="SECRET_KEY"):
        make_app(model_dir, ENV_NAME="production", SECRET_KEY=None)


def test_production_refuses_the_development_card_key(model_dir):
    assert ml.datasets.CARD_ID_KEY == ml.datasets.DEFAULT_CARD_ID_KEY
    with pytest.raises(ConfigError, match="CARD_ID_KEY is not set"):
        make_app(model_dir, ENV_NAME="production", SECRET_KEY=PRODUCTION_SECRET)


@pytest.mark.parametrize("secret", ["short", "replace-with-a-long-random-value"])
def test_production_refuses_weak_or_example_secrets(model_dir, production_key, secret):
    with pytest.raises(ConfigError, match="SECRET_KEY is too short or still the example value"):
        make_app(model_dir, ENV_NAME="production", SECRET_KEY=secret)


def test_production_forces_secure_cookies_and_hsts(model_dir, production_key):
    app = make_app(model_dir, ENV_NAME="production", SECRET_KEY=PRODUCTION_SECRET, SESSION_COOKIE_SECURE=False)
    assert app.config["SESSION_COOKIE_SECURE"] and app.config["REMEMBER_COOKIE_SECURE"]
    client = app.test_client()
    response = login(client)
    assert "Secure" in response.headers["Set-Cookie"]
    assert response.headers["Strict-Transport-Security"] == "max-age=31536000"


def test_development_sends_no_hsts(client):
    response = client.get("/login")
    assert "Strict-Transport-Security" not in response.headers


# --- Reverse proxy ------------------------------------------------------------------

def failed_login_ip(app, headers):
    client = app.test_client()
    client.post("/api/auth/login", json={"email": "admin@test.local", "password": "wrong-password"},
                headers=headers)
    with app.app_context():
        return db.session.query(AuditLog).filter_by(action="login_failed").one().ip_address


def test_behind_a_proxy_the_visitor_address_is_logged(model_dir):
    app = make_app(model_dir, TRUST_PROXY=True)
    assert failed_login_ip(app, {"X-Forwarded-For": "203.0.113.7"}) == "203.0.113.7"


def test_forwarded_address_is_ignored_without_a_proxy(model_dir):
    # Otherwise anyone could dodge the login limit by sending a made-up address.
    app = make_app(model_dir, TRUST_PROXY=False)
    assert failed_login_ip(app, {"X-Forwarded-For": "203.0.113.7"}) == "127.0.0.1"


# --- Health check and errors ------------------------------------------------------------

def test_health_check_reports_the_database(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "database": "ok"}


def test_health_check_fails_when_the_database_is_down(client, monkeypatch):
    def broken(*args, **kwargs):
        raise sqlalchemy.exc.OperationalError("SELECT 1", {}, Exception("disk I/O error"))

    monkeypatch.setattr(db.session, "execute", broken)
    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.get_json()["database"] == "unreachable"


def test_unexpected_errors_are_logged_and_hidden(model_dir, caplog):
    app = make_app(model_dir, TESTING=False)

    @app.get("/explode")
    def explode():
        raise RuntimeError("secret detail")

    response = app.test_client().get("/explode")
    assert response.status_code == 500
    assert b"Something went wrong" in response.data
    assert b"secret detail" not in response.data
    assert "secret detail" in caplog.text


# --- Backups -------------------------------------------------------------------------

def test_backup_copies_the_database_and_keeps_the_newest(model_dir, tmp_path):
    backups = tmp_path / "backups"
    backups.mkdir()
    for stamp in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
        (backups / f"securebank-{stamp}.db").write_bytes(b"old")
    app = make_app(model_dir, SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / 'live.db'}", BACKUP_DIR=backups)

    result = app.test_cli_runner().invoke(args=["backup-db", "--keep", "2"])
    assert result.exit_code == 0, result.output
    assert "removed 2 older backups" in result.output

    kept = sorted(backups.glob("securebank-*.db"))
    assert [p.name for p in kept][0] == "securebank-20260103T000000Z.db"
    assert len(kept) == 2
    with sqlite3.connect(kept[-1]) as copy:
        assert copy.execute("SELECT email FROM users").fetchall() == [("admin@test.local",)]


def test_backup_needs_a_database_file(app):
    result = app.test_cli_runner().invoke(args=["backup-db"])
    assert result.exit_code != 0
    assert "SQLite database file" in result.output
