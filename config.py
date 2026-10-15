"""App settings. Values can be overridden with environment variables."""

import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    # "production" turns on the checks and settings a public server needs (see app.py).
    ENV_NAME = os.environ.get("SECUREBANK_ENV", "development")
    # Behind a reverse proxy (Caddy, nginx), trust its X-Forwarded-For and -Proto headers,
    # so the login limit and the audit log see the visitor's address, not the proxy's.
    TRUST_PROXY = os.environ.get("TRUST_PROXY", "0") == "1"
    BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", BASE_DIR / "backups"))
    # When unset, create_app generates a random key and warns.
    SECRET_KEY = os.environ.get("SECRET_KEY")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'instance' / 'securebank.db'}"
    )
    MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "ml" / "artifacts"))

    # Uploads: the full Kaggle file is ~150 MB.
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024

    # The Kaggle dataset's Time column counts seconds from its first transaction,
    # which the dataset authors place in September 2013. Times are stored as UTC.
    DATASET_START = datetime(2013, 9, 1)

    # Session cookies: not readable by JavaScript, not sent on cross-site requests.
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "0") == "1"

    # The live API declines outright at or above this risk score; between the model's
    # alert threshold and this, it approves but opens a case for review.
    DECLINE_SCORE = float(os.environ.get("DECLINE_SCORE", 0.98))

    # Admins must set up two-factor login before they can use anything else.
    REQUIRE_2FA_FOR_ADMINS = os.environ.get("REQUIRE_2FA_FOR_ADMINS", "1") == "1"

    PER_PAGE_DEFAULT = 25
    PER_PAGE_MAX = 100
    # Largest CSV download allowed, so one click cannot stream a million rows.
    EXPORT_MAX_ROWS = int(os.environ.get("EXPORT_MAX_ROWS", 100_000))


class TestConfig(Config):
    TESTING = True
    # Tests of two-factor login turn this on themselves.
    REQUIRE_2FA_FOR_ADMINS = False
    SECRET_KEY = "test-secret"
    SQLALCHEMY_DATABASE_URI = "sqlite://"
