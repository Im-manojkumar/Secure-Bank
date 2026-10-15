"""SecureBank Flask application.

Run locally:
    flask --app app init-db
    flask --app app create-user admin@securebank.local --role admin
    flask --app app import-csv data/raw/creditcard.csv
    flask --app app run
"""

import logging
import secrets
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import click
import sqlalchemy
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager
from sqlalchemy import event
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from config import Config
from models import ROLES, User, db
from services import audit

login_manager = LoginManager()


def use_sqlite_wal(engine):
    """Let pages keep reading while another request writes.

    By default SQLite makes a write wait for every running read, so a slow
    report over a million rows could make a login fail with "database is
    locked". Write-ahead logging lets reads and one writer run together.
    """
    if engine.dialect.name != "sqlite" or engine.url.database in (None, "", ":memory:"):
        return

    @event.listens_for(engine, "connect")
    def set_pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=15000")
        cursor.close()


class ConfigError(RuntimeError):
    """The app is set to production but a required setting is missing."""


def weak_secret(value):
    """Shorter than 32 characters, or the placeholder from deploy/env.example."""
    return len(value) < 32 or value.startswith("replace-with")


def check_production(app):
    """Refuse to start a production server with development secrets, and force secure cookies."""
    from ml.datasets import CARD_ID_KEY, DEFAULT_CARD_ID_KEY

    problems = []
    secret = app.config.get("SECRET_KEY") or ""
    if not secret:
        problems.append("SECRET_KEY is not set")
    elif weak_secret(secret):
        problems.append("SECRET_KEY is too short or still the example value")
    if CARD_ID_KEY == DEFAULT_CARD_ID_KEY:
        problems.append("CARD_ID_KEY is not set")
    elif weak_secret(CARD_ID_KEY.decode()):
        problems.append("CARD_ID_KEY is too short or still the example value")
    if problems:
        raise ConfigError("Production settings missing: " + "; ".join(problems)
                          + ". Set them to long random values (see docs/DEPLOYMENT.md).")
    app.config.update(SESSION_COOKIE_SECURE=True, REMEMBER_COOKIE_SECURE=True)


def create_app(config_class=Config):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    production = app.config["ENV_NAME"] == "production"
    if production:
        check_production(app)
        # One line per log record on stdout, where Docker and gunicorn collect it.
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if app.config["TRUST_PROXY"]:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    if not app.config.get("SECRET_KEY"):
        warnings.warn(
            "SECRET_KEY is not set; using a random key, so logins reset when the app restarts. "
            "Set SECRET_KEY in the environment for anything beyond local development."
        )
        app.config["SECRET_KEY"] = secrets.token_hex(32)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    with app.app_context():
        use_sqlite_wal(db.engine)
    login_manager.init_app(app)

    from routes import alerts, audit as audit_routes, auth, dashboard, live, pages, reports, transactions

    for module in (auth, transactions, alerts, dashboard, reports, audit_routes, live, pages):
        app.register_blueprint(module.bp)

    register_error_handlers(app)
    register_security(app)
    register_commands(app)

    @app.get("/api/health")
    def health():
        """For load balancers and uptime checks: is the database reachable?"""
        try:
            db.session.execute(sqlalchemy.text("SELECT 1"))
        except sqlalchemy.exc.SQLAlchemyError:
            app.logger.exception("Health check could not reach the database")
            return jsonify(status="error", database="unreachable"), 503
        return jsonify(status="ok", database="ok")

    return app


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


@login_manager.unauthorized_handler
def unauthorized():
    # API calls get JSON; pages send the browser to the login page.
    if request.path.startswith("/api/"):
        return jsonify(error="Please log in first."), 401
    return redirect(url_for("pages.login", next=request.full_path.rstrip("?")))


CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'self'; object-src 'none'"
)


def register_security(app):
    @app.before_request
    def require_two_factor_setup():
        """An admin without two-factor login can only reach the setup page until they turn it on."""
        from flask_login import current_user

        from routes.auth import two_factor_missing

        if not current_user.is_authenticated or not two_factor_missing(current_user):
            return None
        path = request.path
        if path.startswith(("/api/auth/", "/static/", "/api/v1/")) or path in ("/account", "/favicon.ico"):
            return None
        if path.startswith("/api/"):
            return jsonify(error="Set up two-factor login on the Account page first.",
                           two_factor_setup_required=True), 403
        return redirect(url_for("pages.account"))

    @app.before_request
    def reject_cross_site_writes():
        """Block state-changing API calls sent from another website (CSRF).

        Browsers send an Origin header on POST/PATCH/DELETE. If it names a different
        site, the request did not come from SecureBank's own pages. SameSite=Lax
        cookies already stop most of these; this check covers the rest.
        """
        if request.method in ("GET", "HEAD", "OPTIONS") or not request.path.startswith("/api/"):
            return None
        origin = request.headers.get("Origin")
        if origin is None:
            # Non-browser clients such as curl and the test client send no Origin.
            return None
        if urlparse(origin).netloc != request.host:
            return jsonify(error="Cross-site request blocked."), 403
        return None

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if app.config["SESSION_COOKIE_SECURE"]:
            # Served over HTTPS: tell browsers never to use plain HTTP for this site.
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        if request.path.startswith("/api/"):
            # API responses can contain transaction data: never cache them.
            response.headers.setdefault("Cache-Control", "no-store")
        return response


def register_error_handlers(app):
    @app.errorhandler(HTTPException)
    def http_error(exc):
        # API callers get JSON instead of an HTML error page.
        if request.path.startswith("/api/"):
            messages = {413: "The file is too large."}
            return jsonify(error=messages.get(exc.code, exc.description)), exc.code
        pages = {
            403: ("No access", "Your role does not allow this page."),
            404: ("Page not found", "This page does not exist."),
            # Flask has already logged the error with its traceback; the visitor sees none of it.
            500: ("Something went wrong", "The error has been logged. Try again, or tell an admin if it keeps happening."),
        }
        if exc.code in pages:
            title, message = pages[exc.code]
            return render_template("error.html", code=exc.code, title=title, message=message), exc.code
        return exc


def add_missing_columns():
    """Add columns introduced after a database was created (create_all skips existing tables).

    Only nullable columns are added, so existing rows stay valid.
    """
    inspector = sqlalchemy.inspect(db.engine)
    for table in db.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        present = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present or not column.nullable:
                continue
            kind = column.type.compile(dialect=db.engine.dialect)
            with db.engine.begin() as conn:
                conn.execute(sqlalchemy.text(f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {kind}'))
                if column.index:
                    conn.execute(sqlalchemy.text(
                        f'CREATE INDEX IF NOT EXISTS ix_{table.name}_{column.name} ON {table.name} ("{column.name}")'
                    ))
            click.echo(f"Added column {table.name}.{column.name}")


def register_commands(app):
    @app.cli.command("init-db")
    def init_db():
        """Create the database tables, and add columns that newer versions need."""
        db.create_all()
        add_missing_columns()
        click.echo(f"Database ready at {app.config['SQLALCHEMY_DATABASE_URI']}")

    @app.cli.command("create-user")
    @click.argument("email")
    @click.option("--role", type=click.Choice(ROLES), default="analyst")
    @click.password_option()
    def create_user(email, role, password):
        """Add a user who can log in."""
        email = email.strip().lower()
        if User.query.filter_by(email=email).first():
            raise click.ClickException(f"{email} already exists")
        if len(password) < 8:
            raise click.ClickException("Use a password of at least 8 characters")
        user = User(email=email, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.flush()
        audit.record("user_created", subject=email, details={"role": role, "via": "cli"})
        db.session.commit()
        click.echo(f"Created {role} {email}")

    @app.cli.command("create-api-key")
    @click.argument("name")
    def create_api_key(name):
        """Make a key another system uses to call POST /api/v1/score."""
        from services.live import create_key

        row, key = create_key(name.strip()[:120])
        audit.record("api_key_created", subject=f"api_key:{row.id}", details={"name": row.name, "prefix": row.prefix})
        db.session.commit()
        click.echo(f"API key for {row.name} (shown only once, store it safely):\n{key}")

    @app.cli.command("revoke-api-key")
    @click.argument("prefix")
    def revoke_api_key(prefix):
        """Stop a key from working, by the first characters shown when it was made."""
        from models import ApiKey, utcnow

        keys = ApiKey.query.filter(ApiKey.prefix == prefix[:12], ApiKey.revoked_at.is_(None)).all()
        if len(keys) != 1:
            raise click.ClickException(f"No single active key starts with {prefix[:12]}")
        keys[0].revoked_at = utcnow()
        audit.record("api_key_revoked", subject=f"api_key:{keys[0].id}", details={"name": keys[0].name})
        db.session.commit()
        click.echo(f"Revoked the key for {keys[0].name}")

    @app.cli.command("retrain")
    @click.option("--dataset", required=True, help="Format to retrain: creditcard, sparkov or paysim")
    @click.option("--data", "data_path", required=True, type=click.Path(exists=True, dir_okay=False),
                  help="The original training CSV")
    @click.option("--test-data", type=click.Path(exists=True, dir_okay=False), help="Separate test CSV (Sparkov)")
    @click.option("--max-train-rows", type=int, help="Cap on training rows, as in python -m ml.train")
    def retrain(dataset, data_path, test_data, max_train_rows):
        """Retrain a model with the analysts' case decisions added to its training data.

        The current model is kept in a previous/ folder so it can be put back.
        Restart the app afterwards to load the new model.
        """
        import shutil

        from ml.datasets import get_profile
        from ml.train import train
        from services.feedback import reviewed_cases

        try:
            get_profile(dataset)
        except ValueError as exc:
            raise click.ClickException(str(exc))
        reviewed = reviewed_cases(dataset)
        click.echo(f"{len(reviewed):,} resolved cases with a verdict for {dataset}.")

        model_dir = Path(app.config["MODEL_DIR"]) / dataset
        meta_path = model_dir / "model_meta.json"
        if meta_path.exists():
            import json

            stamp = json.loads(meta_path.read_text())["trained_at"].replace(":", "-")
            backup = model_dir / "previous" / stamp
            backup.mkdir(parents=True, exist_ok=True)
            for item in model_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, backup / item.name)
            click.echo(f"Kept the current model in {backup}")

        best, _, test_results = train(data_path, dataset=dataset, test_path=test_data,
                                      max_train_rows=max_train_rows, out_dir=model_dir, reviewed=reviewed)
        audit.record("model_retrained", subject=f"model:{dataset}", details={
            "model": best, "reviewed_cases": len(reviewed),
            "test_pr_auc": test_results[best]["pr_auc"],
        })
        db.session.commit()
        click.echo("Restart the app to start scoring with the new model.")

    @app.cli.command("backup-db")
    @click.option("--keep", default=14, show_default=True, help="How many backups to keep; older ones are deleted")
    def backup_db(keep):
        """Copy the SQLite database to BACKUP_DIR, safely, even while the app is running."""
        path = db.engine.url.database
        if db.engine.dialect.name != "sqlite" or path in (None, "", ":memory:"):
            raise click.ClickException("backup-db copies a SQLite database file; use your database's own tools otherwise.")
        folder = Path(app.config["BACKUP_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = folder / f"securebank-{stamp}.db"
        # SQLite's backup API copies a consistent snapshot while other connections keep writing.
        with sqlite3.connect(path) as source, sqlite3.connect(target) as copy:
            source.backup(copy)
        old = sorted(folder.glob("securebank-*.db"))[:-keep] if keep > 0 else []
        for item in old:
            item.unlink()
        click.echo(f"Backed up to {target}" + (f"; removed {len(old)} older backups" if old else ""))

    @app.cli.command("import-csv")
    @click.argument("path", type=click.Path(exists=True, dir_okay=False))
    def import_csv(path):
        """Import a transactions CSV from disk (for files too big to upload)."""
        from models import DataSource
        from services.ingestion import CSVImportError, import_transactions, record_failed_import
        from services.prediction import ModelNotReady

        name = Path(path).name
        source = DataSource(name=name, source_type="csv_file", status="processing")
        db.session.add(source)
        db.session.commit()
        source_id = source.id
        try:
            with open(path, "rb") as f:
                source = import_transactions(f, name, source=source)
        except (CSVImportError, ModelNotReady) as exc:
            db.session.rollback()
            source = record_failed_import(name, str(exc), source=db.session.get(DataSource, source_id))
            raise click.ClickException(source.message)
        click.echo(source.message)

    @app.cli.command("import-worker")
    @click.option("--once", is_flag=True, help="Run the queued imports, then stop")
    def import_worker(once):
        """Import uploaded files as they are queued (for IMPORT_MODE=worker)."""
        from services.jobs import work

        click.echo("Waiting for imports. Press Ctrl+C to stop." if not once else "Running queued imports.")
        ran = work(app, once=once)
        if once:
            click.echo(f"Ran {ran} imports.")

