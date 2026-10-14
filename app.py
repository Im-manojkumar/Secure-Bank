"""SecureBank Flask application.

Run locally:
    flask --app app init-db
    flask --app app create-user admin@securebank.local --role admin
    flask --app app import-csv data/raw/creditcard.csv
    flask --app app run
"""

import secrets
import warnings
from pathlib import Path

import click
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager
from werkzeug.exceptions import HTTPException

from config import Config
from models import ROLES, User, db

login_manager = LoginManager()


def create_app(config_class=Config):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    if not app.config.get("SECRET_KEY"):
        warnings.warn(
            "SECRET_KEY is not set; using a random key, so logins reset when the app restarts. "
            "Set SECRET_KEY in the environment for anything beyond local development."
        )
        app.config["SECRET_KEY"] = secrets.token_hex(32)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    login_manager.init_app(app)

    from routes import alerts, auth, dashboard, pages, transactions

    for module in (auth, transactions, alerts, dashboard, pages):
        app.register_blueprint(module.bp)

    register_error_handlers(app)
    register_commands(app)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

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
        }
        if exc.code in pages:
            title, message = pages[exc.code]
            return render_template("error.html", code=exc.code, title=title, message=message), exc.code
        return exc


def register_commands(app):
    @app.cli.command("init-db")
    def init_db():
        """Create the database tables."""
        db.create_all()
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
        db.session.commit()
        click.echo(f"Created {role} {email}")

    @app.cli.command("import-csv")
    @click.argument("path", type=click.Path(exists=True, dir_okay=False))
    def import_csv(path):
        """Import a transactions CSV from disk (for files too big to upload)."""
        from services.ingestion import CSVImportError, import_transactions
        from services.prediction import ModelNotReady

        try:
            with open(path, "rb") as f:
                source = import_transactions(f, Path(path).name, source_type="csv_file")
        except (CSVImportError, ModelNotReady) as exc:
            db.session.rollback()
            raise click.ClickException(str(exc))
        click.echo(source.message)

