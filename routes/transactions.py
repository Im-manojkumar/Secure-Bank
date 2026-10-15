"""Transaction import, search and single-transaction prediction."""

from pathlib import Path

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required
from sqlalchemy import select

from ml.datasets import DEFAULT_PROFILE, PROFILES
from models import DataSource, Transaction, db
from routes.auth import role_required
from routes.pagination import paginate
from services.ingestion import CSVImportError, check_file, record_failed_import
from services.jobs import ensure_worker, queue_import
from services.prediction import ModelNotReady, describe_models, get_model, predict_one
from services.search import FilterError, clean_filters, transaction_query

bp = Blueprint("transactions", __name__, url_prefix="/api")


@bp.post("/transactions/import")
@role_required("admin")
def import_csv():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify(error="Attach a CSV file in the 'file' field."), 400
    name = Path(upload.filename).name
    if not name.lower().endswith(".csv"):
        return jsonify(error="Only .csv files can be imported."), 400

    try:
        profile = check_file(upload.stream)
        get_model(profile.key)
    except ModelNotReady as exc:
        return jsonify(error=str(exc)), 503
    except CSVImportError as exc:
        db.session.rollback()
        source = record_failed_import(name, str(exc), current_user)
        return jsonify(error=str(exc), source=source.to_dict()), 400

    # The worker imports it; the page polls GET /api/sources/<id> for progress.
    source = queue_import(upload, name, profile, current_user)
    ensure_worker()
    db.session.refresh(source)
    if source.status == "failed":
        return jsonify(error=source.message, source=source.to_dict()), 400
    return jsonify(source=source.to_dict()), 201 if source.status == "completed" else 202


@bp.get("/sources")
@login_required
def list_sources():
    sources = db.session.scalars(select(DataSource).order_by(DataSource.created_at.desc())).all()
    if any(s.status == "queued" for s in sources):
        # Imports queued before a restart get a worker when someone opens the import page.
        ensure_worker()
    return jsonify(items=[s.to_dict() for s in sources])


@bp.get("/sources/<int:source_id>")
@login_required
def get_source(source_id):
    """One import, for following its progress."""
    source = db.session.get(DataSource, source_id)
    if source is None:
        return jsonify(error="Import not found."), 404
    return jsonify(source=source.to_dict())


@bp.get("/transactions")
@login_required
def list_transactions():
    """Search transactions.

    Filters (all optional): source_id, dataset, date_from, date_to, min_amount,
    max_amount, prediction (suspicious | low_risk), status (a case status).
    """
    try:
        query = transaction_query(clean_filters(request.args))
    except FilterError as exc:
        return jsonify(error=str(exc)), 400
    return paginate(query, lambda t: t.to_dict())


@bp.get("/transactions/<int:transaction_id>")
@login_required
def get_transaction(transaction_id):
    transaction = db.get_or_404(Transaction, transaction_id)
    return jsonify(transaction=transaction.to_dict(include_features=True))


@bp.get("/models")
@login_required
def list_models():
    """The dataset formats SecureBank understands, with each one's trained model if any."""
    return jsonify(items=describe_models())


@bp.post("/predict")
@login_required
def predict():
    """Score one transaction. The body holds its input columns, plus an optional
    "dataset" naming the format (default: creditcard)."""
    payload = request.get_json(silent=True)
    dataset = payload.get("dataset", DEFAULT_PROFILE) if isinstance(payload, dict) else DEFAULT_PROFILE
    if dataset not in PROFILES:
        return jsonify(error=f"dataset must be one of: {', '.join(PROFILES)}"), 400
    row, errors = PROFILES[dataset].parse_input(payload)
    if errors:
        return jsonify(error="Invalid transaction.", details=errors), 400
    try:
        return jsonify(predict_one(dataset, row))
    except ModelNotReady as exc:
        return jsonify(error=str(exc)), 503
