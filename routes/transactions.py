"""Transaction import, search and single-transaction prediction."""

from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required
from sqlalchemy import select

from ml.datasets import DEFAULT_PROFILE, PROFILES
from models import DataSource, FraudCase, Prediction, Transaction, db
from routes.auth import role_required
from routes.pagination import paginate
from services.ingestion import CSVImportError, import_transactions, record_failed_import
from services.prediction import ModelNotReady, describe_models, predict_one

bp = Blueprint("transactions", __name__, url_prefix="/api")


class FilterError(ValueError):
    pass


def _parse_date(value, name):
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise FilterError(f"{name} must be a date like 2013-09-01 or 2013-09-01T10:30:00")


def _parse_float(value, name):
    try:
        return float(value)
    except ValueError:
        raise FilterError(f"{name} must be a number")


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
        source = import_transactions(upload.stream, name, current_user)
    except ModelNotReady as exc:
        db.session.rollback()
        return jsonify(error=str(exc)), 503
    except CSVImportError as exc:
        db.session.rollback()
        source = record_failed_import(name, str(exc), current_user)
        return jsonify(error=str(exc), source=source.to_dict()), 400
    except Exception:
        db.session.rollback()
        current_app.logger.exception("Import of %s failed", name)
        source = record_failed_import(name, "Unexpected error during import.", current_user)
        return jsonify(error="The import failed unexpectedly.", source=source.to_dict()), 500
    return jsonify(source=source.to_dict()), 201


@bp.get("/sources")
@login_required
def list_sources():
    sources = db.session.scalars(select(DataSource).order_by(DataSource.created_at.desc())).all()
    return jsonify(items=[s.to_dict() for s in sources])


@bp.get("/transactions")
@login_required
def list_transactions():
    """Search transactions.

    Filters (all optional): source_id, dataset, date_from, date_to, min_amount,
    max_amount, prediction (suspicious | low_risk), status (a case status).
    """
    args = request.args
    query = select(Transaction).join(Prediction, Prediction.transaction_id == Transaction.id, isouter=True)
    try:
        if "dataset" in args:
            if args["dataset"] not in PROFILES:
                raise FilterError(f"dataset must be one of: {', '.join(PROFILES)}")
            query = query.join(DataSource, DataSource.id == Transaction.source_id).where(
                DataSource.dataset == args["dataset"]
            )
        if "source_id" in args:
            query = query.where(Transaction.source_id == int(_parse_float(args["source_id"], "source_id")))
        if "date_from" in args:
            query = query.where(Transaction.timestamp >= _parse_date(args["date_from"], "date_from"))
        if "date_to" in args:
            query = query.where(Transaction.timestamp <= _parse_date(args["date_to"], "date_to"))
        if "min_amount" in args:
            query = query.where(Transaction.amount >= _parse_float(args["min_amount"], "min_amount"))
        if "max_amount" in args:
            query = query.where(Transaction.amount <= _parse_float(args["max_amount"], "max_amount"))
        if "prediction" in args:
            labels = {"suspicious": 1, "low_risk": 0}
            if args["prediction"] not in labels:
                raise FilterError("prediction must be suspicious or low_risk")
            query = query.where(Prediction.predicted_label == labels[args["prediction"]])
        if "status" in args:
            query = query.join(FraudCase, FraudCase.transaction_id == Transaction.id).where(
                FraudCase.status == args["status"]
            )
    except FilterError as exc:
        return jsonify(error=str(exc)), 400

    query = query.order_by(Transaction.timestamp.desc(), Transaction.id.desc())
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
