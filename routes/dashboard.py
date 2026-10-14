"""Summary numbers for the dashboard."""

from flask import Blueprint, jsonify, request
from flask_login import login_required
from sqlalchemy import func, select

from ml.datasets import PROFILES
from models import CASE_STATUSES, DataSource, FraudCase, Prediction, Transaction, currency_of, db

bp = Blueprint("dashboard", __name__, url_prefix="/api/dashboard")


@bp.get("/summary")
@login_required
def summary():
    """Dashboard numbers, for all data or one dataset with ?dataset=sparkov."""
    dataset = request.args.get("dataset")
    if dataset and dataset not in PROFILES:
        return jsonify(error=f"dataset must be one of: {', '.join(PROFILES)}"), 400

    def scoped(query):
        """Limit a query that already involves Transaction to the chosen dataset."""
        if not dataset:
            return query
        return query.join(DataSource, DataSource.id == Transaction.source_id).where(DataSource.dataset == dataset)

    total = db.session.scalar(scoped(select(func.count(Transaction.id))))
    flagged = db.session.scalar(scoped(
        select(func.count(Prediction.id))
        .join(Transaction, Transaction.id == Prediction.transaction_id)
        .where(Prediction.predicted_label == 1)
    ))
    # Datasets use different currencies, so flagged amounts are totalled per currency.
    by_dataset = (
        select(DataSource.dataset, func.sum(Transaction.amount))
        .join(Transaction, Transaction.source_id == DataSource.id)
        .join(Prediction, Prediction.transaction_id == Transaction.id)
        .where(Prediction.predicted_label == 1)
    )
    if dataset:
        by_dataset = by_dataset.where(DataSource.dataset == dataset)
    flagged_by_dataset = db.session.execute(by_dataset.group_by(DataSource.dataset)).all()
    flagged_amounts = {}
    for key, amount in flagged_by_dataset:
        currency = currency_of(key)
        flagged_amounts[currency] = flagged_amounts.get(currency, 0) + float(amount or 0)

    status_counts = dict(db.session.execute(scoped(
        select(FraudCase.status, func.count())
        .join(Transaction, Transaction.id == FraudCase.transaction_id)
    ).group_by(FraudCase.status)).all())
    outcome_counts = dict(db.session.execute(scoped(
        select(FraudCase.outcome, func.count())
        .join(Transaction, Transaction.id == FraudCase.transaction_id)
        .where(FraudCase.outcome.is_not(None))
    ).group_by(FraudCase.outcome)).all())

    # ?granularity=hour groups the trend by hour (the credit card data covers only two days).
    # strftime is SQLite-specific; a PostgreSQL move would use date_trunc instead.
    if request.args.get("granularity") == "hour":
        bucket = func.strftime("%Y-%m-%d %H:00", Transaction.timestamp)
    else:
        bucket = func.date(Transaction.timestamp)
    trend = db.session.execute(scoped(
        select(bucket, func.count(Transaction.id), func.sum(Prediction.predicted_label))
        .join(Prediction, Prediction.transaction_id == Transaction.id)
    ).group_by(bucket).order_by(bucket)).all()

    return jsonify(
        total_transactions=total,
        predicted_fraud=flagged,
        predicted_fraud_rate=round(flagged / total, 6) if total else 0,
        predicted_fraud_amounts=[
            {"currency": c, "amount": round(a, 2)} for c, a in sorted(flagged_amounts.items(), key=lambda x: x[0] or "")
        ],
        cases_by_status={s: status_counts.get(s, 0) for s in CASE_STATUSES},
        pending_review=status_counts.get("Open", 0) + status_counts.get("Under Review", 0),
        resolved_outcomes=outcome_counts,
        trend=[
            {"period": str(p), "transactions": n, "suspicious": int(s or 0)} for p, n, s in trend
        ],
    )
