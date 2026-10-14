"""Summary numbers for the dashboard."""

from flask import Blueprint, jsonify, request
from flask_login import login_required
from sqlalchemy import func, select

from models import CASE_STATUSES, FraudCase, Prediction, Transaction, db

bp = Blueprint("dashboard", __name__, url_prefix="/api/dashboard")


@bp.get("/summary")
@login_required
def summary():
    total = db.session.scalar(select(func.count(Transaction.id)))
    flagged = db.session.scalar(select(func.count(Prediction.id)).where(Prediction.predicted_label == 1))
    flagged_amount = db.session.scalar(
        select(func.coalesce(func.sum(Transaction.amount), 0))
        .join(Prediction, Prediction.transaction_id == Transaction.id)
        .where(Prediction.predicted_label == 1)
    )

    status_counts = dict(
        db.session.execute(select(FraudCase.status, func.count()).group_by(FraudCase.status)).all()
    )
    outcome_counts = dict(
        db.session.execute(
            select(FraudCase.outcome, func.count())
            .where(FraudCase.outcome.is_not(None))
            .group_by(FraudCase.outcome)
        ).all()
    )

    # ?granularity=hour groups the trend by hour (the Kaggle data covers only two days).
    # strftime is SQLite-specific; a PostgreSQL move would use date_trunc instead.
    if request.args.get("granularity") == "hour":
        bucket = func.strftime("%Y-%m-%d %H:00", Transaction.timestamp)
    else:
        bucket = func.date(Transaction.timestamp)
    trend = db.session.execute(
        select(bucket, func.count(Transaction.id), func.sum(Prediction.predicted_label))
        .join(Prediction, Prediction.transaction_id == Transaction.id)
        .group_by(bucket)
        .order_by(bucket)
    ).all()

    return jsonify(
        total_transactions=total,
        predicted_fraud=flagged,
        predicted_fraud_rate=round(flagged / total, 6) if total else 0,
        predicted_fraud_amount=round(float(flagged_amount), 2),
        cases_by_status={s: status_counts.get(s, 0) for s in CASE_STATUSES},
        pending_review=status_counts.get("Open", 0) + status_counts.get("Under Review", 0),
        resolved_outcomes=outcome_counts,
        trend=[
            {"period": str(p), "transactions": n, "suspicious": int(s or 0)} for p, n, s in trend
        ],
    )
