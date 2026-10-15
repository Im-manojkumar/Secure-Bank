"""Fraud alerts and case review."""

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required
from sqlalchemy import select

from models import CASE_STATUSES, FraudCase, Prediction, Transaction, db
from routes.auth import role_required
from routes.pagination import paginate
from services import audit
from services.alerts import CaseUpdateError, apply_case_update

bp = Blueprint("alerts", __name__, url_prefix="/api")


@bp.get("/alerts")
@login_required
def list_alerts():
    """List fraud cases, highest risk first. Optional filter: ?status=Open."""
    query = (
        select(FraudCase)
        .join(Transaction, Transaction.id == FraudCase.transaction_id)
        .join(Prediction, Prediction.transaction_id == Transaction.id)
    )
    status = request.args.get("status")
    if status:
        if status not in CASE_STATUSES:
            return jsonify(error=f"status must be one of: {', '.join(CASE_STATUSES)}"), 400
        query = query.where(FraudCase.status == status)
    query = query.order_by(Prediction.risk_score.desc(), FraudCase.id)
    return paginate(query, lambda c: c.to_dict())


@bp.get("/cases/<int:case_id>")
@login_required
def get_case(case_id):
    return jsonify(case=db.get_or_404(FraudCase, case_id).to_dict())


@bp.patch("/cases/<int:case_id>")
@role_required("admin", "analyst")
def update_case(case_id):
    case = db.get_or_404(FraudCase, case_id)
    before = {"status": case.status, "outcome": case.outcome}
    try:
        apply_case_update(case, request.get_json(silent=True), current_user)
    except CaseUpdateError as exc:
        return jsonify(error=str(exc)), 400
    audit.record("case_updated", user=current_user, subject=f"case:{case.id}", details={
        "before": before,
        "after": {"status": case.status, "outcome": case.outcome},
        "notes_changed": "review_notes" in (request.get_json(silent=True) or {}),
    })
    db.session.commit()
    return jsonify(case=case.to_dict())
