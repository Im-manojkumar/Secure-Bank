"""Audit log viewer for admins."""

from flask import Blueprint, jsonify, request
from sqlalchemy import select

from models import AuditLog
from routes.auth import role_required
from routes.pagination import paginate
from services.audit import ACTIONS

bp = Blueprint("audit", __name__, url_prefix="/api")


@bp.get("/audit")
@role_required("admin")
def list_audit():
    """Newest first. Optional filters: ?action=login_failed&subject=someone@example.com"""
    query = select(AuditLog)
    action = request.args.get("action")
    if action:
        if action not in ACTIONS:
            return jsonify(error=f"action must be one of: {', '.join(ACTIONS)}"), 400
        query = query.where(AuditLog.action == action)
    subject = request.args.get("subject")
    if subject:
        query = query.where(AuditLog.subject.contains(subject.strip().lower()))
    return paginate(query.order_by(AuditLog.created_at.desc(), AuditLog.id.desc()), lambda a: a.to_dict())
