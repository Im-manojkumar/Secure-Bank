"""Audit logging for sensitive actions.

record() adds a row to the current database session; the caller commits it
together with the change it describes, so the log and the data never disagree.
"""

from datetime import timedelta

from flask import has_request_context, request
from sqlalchemy import func, select

from models import AuditLog, db, utcnow

ACTIONS = (
    "login_success",
    "login_failed",
    "login_blocked",
    "logout",
    "import_completed",
    "import_failed",
    "case_updated",
    "user_created",
    "report_saved",
    "report_deleted",
    "report_exported",
)

# Failed logins allowed per email and IP address within the window before blocking.
MAX_FAILED_LOGINS = 5
LOGIN_WINDOW = timedelta(minutes=15)


def record(action, user=None, subject=None, details=None):
    if action not in ACTIONS:
        raise ValueError(f"Unknown audit action: {action}")
    entry = AuditLog(
        action=action,
        user_id=user.id if user is not None else None,
        subject=subject,
        details=details,
        ip_address=request.remote_addr if has_request_context() else None,
    )
    db.session.add(entry)
    return entry


def recent_failed_logins(email, ip_address):
    """Failed logins for this email from this IP in the window, since the last success."""
    since = utcnow() - LOGIN_WINDOW
    last_success = db.session.scalar(
        select(func.max(AuditLog.created_at)).where(
            AuditLog.action == "login_success",
            AuditLog.subject == email,
            AuditLog.ip_address == ip_address,
        )
    )
    if last_success and last_success > since:
        since = last_success
    return db.session.scalar(
        select(func.count(AuditLog.id)).where(
            AuditLog.action == "login_failed",
            AuditLog.subject == email,
            AuditLog.ip_address == ip_address,
            AuditLog.created_at >= since,
        )
    )
