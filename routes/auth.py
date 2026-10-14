"""Login, logout and role checks."""

from functools import wraps

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required, login_user, logout_user

from models import User, db
from services import audit

bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def role_required(*roles):
    """Allow the view only for logged-in users whose role is in roles."""

    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if current_user.role not in roles:
                return jsonify(error="You do not have permission to do this."), 403
            return view(*args, **kwargs)

        return wrapped

    return decorator


@bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()[:255]
    password = str(data.get("password", ""))

    if audit.recent_failed_logins(email, request.remote_addr) >= audit.MAX_FAILED_LOGINS:
        audit.record("login_blocked", subject=email)
        db.session.commit()
        minutes = int(audit.LOGIN_WINDOW.total_seconds() // 60)
        return jsonify(error=f"Too many failed attempts. Try again in {minutes} minutes."), 429

    user = User.query.filter_by(email=email).first()
    # Same message for unknown email and wrong password, so emails cannot be probed.
    if user is None or not user.check_password(password):
        audit.record("login_failed", subject=email)
        db.session.commit()
        return jsonify(error="Invalid email or password."), 401

    login_user(user)
    audit.record("login_success", user=user, subject=email)
    db.session.commit()
    return jsonify(user=user.to_dict())


@bp.post("/logout")
@login_required
def logout():
    audit.record("logout", user=current_user, subject=current_user.email)
    db.session.commit()
    logout_user()
    return jsonify(message="Logged out.")


@bp.get("/me")
@login_required
def me():
    return jsonify(user=current_user.to_dict())
