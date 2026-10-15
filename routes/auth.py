"""Login, logout and role checks."""

from functools import wraps

from flask import Blueprint, current_app, jsonify, request, session
from flask_login import current_user, login_required, login_user, logout_user

from models import User, db
from services import audit, totp

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

    if user.two_factor_enabled:
        code = data.get("code")
        if not code:
            # Not a failed attempt: the page now asks for the code.
            return jsonify(error="Enter the 6-digit code from your authenticator app.",
                           two_factor_required=True), 401
        step = totp.matching_step(user.totp_secret, code, user.totp_last_step)
        if step is None:
            audit.record("login_failed", subject=email, details={"reason": "wrong two-factor code"})
            db.session.commit()
            return jsonify(error="That code is wrong or has expired. Try the newest one.",
                           two_factor_required=True), 401
        user.totp_last_step = step

    login_user(user)
    audit.record("login_success", user=user, subject=email)
    db.session.commit()
    return jsonify(user=user.to_dict(), must_set_up_two_factor=two_factor_missing(user))


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


def two_factor_missing(user):
    """True when this user must set up two-factor login before doing anything else."""
    return user.is_admin and not user.two_factor_enabled and current_app.config["REQUIRE_2FA_FOR_ADMINS"]


@bp.post("/2fa/setup")
@login_required
def two_factor_setup():
    """Start setup: a new secret to add to an authenticator app. Nothing changes until it is confirmed."""
    if current_user.two_factor_enabled:
        return jsonify(error="Two-factor login is already on."), 400
    secret = totp.new_secret()
    session["pending_totp_secret"] = secret
    return jsonify(secret=secret, uri=totp.setup_uri(secret, current_user.email))


@bp.post("/2fa/enable")
@login_required
def two_factor_enable():
    """Finish setup by proving the app shows the right code."""
    secret = session.get("pending_totp_secret")
    if current_user.two_factor_enabled:
        return jsonify(error="Two-factor login is already on."), 400
    if not secret:
        return jsonify(error="Start the setup first."), 400
    step = totp.matching_step(secret, (request.get_json(silent=True) or {}).get("code"))
    if step is None:
        return jsonify(error="That code does not match. Check the app and try the newest code."), 400
    current_user.totp_secret, current_user.totp_last_step = secret, step
    session.pop("pending_totp_secret", None)
    audit.record("two_factor_enabled", user=current_user, subject=current_user.email)
    db.session.commit()
    return jsonify(user=current_user.to_dict())


@bp.post("/2fa/disable")
@login_required
def two_factor_disable():
    """Turn two-factor login off, with the password and a current code. Admins cannot when it is required."""
    if not current_user.two_factor_enabled:
        return jsonify(error="Two-factor login is not on."), 400
    if current_user.is_admin and current_app.config["REQUIRE_2FA_FOR_ADMINS"]:
        return jsonify(error="Admins must keep two-factor login on."), 403
    data = request.get_json(silent=True) or {}
    step = totp.matching_step(current_user.totp_secret, data.get("code"), current_user.totp_last_step)
    if not current_user.check_password(str(data.get("password", ""))) or step is None:
        return jsonify(error="Wrong password or code."), 400
    current_user.totp_secret = current_user.totp_last_step = None
    audit.record("two_factor_disabled", user=current_user, subject=current_user.email)
    db.session.commit()
    return jsonify(user=current_user.to_dict())
