"""Login, logout and role checks."""

from functools import wraps

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required, login_user, logout_user

from models import User

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
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    user = User.query.filter_by(email=email).first()
    # Same message for unknown email and wrong password, so emails cannot be probed.
    if user is None or not user.check_password(password):
        return jsonify(error="Invalid email or password."), 401
    login_user(user)
    return jsonify(user=user.to_dict())


@bp.post("/logout")
@login_required
def logout():
    logout_user()
    return jsonify(message="Logged out.")


@bp.get("/me")
@login_required
def me():
    return jsonify(user=current_user.to_dict())
