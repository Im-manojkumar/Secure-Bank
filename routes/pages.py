"""HTML pages. Each page is a template; its data is loaded from the JSON API by JavaScript."""

from urllib.parse import urlparse

from flask import Blueprint, abort, redirect, render_template, request, url_for
from flask_login import current_user, login_required

bp = Blueprint("pages", __name__)


def _safe_next(target):
    """Only follow ?next= to a path on this site, never to another domain."""
    if target and urlparse(target).netloc == "" and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for("pages.dashboard")


@bp.get("/favicon.ico")
def favicon():
    # Browsers ask for /favicon.ico before reading the page's <link rel="icon">.
    return redirect(url_for("static", filename="favicon.svg"))


@bp.get("/login")
def login():
    if current_user.is_authenticated:
        return redirect(_safe_next(request.args.get("next")))
    return render_template("login.html", next_url=_safe_next(request.args.get("next")))


@bp.get("/")
@login_required
def dashboard():
    return render_template("dashboard.html", active="dashboard")


@bp.get("/account")
@login_required
def account():
    return render_template("account.html", active="account")


@bp.get("/transactions")
@login_required
def transactions():
    return render_template("transactions.html", active="transactions")


@bp.get("/transactions/<int:transaction_id>")
@login_required
def transaction_detail(transaction_id):
    return render_template("transaction_detail.html", active="transactions", transaction_id=transaction_id)


@bp.get("/alerts")
@login_required
def alerts():
    return render_template("alerts.html", active="alerts")


@bp.get("/cases/<int:case_id>")
@login_required
def case_detail(case_id):
    return render_template("case.html", active="alerts", case_id=case_id)


@bp.get("/reports")
@login_required
def reports():
    return render_template("reports.html", active="reports")


@bp.get("/reports/<int:report_id>")
@login_required
def report_detail(report_id):
    return render_template("report.html", active="reports", report_id=report_id)


@bp.get("/predict")
@login_required
def predict():
    return render_template("predict.html", active="predict")


@bp.get("/import")
@login_required
def import_page():
    if not current_user.is_admin:
        abort(403)
    return render_template("import.html", active="import")


@bp.get("/audit")
@login_required
def audit_page():
    if not current_user.is_admin:
        abort(403)
    return render_template("audit.html", active="audit")
