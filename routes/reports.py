"""Saved reports: named transaction searches with totals and a CSV download."""

import re

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from flask_login import current_user, login_required
from sqlalchemy import func, select

from models import SavedReport, db
from routes.auth import role_required
from services import audit
from services.search import FilterError, clean_filters, count_matches, export_csv, summarize

bp = Blueprint("reports", __name__, url_prefix="/api")

NAME_MAX = 120


def _report_dict(report):
    data = report.to_dict()
    data["can_delete"] = current_user.is_admin or report.created_by == current_user.id
    return data


def _csv_response(filters, filename, subject):
    """Stream the matching transactions as a CSV download, within the row limit.

    With ?check=1 it only says whether the download would be allowed, so the
    page can explain a refusal instead of the browser saving an error as a file.
    """
    limit = current_app.config["EXPORT_MAX_ROWS"]
    matches = count_matches(filters)
    if request.args.get("check"):
        return jsonify(rows=matches, limit=limit, allowed=matches <= limit)
    if matches > limit:
        return jsonify(error=f"This search matches {matches:,} transactions. Narrow the filters to "
                             f"{limit:,} or fewer to download it."), 400
    audit.record("report_exported", user=current_user, subject=subject,
                 details={"filters": filters, "rows": matches})
    db.session.commit()
    return Response(
        stream_with_context(export_csv(filters)),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'},
    )


def _filename(name):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return f"securebank-{slug or 'report'}"


@bp.get("/reports")
@login_required
def list_reports():
    reports = db.session.scalars(select(SavedReport).order_by(SavedReport.created_at.desc(), SavedReport.id.desc()))
    return jsonify(items=[_report_dict(r) for r in reports])


@bp.post("/reports")
@role_required("admin", "analyst")
def create_report():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name") or "").strip()
    if not name:
        return jsonify(error="Give the report a name."), 400
    if len(name) > NAME_MAX:
        return jsonify(error=f"The name can be at most {NAME_MAX} characters."), 400
    if not isinstance(payload.get("filters", {}), dict):
        return jsonify(error="filters must be an object."), 400
    try:
        filters = clean_filters(payload.get("filters") or {})
    except FilterError as exc:
        return jsonify(error=str(exc)), 400
    taken = db.session.scalar(select(SavedReport.id).where(func.lower(SavedReport.name) == name.lower()))
    if taken:
        return jsonify(error="A report with this name already exists."), 409

    report = SavedReport(name=name, filters=filters, created_by=current_user.id)
    db.session.add(report)
    db.session.flush()
    audit.record("report_saved", user=current_user, subject=f"report:{report.id}",
                 details={"name": name, "filters": filters})
    db.session.commit()
    return jsonify(report=_report_dict(report)), 201


@bp.get("/reports/<int:report_id>")
@login_required
def get_report(report_id):
    report = db.get_or_404(SavedReport, report_id)
    return jsonify(report=_report_dict(report), summary=summarize(report.filters))


@bp.delete("/reports/<int:report_id>")
@role_required("admin", "analyst")
def delete_report(report_id):
    report = db.get_or_404(SavedReport, report_id)
    if not (current_user.is_admin or report.created_by == current_user.id):
        return jsonify(error="Only the person who saved a report, or an admin, can delete it."), 403
    audit.record("report_deleted", user=current_user, subject=f"report:{report.id}", details={"name": report.name})
    db.session.delete(report)
    db.session.commit()
    return "", 204


@bp.get("/reports/<int:report_id>/export")
@login_required
def export_report(report_id):
    report = db.get_or_404(SavedReport, report_id)
    return _csv_response(report.filters, _filename(report.name), f"report:{report.id}")


@bp.get("/transactions/export")
@login_required
def export_search():
    """Download the current transaction search, with the same filters as /api/transactions."""
    try:
        filters = clean_filters(request.args)
    except FilterError as exc:
        return jsonify(error=str(exc)), 400
    return _csv_response(filters, "securebank-transactions", "search")
