"""Shared page/per_page handling for list endpoints."""

from flask import current_app, jsonify, request

from models import db


def paginate(query, serialize):
    """Run a select() with ?page= and ?per_page= and return a JSON response."""
    try:
        page = max(int(request.args.get("page", 1)), 1)
        per_page = int(request.args.get("per_page", current_app.config["PER_PAGE_DEFAULT"]))
    except ValueError:
        return jsonify(error="page and per_page must be whole numbers"), 400
    per_page = min(max(per_page, 1), current_app.config["PER_PAGE_MAX"])

    result = db.paginate(query, page=page, per_page=per_page, error_out=False)
    return jsonify(
        items=[serialize(item) for item in result.items],
        page=page,
        per_page=per_page,
        total=result.total,
        pages=result.pages,
    )
