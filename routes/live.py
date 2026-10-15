"""The live scoring API, for other systems rather than people.

    POST /api/v1/score
    Authorization: Bearer sbk_...
    {"dataset": "sparkov", "transaction": {...the format's columns...}, "card_number": "4111..."}

Keys are made with `flask --app app create-api-key NAME`. card_number is
optional and only used by formats that compare a card with its history
(Sparkov); it is never stored, only its keyed hash and last four digits.
"""

from flask import Blueprint, jsonify, request

from ml.datasets import PROFILES
from services.live import find_key, score_live
from services.prediction import ModelNotReady

bp = Blueprint("live", __name__, url_prefix="/api/v1")


@bp.post("/score")
def score():
    header = request.headers.get("Authorization", "")
    key = find_key(header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else None)
    if key is None:
        return jsonify(error="Send a valid API key as 'Authorization: Bearer <key>'."), 401

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Body must be a JSON object with dataset and transaction."), 400
    dataset = payload.get("dataset")
    if dataset not in PROFILES:
        return jsonify(error=f"dataset must be one of: {', '.join(PROFILES)}"), 400
    row, errors = PROFILES[dataset].parse_input(payload.get("transaction"))
    if errors:
        return jsonify(error="Invalid transaction.", details=errors), 400
    card_number = payload.get("card_number")
    if card_number is not None and (not isinstance(card_number, str) or not card_number.strip().isdigit()):
        return jsonify(error="card_number must be a string of digits."), 400
    try:
        return jsonify(score_live(key, dataset, row, card_number))
    except ModelNotReady as exc:
        return jsonify(error=str(exc)), 503
