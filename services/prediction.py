"""Access to the trained fraud model from inside the Flask app."""

import math

import pandas as pd
from flask import current_app

from ml.predict import FraudModel
from ml.preprocess import FEATURES


class ModelNotReady(RuntimeError):
    pass


def get_model():
    """Load the model once per app and reuse it."""
    model = current_app.extensions.get("fraud_model")
    if model is None:
        try:
            model = FraudModel(current_app.config["MODEL_DIR"])
        except FileNotFoundError as exc:
            raise ModelNotReady(str(exc)) from exc
        current_app.extensions["fraud_model"] = model
    return model


def model_version(model):
    meta = model.meta
    return f"{meta['model_name']} @ {meta['trained_at']}"


def parse_features(payload):
    """Validate a JSON object of model inputs. Returns (row dict, list of errors)."""
    if not isinstance(payload, dict):
        return None, ["Body must be a JSON object with the fields " + ", ".join(FEATURES)]
    errors, row = [], {}
    for name in FEATURES:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{name} must be a number")
        elif not math.isfinite(value):
            errors.append(f"{name} must be finite")
        else:
            row[name] = float(value)
    if not errors and row["Amount"] < 0:
        errors.append("Amount cannot be negative")
    return row, errors


def predict_one(row):
    model = get_model()
    scores, flagged = model.score(pd.DataFrame([row], columns=FEATURES))
    return {
        "predicted_label": int(flagged[0]),
        "result": "Suspicious" if flagged[0] else "Low risk",
        "risk_score": round(float(scores[0]), 6),
        "threshold": model.threshold,
        "model_version": model_version(model),
        "note": "The score ranks how unusual a transaction is; it is not proof of fraud.",
    }
