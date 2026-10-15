"""Access to the trained fraud models from inside the Flask app.

Each dataset format has its own model, saved under MODEL_DIR/<dataset>/.
"""

import pandas as pd
from flask import current_app
from sqlalchemy import select

from ml.datasets import DEFAULT_PROFILE, PROFILES, get_profile
from ml.predict import FraudModel
from models import Transaction, db

HISTORY_CHUNK = 500


class ModelNotReady(RuntimeError):
    pass


def get_model(dataset=DEFAULT_PROFILE):
    """Load a dataset's model once per app and reuse it."""
    models = current_app.extensions.setdefault("fraud_models", {})
    if dataset not in models:
        try:
            models[dataset] = FraudModel(current_app.config["MODEL_DIR"] / dataset)
        except FileNotFoundError:
            raise ModelNotReady(
                f"No trained model for the {get_profile(dataset).title} format yet. "
                f"Run `python -m ml.train --dataset {dataset}` first."
            ) from None
    return models[dataset]


def card_history(card_ids):
    """Stored payments of these cards, in the columns the card-history features read.

    Returns None when there are no cards to look up.
    """
    card_ids = [c for c in dict.fromkeys(card_ids) if isinstance(c, str)]
    if not card_ids:
        return None
    rows = []
    for start in range(0, len(card_ids), HISTORY_CHUNK):
        chunk = card_ids[start:start + HISTORY_CHUNK]
        rows += db.session.execute(
            select(Transaction.card_id, Transaction.timestamp, Transaction.amount)
            .where(Transaction.card_id.in_(chunk))
        ).all()
    return pd.DataFrame(rows, columns=["card_id", "trans_date_trans_time", "amt"])


def model_version(model):
    meta = model.meta
    return f"{meta['model_name']} @ {meta['trained_at']}"


def describe_models():
    """Every dataset format, with its trained model's details when there is one."""
    items = []
    for key, profile in PROFILES.items():
        item = profile.to_dict()
        try:
            model = get_model(key)
        except ModelNotReady:
            item["model"] = None
        else:
            meta = model.meta
            item["model"] = {
                "name": meta["model_name"], "version": model_version(model),
                "threshold": meta["threshold"], "trained_at": meta["trained_at"],
                "test_metrics": meta.get("test_metrics"),
            }
        items.append(item)
    return items


def predict_one(dataset, row):
    """Score one validated row (from the profile's parse_input) and explain the score."""
    model = get_model(dataset)
    df = pd.DataFrame([row])
    scores, flagged = model.score(df)
    return {
        "dataset": dataset,
        "predicted_label": int(flagged[0]),
        "result": "Suspicious" if flagged[0] else "Low risk",
        "risk_score": round(float(scores[0]), 6),
        "threshold": model.threshold,
        "model_version": model_version(model),
        "explanation": model.explain(df)[0],
        "note": "The score ranks how unusual a transaction is; it is not proof of fraud.",
    }
