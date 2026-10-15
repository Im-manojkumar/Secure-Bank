"""Score one transaction as it happens, for another system calling the API.

A payment system sends a transaction before approving it and gets back a
decision: approve, review (approve, but open a case for an analyst) or
decline. The transaction, its prediction and any case are stored exactly as
an imported one would be, so it shows up on the dashboard, in searches and
in the alert queue.
"""

import hashlib
import secrets
import time

import pandas as pd
from flask import current_app
from sqlalchemy import select

from ml.datasets import LABEL, PROFILES, card_ids
from models import ApiKey, DataSource, FraudCase, Prediction, Transaction, db, utcnow
from services.ingestion import row_hashes
from services.prediction import card_history, get_model, model_version

KEY_PREFIX = "sbk_"


def hash_key(key):
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(name):
    """Make a new API key. Returns (ApiKey row, the key itself, shown only now)."""
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    row = ApiKey(name=name, prefix=key[:12], key_hash=hash_key(key))
    db.session.add(row)
    db.session.flush()
    return row, key


def find_key(key):
    """The active ApiKey for this key, or None."""
    if not key or not key.startswith(KEY_PREFIX):
        return None
    row = db.session.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(key)))
    if row is None or row.revoked_at is not None:
        return None
    return row


def api_source(api_key, dataset):
    """The data source that collects one key's live transactions of one format."""
    name = f"Live API: {api_key.name}"
    source = db.session.scalar(select(DataSource).where(
        DataSource.name == name, DataSource.dataset == dataset, DataSource.source_type == "api"))
    if source is None:
        source = DataSource(name=name, source_type="api", dataset=dataset, status="completed",
                            rows_received=0, rows_imported=0, rows_skipped=0,
                            message="Transactions scored one at a time through the live API.")
        db.session.add(source)
        db.session.flush()
    return source


def decide(score, flagged):
    if score >= current_app.config["DECLINE_SCORE"]:
        return "decline"
    return "review" if flagged else "approve"


def score_live(api_key, dataset, row, card_number=None):
    """Score, store and decide on one validated transaction. Returns the response body."""
    started = time.perf_counter()
    profile = PROFILES[dataset]
    model = get_model(dataset)
    df = pd.DataFrame([row])
    history = None
    if card_number and "cc_num" in profile.raw_only:
        df["card_last4"] = str(card_number).strip()[-4:]
        df["card_id"] = card_ids(pd.Series([card_number])).to_numpy()
        history = card_history(df["card_id"])

    stored = profile.stored_columns(df)
    row_hash = row_hashes(df, stored).iloc[0]
    existing = db.session.scalar(select(Transaction).where(Transaction.row_hash == row_hash))
    if existing is not None and existing.prediction is not None:
        # The same transaction sent twice (a retry): answer as before, store nothing new.
        p = existing.prediction
        return {"transaction_id": existing.id, "duplicate": True, "risk_score": round(p.risk_score, 6),
                "decision": decide(p.risk_score, bool(p.predicted_label)), "threshold": model.threshold,
                "reasons": p.explanation or [], "model_version": p.model_version,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    scores, flagged = model.score(df, history)
    score, is_flagged = float(scores[0]), bool(flagged[0])
    reasons = model.explain(df, history)[0] if is_flagged else []

    source = api_source(api_key, dataset)
    transaction = Transaction(
        source_id=source.id, amount=float(profile.amounts(df)[0]),
        timestamp=profile.timestamps(df, current_app.config)[0],
        transaction_type=profile.transaction_types(df)[0],
        raw_data_reference=f"api:{api_key.prefix}", features=df[stored].to_dict("records")[0],
        row_hash=row_hash, card_id=df["card_id"].iloc[0] if "card_id" in df.columns else None,
    )
    db.session.add(transaction)
    db.session.flush()
    db.session.add(Prediction(transaction_id=transaction.id, model_version=model_version(model),
                              predicted_label=int(is_flagged), risk_score=score, explanation=reasons or None))
    if is_flagged:
        db.session.add(FraudCase(transaction_id=transaction.id, status="Open", review_notes=""))
    source.rows_received += 1
    source.rows_imported += 1
    api_key.last_used_at = utcnow()
    db.session.commit()

    return {
        "transaction_id": transaction.id,
        "duplicate": False,
        "risk_score": round(score, 6),
        "decision": decide(score, is_flagged),
        "threshold": model.threshold,
        "reasons": reasons,
        "model_version": model_version(model),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
