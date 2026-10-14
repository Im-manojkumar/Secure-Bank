"""Import transaction CSV files: validate, de-duplicate, score and store.

Each import creates a DataSource row recording what happened. Rows the
model flags as suspicious get an Open fraud case for review.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from flask import current_app
from sqlalchemy import insert, select

from ml.preprocess import FEATURES, TARGET
from models import DataSource, FraudCase, Prediction, Transaction, db, utcnow
from services import audit
from services.prediction import get_model, model_version

CHUNK_SIZE = 5000


class CSVImportError(ValueError):
    """The file cannot be imported at all (wrong format or columns)."""


def read_and_validate(file_obj):
    """Read a CSV and return (clean DataFrame, rows received, rows skipped as invalid)."""
    try:
        df = pd.read_csv(file_obj)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise CSVImportError(f"Could not read the file as CSV: {exc}") from exc

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise CSVImportError(f"Missing required columns: {', '.join(missing)}")

    received = len(df)
    if received == 0:
        raise CSVImportError("The file has no rows")

    columns = FEATURES + ([TARGET] if TARGET in df.columns else [])
    # Always float64, so the same values hash the same whichever way pandas read them.
    df = df[columns].apply(pd.to_numeric, errors="coerce").astype(float)

    valid = np.isfinite(df[FEATURES]).all(axis=1) & (df["Amount"] >= 0)
    if TARGET in df.columns:
        # A label is optional per row, but when present it must be 0 or 1.
        valid &= df[TARGET].isna() | df[TARGET].isin([0, 1])
    df = df[valid]
    return df, received, received - len(df)


def row_hashes(df):
    """A fingerprint of each row's feature values, as 16-character hex strings."""
    return pd.util.hash_pandas_object(df[FEATURES], index=False).map(lambda h: f"{h:016x}")


def existing_hashes(hashes):
    found = set()
    for start in range(0, len(hashes), CHUNK_SIZE):
        chunk = hashes[start:start + CHUNK_SIZE]
        found.update(db.session.scalars(select(Transaction.row_hash).where(Transaction.row_hash.in_(chunk))))
    return found


def import_transactions(file_obj, name, user=None, source_type="csv_upload"):
    """Import a CSV file object. Returns the DataSource row describing the result.

    Raises CSVImportError for files that cannot be imported, and ModelNotReady
    if no trained model exists yet.
    """
    model = get_model()
    version = model_version(model)
    start_time = current_app.config["DATASET_START"]

    df, received, invalid = read_and_validate(file_obj)

    # Remember the original row numbers (1 = first data row) before filtering further.
    df = df.assign(_row=df.index + 1, _hash=row_hashes(df))
    df = df.drop_duplicates("_hash")
    already = existing_hashes(df["_hash"].tolist())
    df = df[~df["_hash"].isin(already)]
    duplicates = received - invalid - len(df)

    source = DataSource(
        name=name, source_type=source_type, rows_received=received,
        imported_by=user.id if user else None,
    )
    db.session.add(source)
    db.session.flush()

    flagged_count = 0
    if len(df):
        scores, flagged = model.score(df)
        flagged_count = int(flagged.sum())
        has_label = TARGET in df.columns
        now = utcnow()

        records = df[FEATURES].to_dict("records")
        amounts = df["Amount"].to_numpy(dtype=float)
        times = df["Time"].to_numpy(dtype=float)
        row_numbers = df["_row"].to_numpy()
        hashes = df["_hash"].to_numpy()
        labels = df[TARGET].to_numpy() if has_label else [None] * len(df)

        for start in range(0, len(df), CHUNK_SIZE):
            end = start + CHUNK_SIZE
            rows = [
                {
                    "source_id": source.id,
                    "amount": float(amounts[i]),
                    "timestamp": start_time + timedelta(seconds=float(times[i])),
                    "transaction_type": "card",
                    "raw_data_reference": f"{name}#row{int(row_numbers[i])}",
                    "features": records[i],
                    "dataset_label": None if labels[i] is None or pd.isna(labels[i]) else int(labels[i]),
                    "row_hash": hashes[i],
                    "created_at": now,
                }
                for i in range(start, min(end, len(df)))
            ]
            ids = db.session.scalars(
                insert(Transaction).returning(Transaction.id, sort_by_parameter_order=True), rows
            ).all()

            db.session.execute(insert(Prediction), [
                {
                    "transaction_id": tid,
                    "model_version": version,
                    "predicted_label": int(flag),
                    "risk_score": float(score),
                    "created_at": now,
                }
                for tid, score, flag in zip(ids, scores[start:end], flagged[start:end])
            ])
            cases = [
                {"transaction_id": tid, "status": "Open", "review_notes": "", "created_at": now, "updated_at": now}
                for tid, flag in zip(ids, flagged[start:end]) if flag
            ]
            if cases:
                db.session.execute(insert(FraudCase), cases)

    source.rows_imported = len(df)
    source.rows_skipped = invalid + duplicates
    source.status = "completed"
    source.message = (
        f"Imported {len(df):,} of {received:,} rows; skipped {invalid:,} invalid and "
        f"{duplicates:,} duplicate rows. Flagged {flagged_count:,} as suspicious."
    )
    audit.record("import_completed", user=user, subject=f"source:{source.id}", details={
        "file": name, "rows_imported": source.rows_imported, "rows_skipped": source.rows_skipped,
        "flagged": flagged_count,
    })
    db.session.commit()
    return source


def record_failed_import(name, message, user=None, source_type="csv_upload"):
    source = DataSource(
        name=name, source_type=source_type, status="failed", message=message,
        imported_by=user.id if user else None,
    )
    db.session.add(source)
    db.session.flush()
    audit.record("import_failed", user=user, subject=f"source:{source.id}", details={"file": name, "error": message})
    db.session.commit()
    return source
