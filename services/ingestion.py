"""Import transaction CSV files: validate, de-duplicate, score and store.

The file's format is recognised from its columns (see ml/datasets.py), and
the model trained on that format scores it. Each import creates a
DataSource row recording what happened. Rows the model flags as suspicious
get an Open fraud case for review, with the inputs that raised their score.
"""

import pandas as pd
from flask import current_app
from sqlalchemy import func, insert, select

from ml import drift
from ml.datasets import LABEL, PROFILES, detect_profile
from models import DataSource, FraudCase, Prediction, Transaction, db, utcnow
from services import audit
from services.prediction import card_history, get_model, model_version

CHUNK_SIZE = 5000


class CSVImportError(ValueError):
    """The file cannot be imported at all (wrong format or columns)."""


def read_csv(file_obj, **options):
    try:
        return pd.read_csv(file_obj, **options)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise CSVImportError(f"Could not read the file as CSV: {exc}") from exc


def format_of(df):
    """The format a CSV's columns match. Raises CSVImportError if none does, or it has no rows."""
    profile, missing = detect_profile(df.columns)
    if len(missing) > len(profile.required_columns) / 2:
        titles = ", ".join(p.title for p in PROFILES.values())
        raise CSVImportError(f"The columns do not match any supported format ({titles}).")
    if missing:
        raise CSVImportError(f"Missing required columns for the {profile.title} format: {', '.join(missing)}")
    if df.empty:
        raise CSVImportError("The file has no rows")
    return profile


def check_file(file_obj):
    """Check a CSV's columns and first rows before it is queued, and rewind it. Returns its format.

    A wrong file is refused at once rather than after waiting for the import
    worker; read_and_validate checks every row later.
    """
    try:
        return format_of(read_csv(file_obj, nrows=5))
    finally:
        file_obj.seek(0)


def read_and_validate(file_obj):
    """Read a CSV. Returns (profile, clean DataFrame, rows received, rows skipped as invalid)."""
    df = read_csv(file_obj)
    profile = format_of(df)
    clean, invalid = profile.clean(df)
    return profile, clean, len(df), invalid


def row_hashes(df, columns):
    """A fingerprint of each row's stored values, as 16-character hex strings."""
    return pd.util.hash_pandas_object(df[columns], index=False).map(lambda h: f"{h:016x}")


def existing_hashes(hashes):
    found = set()
    for start in range(0, len(hashes), CHUNK_SIZE):
        chunk = hashes[start:start + CHUNK_SIZE]
        found.update(db.session.scalars(select(Transaction.row_hash).where(Transaction.row_hash.in_(chunk))))
    return found


def import_transactions(file_obj, name, user=None, source_type="csv_upload", source=None):
    """Import a CSV file object. Returns the DataSource row describing the result.

    Pass `source` to fill in a queued import's row (services/jobs.py); otherwise
    a new one is made. Rows are saved and committed CHUNK_SIZE at a time, with
    `rows_processed` updated, so the import page can show progress and SQLite's
    single write lock is never held for long: a one-minute import would
    otherwise block every login, which also writes, until it finished. If an
    import fails part-way, the rows already saved stay, and importing the file
    again adds the rest, because saved rows are skipped as duplicates.

    Raises CSVImportError for files that cannot be imported, and ModelNotReady
    if no model has been trained for the file's format yet.
    """
    profile, df, received, invalid = read_and_validate(file_obj)
    model = get_model(profile.key)
    version = model_version(model)
    stored = profile.stored_columns(df)

    # Remember the original row numbers (1 = first data row) before filtering further.
    df = df.assign(_row=df.index + 1, _hash=row_hashes(df, stored))
    df = df.drop_duplicates("_hash")
    already = existing_hashes(df["_hash"].tolist())
    df = df[~df["_hash"].isin(already)].reset_index(drop=True)
    duplicates = received - invalid - len(df)

    if source is None:
        source = DataSource(name=name, source_type=source_type, imported_by=user.id if user else None)
        db.session.add(source)
    source.dataset = profile.key
    source.status = "processing"
    source.rows_received = received
    source.rows_processed = 0
    db.session.commit()

    flagged_count = 0
    if len(df):
        # Earlier stored payments of the same cards, so each payment is compared with its card's past.
        history = card_history(df["card_id"]) if "card_id" in df.columns else None
        scores, flagged = model.score(df, history)
        flagged_count = int(flagged.sum())
        source.drift = model.check_drift(df, scores, history)
        explanations = [None] * len(df)
        for i, reasons in zip(flagged.nonzero()[0], model.explain(df[flagged], history)):
            explanations[i] = reasons
        now = utcnow()

        records = df[stored].to_dict("records")
        amounts = profile.amounts(df)
        times = profile.timestamps(df, current_app.config)
        types = profile.transaction_types(df)
        row_numbers = df["_row"].to_numpy()
        hashes = df["_hash"].to_numpy()
        labels = df[LABEL].to_numpy() if LABEL in df.columns else [None] * len(df)
        cards = df["card_id"].to_numpy() if "card_id" in df.columns else [None] * len(df)

        for start in range(0, len(df), CHUNK_SIZE):
            end = start + CHUNK_SIZE
            rows = [
                {
                    "source_id": source.id,
                    "amount": float(amounts[i]),
                    "timestamp": times[i],
                    "transaction_type": types[i],
                    "raw_data_reference": f"{name}#row{int(row_numbers[i])}",
                    "features": records[i],
                    "dataset_label": None if labels[i] is None or pd.isna(labels[i]) else int(labels[i]),
                    "row_hash": hashes[i],
                    "card_id": cards[i],
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
                    "predicted_label": int(flagged[i]),
                    "risk_score": float(scores[i]),
                    "explanation": explanations[i],
                    "created_at": now,
                }
                for tid, i in zip(ids, range(start, end))
            ])
            cases = [
                {"transaction_id": tid, "status": "Open", "review_notes": "", "created_at": now, "updated_at": now}
                for tid, i in zip(ids, range(start, end)) if flagged[i]
            ]
            if cases:
                db.session.execute(insert(FraudCase), cases)
            source.rows_processed = min(end, len(df))
            db.session.commit()

    source.rows_imported = len(df)
    source.rows_skipped = invalid + duplicates
    source.status = "completed"
    source.message = (
        f"Imported {len(df):,} of {received:,} rows as {profile.title}; skipped {invalid:,} invalid and "
        f"{duplicates:,} duplicate rows. Flagged {flagged_count:,} as suspicious."
    )
    warning = drift.describe(source.drift)
    if warning:
        source.message += " " + warning
    audit.record("import_completed", user=user, subject=f"source:{source.id}", details={
        "file": name, "dataset": profile.key, "rows_imported": source.rows_imported,
        "rows_skipped": source.rows_skipped, "flagged": flagged_count,
        "drift": source.drift["level"] if source.drift else None,
    })
    db.session.commit()
    return source


def record_failed_import(name, message, user=None, source_type="csv_upload", source=None):
    """Mark an import failed (making its DataSource row if it has none) and audit it.

    Call after rolling back. Rows an import saved before it failed are counted
    and mentioned, since they stay in the database.
    """
    if source is None:
        source = DataSource(name=name, source_type=source_type, imported_by=user.id if user else None)
        db.session.add(source)
        db.session.flush()
    saved = db.session.scalar(select(func.count(Transaction.id)).where(Transaction.source_id == source.id))
    if saved:
        message += (f" {saved:,} rows were saved before it stopped; import the file again to add the rest"
                    " (saved rows are skipped as duplicates).")
    source.status = "failed"
    source.message = message
    source.rows_imported = saved
    source.file_path = None
    audit.record("import_failed", user=user, subject=f"source:{source.id}", details={"file": name, "error": message})
    db.session.commit()
    return source
