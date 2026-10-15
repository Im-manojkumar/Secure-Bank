"""Transaction search, shared by the transaction list, saved reports and CSV export.

A search is a dict of filters, the same ones /api/transactions takes in its
query string. A saved report stores that dict and runs it again on demand,
so its numbers always reflect the current data.
"""

import csv
import io
from datetime import datetime

from sqlalchemy import case, func, select

from ml.datasets import PROFILES
from models import CASE_STATUSES, DataSource, FraudCase, Prediction, Transaction, currency_of, db, utcnow

FILTER_KEYS = ("dataset", "source_id", "date_from", "date_to", "min_amount", "max_amount", "prediction", "status")
PREDICTIONS = {"suspicious": 1, "low_risk": 0}

EXPORT_COLUMNS = [
    "transaction_id", "time_utc", "dataset", "type", "amount", "currency", "risk_score", "prediction",
    "case_id", "case_status", "case_outcome", "top_reason", "source_row",
]


class FilterError(ValueError):
    pass


def _parse_date(value, name):
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise FilterError(f"{name} must be a date like 2013-09-01 or 2013-09-01T10:30:00") from None


def _parse_float(value, name):
    try:
        return float(value)
    except ValueError:
        raise FilterError(f"{name} must be a number") from None


def clean_filters(args):
    """Keep only known filter keys with a value, as strings. Validates them too."""
    filters = {k: str(args[k]).strip() for k in FILTER_KEYS if args.get(k) not in (None, "")}
    filters = {k: v for k, v in filters.items() if v}
    transaction_query(filters)  # raises FilterError if any value is invalid
    return filters


def transaction_query(filters):
    """A select() of the transactions matching the filters, newest first. Raises FilterError."""
    query = select(Transaction).join(Prediction, Prediction.transaction_id == Transaction.id, isouter=True)
    if "dataset" in filters:
        if filters["dataset"] not in PROFILES:
            raise FilterError(f"dataset must be one of: {', '.join(PROFILES)}")
        query = query.join(DataSource, DataSource.id == Transaction.source_id).where(
            DataSource.dataset == filters["dataset"]
        )
    if "source_id" in filters:
        query = query.where(Transaction.source_id == int(_parse_float(filters["source_id"], "source_id")))
    if "date_from" in filters:
        query = query.where(Transaction.timestamp >= _parse_date(filters["date_from"], "date_from"))
    if "date_to" in filters:
        query = query.where(Transaction.timestamp <= _parse_date(filters["date_to"], "date_to"))
    if "min_amount" in filters:
        query = query.where(Transaction.amount >= _parse_float(filters["min_amount"], "min_amount"))
    if "max_amount" in filters:
        query = query.where(Transaction.amount <= _parse_float(filters["max_amount"], "max_amount"))
    if "prediction" in filters:
        if filters["prediction"] not in PREDICTIONS:
            raise FilterError("prediction must be suspicious or low_risk")
        query = query.where(Prediction.predicted_label == PREDICTIONS[filters["prediction"]])
    if "status" in filters:
        if filters["status"] not in CASE_STATUSES:
            raise FilterError(f"status must be one of: {', '.join(CASE_STATUSES)}")
        query = query.join(FraudCase, FraudCase.transaction_id == Transaction.id).where(
            FraudCase.status == filters["status"]
        )
    return query.order_by(Transaction.timestamp.desc(), Transaction.id.desc())


def _matching_ids(filters):
    return transaction_query(filters).with_only_columns(Transaction.id).order_by(None).subquery()


def count_matches(filters):
    return db.session.scalar(select(func.count()).select_from(_matching_ids(filters)))


def summarize(filters):
    """Totals for the matching transactions: per dataset, and their fraud cases."""
    ids = _matching_ids(filters)
    flagged = Prediction.predicted_label == 1
    per_dataset = db.session.execute(
        select(
            DataSource.dataset,
            func.count(Transaction.id),
            func.sum(Transaction.amount),
            func.sum(case((flagged, 1), else_=0)),
            func.sum(case((flagged, Transaction.amount), else_=0)),
        )
        .select_from(Transaction)
        .join(ids, ids.c.id == Transaction.id)
        .join(DataSource, DataSource.id == Transaction.source_id)
        .join(Prediction, Prediction.transaction_id == Transaction.id, isouter=True)
        .group_by(DataSource.dataset)
        .order_by(DataSource.dataset)
    ).all()
    cases = db.session.execute(
        select(FraudCase.status, FraudCase.outcome, func.count())
        .join(ids, ids.c.id == FraudCase.transaction_id)
        .group_by(FraudCase.status, FraudCase.outcome)
    ).all()

    by_status = {s: 0 for s in CASE_STATUSES}
    outcomes = {}
    for status, outcome, n in cases:
        by_status[status] = by_status.get(status, 0) + n
        if outcome:
            outcomes[outcome] = outcomes.get(outcome, 0) + n

    datasets = [
        {
            "dataset": key,
            "currency": currency_of(key),
            "transactions": n,
            "amount": round(float(amount or 0), 2),
            "suspicious": int(n_flagged or 0),
            "suspicious_amount": round(float(flagged_amount or 0), 2),
        }
        for key, n, amount, n_flagged, flagged_amount in per_dataset
    ]
    return {
        "generated_at": utcnow().isoformat(timespec="seconds"),
        "transactions": sum(d["transactions"] for d in datasets),
        "suspicious": sum(d["suspicious"] for d in datasets),
        "by_dataset": datasets,
        "cases_by_status": by_status,
        "outcomes": outcomes,
    }


def _safe_cell(value):
    """Stop spreadsheet apps from running a text cell as a formula (CSV injection)."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _top_reason(explanation):
    if not explanation:
        return ""
    first = explanation[0]
    return f"{first['feature']}: {first['value']} (typical {first['typical']})"


def export_csv(filters, batch_size=2000):
    """Yield the matching transactions as CSV text, a batch of rows at a time."""
    ids = _matching_ids(filters)
    query = (
        select(
            Transaction.id, Transaction.timestamp, DataSource.dataset, Transaction.transaction_type,
            Transaction.amount, Prediction.risk_score, Prediction.predicted_label, FraudCase.id,
            FraudCase.status, FraudCase.outcome, Prediction.explanation, Transaction.raw_data_reference,
        )
        .select_from(Transaction)
        .join(ids, ids.c.id == Transaction.id)
        .join(DataSource, DataSource.id == Transaction.source_id)
        .join(Prediction, Prediction.transaction_id == Transaction.id, isouter=True)
        .join(FraudCase, FraudCase.transaction_id == Transaction.id, isouter=True)
        .order_by(Transaction.timestamp.desc(), Transaction.id.desc())
        .execution_options(yield_per=batch_size)
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(EXPORT_COLUMNS)
    for partition in db.session.execute(query).partitions():
        for (tid, timestamp, dataset, ttype, amount, score, label, case_id, status, outcome, explanation,
             source_row) in partition:
            writer.writerow([
                tid, timestamp.isoformat(sep=" "), dataset, ttype, f"{amount:.2f}", currency_of(dataset) or "",
                "" if score is None else f"{score:.6f}",
                "" if label is None else ("suspicious" if label else "low risk"),
                case_id or "", status or "", outcome or "",
                _safe_cell(_top_reason(explanation)), _safe_cell(source_row or ""),
            ])
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate()
    yield buffer.getvalue()
