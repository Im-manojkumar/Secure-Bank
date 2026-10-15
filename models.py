"""Database tables.

The imported transaction, the model's prediction and the analyst's case
outcome are stored in separate tables: a prediction is not proof of fraud,
and a reviewer's decision must not overwrite what the model said.
"""

from datetime import datetime, timezone

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import JSON
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()

ROLES = ("admin", "analyst")
CASE_STATUSES = ("Open", "Under Review", "Resolved")
CASE_OUTCOMES = ("Confirmed fraud", "Legitimate")


def utcnow():
    """Current UTC time without tzinfo; SQLite stores datetimes without a zone, so all times are UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="analyst")
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == "admin"

    def to_dict(self):
        return {"id": self.id, "email": self.email, "role": self.role}


class DataSource(db.Model):
    """One imported file or feed, with how its import went."""

    __tablename__ = "data_sources"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    source_type = db.Column(db.String(50), nullable=False, default="csv_upload")
    status = db.Column(db.String(20), nullable=False, default="processing")
    rows_received = db.Column(db.Integer, default=0)
    rows_imported = db.Column(db.Integer, default=0)
    rows_skipped = db.Column(db.Integer, default=0)
    message = db.Column(db.Text)
    imported_by = db.Column(db.Integer, db.ForeignKey("users.id"))
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "status": self.status,
            "rows_received": self.rows_received,
            "rows_imported": self.rows_imported,
            "rows_skipped": self.rows_skipped,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Transaction(db.Model):
    __tablename__ = "transactions"

    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey("data_sources.id"), nullable=False, index=True)
    amount = db.Column(db.Float, nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    transaction_type = db.Column(db.String(50), nullable=False, default="card")
    # Row number in the original file, so a record can be traced back to its source.
    raw_data_reference = db.Column(db.String(100))
    # Model inputs (Time, V1..V28, Amount) exactly as imported.
    features = db.Column(JSON, nullable=False)
    # The label shipped with the dataset, when it has one. Kept for evaluation only.
    dataset_label = db.Column(db.Integer)
    # Fingerprint of the row's values, used to skip duplicates across imports.
    row_hash = db.Column(db.String(40), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    source = db.relationship("DataSource")
    prediction = db.relationship(
        "Prediction", uselist=False, back_populates="transaction",
        order_by="desc(Prediction.created_at)",
    )
    case = db.relationship("FraudCase", uselist=False, back_populates="transaction")

    def to_dict(self, include_features=False):
        data = {
            "id": self.id,
            "source_id": self.source_id,
            "amount": self.amount,
            "timestamp": self.timestamp.isoformat(),
            "transaction_type": self.transaction_type,
            "raw_data_reference": self.raw_data_reference,
            "dataset_label": self.dataset_label,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "case": self.case.to_dict(include_transaction=False) if self.case else None,
        }
        if include_features:
            data["features"] = self.features
        return data


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.Integer, db.ForeignKey("transactions.id"), nullable=False, index=True)
    model_version = db.Column(db.String(100), nullable=False)
    predicted_label = db.Column(db.Integer, nullable=False, index=True)
    risk_score = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    transaction = db.relationship("Transaction", back_populates="prediction")

    def to_dict(self):
        return {
            "model_version": self.model_version,
            "predicted_label": self.predicted_label,
            "risk_score": round(self.risk_score, 6),
            "created_at": self.created_at.isoformat(),
        }


class FraudCase(db.Model):
    """An alert raised for a suspicious transaction and its investigation."""

    __tablename__ = "fraud_cases"

    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.Integer, db.ForeignKey("transactions.id"), unique=True, nullable=False)
    status = db.Column(db.String(20), nullable=False, default="Open", index=True)
    # Set when a case is resolved: the reviewer's finding, separate from the prediction.
    outcome = db.Column(db.String(30))
    review_notes = db.Column(db.Text, default="")
    reviewed_by = db.Column(db.Integer, db.ForeignKey("users.id"))
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    transaction = db.relationship("Transaction", back_populates="case")
    reviewer = db.relationship("User")

    def to_dict(self, include_transaction=True):
        data = {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "status": self.status,
            "outcome": self.outcome,
            "review_notes": self.review_notes,
            "reviewed_by": self.reviewer.email if self.reviewer else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if include_transaction:
            t = self.transaction
            data["transaction"] = {
                "id": t.id,
                "amount": t.amount,
                "timestamp": t.timestamp.isoformat(),
                "risk_score": round(t.prediction.risk_score, 6) if t.prediction else None,
            }
        return data


class AuditLog(db.Model):
    """Who did what and when, for sensitive actions. Rows are only ever added."""

    __tablename__ = "audit_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), index=True)
    action = db.Column(db.String(50), nullable=False, index=True)
    # What the action was about, e.g. the email tried at login or "case:12".
    subject = db.Column(db.String(255), index=True)
    details = db.Column(JSON)
    ip_address = db.Column(db.String(45))
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False, index=True)

    user = db.relationship("User")

    def to_dict(self):
        return {
            "id": self.id,
            "user": self.user.email if self.user else None,
            "action": self.action,
            "subject": self.subject,
            "details": self.details,
            "ip_address": self.ip_address,
            "created_at": self.created_at.isoformat(),
        }
