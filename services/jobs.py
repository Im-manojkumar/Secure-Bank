"""Run uploaded CSV imports in the background.

An upload is checked (format, columns, a trained model), saved to UPLOAD_DIR
and queued as a DataSource row with status "queued". A worker claims queued
imports one at a time and runs them, so a large file no longer holds the
upload request open, and the import page polls GET /api/sources/<id> for
progress. The queue is the data_sources table itself, so no extra service
(Redis, Celery) is needed.

IMPORT_MODE decides who runs the worker:
- "thread": a background thread in each web process, started by the first upload
  or visit to the import page. Good for `flask run` and a single server.
- "worker": a separate `flask import-worker` process (docker-compose.yml runs one).
  Web processes only queue, so restarting them never cuts an import short.
- "inline": the upload request runs the import itself before answering (tests).
"""

import logging
import threading
import time
import uuid
from datetime import timedelta
from pathlib import Path

from flask import current_app
from sqlalchemy import select, update

from models import DataSource, User, db, utcnow
from services.ingestion import CSVImportError, import_transactions, record_failed_import
from services.prediction import ModelNotReady

log = logging.getLogger(__name__)

# An import that has not saved progress for this long was cut off, say by a restart.
STALE_AFTER = timedelta(minutes=30)
POLL_SECONDS = 2


def queue_import(upload, name, profile, user):
    """Save an uploaded file and queue it. Returns the queued DataSource row."""
    folder = Path(current_app.config["UPLOAD_DIR"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}.csv"
    upload.save(path)
    source = DataSource(name=name, source_type="csv_upload", dataset=profile.key, status="queued",
                        file_path=str(path), imported_by=user.id if user else None)
    db.session.add(source)
    db.session.commit()
    return source


def fail_stale_imports():
    """Mark imports failed whose worker stopped mid-way (a crash or restart)."""
    cutoff = utcnow() - STALE_AFTER
    for source in db.session.scalars(select(DataSource).where(
            DataSource.status == "processing", DataSource.updated_at < cutoff)):
        if source.file_path:
            Path(source.file_path).unlink(missing_ok=True)
        user = db.session.get(User, source.imported_by) if source.imported_by else None
        record_failed_import(source.name, "The import stopped part-way, probably because the server restarted.",
                             user, source=source)


def claim_next():
    """Take the oldest queued import, so no other worker runs it too. Returns it, or None."""
    while True:
        candidate = db.session.scalar(
            select(DataSource.id).where(DataSource.status == "queued").order_by(DataSource.id).limit(1))
        if candidate is None:
            return None
        # Only one worker's update can still find the row queued.
        claimed = db.session.execute(
            update(DataSource).where(DataSource.id == candidate, DataSource.status == "queued")
            .values(status="processing", updated_at=utcnow())
        ).rowcount
        db.session.commit()
        if claimed:
            return db.session.get(DataSource, candidate)


def run_import(source):
    """Import a claimed upload, record how it went, and delete the file."""
    user = db.session.get(User, source.imported_by) if source.imported_by else None
    source_id, name, path = source.id, source.name, Path(source.file_path)
    try:
        with open(path, "rb") as f:
            import_transactions(f, name, user, source=source)
    except (CSVImportError, ModelNotReady, FileNotFoundError) as exc:
        db.session.rollback()
        message = "The uploaded file is missing." if isinstance(exc, FileNotFoundError) else str(exc)
        record_failed_import(name, message, user, source=db.session.get(DataSource, source_id))
    except Exception:
        db.session.rollback()
        log.exception("Import of %s (source %s) failed", name, source_id)
        record_failed_import(name, "Unexpected error during import.", user, source=db.session.get(DataSource, source_id))
    finally:
        path.unlink(missing_ok=True)
    source = db.session.get(DataSource, source_id)
    source.file_path = None
    db.session.commit()
    return source


def run_pending():
    """Run queued imports until none are left. Returns how many ran."""
    fail_stale_imports()
    count = 0
    while (source := claim_next()) is not None:
        run_import(source)
        count += 1
    return count


def work(app, once=False):
    """The worker loop for `flask import-worker`."""
    while True:
        with app.app_context():
            ran = run_pending()
        if once:
            return ran
        if not ran:
            time.sleep(POLL_SECONDS)


_thread = None
_thread_lock = threading.Lock()
_wake = threading.Event()


def ensure_worker():
    """Make sure queued imports will run: start the thread, or run them now, by IMPORT_MODE."""
    global _thread
    mode = current_app.config["IMPORT_MODE"]
    if mode == "inline":
        run_pending()
    elif mode == "thread":
        app = current_app._get_current_object()
        with _thread_lock:
            if _thread is None or not _thread.is_alive():
                _thread = threading.Thread(target=_thread_loop, args=(app,), name="import-worker", daemon=True)
                _thread.start()
        _wake.set()


def _thread_loop(app):
    while True:
        _wake.clear()
        try:
            with app.app_context():
                ran = run_pending()
        except Exception:
            log.exception("The import thread hit an error")
            ran = 0
        if not ran:
            _wake.wait(POLL_SECONDS * 5)
