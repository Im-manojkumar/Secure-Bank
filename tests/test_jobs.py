"""Background imports: queueing, the worker, progress, and imports cut off part-way."""

import time
from datetime import timedelta

import pytest

import services.ingestion
from app import create_app
from config import TestConfig
from ml.sample_data import make_sample
from models import AuditLog, DataSource, Transaction, User, db, utcnow
from services import jobs
from tests.conftest import csv_upload, login


def upload(client, df, name="queued.csv"):
    return client.post("/api/transactions/import", data=csv_upload(df, name), content_type="multipart/form-data")


@pytest.fixture
def worker_mode(app):
    app.config["IMPORT_MODE"] = "worker"
    return app


def test_upload_is_queued_until_the_worker_runs_it(worker_mode, admin):
    app = worker_mode
    response = upload(admin, make_sample(rows=300, fraud_rate=0.05, random_state=11))
    assert response.status_code == 202
    source = response.get_json()["source"]
    assert source["status"] == "queued" and source["dataset"] == "creditcard"
    assert len(list(app.config["UPLOAD_DIR"].glob("*.csv"))) == 1
    with app.app_context():
        assert db.session.query(Transaction).count() == 0

    result = app.test_cli_runner().invoke(args=["import-worker", "--once"])
    assert "Ran 1 imports." in result.output

    done = admin.get(f"/api/sources/{source['id']}").get_json()["source"]
    assert done["status"] == "completed"
    assert done["rows_imported"] == done["rows_processed"] == 300
    assert list(app.config["UPLOAD_DIR"].glob("*.csv")) == []
    with app.app_context():
        entry = db.session.query(AuditLog).filter_by(action="import_completed").one()
        assert entry.user.email == "admin@test.local"


def test_wrong_files_are_refused_before_queueing(worker_mode, admin):
    response = upload(admin, make_sample(rows=5).drop(columns=["V1"]))
    assert response.status_code == 400
    assert response.get_json()["source"]["status"] == "failed"
    assert list(worker_mode.config["UPLOAD_DIR"].glob("*.csv")) == []


def test_anyone_logged_in_can_follow_an_import(analyst, imported):
    source_id = analyst.get("/api/sources").get_json()["items"][0]["id"]
    assert analyst.get(f"/api/sources/{source_id}").get_json()["source"]["status"] == "completed"
    assert analyst.get("/api/sources/9999").status_code == 404


def test_imports_are_claimed_once_oldest_first(app):
    with app.app_context():
        first, second = DataSource(name="a.csv", status="queued"), DataSource(name="b.csv", status="queued")
        db.session.add_all([first, second])
        db.session.commit()
        assert jobs.claim_next().id == first.id
        assert jobs.claim_next().id == second.id
        assert jobs.claim_next() is None
        assert {s.status for s in db.session.query(DataSource)} == {"processing"}


def test_an_import_cut_off_part_way_is_marked_failed(app):
    with app.app_context():
        stuck = DataSource(name="stuck.csv", status="processing", updated_at=utcnow() - timedelta(hours=1))
        running = DataSource(name="running.csv", status="processing")
        db.session.add_all([stuck, running])
        db.session.commit()
        jobs.run_pending()
        assert db.session.get(DataSource, stuck.id).status == "failed"
        assert "stopped part-way" in db.session.get(DataSource, stuck.id).message
        assert db.session.get(DataSource, running.id).status == "processing"


def test_a_failed_import_keeps_saved_rows_and_can_be_finished(app, admin, monkeypatch):
    """Rows are committed in chunks; importing the file again adds only the rest."""
    df = make_sample(rows=500, fraud_rate=0.05, random_state=12)
    monkeypatch.setattr(services.ingestion, "CHUNK_SIZE", 100)
    real_insert = services.ingestion.insert
    chunks = []

    def failing_insert(table):
        if table is Transaction:
            chunks.append(1)
            if len(chunks) == 3:
                raise RuntimeError("disk full")
        return real_insert(table)

    monkeypatch.setattr(services.ingestion, "insert", failing_insert)
    response = upload(admin, df, "big.csv")
    assert response.status_code == 400
    failed = response.get_json()["source"]
    assert failed["status"] == "failed" and failed["rows_imported"] == 200
    assert "200 rows were saved before it stopped" in failed["message"]

    monkeypatch.setattr(services.ingestion, "insert", real_insert)
    again = upload(admin, df, "big.csv").get_json()["source"]
    assert again["status"] == "completed"
    assert again["rows_imported"] == 300 and again["rows_skipped"] == 200
    with app.app_context():
        assert db.session.query(Transaction).count() == 500


def test_thread_mode_imports_in_the_background(model_dir, tmp_path):
    class Config(TestConfig):
        MODEL_DIR = model_dir
        UPLOAD_DIR = tmp_path / "uploads"
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path / 'thread.db'}"
        IMPORT_MODE = "thread"

    app = create_app(Config)
    with app.app_context():
        db.create_all()
        user = User(email="admin@test.local", role="admin")
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()
    client = app.test_client()
    login(client)
    response = upload(client, make_sample(rows=300, fraud_rate=0.05, random_state=13))
    assert response.status_code == 202
    source_id = response.get_json()["source"]["id"]

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        source = client.get(f"/api/sources/{source_id}").get_json()["source"]
        if source["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert source["status"] == "completed", source
    assert source["rows_imported"] == 300
