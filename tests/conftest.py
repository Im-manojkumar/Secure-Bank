import io

import pytest

from app import create_app
from config import TestConfig
from ml.sample_data import make_paysim_sample, make_sample, make_sparkov_sample
from ml.train import train
from models import User, db


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Train a small model per dataset format on fake data, once for the whole test run."""
    data_dir = tmp_path_factory.mktemp("data")
    out = tmp_path_factory.mktemp("models")
    for dataset, maker, rows in (("creditcard", make_sample, 4000), ("sparkov", make_sparkov_sample, 3000),
                                 ("paysim", make_paysim_sample, 3000)):
        maker(rows=rows, fraud_rate=0.03).to_csv(data_dir / f"{dataset}.csv", index=False)
        train(data_dir / f"{dataset}.csv", out_dir=out / dataset, dataset=dataset)
    return out


@pytest.fixture
def app(model_dir, tmp_path):
    class Config(TestConfig):
        MODEL_DIR = model_dir
        UPLOAD_DIR = tmp_path / "uploads"

    app = create_app(Config)
    # Tests must not hold an app context open while making requests: Flask would
    # reuse it, and Flask-Login's cached user in `g` would leak between requests.
    with app.app_context():
        db.create_all()
        for email, role in (("admin@test.local", "admin"), ("analyst@test.local", "analyst")):
            user = User(email=email, role=role)
            user.set_password("password123")
            db.session.add(user)
        db.session.commit()
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def login(client, role="admin"):
    response = client.post("/api/auth/login", json={"email": f"{role}@test.local", "password": "password123"})
    assert response.status_code == 200
    return response


@pytest.fixture
def admin(client):
    login(client, "admin")
    return client


@pytest.fixture
def analyst(client):
    login(client, "analyst")
    return client


def csv_upload(df, name="upload.csv"):
    buffer = io.BytesIO(df.to_csv(index=False).encode())
    return {"file": (buffer, name)}


@pytest.fixture
def imported(app):
    """Import 1,000 fake transactions with a high fraud rate so there are alerts.

    Uses its own client so the test's logged-in user is not replaced by the admin.
    """
    admin = app.test_client()
    login(admin, "admin")
    df = make_sample(rows=1000, fraud_rate=0.05, random_state=3)
    response = admin.post("/api/transactions/import", data=csv_upload(df), content_type="multipart/form-data")
    assert response.status_code == 201, response.get_json()
    return response.get_json()["source"]
