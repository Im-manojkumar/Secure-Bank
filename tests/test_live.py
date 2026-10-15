"""The live scoring API that payment systems call with an API key."""

import pytest

from models import ApiKey, AuditLog, FraudCase, Transaction, db, utcnow
from services.live import create_key


@pytest.fixture
def api_key(app):
    with app.app_context():
        _, key = create_key("Test switch")
        db.session.commit()
    return key


def risky(**changes):
    """A Sparkov payment shaped like the fake data's planted fraud: a big online buy late at night."""
    tx = {"trans_date_trans_time": "2019-06-01 23:30:00", "category": "shopping_net", "amt": 950.0,
          "dob": "1970-05-01", "city_pop": 25000, "lat": 40.0, "long": -100.0,
          "merch_lat": 40.5, "merch_long": -100.5}
    return tx | changes


def call(client, key, body):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return client.post("/api/v1/score", json=body, headers=headers)


def test_score_needs_a_valid_active_key(app, client, api_key):
    body = {"dataset": "sparkov", "transaction": risky()}
    assert call(client, None, body).status_code == 401
    assert call(client, "sbk_wrong", body).status_code == 401
    assert call(client, api_key, body).status_code == 200
    with app.app_context():
        db.session.query(ApiKey).one().revoked_at = utcnow()
        db.session.commit()
    assert call(client, api_key, {"dataset": "sparkov", "transaction": risky(amt=951.0)}).status_code == 401


def test_live_score_decides_stores_and_opens_a_case(app, client, admin, api_key):
    response = call(client, api_key, {"dataset": "sparkov", "transaction": risky(), "card_number": "4111111111111111"})
    assert response.status_code == 200, response.get_json()
    result = response.get_json()
    assert result["decision"] in ("review", "decline") and result["reasons"]
    assert result["duplicate"] is False and result["latency_ms"] >= 0

    with app.app_context():
        t = db.session.get(Transaction, result["transaction_id"])
        assert t.source.source_type == "api" and t.source.name == "Live API: Test switch"
        assert t.card_id and t.features["card_last4"] == "1111"
        assert "4111111111111111" not in str(t.features)  # the card number is never stored
        assert db.session.query(FraudCase).filter_by(transaction_id=t.id).count() == 1
        assert db.session.query(ApiKey).one().last_used_at is not None
    alerts = admin.get("/api/alerts?dataset=sparkov").get_json()["items"]
    assert any(a["transaction_id"] == result["transaction_id"] for a in alerts)


def test_a_retried_call_is_answered_without_storing_twice(app, client, api_key):
    body = {"dataset": "sparkov", "transaction": risky(), "card_number": "4111111111111111"}
    first = call(client, api_key, body).get_json()
    again = call(client, api_key, body).get_json()
    assert again["duplicate"] is True and again["transaction_id"] == first["transaction_id"]
    assert again["decision"] == first["decision"]
    with app.app_context():
        assert db.session.query(Transaction).count() == 1


def test_payments_on_the_same_card_build_its_history(app, client, api_key):
    for minute in range(3):
        tx = risky(trans_date_trans_time=f"2019-06-01 10:0{minute}:00", amt=20.0 + minute, category="grocery_pos")
        assert call(client, api_key, {"dataset": "sparkov", "transaction": tx, "card_number": "5500000000000004"}).status_code == 200
    with app.app_context():
        from ml.datasets import card_history
        from services.prediction import card_history as stored
        history = stored(db.session.scalars(db.select(Transaction.card_id)).all())
        assert len(history) == 3 and history["card_id"].nunique() == 1
        latest = history.iloc[[-1]].assign(trans_date_trans_time="2019-06-01 10:03:00")
        assert card_history(latest, history).iloc[0]["card_txn_1h"] == 3


def test_very_high_scores_are_declined(app, client, api_key):
    app.config["DECLINE_SCORE"] = 0.0
    result = call(client, api_key, {"dataset": "sparkov", "transaction": risky()}).get_json()
    assert result["decision"] == "decline"


def test_bad_requests_are_explained(client, api_key):
    assert call(client, api_key, {"dataset": "bank", "transaction": {}}).status_code == 400
    response = call(client, api_key, {"dataset": "sparkov", "transaction": risky(amt=-5)})
    assert response.status_code == 400 and response.get_json()["details"]
    bad_card = call(client, api_key, {"dataset": "sparkov", "transaction": risky(), "card_number": "4111-1111"})
    assert bad_card.status_code == 400


def test_api_keys_are_made_and_revoked_from_the_command_line(app):
    runner = app.test_cli_runner()
    made = runner.invoke(args=["create-api-key", "Branch switch"])
    assert made.exit_code == 0 and "sbk_" in made.output
    key = made.output.strip().splitlines()[-1]
    with app.app_context():
        row = db.session.query(ApiKey).one()
        assert row.key_hash != key and key not in row.key_hash  # only a hash is stored
        prefix = row.prefix
    revoked = runner.invoke(args=["revoke-api-key", prefix])
    assert revoked.exit_code == 0, revoked.output
    with app.app_context():
        assert db.session.query(ApiKey).one().revoked_at is not None
        actions = {a.action for a in db.session.query(AuditLog)}
        assert {"api_key_created", "api_key_revoked"} <= actions
