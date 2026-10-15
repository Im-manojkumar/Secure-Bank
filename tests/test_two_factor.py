"""Two-factor login with authenticator-app codes."""

from models import AuditLog, User, db
from services import totp

# RFC 6238's SHA-1 test secret ("12345678901234567890"), in base32.
RFC_SECRET = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"


def test_codes_match_the_rfc_test_vectors():
    # RFC 6238 appendix B: time 59 s -> 94287082, 1111111109 s -> 07081804 (8 digits; we use the last 6).
    assert totp.code_at(RFC_SECRET, 59 // 30) == "287082"
    assert totp.code_at(RFC_SECRET, 1111111109 // 30) == "081804"


def test_a_code_works_once_within_its_window():
    now = 1_700_000_000
    code = totp.code_at(RFC_SECRET, totp.current_step(now))
    step = totp.matching_step(RFC_SECRET, code, now=now)
    assert step == totp.current_step(now)
    assert totp.matching_step(RFC_SECRET, code, last_used_step=step, now=now) is None  # no replay
    assert totp.matching_step(RFC_SECRET, code, now=now + 30) == step  # one step late is fine
    assert totp.matching_step(RFC_SECRET, code, now=now + 90) is None  # too late
    assert totp.matching_step(RFC_SECRET, "12345", now=now) is None


def enable(client):
    secret = client.post("/api/auth/2fa/setup", json={}).get_json()["secret"]
    code = totp.code_at(secret, totp.current_step())
    response = client.post("/api/auth/2fa/enable", json={"code": code})
    assert response.status_code == 200 and response.get_json()["user"]["two_factor_enabled"]
    return secret


def test_setup_needs_a_correct_code(analyst):
    setup = analyst.post("/api/auth/2fa/setup", json={}).get_json()
    assert setup["uri"].startswith("otpauth://totp/SecureBank:analyst%40test.local?secret=")
    assert analyst.post("/api/auth/2fa/enable", json={"code": "000000"}).status_code == 400
    assert analyst.get("/api/auth/me").get_json()["user"]["two_factor_enabled"] is False


def test_login_asks_for_the_code_once_it_is_on(app, analyst):
    secret = enable(analyst)
    analyst.post("/api/auth/logout")
    client = app.test_client()
    creds = {"email": "analyst@test.local", "password": "password123"}

    asked = client.post("/api/auth/login", json=creds)
    assert asked.status_code == 401 and asked.get_json()["two_factor_required"]
    assert client.get("/api/auth/me").status_code == 401

    wrong = client.post("/api/auth/login", json=creds | {"code": "000000"})
    assert wrong.status_code == 401
    with app.app_context():
        failed = db.session.query(AuditLog).filter_by(action="login_failed").one()
        assert failed.details == {"reason": "wrong two-factor code"}
        # The code used at setup cannot be used again.
        user = db.session.query(User).filter_by(email="analyst@test.local").one()
        next_step = user.totp_last_step + 1
    # The next step's code is accepted: the window allows one step of clock difference.
    good = client.post("/api/auth/login", json=creds | {"code": totp.code_at(secret, next_step)})
    assert good.status_code == 200
    assert client.get("/api/auth/me").status_code == 200


def test_admins_must_set_it_up_before_anything_else(app, admin):
    app.config["REQUIRE_2FA_FOR_ADMINS"] = True
    blocked = admin.get("/api/alerts")
    assert blocked.status_code == 403 and blocked.get_json()["two_factor_setup_required"]
    page = admin.get("/transactions")
    assert page.status_code == 302 and page.headers["Location"].endswith("/account")
    assert admin.get("/account").status_code == 200
    assert admin.get("/api/auth/me").status_code == 200

    enable(admin)
    assert admin.get("/api/alerts").status_code == 200
    # When it is required, an admin cannot turn it off.
    assert admin.post("/api/auth/2fa/disable", json={"password": "password123", "code": "1"}).status_code == 403


def test_analyst_can_turn_it_off_with_password_and_code(app, analyst):
    secret = enable(analyst)
    with app.app_context():
        step = db.session.query(User).filter_by(email="analyst@test.local").one().totp_last_step
    assert analyst.post("/api/auth/2fa/disable", json={"password": "wrong", "code": totp.code_at(secret, step + 1)}).status_code == 400
    response = analyst.post("/api/auth/2fa/disable", json={"password": "password123", "code": totp.code_at(secret, step + 1)})
    assert response.status_code == 200
    assert response.get_json()["user"]["two_factor_enabled"] is False
    with app.app_context():
        actions = {a.action for a in db.session.query(AuditLog)}
        assert {"two_factor_enabled", "two_factor_disabled"} <= actions
