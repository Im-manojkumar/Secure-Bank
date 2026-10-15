import pytest

from tests.conftest import login

PAGES = ["/", "/transactions", "/transactions/1", "/alerts", "/cases/1", "/predict", "/reports", "/reports/1"]


@pytest.mark.parametrize("path", PAGES + ["/import"])
def test_pages_redirect_to_login_when_logged_out(client, path):
    response = client.get(path)
    assert response.status_code == 302
    assert response.headers["Location"].startswith("/login?next=")


def test_login_page_renders(client):
    response = client.get("/login")
    assert response.status_code == 200
    assert b"Log in" in response.data


@pytest.mark.parametrize("path", PAGES)
def test_pages_render_for_analyst(analyst, path):
    response = analyst.get(path)
    assert response.status_code == 200
    assert b"analyst@test.local" in response.data
    # Analysts do not see the import link.
    assert b'href="/import"' not in response.data


def test_import_page_is_admin_only(client):
    login(client, "analyst")
    response = client.get("/import")
    assert response.status_code == 403
    assert b"No access" in response.data


def test_admin_sees_import_page(admin):
    response = admin.get("/import")
    assert response.status_code == 200
    assert b'href="/import"' in response.data


def test_login_page_ignores_offsite_next(client):
    response = client.get("/login?next=https://evil.example/")
    assert b"evil.example" not in response.data
    response = client.get("/login?next=//evil.example/")
    assert b"evil.example" not in response.data


def test_logged_in_user_is_sent_on_from_login(admin):
    response = admin.get("/login?next=/alerts")
    assert response.status_code == 302
    assert response.headers["Location"] == "/alerts"


def test_unknown_page_shows_html_404(admin):
    response = admin.get("/no-such-page")
    assert response.status_code == 404
    assert b"Page not found" in response.data


def test_static_files_served(client):
    for path in ("/static/css/style.css", "/static/js/common.js", "/static/js/chart.js"):
        assert client.get(path).status_code == 200
