"""Browser-origin protection for state-changing local UI routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from trading_system.ui import app as ui_app


def test_cross_site_form_post_is_rejected_before_route(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    called: list[bool] = []
    monkeypatch.setattr(ui_app, "upsert_env", lambda *args, **kwargs: called.append(True))
    client = TestClient(ui_app.create_app(), follow_redirects=False)

    response = client.post(
        "/settings/appetite",
        data={"risk_appetite": "moderate"},
        headers={"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"},
    )

    assert response.status_code == 403
    assert called == []


def test_untrusted_origin_is_rejected_without_fetch_metadata(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    called: list[bool] = []
    monkeypatch.setattr(ui_app, "upsert_env", lambda *args, **kwargs: called.append(True))
    client = TestClient(ui_app.create_app(), follow_redirects=False)

    response = client.post(
        "/settings/appetite",
        data={"risk_appetite": "moderate"},
        headers={"Origin": "https://attacker.example"},
    )

    assert response.status_code == 403
    assert called == []


def test_same_origin_and_vite_development_posts_are_allowed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    called: list[bool] = []
    monkeypatch.setattr(ui_app, "upsert_env", lambda *args, **kwargs: called.append(True))
    client = TestClient(ui_app.create_app(), follow_redirects=False)

    same_origin = client.post(
        "/settings/appetite",
        data={"risk_appetite": "moderate"},
        headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
    )
    vite = client.post(
        "/settings/appetite",
        data={"risk_appetite": "moderate"},
        headers={"Origin": "http://127.0.0.1:5173", "Sec-Fetch-Site": "same-site"},
    )

    assert same_origin.status_code == 303
    assert vite.status_code == 303
    assert called == [True, True]


def test_local_non_browser_post_remains_supported(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    called: list[bool] = []
    monkeypatch.setattr(ui_app, "upsert_env", lambda *args, **kwargs: called.append(True))
    client = TestClient(ui_app.create_app(), follow_redirects=False)

    response = client.post("/settings/appetite", data={"risk_appetite": "moderate"})

    assert response.status_code == 303
    assert called == [True]
