"""One-button connection check: cheapest probe per provider, no fabricated status."""

from __future__ import annotations

import httpx
import pytest

from trading_system import connectivity
from trading_system.connectivity import (
    ANTHROPIC_PING_MODEL,
    PING_MODEL,
    STATUS_AUTH,
    STATUS_NOT_CONFIGURED,
    STATUS_OK,
    STATUS_QUOTA,
    STATUS_RATE_LIMITED,
    STATUS_UNAVAILABLE,
    check_all,
    probe_alpaca,
    probe_anthropic,
    probe_gemini,
    probe_openai,
    probe_sec,
)


def _response(status: int, url: str, *, json_body: dict | None = None, headers: dict[str, str] | None = None):
    request = httpx.Request("GET", url)
    return httpx.Response(status, json=json_body, headers=headers or {}, request=request)


def test_missing_keys_are_not_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("no request should be sent without a key")

    monkeypatch.setattr(httpx.Client, "get", boom)
    monkeypatch.setattr(httpx.Client, "post", boom)

    assert probe_openai(None).status == STATUS_NOT_CONFIGURED
    assert probe_alpaca("key", None).status == STATUS_NOT_CONFIGURED
    assert probe_gemini("").status == STATUS_NOT_CONFIGURED
    assert probe_anthropic(None).status == STATUS_NOT_CONFIGURED


def test_openai_ping_is_a_single_one_token_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_post(self, url, **kwargs):  # noqa: ANN001
        calls.append({"url": url, "json": kwargs.get("json")})
        return _response(
            200,
            url,
            json_body={"usage": {"total_tokens": 2}},
            headers={"x-ratelimit-remaining-tokens": "199000", "x-ratelimit-limit-tokens": "200000"},
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    result = probe_openai("sk-test")
    assert result.status == STATUS_OK
    assert len(calls) == 1
    body = calls[0]["json"]
    assert body["model"] == PING_MODEL
    assert body["max_tokens"] == 1
    assert result.usage is not None
    assert "2 토큰" in result.usage
    assert "199,000" in result.usage


def test_openai_quota_and_rate_limit_are_distinguished(monkeypatch: pytest.MonkeyPatch) -> None:
    def quota(self, url, **_k):  # noqa: ANN001
        return _response(429, url, json_body={"error": {"code": "insufficient_quota", "type": "insufficient_quota"}})

    monkeypatch.setattr(httpx.Client, "post", quota)
    assert probe_openai("sk-test").status == STATUS_QUOTA

    def throttled(self, url, **_k):  # noqa: ANN001
        return _response(429, url, json_body={"error": {"code": "rate_limit_exceeded", "type": "rate_limit_error"}})

    monkeypatch.setattr(httpx.Client, "post", throttled)
    assert probe_openai("sk-test").status == STATUS_RATE_LIMITED

    def unauthorized(self, url, **_k):  # noqa: ANN001
        return _response(401, url, json_body={"error": {"code": "invalid_api_key"}})

    monkeypatch.setattr(httpx.Client, "post", unauthorized)
    assert probe_openai("sk-bad").status == STATUS_AUTH


def test_openai_probe_never_leaks_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def unauthorized(self, url, **_k):  # noqa: ANN001
        return _response(401, url, json_body={"error": {"code": "invalid_api_key"}})

    monkeypatch.setattr(httpx.Client, "post", unauthorized)
    result = probe_openai("sk-super-secret")
    assert "sk-super-secret" not in f"{result.detail} {result.usage} {result.label}"


def test_alpaca_falls_back_to_the_live_host(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_get(self, url, **_k):  # noqa: ANN001
        seen.append(url)
        if "paper-api" in url:
            return _response(401, url)
        return _response(200, url, json_body={"is_open": False})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    result = probe_alpaca("key", "secret")
    assert result.status == STATUS_OK
    assert len(seen) == 2
    assert all(u.endswith("/v2/clock") for u in seen)


def test_alpaca_auth_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **_k: _response(403, url))
    assert probe_alpaca("key", "secret").status == STATUS_AUTH


def test_anthropic_models_list_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(self, url, **_k):  # noqa: ANN001
        assert url == "https://api.anthropic.com/v1/models"
        return _response(200, url, json_body={"data": [{"id": "claude-haiku-4-5"}]})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no token ping")))
    result = probe_anthropic("sk-ant-test")
    assert result.status == STATUS_OK
    assert result.usage is None


def test_anthropic_401_explains_rejected_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda self, url, **_k: _response(401, url, json_body={"error": {"type": "authentication_error"}}),
    )
    result = probe_anthropic("sk-ant-bad")
    assert result.status == STATUS_AUTH
    assert "sk-ant-" in result.detail
    assert "sk-ant-bad" not in result.detail


def test_anthropic_404_falls_back_to_one_token_haiku(monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list[dict] = []

    def fake_get(self, url, **_k):  # noqa: ANN001
        return _response(404, url, json_body={"error": {"type": "not_found_error"}})

    def fake_post(self, url, **kwargs):  # noqa: ANN001
        posts.append({"url": url, "json": kwargs.get("json")})
        return _response(200, url, json_body={"id": "msg_1"})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    monkeypatch.setattr(httpx.Client, "post", fake_post)
    result = probe_anthropic("sk-ant-test")
    assert result.status == STATUS_OK
    assert posts and posts[0]["json"]["model"] == ANTHROPIC_PING_MODEL
    assert posts[0]["json"]["max_tokens"] == 1


def test_network_error_is_unavailable_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def blow_up(self, url, **_k):  # noqa: ANN001
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx.Client, "get", blow_up)
    result = probe_gemini("key")
    assert result.status == STATUS_UNAVAILABLE
    assert "ConnectError" in result.detail


def test_sec_probe_uses_head_and_flags_403(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_head(self, url, **kwargs):  # noqa: ANN001
        seen.append(url)
        assert "InvestAssist" in kwargs["headers"]["User-Agent"]
        return _response(403, url)

    monkeypatch.setattr(httpx.Client, "head", fake_head)
    result = probe_sec()
    assert result.status == STATUS_UNAVAILABLE
    assert seen and seen[0].startswith("https://data.sec.gov/submissions/")
    assert "API 키" in result.detail


def test_check_all_runs_every_provider_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connectivity, "probe_sec", lambda: connectivity.ProbeResult("sec", "SEC EDGAR", STATUS_OK, "ok"))
    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **_k: _response(200, url, json_body={}))
    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **_k: _response(200, url, json_body={}))

    report = check_all(
        {
            "ALPACA_API_KEY": "a",
            "ALPACA_SECRET_KEY": "b",
            "OPENAI_API_KEY": "c",
            "GEMINI_API_KEY": "d",
            "ANTHROPIC_API_KEY": "e",
        }
    )
    assert [r.key for r in report.results] == list(connectivity.PROBE_ORDER)
    assert report.failures == []
    payload = report.as_dict()
    assert payload["ok"] is True
    assert len(payload["results"]) == len(connectivity.PROBE_ORDER)


def _client(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    from trading_system.ui.app import create_app

    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "ui.duckdb"))
    return TestClient(create_app())


def test_connection_state_starts_unchecked(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    body = client.get("/api/v1/connection-state").json()
    assert body["checked"] is False
    assert body["results"] == []


def test_one_click_check_populates_every_provider(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_system.ui import app as ui_app

    report = connectivity.ConnectionReport(
        checked_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        results=[
            connectivity.ProbeResult("alpaca", "Alpaca 시장 데이터", STATUS_OK, "키가 유효합니다."),
            connectivity.ProbeResult("openai", "OpenAI", STATUS_QUOTA, "HTTP 429 (insufficient_quota)"),
        ],
    )
    monkeypatch.setattr(ui_app, "check_all", lambda _env: report)
    client = _client(tmp_path, monkeypatch)

    body = client.post("/api/v1/connection-check").json()
    assert body["ok"] is False
    assert [r["key"] for r in body["results"]] == ["alpaca", "openai"]
    assert body["checked_at_label"]

    cached = client.get("/api/v1/connection-state").json()
    assert cached["checked"] is True
    assert cached["results"][1]["status"] == STATUS_QUOTA


def test_dashboard_warns_before_wasting_a_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_system.ui import app as ui_app

    monkeypatch.setattr(ui_app, "read_env_map", lambda _p: {"OPENAI_API_KEY": "sk-x"})
    client = _client(tmp_path, monkeypatch)

    unchecked = client.get("/").text
    assert "연결 확인 버튼은 선택입니다" in unchecked
    assert "legacy-b 방식의 GPT 연속 분석" in unchecked
    assert "API 연결을 아직 확인하지 않았습니다" not in unchecked
    assert "그래도 스캔을 실행할까요?" not in unchecked

    monkeypatch.setattr(
        ui_app,
        "check_all",
        lambda _env: connectivity.ConnectionReport(
            checked_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            results=[connectivity.ProbeResult("openai", "OpenAI", STATUS_OK, "정상")],
        ),
    )
    client.post("/api/v1/connection-check")
    cleared = client.get("/").text
    assert "연결 확인 버튼은 선택입니다" not in cleared
    assert "연결되지 않은 API가 있습니다." not in cleared


def test_failures_only_count_configured_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        connectivity, "probe_sec", lambda: connectivity.ProbeResult("sec", "SEC EDGAR", STATUS_OK, "ok")
    )
    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **_k: _response(401, url, json_body={}))
    report = check_all({"OPENAI_API_KEY": "c"})
    keys = {r.key for r in report.failures}
    assert keys == {"openai"}
    assert report.as_dict()["ok"] is False


def test_universe_preset_saves_without_500(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from trading_system.credentials import read_env_map
    from trading_system.ui.app import create_app

    monkeypatch.setattr("trading_system.ui.app.PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "ui.duckdb"))
    client = TestClient(create_app(), follow_redirects=False)
    res = client.post("/universe/preset", data={"preset": "fast_100"})
    assert res.status_code == 303
    assert "universe=saved" in res.headers["location"]
    assert read_env_map(tmp_path / ".env")["SCAN_UNIVERSE_PRESET"] == "fast_100"
    html = client.get("/settings?universe=saved").text
    assert "스캔 범위를 저장했습니다." in html
    assert "Internal Server Error" not in html
