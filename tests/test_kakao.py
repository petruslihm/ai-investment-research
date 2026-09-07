"""KakaoTalk Send-to-Me notifier — mocked HTTP only."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from trading_system.alert_engine import collect_cycle_notifications, format_recommendation_message
from trading_system.alerts import AlertKind, AlertRecord
from trading_system.kakao import (
    DEFAULT_REDIRECT_URI,
    KakaoStatus,
    connection_status,
    exchange_code,
    format_kakao_text,
    notify_kakao,
    persist_tokens,
    refresh_access_token,
    send_to_me,
)
from trading_system.recommendations import (
    HorizonOutlook,
    RecommendationAction,
    RecommendationRecord,
    RecommendationSource,
)


def _env(tmp_path: Path, text: str) -> Path:
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return p


def test_not_configured_skips_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = _env(tmp_path, "")
    calls: list[str] = []

    def boom(*_a, **_k):
        calls.append("hit")
        raise AssertionError("must not call Kakao")

    monkeypatch.setattr("trading_system.kakao.kakao_request", boom)
    result = notify_kakao(
        AlertRecord(kind=AlertKind.RUNTIME, message="hi"),
        env_file=env_file,
    )
    assert connection_status({}) is KakaoStatus.NOT_CONFIGURED
    assert result.ok is True
    assert result.value is KakaoStatus.NOT_CONFIGURED
    assert calls == []


def test_auth_required_without_tokens(tmp_path: Path) -> None:
    env_file = _env(tmp_path, "KAKAO_REST_API_KEY=rest_key\n")
    result = notify_kakao(
        AlertRecord(kind=AlertKind.RUNTIME, message="hi"),
        env_file=env_file,
    )
    assert result.ok is True
    assert result.value is KakaoStatus.AUTH_REQUIRED


def test_exchange_code_persists_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = _env(tmp_path, "KAKAO_REST_API_KEY=rest_key\n")
    seen: list[str] = []

    def fake(method: str, url: str, **kwargs):
        seen.append(url)
        if "oauth/token" in url:
            return httpx.Response(
                200,
                json={
                    "access_token": "access_secret_token",
                    "refresh_token": "refresh_secret_token",
                    "expires_in": 3600,
                },
                request=httpx.Request(method, url),
            )
        if "user/me" in url:
            return httpx.Response(
                200,
                json={"properties": {"nickname": "테스터"}},
                request=httpx.Request(method, url),
            )
        raise AssertionError(url)

    monkeypatch.setattr("trading_system.kakao.kakao_request", fake)
    from trading_system.credentials import read_env_map

    out = exchange_code(read_env_map(env_file), "auth-code", env_file=env_file)
    saved = read_env_map(env_file)
    assert out["nickname"] == "테스터"
    assert saved["KAKAO_ACCESS_TOKEN"] == "access_secret_token"
    assert saved["KAKAO_REFRESH_TOKEN"] == "refresh_secret_token"
    assert "oauth/token" in seen[0]


def test_refresh_access_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = _env(
        tmp_path,
        "KAKAO_REST_API_KEY=rest\nKAKAO_REFRESH_TOKEN=old_refresh\nKAKAO_ACCESS_TOKEN=old\nKAKAO_ACCESS_EXPIRES_AT=1\n",
    )

    def fake(method: str, url: str, **kwargs):
        assert "oauth/token" in url
        assert kwargs["data"]["grant_type"] == "refresh_token"
        return httpx.Response(
            200,
            json={"access_token": "new_access", "expires_in": 3600},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr("trading_system.kakao.kakao_request", fake)
    from trading_system.credentials import read_env_map

    merged = refresh_access_token(read_env_map(env_file), env_file=env_file)
    assert merged["KAKAO_ACCESS_TOKEN"] == "new_access"
    assert read_env_map(env_file)["KAKAO_ACCESS_TOKEN"] == "new_access"


def test_send_to_me_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = _env(
        tmp_path,
        "KAKAO_REST_API_KEY=rest\nKAKAO_ACCESS_TOKEN=tok\nKAKAO_REFRESH_TOKEN=r\nKAKAO_ACCESS_EXPIRES_AT=9999999999\n",
    )
    posts: list[str] = []

    def fake(method: str, url: str, **kwargs):
        posts.append(url)
        assert "memo/default/send" in url
        assert "tok" not in str(kwargs.get("data"))
        return httpx.Response(200, json={"result_code": 0}, request=httpx.Request(method, url))

    monkeypatch.setattr("trading_system.kakao.kakao_request", fake)
    from trading_system.credentials import read_env_map

    st = send_to_me(
        read_env_map(env_file),
        AlertRecord(kind=AlertKind.RUNTIME, message="hello"),
        env_file=env_file,
    )
    assert st is KakaoStatus.CONNECTED
    assert posts


def test_401_then_refresh_then_send(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = _env(
        tmp_path,
        "KAKAO_REST_API_KEY=rest\nKAKAO_ACCESS_TOKEN=stale\nKAKAO_REFRESH_TOKEN=rf\nKAKAO_ACCESS_EXPIRES_AT=9999999999\n",
    )
    n_memo = {"n": 0}

    def fake(method: str, url: str, **kwargs):
        if "memo" in url:
            n_memo["n"] += 1
            if n_memo["n"] == 1:
                return httpx.Response(401, text="expired", request=httpx.Request(method, url))
            auth = kwargs.get("headers", {}).get("Authorization", "")
            assert auth.endswith("fresh")
            return httpx.Response(200, json={"result_code": 0}, request=httpx.Request(method, url))
        if "oauth/token" in url:
            return httpx.Response(
                200,
                json={"access_token": "fresh", "expires_in": 3600},
                request=httpx.Request(method, url),
            )
        raise AssertionError(url)

    monkeypatch.setattr("trading_system.kakao.kakao_request", fake)
    from trading_system.credentials import read_env_map

    st = send_to_me(
        read_env_map(env_file),
        AlertRecord(kind=AlertKind.STOP_LOSS, message="stop"),
        env_file=env_file,
    )
    assert st is KakaoStatus.CONNECTED
    assert n_memo["n"] == 2


def test_timeout_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = _env(
        tmp_path,
        "KAKAO_REST_API_KEY=rest\nKAKAO_ACCESS_TOKEN=tok\nKAKAO_REFRESH_TOKEN=r\nKAKAO_ACCESS_EXPIRES_AT=9999999999\n",
    )

    def boom(*_a, **_k):
        raise httpx.TimeoutException("network down")

    monkeypatch.setattr("trading_system.kakao.kakao_request", boom)
    result = notify_kakao(
        AlertRecord(kind=AlertKind.RUNTIME, message="x"),
        env_file=env_file,
    )
    assert result.ok is False
    assert result.degraded is True
    err = result.error or ""
    assert "tok" not in err
    assert "TimeoutException" in err or "deadline" in err.lower()


def test_redact_and_format_do_not_embed_tokens() -> None:
    from trading_system.kakao import redact

    env = {"KAKAO_ACCESS_TOKEN": "supersecretvalue"}
    assert "[redacted]" in redact("fail supersecretvalue", env)
    text = format_kakao_text(AlertRecord(kind=AlertKind.RUNTIME, message="NVDA ADD"))
    assert "supersecretvalue" not in text
    assert text.startswith("[Stock AI]")


def test_settings_hides_telegram_discord_and_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trading_system.ui.app.PROJECT_ROOT", tmp_path)
    _env(
        tmp_path,
        "OPENAI_API_KEY=sk-live-secret-xyz\nKAKAO_ACCESS_TOKEN=kakao_secret_token\nKAKAO_REST_API_KEY=rest\n",
    )
    from trading_system.ui.app import create_app

    html = TestClient(create_app()).get("/settings").text
    assert "TELEGRAM" not in html
    assert "DISCORD" not in html
    assert "Telegram" not in html
    assert "Discord" not in html
    assert "sk-live-secret-xyz" not in html
    assert "kakao_secret_token" not in html
    assert "입력완료" in html
    assert "카카오톡 알림" in html
    assert DEFAULT_REDIRECT_URI in html
    assert "/auth/kakao/callback" in html


def test_oauth_callback_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trading_system.ui.app.PROJECT_ROOT", tmp_path)
    _env(tmp_path, "KAKAO_REST_API_KEY=rest\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / ".kakao_oauth_state").write_text("state123", encoding="utf-8")

    def fake(method: str, url: str, **kwargs):
        if "oauth/token" in url:
            return httpx.Response(
                200,
                json={"access_token": "a", "refresh_token": "b", "expires_in": 60},
                request=httpx.Request(method, url),
            )
        return httpx.Response(200, json={"properties": {"nickname": "n"}}, request=httpx.Request(method, url))

    monkeypatch.setattr("trading_system.kakao.kakao_request", fake)
    from trading_system.ui.app import create_app

    client = TestClient(create_app(), follow_redirects=False)
    res = client.get("/auth/kakao/callback?code=abc&state=state123")
    assert res.status_code == 303
    assert "connected" in res.headers["location"]
    from trading_system.credentials import read_env_map

    saved = read_env_map(tmp_path / ".env")
    assert saved.get("KAKAO_REFRESH_TOKEN") == "b"


def test_cycle_notifications_skip_hold_and_tiny_delta() -> None:
    rec = RecommendationRecord(
        source=RecommendationSource.QUANT_ONLY,
        tick_id="tick_1",  # type: ignore[arg-type]
        decision_epoch_id="dec_1",  # type: ignore[arg-type]
        feature_snapshot_id="fs_1",  # type: ignore[arg-type]
        instrument_id="inst_nvda",  # type: ignore[arg-type]
        action=RecommendationAction.HOLD,
        current_units=10,
        recommended_units=10,
        delta_units=0,
        horizons=[HorizonOutlook(horizon=5, expected_return=0.01)],
        actionable=True,
    )
    add = rec.model_copy(
        update={
            "action": RecommendationAction.ADD,
            "recommended_units": 20,
            "delta_units": 10,
            "confidence": 0.74,
        }
    )
    out = collect_cycle_notifications(
        recommendations=[rec, add],
        tick_id="tick_1",
        data_status="AVAILABLE",
        llm_status="NOT_CONFIGURED",
        market_status="AVAILABLE",
        alpaca_configured=True,
        health_overall="ok",
    )
    kinds = [a.kind for a in out]
    assert AlertKind.RECOMMENDATION in kinds
    msg = format_recommendation_message(add, data_status="VALID", llm_status="NOT_CONFIGURED")
    assert "NVDA" in msg
    assert "추가" in msg


def test_notify_kakao_missing_env_file(tmp_path: Path) -> None:
    missing = tmp_path / "no-such.env"
    result = notify_kakao(AlertRecord(kind=AlertKind.RUNTIME, message="x"), env_file=missing)
    assert result.ok is True
    assert result.value is KakaoStatus.NOT_CONFIGURED


def test_persist_tokens_never_logs_body(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env_file = _env(tmp_path, "KAKAO_REST_API_KEY=rest\n")
    persist_tokens(
        env_file,
        {"access_token": "hidden_access", "refresh_token": "hidden_refresh", "expires_in": 10},
        nickname="me",
    )
    captured = capsys.readouterr()
    assert "hidden_access" not in captured.out
    assert "hidden_refresh" not in captured.out
