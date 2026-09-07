"""Gemini generate_json status and search fallback. No live API calls."""

from __future__ import annotations

import httpx
import pytest

from trading_system.config import Settings
from trading_system.gemini_client import generate_json


def _settings(**kwargs: object) -> Settings:
    base = {
        "_env_file": None,
        "alpaca_api_key": None,
        "alpaca_secret_key": None,
        "openai_api_key": None,
        "gemini_api_key": "test-gemini",
        "anthropic_api_key": None,
        "gemini_deadline_seconds": 30.0,
        "smoke_universe": ("AAPL", "MSFT", "BTC/USD"),
    }
    base.update(kwargs)
    return Settings(**base)  # type: ignore[arg-type]


def _ok_body(text: str = '{"ticker":"AAPL","summary_ko":"ok"}') -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 2, "totalTokenCount": 4},
    }


def test_urls_from_thinking_text() -> None:
    from trading_system.gemini_client import _urls_from_text

    rows = _urls_from_text(
        "see https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc and https://example.com/x"
    )
    assert len(rows) == 1
    assert "vertexaisearch" in rows[0]["url"]


def test_search_http_ok_marks_search_used_without_grounding_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a, **k):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            assert (json or {}).get("tools")
            return httpx.Response(
                200,
                json=_ok_body(),
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr("trading_system.gemini_client.httpx.Client", FakeClient)
    out = generate_json(_settings(), system="s", user="u", use_search=True)
    assert out["status"] == "AVAILABLE"
    assert out["web_search_used"] is True


def test_search_400_fallback_is_available_not_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list[bool] = []

    class FakeClient:
        def __init__(self, *a, **k):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            search = bool((json or {}).get("tools"))
            posts.append(search)
            if search:
                return httpx.Response(
                    400,
                    json={"error": {"message": "Search grounding is not enabled"}},
                    request=httpx.Request("POST", url),
                )
            return httpx.Response(200, json=_ok_body(), request=httpx.Request("POST", url))

    monkeypatch.setattr("trading_system.gemini_client.httpx.Client", FakeClient)
    out = generate_json(_settings(), system="s", user="u", use_search=True)
    assert posts == [True, False]
    assert out["status"] == "AVAILABLE"
    assert out["web_search_used"] is False
    assert any("Search" in str(q) or "search" in str(q) for q in out["open_questions"])


def test_structured_schema_is_sent_and_required_keys_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    schema = {
        "type": "object",
        "properties": {"summary_ko": {"type": "string"}},
        "required": ["summary_ko"],
    }

    class FakeClient:
        def __init__(self, *a, **k):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            seen["config"] = (json or {}).get("generationConfig")
            return httpx.Response(200, json=_ok_body(), request=httpx.Request("POST", url))

    monkeypatch.setattr("trading_system.gemini_client.httpx.Client", FakeClient)
    out = generate_json(
        _settings(), system="s", user="u", use_search=False, response_schema=schema
    )
    assert out["status"] == "AVAILABLE"
    config = seen["config"]
    assert isinstance(config, dict)
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == schema


def test_structured_response_missing_required_key_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *a, **k):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            return httpx.Response(200, json=_ok_body('{"other":"value"}'), request=httpx.Request("POST", url))

    monkeypatch.setattr("trading_system.gemini_client.httpx.Client", FakeClient)
    out = generate_json(
        _settings(),
        system="s",
        user="u",
        response_schema={"type": "object", "required": ["summary_ko"]},
    )
    assert out["status"] == "UNAVAILABLE"
