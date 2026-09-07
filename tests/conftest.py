"""Keep unit tests off the developer's live .env (uv injects it into os.environ)."""

from __future__ import annotations

import pytest

_PROVIDER_ENV = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "KAKAO_REST_API_KEY",
    "KAKAO_CLIENT_SECRET",
    "KAKAO_ACCESS_TOKEN",
    "KAKAO_REFRESH_TOKEN",
)


@pytest.fixture(autouse=True)
def _isolate_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _PROVIDER_ENV:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _reset_auto_ledger_failure_latch() -> None:
    """llm_budget's fail-closed latch for automatic ledger-write failures is
    in-process global state by design (it must survive across DB connections within
    one process). Left set by one test, it would silently force every later test's
    daily_auto_spend_usd() to infinity regardless of that test's own setup."""
    from trading_system import llm_budget

    llm_budget._auto_ledger_write_failed_day = None
    yield
    llm_budget._auto_ledger_write_failed_day = None
