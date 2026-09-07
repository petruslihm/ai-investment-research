"""Phase 03 market data tests (smoke universe + fixtures + mocked failures)."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from trading_system.config import Settings
from trading_system.ids import InstrumentId
from trading_system.market.calendar import is_trading_day
from trading_system.market.registry import bootstrap_smoke_universe, stable_instrument_id
from trading_system.market.repository import latest_lkg_quote
from trading_system.market.service import MarketDataService
from trading_system.market.status import DataStatus, is_canonical_decision
from trading_system.providers.alpaca import AlpacaMarketDataProvider
from trading_system.providers.fallback import FallbackChainProvider
from trading_system.providers.fixture import (
    FixtureMarketDataProvider,
    FixtureProviderConfig,
    FixtureQuoteSpec,
    make_smoke_fixture_bars,
)
from trading_system.providers.never_block import RetryPolicy, run_with_deadline
from trading_system.storage import Store


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        _env_file=None,
        smoke_universe=("SPY", "AAPL", "MSFT", "BTC/USD"),
        history_years=1,
        max_live_equity_symbols=4,
    )


@pytest.fixture()
def store(db_path: Path) -> Store:
    s = Store(db_path)
    s.open(acquire_writer=True)
    yield s
    s.close()


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "market.duckdb"


def _make_fixture_provider(settings: Settings, *, fail_symbols: set[str] | None = None) -> FixtureMarketDataProvider:
    end = date(2024, 6, 14)
    equity_symbols = tuple(s for s in settings.smoke_universe if s != settings.btc_symbol)
    bars = make_smoke_fixture_bars(equity_symbols, end=end, days=5)
    btc_bars = make_smoke_fixture_bars((settings.btc_symbol,), end=end, days=5)[settings.btc_symbol]
    quotes = {
        sym: FixtureQuoteSpec(
            price=100.0 + i,
            event_ts=datetime(2024, 6, 14, 15, 0, tzinfo=timezone.utc),
        )
        for i, sym in enumerate(equity_symbols)
    }
    quotes[settings.btc_symbol] = FixtureQuoteSpec(
        price=65000.0,
        event_ts=datetime(2024, 6, 14, 12, 0, tzinfo=timezone.utc),
    )
    cfg = FixtureProviderConfig(
        equity_bars=bars,
        btc_bars=btc_bars,
        quotes=quotes,
        fail_symbols=fail_symbols or set(),
    )
    return FixtureMarketDataProvider(cfg)


def _service(store: Store, settings: Settings, provider: FixtureMarketDataProvider) -> MarketDataService:
    return MarketDataService(
        store.conn,
        settings,
        provider,
        provider_name=provider.provider_name(),
    )


def test_schema_market_tables_exist(store: Store) -> None:
    tables = set(store.list_tables())
    for name in (
        "equity_daily_bars",
        "btc_daily_bars",
        "live_observations",
        "market_coverage",
        "provider_state",
        "live_watchlist",
    ):
        assert name in tables


def test_startup_backfill_persists_bars(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    rs = svc.startup_backfill(end=date(2024, 6, 14))
    eq_count = store.conn.execute("SELECT COUNT(*) FROM equity_daily_bars").fetchone()[0]
    btc_count = store.conn.execute("SELECT COUNT(*) FROM btc_daily_bars").fetchone()[0]
    assert eq_count > 0
    assert btc_count > 0
    cov = store.conn.execute(
        "SELECT COUNT(*) FROM market_coverage WHERE request_set_id = ?", [rs]
    ).fetchone()[0]
    assert cov > 0


def test_spy_and_btc_history_present(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    svc.startup_backfill(end=date(2024, 6, 14))
    spy_id = stable_instrument_id("SPY")
    spy_rows = store.conn.execute(
        "SELECT COUNT(*) FROM equity_daily_bars WHERE instrument_id = ?", [spy_id]
    ).fetchone()[0]
    assert spy_rows >= 5
    btc_rows = store.conn.execute("SELECT COUNT(*) FROM btc_daily_bars").fetchone()[0]
    assert btc_rows >= 5


def test_live_watchlist_includes_spy(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    svc.refresh_live_watchlist()
    spy_id = stable_instrument_id("SPY")
    row = store.conn.execute(
        "SELECT reason FROM live_watchlist WHERE instrument_id = ?", [spy_id]
    ).fetchone()
    assert row is not None


def test_live_watchlist_refresh_persists_observations(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    svc.refresh_live_watchlist()
    count = store.conn.execute("SELECT COUNT(*) FROM live_observations").fetchone()[0]
    assert count >= 3


def test_btc_refresh_persists(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    svc.refresh_btc()
    row = store.conn.execute(
        """
        SELECT data_status FROM live_observations
        WHERE asset_class = 'btc' ORDER BY receive_ts DESC LIMIT 1
        """
    ).fetchone()
    assert row is not None
    assert row[0] in {DataStatus.VALID.value, DataStatus.DEGRADED.value}


def test_one_failed_symbol_does_not_freeze_universe(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings, fail_symbols={"AAPL"})
    svc = _service(store, settings, provider)
    svc.refresh_live_watchlist()
    spy_id = stable_instrument_id("SPY")
    price, status, _, _ = latest_lkg_quote(store.conn, spy_id)
    assert price is not None
    aapl_id = stable_instrument_id("AAPL")
    aapl_price, _, _, _ = latest_lkg_quote(store.conn, aapl_id)
    assert aapl_price is None


def test_lkg_display_after_refresh_failure(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    svc.refresh_live_watchlist()
    spy_id = stable_instrument_id("SPY")
    first = svc.display_quote(spy_id)
    assert first["price"] is not None

    # Fail all quotes on second refresh
    provider.config.fail_symbols = set(settings.smoke_universe) - {settings.btc_symbol}
    svc.refresh_live_watchlist()
    second = svc.display_quote(spy_id)
    assert second["price"] is not None
    assert "failed" in str(second["display"]).lower() or second.get("lkg")


def test_unknown_event_time_stored_as_degraded(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    provider.config.quotes["SPY"] = FixtureQuoteSpec(price=450.0, event_ts=None, degraded=True)
    svc = _service(store, settings, provider)
    svc.refresh_live_watchlist()
    row = store.conn.execute(
        """
        SELECT data_status, is_canonical FROM live_observations
        WHERE instrument_id = ? ORDER BY receive_ts DESC LIMIT 1
        """,
        [stable_instrument_id("SPY")],
    ).fetchone()
    assert row[0] == DataStatus.DEGRADED.value
    assert row[1] is False
    assert not is_canonical_decision(DataStatus.DEGRADED, has_event_ts=False)


def test_fallback_provider_success(store: Store, settings: Settings) -> None:
    primary = FixtureMarketDataProvider(
        FixtureProviderConfig(always_fail=True, provider_name="primary"),
    )
    fallback = _make_fixture_provider(settings)
    fallback.config.provider_name = "fallback"
    chain = FallbackChainProvider(primary=primary, fallback=fallback)
    svc = MarketDataService(store.conn, settings, chain, provider_name="fallback")
    rs = svc.startup_backfill(end=date(2024, 6, 14))
    count = store.conn.execute("SELECT COUNT(*) FROM equity_daily_bars").fetchone()[0]
    assert count > 0
    assert rs


def test_provider_hang_bounded(store: Store, settings: Settings) -> None:
    hang = FixtureMarketDataProvider(
        FixtureProviderConfig(hang_seconds=0.05, provider_name="hang"),
    )
    started = time.monotonic()
    result = hang.fetch_quotes(
        [stable_instrument_id("SPY")],
        policy=RetryPolicy(max_attempts=1, total_deadline_seconds=0.01),
    )
    elapsed = time.monotonic() - started
    assert elapsed < 0.5
    assert result.ok is False


def test_no_infinite_retry() -> None:
    calls = {"n": 0}

    def always_fail() -> None:
        calls["n"] += 1
        raise TimeoutError("nope")

    result = run_with_deadline(
        always_fail,
        policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.0, jitter=0.0, total_deadline_seconds=1.0),
        sleep=lambda _s: None,
    )
    assert result.ok is False
    assert calls["n"] == 3


def test_equity_calendar_separate_from_btc(store: Store, settings: Settings) -> None:
    # Saturday is not an equity trading day but BTC refresh still works
    assert not is_trading_day(date(2024, 6, 15))  # Saturday
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    svc.refresh_btc()
    row = store.conn.execute(
        "SELECT COUNT(*) FROM live_observations WHERE asset_class = 'btc'"
    ).fetchone()[0]
    assert row == 1


def test_data_health_snapshot(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    rs = svc.startup_backfill(end=date(2024, 6, 14))
    snap = svc.build_data_health_snapshot(request_set_id=rs)
    assert snap.overall.value in {"ok", "degraded", "critical", "unknown"}
    persisted = store.conn.execute("SELECT COUNT(*) FROM data_health_snapshots").fetchone()[0]
    assert persisted == 1


def test_runtime_events_during_recovery(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings, fail_symbols={"MSFT"})
    svc = _service(store, settings, provider)
    svc.refresh_live_watchlist()
    events = store.conn.execute("SELECT COUNT(*) FROM runtime_events").fetchone()[0]
    assert events >= 1


def test_idempotent_bar_ingestion(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    svc.startup_backfill(end=date(2024, 6, 14))
    count1 = store.conn.execute("SELECT COUNT(*) FROM equity_daily_bars").fetchone()[0]
    svc.startup_backfill(end=date(2024, 6, 14))
    count2 = store.conn.execute("SELECT COUNT(*) FROM equity_daily_bars").fetchone()[0]
    assert count1 == count2


def test_second_backfill_uses_short_window(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    calls: list[tuple[date, date, int]] = []
    orig = provider.fetch_daily_bars

    def wrapped(instrument_ids, start, end, **kwargs):
        calls.append((start, end, len(instrument_ids)))
        return orig(instrument_ids, start, end, **kwargs)

    provider.fetch_daily_bars = wrapped  # type: ignore[method-assign]
    svc = _service(store, settings, provider)
    end = date(2024, 6, 14)
    svc.startup_backfill(end=end)
    assert calls
    assert any((end - start).days > 30 for start, _stop, _n in calls)
    calls.clear()
    svc.startup_backfill(end=end)
    assert calls, "second backfill should still refresh a short tail"
    for start, stop, _n in calls:
        assert stop == end
        assert (end - start).days <= 14
    started = store.conn.execute(
        "SELECT message FROM runtime_events WHERE kind = 'fetch_us_daily' AND status = 'started' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()[0]
    assert "incremental" in str(started)


def test_broad_universe_scan(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    svc.broad_universe_scan()
    count = store.conn.execute("SELECT COUNT(*) FROM equity_daily_bars").fetchone()[0]
    assert count > 0


def test_trading_day_calendar() -> None:
    # Known Monday after Juneteenth weekend 2024
    d = date(2024, 6, 17)
    assert is_trading_day(d)
    from trading_system.market.calendar import next_trading_day

    assert next_trading_day(date(2024, 6, 14)) == d  # Friday -> Monday


def test_special_closure_2025_01_09() -> None:
    assert not is_trading_day(date(2025, 1, 9))


def test_invalid_quote_not_canonical(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    future = datetime.now(timezone.utc) + timedelta(days=1)
    provider.config.quotes["SPY"] = FixtureQuoteSpec(price=-1.0, event_ts=future)
    svc = _service(store, settings, provider)
    svc.refresh_live_watchlist()
    svc.refresh_live_watchlist()
    rows = store.conn.execute(
        """
        SELECT data_status, is_canonical FROM live_observations
        WHERE instrument_id = ? ORDER BY receive_ts
        """,
        [stable_instrument_id("SPY")],
    ).fetchall()
    assert all(row[0] == DataStatus.DEGRADED.value for row in rows)
    assert all(row[1] is False for row in rows)


def test_total_outage_reports_critical(store: Store, settings: Settings) -> None:
    provider = FixtureMarketDataProvider(FixtureProviderConfig(always_fail=True))
    svc = MarketDataService(store.conn, settings, provider, provider_name="fixture")
    rs = svc.startup_backfill(end=date(2024, 6, 14))
    snap = svc.build_data_health_snapshot(rs)
    assert snap.overall.value == "critical"
    assert snap.equity_feed.value == "critical"


def test_lkg_age_not_reset_on_repeated_failure(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    svc.refresh_live_watchlist()
    spy_id = stable_instrument_id("SPY")
    first_trusted = store.conn.execute(
        """
        SELECT receive_ts FROM live_observations
        WHERE instrument_id = ? AND lkg_fallback = FALSE
        ORDER BY receive_ts DESC LIMIT 1
        """,
        [spy_id],
    ).fetchone()[0]

    time.sleep(1.1)
    provider.config.fail_symbols = {"SPY"}
    svc.refresh_live_watchlist()
    svc.refresh_live_watchlist()
    display = svc.display_quote(spy_id)
    assert display["price"] is not None
    assert "0s ago" not in str(display["display"])
    _, _, trusted_at, _ = latest_lkg_quote(store.conn, spy_id)
    assert trusted_at == first_trusted


def test_alpaca_malformed_quote_degrades() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"quote": {"t": "2026-08-28T00:00:00Z", "ap": "not-a-number"}},
            request=request,
        )

    settings = Settings(_env_file=None, alpaca_api_key="x", alpaca_secret_key="y")
    provider = AlpacaMarketDataProvider(settings, symbol_resolver=lambda _: "SPY")
    provider._client.close()
    provider._client = httpx.Client(
        base_url="https://data.alpaca.markets",
        transport=httpx.MockTransport(handler),
        timeout=0.1,
    )
    try:
        result = provider.fetch_quotes([InstrumentId("inst_spy")])
        assert result.ok is False or (
            result.value and not getattr(result.value, "quotes", [])
        )
    finally:
        provider.close()


def test_missing_values_not_zero_filled(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings, fail_symbols={"AAPL"})
    svc = _service(store, settings, provider)
    svc.refresh_live_watchlist()
    row = store.conn.execute(
        """
        SELECT price FROM live_observations
        WHERE instrument_id = ? AND price IS NULL
        ORDER BY receive_ts DESC LIMIT 1
        """,
        [stable_instrument_id("AAPL")],
    ).fetchone()
    assert row is not None
    assert row[0] is None


def test_alpaca_malformed_bars_degrades() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": 7}, request=request)

    settings = Settings(_env_file=None, alpaca_api_key="x", alpaca_secret_key="y")
    provider = AlpacaMarketDataProvider(settings, symbol_resolver=lambda _: "SPY")
    provider._client.close()
    provider._client = httpx.Client(
        base_url="https://data.alpaca.markets",
        transport=httpx.MockTransport(handler),
        timeout=0.1,
    )
    try:
        result = provider.fetch_daily_bars(
            [InstrumentId("inst_spy")],
            date(2024, 1, 1),
            date(2024, 1, 5),
            request_set_id="rs_malformed_bars",
        )
        assert result.ok is False
        batch = result.value
        assert batch is not None
        assert not batch.bars
        assert InstrumentId("inst_spy") in batch.failed_instruments
    finally:
        provider.close()


def test_alpaca_btc_bars_follows_pagination() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "bars": {
                        "BTCUSD": [
                            {
                                "t": "2024-01-01T00:00:00Z",
                                "o": 1,
                                "h": 1,
                                "l": 1,
                                "c": 1,
                                "v": 1,
                            }
                        ]
                    },
                    "next_page_token": "page2",
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "bars": {
                    "BTCUSD": [
                        {
                            "t": "2024-01-02T00:00:00Z",
                            "o": 2,
                            "h": 2,
                            "l": 2,
                            "c": 2,
                            "v": 2,
                        }
                    ]
                }
            },
            request=request,
        )

    settings = Settings(_env_file=None, alpaca_api_key="x", alpaca_secret_key="y")
    provider = AlpacaMarketDataProvider(settings)
    provider._client.close()
    provider._client = httpx.Client(
        base_url="https://data.alpaca.markets",
        transport=httpx.MockTransport(handler),
        timeout=0.1,
    )
    try:
        result = provider.fetch_btc_bars(
            date(2024, 1, 1),
            date(2024, 1, 2),
            request_set_id="rs_btc_pages",
        )
        assert result.ok is True
        assert len(result.value.bars) == 2
        assert calls["n"] == 2
    finally:
        provider.close()


def test_alpaca_btc_bars_stops_on_repeated_pagination_token() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "bars": {
                    "BTCUSD": [
                        {"t": "2024-01-02T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                    ]
                },
                "next_page_token": "stuck",
            },
            request=request,
        )

    settings = Settings(_env_file=None, alpaca_api_key="x", alpaca_secret_key="y")
    provider = AlpacaMarketDataProvider(settings)
    provider._client.close()
    provider._client = httpx.Client(
        base_url="https://data.alpaca.markets",
        transport=httpx.MockTransport(handler),
        timeout=0.1,
    )
    try:
        result = provider.fetch_btc_bars(
            date(2024, 1, 1),
            date(2024, 1, 2),
            request_set_id="rs_btc_stuck_page",
            policy=RetryPolicy(max_attempts=1, total_deadline_seconds=None),
        )
        assert result.ok is True
        assert result.degraded is True
        assert "pagination token repeated" in str(result.error)
        assert calls["n"] == 2
    finally:
        provider.close()


def test_alpaca_equity_bars_stops_on_repeated_pagination_token() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "bars": {
                    "SPY": [
                        {"t": "2024-01-02T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
                    ]
                },
                "next_page_token": "stuck",
            },
            request=request,
        )

    settings = Settings(_env_file=None, alpaca_api_key="x", alpaca_secret_key="y")
    provider = AlpacaMarketDataProvider(settings, symbol_resolver=lambda _: "SPY")
    provider._client.close()
    provider._client = httpx.Client(
        base_url="https://data.alpaca.markets",
        transport=httpx.MockTransport(handler),
        timeout=0.1,
    )
    try:
        result = provider.fetch_daily_bars(
            [InstrumentId("inst_spy")],
            date(2024, 1, 1),
            date(2024, 1, 2),
            request_set_id="rs_equity_stuck_page",
            policy=RetryPolicy(max_attempts=1, total_deadline_seconds=None),
        )
        assert result.ok is True
        assert result.degraded is True
        assert "pagination token repeated" in str(result.error)
        assert calls["n"] == 2
    finally:
        provider.close()


def test_alpaca_equity_bars_stops_on_unbounded_unique_pagination_tokens() -> None:
    """A provider that always returns a fresh (never-repeating) next_page_token, with
    no deadline set, must still terminate -- the repeated-token guard alone can't catch
    this, since the token is never actually repeated."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "bars": {"SPY": [{"t": "2024-01-02T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]},
                "next_page_token": f"unique_{calls['n']}",
            },
            request=request,
        )

    settings = Settings(_env_file=None, alpaca_api_key="x", alpaca_secret_key="y")
    provider = AlpacaMarketDataProvider(settings, symbol_resolver=lambda _: "SPY")
    provider._client.close()
    provider._client = httpx.Client(
        base_url="https://data.alpaca.markets",
        transport=httpx.MockTransport(handler),
        timeout=0.1,
    )
    try:
        result = provider.fetch_daily_bars(
            [InstrumentId("inst_spy")],
            date(2024, 1, 1),
            date(2024, 1, 2),
            request_set_id="rs_equity_unbounded_pages",
            policy=RetryPolicy(max_attempts=1, total_deadline_seconds=None),
        )
        assert result.ok is True
        assert "max page count" in str(result.error)
        assert calls["n"] < 10_000  # loop actually terminated, not hung
    finally:
        provider.close()


def test_alpaca_btc_bars_stops_on_unbounded_unique_pagination_tokens() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "bars": {"BTCUSD": [{"t": "2024-01-02T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]},
                "next_page_token": f"unique_{calls['n']}",
            },
            request=request,
        )

    settings = Settings(_env_file=None, alpaca_api_key="x", alpaca_secret_key="y")
    provider = AlpacaMarketDataProvider(settings)
    provider._client.close()
    provider._client = httpx.Client(
        base_url="https://data.alpaca.markets",
        transport=httpx.MockTransport(handler),
        timeout=0.1,
    )
    try:
        result = provider.fetch_btc_bars(
            date(2024, 1, 1),
            date(2024, 1, 2),
            request_set_id="rs_btc_unbounded_pages",
            policy=RetryPolicy(max_attempts=1, total_deadline_seconds=None),
        )
        assert result.ok is True
        assert "max page count" in str(result.error)
        assert calls["n"] < 10_000
    finally:
        provider.close()


def test_broad_scan_partial_preserves_existing_bars(store: Store, settings: Settings) -> None:
    provider = _make_fixture_provider(settings)
    svc = _service(store, settings, provider)
    svc.startup_backfill(end=date(2024, 6, 14))
    aapl_id = stable_instrument_id("AAPL")
    before = store.conn.execute(
        "SELECT COUNT(*) FROM equity_daily_bars WHERE instrument_id = ?",
        [aapl_id],
    ).fetchone()[0]
    assert before >= 5

    provider.config.fail_symbols = {"AAPL"}
    svc.broad_universe_scan()
    after = store.conn.execute(
        "SELECT COUNT(*) FROM equity_daily_bars WHERE instrument_id = ?",
        [aapl_id],
    ).fetchone()[0]
    assert after == before


def test_alpaca_daily_bars_uses_iex_feed() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "bars": [
                    {
                        "t": "2024-01-02T00:00:00Z",
                        "o": 1,
                        "h": 1,
                        "l": 1,
                        "c": 1,
                        "v": 1,
                    }
                ]
            },
            request=request,
        )

    settings = Settings(_env_file=None, alpaca_api_key="x", alpaca_secret_key="y")
    provider = AlpacaMarketDataProvider(settings, symbol_resolver=lambda _: "AAPL")
    provider._client.close()
    provider._client = httpx.Client(
        base_url="https://data.alpaca.markets",
        transport=httpx.MockTransport(handler),
        timeout=0.1,
    )
    try:
        result = provider.fetch_daily_bars(
            [InstrumentId("inst_aapl")],
            date(2024, 1, 2),
            date(2024, 1, 3),
            request_set_id="rs_iex",
        )
        assert "feed=iex" in seen["url"]
        assert result.ok is True
        assert result.value and len(result.value.bars) == 1
    finally:
        provider.close()


def test_alpaca_btc_symbol_keeps_slash() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "bars": {
                    "BTC/USD": [
                        {
                            "t": "2024-01-01T00:00:00Z",
                            "o": 1,
                            "h": 1,
                            "l": 1,
                            "c": 1,
                            "v": 1,
                        }
                    ]
                }
            },
            request=request,
        )

    settings = Settings(_env_file=None, alpaca_api_key="x", alpaca_secret_key="y")
    provider = AlpacaMarketDataProvider(settings)
    provider._client.close()
    provider._client = httpx.Client(
        base_url="https://data.alpaca.markets",
        transport=httpx.MockTransport(handler),
        timeout=0.1,
    )
    try:
        result = provider.fetch_btc_bars(
            date(2024, 1, 1),
            date(2024, 1, 2),
            request_set_id="rs_btc_slash",
        )
        assert "BTC%2FUSD" in seen["url"] or "BTC/USD" in seen["url"]
        assert result.ok is True
        assert result.value and len(result.value.bars) == 1
    finally:
        provider.close()


def test_alpaca_partial_daily_batch_marks_degraded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "AAPL" in str(request.url):
            return httpx.Response(500, json={"message": "fail"}, request=request)
        return httpx.Response(
            200,
            json={
                "bars": [
                    {
                        "t": "2024-01-02T00:00:00Z",
                        "o": 1,
                        "h": 1,
                        "l": 1,
                        "c": 1,
                        "v": 1,
                    }
                ]
            },
            request=request,
        )

    settings = Settings(_env_file=None, alpaca_api_key="x", alpaca_secret_key="y")
    provider = AlpacaMarketDataProvider(
        settings,
        symbol_resolver=lambda inst: "SPY" if str(inst) == "inst_spy" else "AAPL",
    )
    provider._client.close()
    provider._client = httpx.Client(
        base_url="https://data.alpaca.markets",
        transport=httpx.MockTransport(handler),
        timeout=0.1,
    )
    try:
        result = provider.fetch_daily_bars(
            [InstrumentId("inst_spy"), InstrumentId("inst_aapl")],
            date(2024, 1, 1),
            date(2024, 1, 5),
            request_set_id="rs_partial",
        )
        assert result.ok is True
        assert result.degraded is True
        batch = result.value
        assert batch is not None
        assert InstrumentId("inst_aapl") in batch.failed_instruments
        assert batch.bars
    finally:
        provider.close()


def test_bootstrap_smoke_universe(store: Store, settings: Settings) -> None:
    mapping = bootstrap_smoke_universe(store.conn, settings, provider="fixture")
    assert "SPY" in mapping
    assert settings.btc_symbol in mapping
