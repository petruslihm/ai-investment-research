"""Scan universe: presets, held/watchlist overrides, benchmark and BTC separation."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading_system.config import Settings
from trading_system.market.registry import stable_instrument_id
from trading_system.storage import Store
from trading_system.universe import (
    BENCHMARK_SYMBOLS,
    PRESET_CUSTOM,
    PRESET_FAST,
    PRESET_LARGE_LIQUID,
    add_custom_symbols,
    add_watchlist_symbols,
    coverage_stats,
    parse_symbol_input,
    prepare_scan_universe,
    refresh_universe,
    resolve_scan_universe,
    seed_symbols,
    store_universe,
    stored_preset_symbols,
    universe_needs_refresh,
    universe_state,
)


def _settings(**kwargs: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "alpaca_api_key": None,
        "alpaca_secret_key": None,
        "openai_api_key": None,
    }
    base.update(kwargs)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "universe.duckdb")
    s.open(acquire_writer=True)
    yield s
    s.close()


class FakeUniverseProvider:
    source_name = "fake"

    def __init__(self, symbols: list[str], note: str | None = None) -> None:
        self.symbols = symbols
        self.note = note
        self.calls = 0

    def fetch_liquid_symbols(self, limit: int) -> tuple[list[tuple[str, float]], str | None]:
        self.calls += 1
        ranked = [(s, float(10_000 - i)) for i, s in enumerate(self.symbols)]
        return ranked[:limit], self.note


class BrokenUniverseProvider:
    source_name = "broken"

    def fetch_liquid_symbols(self, limit: int):  # noqa: ANN201
        raise RuntimeError("Alpaca 403 forbidden")


def test_universe_tables_exist(store: Store) -> None:
    tables = set(store.list_tables())
    for name in ("universe_members", "universe_state", "user_watchlist", "custom_universe"):
        assert name in tables


def test_default_preset_is_not_the_smoke_list() -> None:
    cfg = _settings()
    assert cfg.scan_universe_preset == PRESET_LARGE_LIQUID
    seed = seed_symbols(PRESET_LARGE_LIQUID)
    assert len(seed) > len(cfg.smoke_universe)


def test_refresh_stores_ranked_members(store: Store) -> None:
    provider = FakeUniverseProvider(["AAPL", "MSFT", "NVDA", "TSLA"])
    out = refresh_universe(store.conn, _settings(), preset=PRESET_FAST, provider=provider)
    assert out["status"] == "AVAILABLE"
    assert stored_preset_symbols(store.conn, PRESET_FAST) == ["AAPL", "MSFT", "NVDA", "TSLA"]
    assert universe_state(store.conn, PRESET_FAST)["source"] == "fake"


def test_failed_refresh_keeps_last_known_universe(store: Store) -> None:
    good = FakeUniverseProvider(["AAPL", "MSFT"])
    refresh_universe(store.conn, _settings(), preset=PRESET_FAST, provider=good)
    out = refresh_universe(store.conn, _settings(), preset=PRESET_FAST, provider=BrokenUniverseProvider())
    assert out["status"] == "DEGRADED"
    assert stored_preset_symbols(store.conn, PRESET_FAST) == ["AAPL", "MSFT"]
    assert "403" in str(universe_state(store.conn, PRESET_FAST)["last_error"])


def test_failed_refresh_without_history_falls_back_to_seed_not_empty(store: Store) -> None:
    out = refresh_universe(store.conn, _settings(), preset=PRESET_FAST, provider=BrokenUniverseProvider())
    stored = stored_preset_symbols(store.conn, PRESET_FAST)
    assert out["source"] == "seed"
    assert stored
    assert "AAPL" in stored


def test_btc_never_enters_equity_universe(store: Store) -> None:
    cfg = _settings()
    provider = FakeUniverseProvider(["AAPL", "BTC/USD", "MSFT"])
    refresh_universe(store.conn, cfg, preset=PRESET_FAST, provider=provider)
    resolved = resolve_scan_universe(store.conn, _settings(scan_universe_preset=PRESET_FAST))
    assert "BTC/USD" not in resolved.candidates
    assert "BTC/USD" not in resolved.scan_symbols
    assert resolved.with_btc(cfg)[-1] == "BTC/USD"


def test_spy_is_benchmark_not_candidate(store: Store) -> None:
    provider = FakeUniverseProvider(["SPY", "AAPL", "MSFT"])
    refresh_universe(store.conn, _settings(), preset=PRESET_FAST, provider=provider)
    resolved = resolve_scan_universe(store.conn, _settings(scan_universe_preset=PRESET_FAST))
    assert "SPY" not in resolved.candidates
    assert resolved.benchmarks == BENCHMARK_SYMBOLS
    assert "SPY" in resolved.scan_symbols


def test_watchlist_symbols_are_always_scanned(store: Store) -> None:
    provider = FakeUniverseProvider(["AAPL", "MSFT"])
    refresh_universe(store.conn, _settings(), preset=PRESET_FAST, provider=provider)
    add_watchlist_symbols(store.conn, ["PLTR"])
    resolved = resolve_scan_universe(store.conn, _settings(scan_universe_preset=PRESET_FAST))
    assert "PLTR" in resolved.candidates
    assert resolved.forced["PLTR"] == "watchlist"


def test_held_symbols_are_always_scanned(store: Store) -> None:
    provider = FakeUniverseProvider(["AAPL", "MSFT"])
    refresh_universe(store.conn, _settings(), preset=PRESET_FAST, provider=provider)
    store.conn.execute(
        """
        INSERT INTO instruments (instrument_id, issuer_id, asset_class, display_name, created_at)
        VALUES ('inst_cost', 'iss_cost', 'us_equity', 'COST', CURRENT_TIMESTAMP)
        """
    )
    store.conn.execute(
        """
        INSERT INTO instrument_aliases (instrument_id, provider, symbol, effective_from)
        VALUES ('inst_cost', 'alpaca', 'COST', DATE '2000-01-01')
        """
    )
    store.conn.execute(
        """
        INSERT INTO portfolio_lots
        (lot_id, instrument_id, acquisition_units, acquisition_price, acquired_on)
        VALUES ('lot_1', 'inst_cost', 1.0, 1.0, DATE '2026-01-02')
        """
    )
    resolved = resolve_scan_universe(store.conn, _settings(scan_universe_preset=PRESET_FAST))
    assert "COST" in resolved.candidates
    assert resolved.forced["COST"] == "held"
    # Forced names lead so downstream caps reach them first.
    assert resolved.candidates[0] == "COST"


def test_custom_preset_uses_only_user_symbols(store: Store) -> None:
    add_custom_symbols(store.conn, parse_symbol_input("nvda, amd  crwd"))
    cfg = _settings(scan_universe_preset=PRESET_CUSTOM)
    refresh_universe(store.conn, cfg, preset=PRESET_CUSTOM)
    resolved = resolve_scan_universe(store.conn, cfg)
    assert set(resolved.candidates) == {"NVDA", "AMD", "CRWD"}


def test_parse_symbol_input_handles_separators() -> None:
    assert parse_symbol_input("aapl, msft\nnvda  aapl") == ["AAPL", "MSFT", "NVDA"]
    assert parse_symbol_input("") == []


def test_needs_refresh_respects_interval(store: Store) -> None:
    cfg = _settings(scan_universe_preset=PRESET_FAST, universe_refresh_days=7)
    assert universe_needs_refresh(store.conn, cfg) is True
    refresh_universe(store.conn, cfg, preset=PRESET_FAST, provider=FakeUniverseProvider(["AAPL"]))
    assert universe_needs_refresh(store.conn, cfg) is False
    store.conn.execute(
        "UPDATE universe_state SET last_refresh_at = ? WHERE preset = ?",
        [datetime.now(timezone.utc) - timedelta(days=30), PRESET_FAST],
    )
    assert universe_needs_refresh(store.conn, cfg) is True


def test_prepare_does_not_raise_when_provider_broken(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trading_system.universe.refresh_universe",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    resolved = prepare_scan_universe(store.conn, _settings(scan_universe_preset=PRESET_FAST))
    assert resolved.candidates
    assert universe_state(store.conn, PRESET_FAST)["status"] in {"DEGRADED", "UNAVAILABLE"}


def test_coverage_stats_counts_missing_and_stale(store: Store) -> None:
    store_universe(
        store.conn,
        preset=PRESET_FAST,
        members=[("AAPL", 1.0), ("MSFT", 1.0), ("NVDA", 1.0)],
        source="fake",
        status="AVAILABLE",
    )
    for symbol, day in (("AAPL", date(2026, 8, 28)), ("MSFT", date(2026, 6, 1))):
        store.conn.execute(
            """
            INSERT INTO equity_daily_bars
            (instrument_id, session_date, open, high, low, close, volume, finality,
             adjustment_revision, provider, receive_ts)
            VALUES (?, ?, 1, 1, 1, 1, 1, 'final', 'rev', 'alpaca', CURRENT_TIMESTAMP)
            """,
            [str(stable_instrument_id(symbol)), day],
        )
    resolved = resolve_scan_universe(store.conn, _settings(scan_universe_preset=PRESET_FAST))
    stats = coverage_stats(store.conn, resolved)
    assert stats["total"] == 4  # SPY benchmark + 3 candidates
    assert stats["ready"] == 1  # AAPL
    assert stats["stale"] == 1  # MSFT
    assert stats["unavailable"] == 2  # NVDA + SPY
    assert stats["coverage_pct"] == 25.0
