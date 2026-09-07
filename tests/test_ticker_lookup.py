"""Confirm portfolio tickers against universe / bars / optional Alpaca."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from trading_system.config import Settings
from trading_system.storage import Store
from trading_system.ticker_lookup import lookup_ticker


def _store(path: Path) -> Store:
    store = Store(path)
    store.open(acquire_writer=True)
    return store


def _add_universe(store: Store, symbol: str, rank: int = 1) -> None:
    store.conn.execute(
        """
        INSERT INTO universe_members
        (preset, symbol, member_rank, dollar_volume, source, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ["large_liquid_500", symbol, rank, 1e9, "test", datetime.now(timezone.utc)],
    )


def test_lookup_confirms_universe_member(tmp_path: Path) -> None:
    store = _store(tmp_path / "t.duckdb")
    try:
        _add_universe(store, "AAPL")
        settings = Settings(_env_file=None)
        got = lookup_ticker(store.conn, settings, "aapl", fetch_remote=lambda _: None)
        assert got.ok is True
        assert got.symbol == "AAPL"
        assert got.instrument_id == "inst_aapl"
        assert got.in_universe is True
        assert got.message.startswith("확인됨: AAPL")
    finally:
        store.close()


def test_lookup_rejects_unknown_and_suggests(tmp_path: Path) -> None:
    store = _store(tmp_path / "t.duckdb")
    try:
        _add_universe(store, "GOOGL")
        settings = Settings(_env_file=None)
        got = lookup_ticker(store.conn, settings, "GOOG", fetch_remote=lambda _: None)
        assert got.ok is False
        assert "GOOGL" in got.suggestions
        assert "찾을 수 없습니다" in got.message
    finally:
        store.close()


def test_lookup_blank_and_bad_format(tmp_path: Path) -> None:
    store = _store(tmp_path / "t.duckdb")
    try:
        settings = Settings(_env_file=None)
        blank = lookup_ticker(store.conn, settings, "  ", fetch_remote=lambda _: None)
        assert blank.ok is False
        bad = lookup_ticker(store.conn, settings, "$$$", fetch_remote=lambda _: None)
        assert bad.ok is False
    finally:
        store.close()


def test_lookup_accepts_alpaca_when_local_miss(tmp_path: Path) -> None:
    store = _store(tmp_path / "t.duckdb")
    try:
        settings = Settings(_env_file=None)
        got = lookup_ticker(
            store.conn,
            settings,
            "ZZZZ",
            fetch_remote=lambda _: {"symbol": "ZZZZ", "name": "Fake Co", "status": "active"},
        )
        assert got.ok is True
        assert got.source == "alpaca"
        assert "Fake Co" in got.message
        assert "유니버스에는 없습니다" in got.message
    finally:
        store.close()


def test_lookup_confirms_bars_without_universe(tmp_path: Path) -> None:
    store = _store(tmp_path / "t.duckdb")
    try:
        store.conn.execute(
            """
            INSERT INTO equity_daily_bars
            (instrument_id, session_date, open, high, low, close, volume,
             finality, adjustment_revision, provider, receive_ts)
            VALUES (?, ?, 1, 1, 1, 1, 1, 'final', 'adj1', 'test', ?)
            """,
            ["inst_hubb", "2024-01-02", datetime.now(timezone.utc)],
        )
        settings = Settings(_env_file=None)
        got = lookup_ticker(store.conn, settings, "HUBB", fetch_remote=lambda _: None)
        assert got.ok is True
        assert got.has_bars is True
        assert got.in_universe is False
        assert got.source == "bars"
    finally:
        store.close()
