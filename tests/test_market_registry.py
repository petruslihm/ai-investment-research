"""register_held_equities: malformed portfolio_lots rows must warn, not crash or vanish silently."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest

from trading_system.config import Settings
from trading_system.market.registry import register_held_equities


@pytest.fixture()
def settings() -> Settings:
    return Settings(_env_file=None, smoke_universe=("SPY", "AAPL", "BTC/USD"))


@pytest.fixture()
def store(tmp_path: Path):
    from trading_system.storage import Store

    s = Store(tmp_path / "registry.duckdb")
    s.open(acquire_writer=True)
    yield s
    s.close()


def _insert_lot(conn, lot_id: str, instrument_id: str) -> None:
    conn.execute(
        "INSERT INTO portfolio_lots (lot_id, instrument_id, acquisition_units, "
        "acquisition_price, acquired_on, notes) VALUES (?, ?, ?, ?, ?, ?)",
        [lot_id, instrument_id, 10.0, 100.0, date(2024, 1, 1), None],
    )


def test_malformed_instrument_id_is_dropped_with_warning_and_does_not_break_others(
    store, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    _insert_lot(store.conn, "lot_bad", "")
    _insert_lot(store.conn, "lot_good", "GME")
    with caplog.at_level(logging.WARNING, logger="trading_system.market.registry"):
        extra = register_held_equities(store.conn, settings, provider="alpaca")
    assert "GME" in extra
    assert any("dropping unparseable held instrument_id" in r.message for r in caplog.records)


def test_no_malformed_rows_emits_no_warning(store, settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    _insert_lot(store.conn, "lot_good", "GME")
    with caplog.at_level(logging.WARNING, logger="trading_system.market.registry"):
        extra = register_held_equities(store.conn, settings, provider="alpaca")
    assert "GME" in extra
    assert not any("dropping unparseable" in r.message for r in caplog.records)
