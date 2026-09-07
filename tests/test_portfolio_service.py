"""portfolio.py / portfolio_service.py: NaN/Infinity must be rejected like any other
non-positive value, and update_lot must not lose a concurrent partial edit."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_system.config import Settings
from trading_system.portfolio import Lot, PortfolioUnits
from trading_system.portfolio_service import add_lot, set_total_base_units, update_lot
from trading_system.storage import Store
from trading_system.ui import app as ui_app


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0, 0.0])
def test_lot_rejects_nan_infinity_and_nonpositive_units(bad: float) -> None:
    with pytest.raises(ValueError):
        Lot(instrument_id="AAPL", acquisition_units=bad, acquisition_price=100.0, acquired_on=date(2024, 1, 1))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0, 0.0])
def test_lot_rejects_nan_infinity_and_nonpositive_price(bad: float) -> None:
    with pytest.raises(ValueError):
        Lot(instrument_id="AAPL", acquisition_units=1.0, acquisition_price=bad, acquired_on=date(2024, 1, 1))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0.0])
def test_portfolio_units_rejects_nan_infinity_total_base_units(bad: float) -> None:
    with pytest.raises(ValueError):
        PortfolioUnits(total_base_units=bad)


def test_set_total_base_units_rejects_nan_and_infinity(tmp_path: Path) -> None:
    store = Store(tmp_path / "meta.duckdb")
    store.open(acquire_writer=True)
    try:
        with pytest.raises(ValueError):
            set_total_base_units(store.conn, float("nan"))
        with pytest.raises(ValueError):
            set_total_base_units(store.conn, float("inf"))
    finally:
        store.close()


def test_update_lot_rejects_nan_and_infinity(tmp_path: Path) -> None:
    store = Store(tmp_path / "lots.duckdb")
    store.open(acquire_writer=True)
    try:
        lot = add_lot(
            store.conn,
            Lot(instrument_id="AAPL", acquisition_units=10.0, acquisition_price=100.0, acquired_on=date(2024, 1, 1)),
        )
        with pytest.raises(ValueError):
            update_lot(store.conn, lot.lot_id, units=float("nan"))
        with pytest.raises(ValueError):
            update_lot(store.conn, lot.lot_id, price=float("inf"))
        row = store.conn.execute(
            "SELECT acquisition_units, acquisition_price FROM portfolio_lots WHERE lot_id = ?", [lot.lot_id]
        ).fetchone()
        assert row == (10.0, 100.0)  # rejected edits must not partially apply
    finally:
        store.close()


def test_update_lot_partial_edit_does_not_lose_the_other_field(tmp_path: Path) -> None:
    """A concurrent units-only edit and a price-only edit must both land -- a naive
    read-then-write-both-columns update can silently drop whichever one lost the race."""
    store = Store(tmp_path / "race.duckdb")
    store.open(acquire_writer=True)
    try:
        lot = add_lot(
            store.conn,
            Lot(instrument_id="AAPL", acquisition_units=5.0, acquisition_price=15.0, acquired_on=date(2024, 1, 1)),
        )
        # Simulates two requests interleaving: both would have read (5.0, 15.0) under
        # the old read-then-write-both-columns approach. Each now only ever touches
        # the column it was actually asked to change.
        update_lot(store.conn, lot.lot_id, units=10.0)
        update_lot(store.conn, lot.lot_id, price=20.0)
        row = store.conn.execute(
            "SELECT acquisition_units, acquisition_price FROM portfolio_lots WHERE lot_id = ?", [lot.lot_id]
        ).fetchone()
        assert row == (10.0, 20.0)
    finally:
        store.close()


def test_portfolio_add_route_rejects_nonpositive_units_without_500(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "ui.duckdb"
    settings = Settings(_env_file=None, duckdb_path=db)
    monkeypatch.setattr(ui_app, "get_settings", lambda: settings)
    client = TestClient(ui_app.create_app(), follow_redirects=False)

    resp = client.post(
        "/portfolio/add",
        data={"instrument_id": "AAPL", "units": "0", "price": "100", "acquired_on": "2024-01-01"},
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/portfolio?err=lot"

    resp_nan = client.post(
        "/portfolio/add",
        data={"instrument_id": "AAPL", "units": "nan", "price": "100", "acquired_on": "2024-01-01"},
    )
    assert resp_nan.status_code == 303
    assert resp_nan.headers["location"] == "/portfolio?err=lot"

    store = Store(db, read_only=True)
    store.open(acquire_writer=False)
    try:
        row = store.conn.execute("SELECT COUNT(*) FROM portfolio_lots").fetchone()
        assert row is not None and int(row[0]) == 0
    finally:
        store.close()


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-5"])
def test_portfolio_meta_route_rejects_nan_infinity_and_nonpositive_without_500(
    bad: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "ui_meta.duckdb"
    settings = Settings(_env_file=None, duckdb_path=db)
    monkeypatch.setattr(ui_app, "get_settings", lambda: settings)
    client = TestClient(ui_app.create_app(), follow_redirects=False)

    from trading_system.portfolio_service import get_total_base_units

    store = Store(db)
    store.open(acquire_writer=True)
    try:
        before = get_total_base_units(store.conn)
    finally:
        store.close()

    resp = client.post("/portfolio/meta", data={"total_base_units": bad})
    assert resp.status_code == 303
    assert resp.headers["location"] == "/portfolio?err=meta"

    store = Store(db, read_only=True)
    store.open(acquire_writer=False)
    try:
        assert get_total_base_units(store.conn) == before  # unchanged, not corrupted
    finally:
        store.close()


def test_portfolio_write_returns_busy_redirect_instead_of_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "ui_busy.duckdb"
    settings = Settings(_env_file=None, duckdb_path=db, writer_lease_stale_seconds=60)
    monkeypatch.setattr(ui_app, "get_settings", lambda: settings)
    owner = Store(db)
    owner.open(acquire_writer=True, stale_seconds=60)
    try:
        client = TestClient(ui_app.create_app(), follow_redirects=False)
        response = client.post("/portfolio/meta", data={"total_base_units": "2500"})
        assert response.status_code == 303
        assert response.headers["location"] == "/portfolio?err=busy"
    finally:
        owner.close()
