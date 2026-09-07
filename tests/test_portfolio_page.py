"""Portfolio page must show the saved total_base_units, not a hardcoded 1000."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_system.config import Settings
from trading_system.portfolio import Lot
from trading_system.portfolio_service import (
    add_lot,
    get_total_base_units,
    set_lots_acquired_on,
    set_total_base_units,
)
from trading_system.storage import Store
from trading_system.ui import app as ui_app
from trading_system.ui.app import format_elapsed_ko


def test_set_total_base_units_round_trips(tmp_path: Path) -> None:
    store = Store(tmp_path / "meta.duckdb")
    store.open(acquire_writer=True)
    try:
        assert get_total_base_units(store.conn) == 1000.0
        set_total_base_units(store.conn, 2500)
        assert get_total_base_units(store.conn) == pytest.approx(2500.0)
    finally:
        store.close()


def test_portfolio_form_shows_saved_base_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "ui.duckdb"
    settings = Settings(_env_file=None, duckdb_path=db)
    monkeypatch.setattr(ui_app, "get_settings", lambda: settings)
    client = TestClient(ui_app.create_app(), follow_redirects=False)

    first = client.get("/portfolio").text
    assert 'name="total_base_units"' in first
    assert 'value="1000"' in first

    saved = client.post("/portfolio/meta", data={"total_base_units": "2500"})
    assert saved.status_code == 303
    assert saved.headers["location"] == "/portfolio"

    again = client.get("/portfolio").text
    assert 'value="2500"' in again
    assert 'value="1000"' not in again


def test_portfolio_date_defaults_to_today(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "ui.duckdb"
    settings = Settings(_env_file=None, duckdb_path=db)
    monkeypatch.setattr(ui_app, "get_settings", lambda: settings)
    client = TestClient(ui_app.create_app(), follow_redirects=False)
    html = client.get("/portfolio").text
    today = settings.today_in_user_tz().isoformat()
    assert f'name="acquired_on"' in html
    assert f'value="{today}"' in html
    assert 'type="date"' in html
    assert "2024-01-02" not in html
    assert 'id="lot-ticker-check"' in html
    assert "확인됨" in html or "확인을 눌러" in html


def test_set_lots_acquired_on_rewrites_every_lot(tmp_path: Path) -> None:
    store = Store(tmp_path / "lots.duckdb")
    store.open(acquire_writer=True)
    try:
        add_lot(
            store.conn,
            Lot(
                instrument_id="FRSH",
                acquisition_units=114,
                acquisition_price=13.1,
                acquired_on=date(2024, 1, 2),
            ),
        )
        n = set_lots_acquired_on(store.conn, date(2026, 9, 1))
        assert n == 1
        row = store.conn.execute("SELECT acquired_on FROM portfolio_lots").fetchone()
        assert row is not None
        got = row[0]
        assert got == date(2026, 9, 1) or str(got)[:10] == "2026-09-01"
    finally:
        store.close()


def test_portfolio_lots_show_confirmation_badge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "ui.duckdb"
    settings = Settings(_env_file=None, duckdb_path=db)
    monkeypatch.setattr(ui_app, "get_settings", lambda: settings)
    monkeypatch.setattr("trading_system.ticker_lookup.fetch_alpaca_asset", lambda *a, **k: None)
    store = Store(db)
    store.open(acquire_writer=True)
    try:
        store.conn.execute(
            """
            INSERT INTO universe_members
            (preset, symbol, member_rank, dollar_volume, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ["large_liquid_500", "FRSH", 1, 1e9, "test", datetime.now(timezone.utc)],
        )
        add_lot(
            store.conn,
            Lot(
                instrument_id="FRSH",
                acquisition_units=114,
                acquisition_price=13.1,
                acquired_on=date(2024, 1, 2),
            ),
        )
        add_lot(
            store.conn,
            Lot(
                instrument_id="NOSUCH",
                acquisition_units=1,
                acquisition_price=1,
                acquired_on=date(2024, 1, 2),
            ),
        )
    finally:
        store.close()
    client = TestClient(ui_app.create_app(), follow_redirects=False)
    html = client.get("/portfolio").text
    assert "lot-verified ok" in html
    assert "확인됨" in html
    assert ">FRSH<" in html
    assert "inst_frsh" not in html
    assert "lot-verified bad" in html
    assert "미확인" in html
    assert "취득일 2024-01-02" in html
    assert "/portfolio/update" in html
    assert "name='units'" in html or 'name="units"' in html
    assert "lot-identity" in html
    assert "lot-actions" in html
    assert "114" in html


def test_portfolio_lot_units_can_be_updated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "ui.duckdb"
    settings = Settings(_env_file=None, duckdb_path=db)
    monkeypatch.setattr(ui_app, "get_settings", lambda: settings)
    monkeypatch.setattr("trading_system.ticker_lookup.fetch_alpaca_asset", lambda *a, **k: None)
    store = Store(db)
    store.open(acquire_writer=True)
    try:
        lot = add_lot(
            store.conn,
            Lot(
                instrument_id="FRSH",
                acquisition_units=114,
                acquisition_price=13.1,
                acquired_on=date(2026, 9, 1),
            ),
        )
        lot_id = lot.lot_id
    finally:
        store.close()
    client = TestClient(ui_app.create_app(), follow_redirects=False)
    add = client.post(
        "/portfolio/update",
        data={"lot_id": lot_id, "units": "150", "price": "13.1"},
    )
    assert add.status_code == 303
    html = client.get("/portfolio").text
    assert "value='150'" in html or 'value="150"' in html
    reduce = client.post(
        "/portfolio/update",
        data={"lot_id": lot_id, "units": "80", "price": "13.1"},
    )
    assert reduce.status_code == 303
    again = client.get("/portfolio").text
    assert "value='80'" in again or 'value="80"' in again
    assert "value='150'" not in again and 'value="150"' not in again
    bad = client.post(
        "/portfolio/update",
        data={"lot_id": lot_id, "units": "0", "price": "13.1"},
    )
    assert bad.status_code == 303
    assert "err=lot" in bad.headers["location"]


def test_ticker_lookup_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "ui.duckdb"
    settings = Settings(_env_file=None, duckdb_path=db)
    monkeypatch.setattr(ui_app, "get_settings", lambda: settings)
    monkeypatch.setattr("trading_system.ticker_lookup.fetch_alpaca_asset", lambda *a, **k: None)
    store = Store(db)
    store.open(acquire_writer=True)
    try:
        store.conn.execute(
            """
            INSERT INTO universe_members
            (preset, symbol, member_rank, dollar_volume, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ["large_liquid_500", "AMD", 1, 1e9, "test", datetime.now(timezone.utc)],
        )
    finally:
        store.close()
    client = TestClient(ui_app.create_app(), follow_redirects=False)
    ok = client.get("/api/v1/ticker-lookup", params={"q": "amd"}).json()
    assert ok["ok"] is True
    assert ok["symbol"] == "AMD"
    assert "확인됨" in ok["message"]
    missing = client.get("/api/v1/ticker-lookup", params={"q": "NOSUCH"}).json()
    assert missing["ok"] is False


def test_format_elapsed_ko() -> None:
    assert format_elapsed_ko(0) == "0초"
    assert format_elapsed_ko(12) == "12초"
    assert format_elapsed_ko(75) == "1분 15초"
    assert format_elapsed_ko(3723) == "1시간 02분 03초"


def test_scan_status_includes_elapsed_clocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "ui.duckdb"
    settings = Settings(_env_file=None, duckdb_path=db)
    monkeypatch.setattr(ui_app, "get_settings", lambda: settings)

    class FakeThread:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(ui_app.threading, "Thread", FakeThread)
    client = TestClient(ui_app.create_app(), follow_redirects=False)
    idle = client.get("/api/v1/scan-status").json()
    assert idle["elapsed_total"] == "0초"
    assert idle["elapsed_step"] == "0초"

    started = client.post("/run-once")
    assert started.status_code == 303
    st = client.get("/api/v1/scan-status").json()
    assert st["status"] == "running"
    assert "elapsed_total_s" in st
    assert "elapsed_step_s" in st
    assert "초" in st["elapsed_total"]
    html = client.get("/").text
    assert "scan-clock" in html
    assert "현재 단계" in html
    assert "전체" in html
