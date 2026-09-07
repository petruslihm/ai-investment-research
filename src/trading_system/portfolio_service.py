"""Manual portfolio CRUD. AI recommendations never mutate lots."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import duckdb

from trading_system.ids import InstrumentId, canonicalize_instrument_id
from trading_system.marked_units import marked_units_from_lots
from trading_system.market.repository import replace_live_watchlist
from trading_system.portfolio import InstrumentPosition, Lot, PortfolioUnits


def get_total_base_units(conn: duckdb.DuckDBPyConnection) -> float:
    row = conn.execute("SELECT value FROM portfolio_meta WHERE key = 'total_base_units'").fetchone()
    return float(row[0]) if row else 1000.0


def set_total_base_units(conn: duckdb.DuckDBPyConnection, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("total_base_units must be a positive finite number")
    conn.execute(
        "INSERT OR REPLACE INTO portfolio_meta (key, value) VALUES ('total_base_units', ?)",
        [str(value)],
    )


def reconcile_portfolio_instrument_ids(conn: duckdb.DuckDBPyConnection) -> int:
    """Rewrite ticker-style lot ids (AMD) to inst_amd so they match bars and scores."""
    rows = conn.execute("SELECT lot_id, instrument_id FROM portfolio_lots").fetchall()
    n = 0
    for lot_id, raw in rows:
        canon = str(canonicalize_instrument_id(raw))
        if canon == str(raw):
            continue
        conn.execute(
            "UPDATE portfolio_lots SET instrument_id = ? WHERE lot_id = ?",
            [canon, lot_id],
        )
        n += 1
    if n:
        _sync_watchlist(conn)
    return n


def add_lot(conn: duckdb.DuckDBPyConnection, lot: Lot) -> Lot:
    conn.execute(
        """
        INSERT INTO portfolio_lots
        (lot_id, instrument_id, acquisition_units, acquisition_price, acquired_on, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            lot.lot_id,
            lot.instrument_id,
            lot.acquisition_units,
            lot.acquisition_price,
            lot.acquired_on,
            lot.notes,
        ],
    )
    _sync_watchlist(conn)
    return lot


def delete_lot(conn: duckdb.DuckDBPyConnection, lot_id: str) -> None:
    conn.execute("DELETE FROM portfolio_lots WHERE lot_id = ?", [lot_id])
    _sync_watchlist(conn)


def update_lot(
    conn: duckdb.DuckDBPyConnection,
    lot_id: str,
    *,
    units: float | None = None,
    price: float | None = None,
) -> None:
    if units is not None and (not math.isfinite(units) or units <= 0):
        raise ValueError("units must be a positive finite number")
    if price is not None and (not math.isfinite(price) or price <= 0):
        raise ValueError("price must be a positive finite number")
    row = conn.execute(
        "SELECT acquisition_units, acquisition_price FROM portfolio_lots WHERE lot_id = ?", [lot_id]
    ).fetchone()
    if not row:
        raise KeyError(lot_id)
    conn.execute(
        "UPDATE portfolio_lots SET acquisition_units = COALESCE(?, acquisition_units), "
        "acquisition_price = COALESCE(?, acquisition_price) WHERE lot_id = ?",
        [units, price, lot_id],
    )


def set_lots_acquired_on(conn: duckdb.DuckDBPyConnection, acquired_on: date) -> int:
    """Rewrite every lot's acquisition date. User-entered lots only; AI never calls this."""
    conn.execute("UPDATE portfolio_lots SET acquired_on = ?", [acquired_on])
    row = conn.execute("SELECT COUNT(*) FROM portfolio_lots").fetchone()
    return int(row[0]) if row else 0


def list_lots(conn: duckdb.DuckDBPyConnection) -> list[Lot]:
    rows = conn.execute(
        "SELECT lot_id, instrument_id, acquisition_units, acquisition_price, acquired_on, notes FROM portfolio_lots"
    ).fetchall()
    return [
        Lot(
            lot_id=r[0],
            instrument_id=InstrumentId(r[1]),
            acquisition_units=float(r[2]),
            acquisition_price=float(r[3]),
            acquired_on=r[4],
            notes=r[5],
        )
        for r in rows
    ]


def load_portfolio(conn: duckdb.DuckDBPyConnection) -> PortfolioUnits:
    reconcile_portfolio_instrument_ids(conn)
    lots = list_lots(conn)
    by: dict[str, list[Lot]] = {}
    for lot in lots:
        by.setdefault(str(lot.instrument_id), []).append(lot)
    positions = [
        InstrumentPosition(instrument_id=InstrumentId(k), lots=v, price_available=True)
        for k, v in by.items()
    ]
    p = PortfolioUnits(total_base_units=get_total_base_units(conn), positions=positions)
    p.recompute_valuation_state()
    return p


def latest_final_close(conn: duckdb.DuckDBPyConnection, instrument_id: object) -> float | None:
    """Latest FINAL equity close, or latest BTC daily close for the BTC sleeve."""
    iid = str(instrument_id)
    row = conn.execute(
        """
        SELECT close FROM equity_daily_bars
        WHERE instrument_id = ? AND finality = 'final'
        ORDER BY session_date DESC LIMIT 1
        """,
        [iid],
    ).fetchone()
    if row is None and "btc" in iid.lower():
        row = conn.execute("SELECT close FROM btc_daily_bars ORDER BY session_date DESC LIMIT 1").fetchone()
    if not row or row[0] is None:
        return None
    try:
        px = float(row[0])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(px) or px <= 0:
        return None
    return px


def mark_positions(conn: duckdb.DuckDBPyConnection, portfolio: PortfolioUnits) -> PortfolioUnits:
    """Official marked units: per-lot FINAL close / purchase price, then sum."""
    for pos in portfolio.positions:
        close = latest_final_close(conn, pos.instrument_id)
        marked = marked_units_from_lots(pos.lots, close)
        if marked is None:
            pos.price_available = False
            pos.marked_units_final = None
            pos.marked_units_intraday_preview = None
            continue
        pos.marked_units_final = marked
        pos.price_available = True
        q = conn.execute(
            """
            SELECT price FROM live_observations
            WHERE instrument_id = ? AND is_canonical = TRUE
            ORDER BY receive_ts DESC LIMIT 1
            """,
            [pos.instrument_id],
        ).fetchone()
        qpx = float(q[0]) if q and q[0] is not None else None
        if qpx is not None and (not math.isfinite(qpx) or qpx <= 0):
            qpx = None
        pos.marked_units_intraday_preview = marked_units_from_lots(pos.lots, qpx)
    portfolio.recompute_valuation_state()
    return portfolio


def _sync_watchlist(conn: duckdb.DuckDBPyConnection) -> None:
    ids = [InstrumentId(r[0]) for r in conn.execute("SELECT DISTINCT instrument_id FROM portfolio_lots").fetchall()]
    entries = [(iid, "held_position", 0) for iid in ids]
    replace_live_watchlist(conn, entries)
