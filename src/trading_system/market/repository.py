"""DuckDB persistence for market data bars, observations, and coverage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import duckdb

from trading_system.ids import InstrumentId, RequestSetId
from trading_system.market.status import DataStatus
from trading_system.providers.interfaces import CoverageRow, DailyBar, LatestQuote


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def upsert_equity_daily_bars(conn: duckdb.DuckDBPyConnection, bars: list[DailyBar]) -> int:
    if not bars:
        return 0
    for bar in bars:
        conn.execute(
            """
            INSERT OR REPLACE INTO equity_daily_bars
            (instrument_id, session_date, open, high, low, close, volume,
             finality, adjustment_revision, provider, receive_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                bar.instrument_id,
                bar.session_date,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.finality.value,
                bar.adjustment_revision,
                bar.provider,
                bar.receive_ts,
            ],
        )
    return len(bars)


def upsert_btc_daily_bars(conn: duckdb.DuckDBPyConnection, bars: list[DailyBar]) -> int:
    if not bars:
        return 0
    for bar in bars:
        conn.execute(
            """
            INSERT OR REPLACE INTO btc_daily_bars
            (session_date, open, high, low, close, volume,
             finality, adjustment_revision, provider, receive_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                bar.session_date,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.finality.value,
                bar.adjustment_revision,
                bar.provider,
                bar.receive_ts,
            ],
        )
    return len(bars)


def upsert_coverage_rows(conn: duckdb.DuckDBPyConnection, rows: list[CoverageRow]) -> int:
    now = _utcnow()
    for row in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO market_coverage
            (provider, adjustment_revision, instrument_id, session, request_set_id, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row.provider,
                row.adjustment_revision,
                row.instrument_id,
                row.session,
                row.request_set_id,
                row.status.value,
                now,
            ],
        )
    return len(rows)


def insert_live_observation(
    conn: duckdb.DuckDBPyConnection,
    *,
    instrument_id: InstrumentId,
    asset_class: str,
    quote: LatestQuote | None,
    data_status: DataStatus,
    request_set_id: RequestSetId | None = None,
    lkg_fallback: bool = False,
    is_canonical: bool | None = None,
    metadata: dict[str, object] | None = None,
) -> str:
    obs_id = f"obs_{uuid4().hex}"
    receive_ts = quote.receive_ts if quote else _utcnow()
    canonical = (
        is_canonical
        if is_canonical is not None
        else (
            data_status == DataStatus.VALID
            and quote is not None
            and quote.event_ts is not None
            and not lkg_fallback
        )
    )
    conn.execute(
        """
        INSERT INTO live_observations
        (observation_id, instrument_id, asset_class, price, volume, event_ts, receive_ts,
         provider, data_status, degraded, is_canonical, request_set_id, lkg_fallback, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            obs_id,
            instrument_id,
            asset_class,
            quote.price if quote else None,
            quote.volume if quote else None,
            quote.event_ts if quote else None,
            receive_ts,
            quote.provider if quote else "none",
            data_status.value,
            quote.degraded if quote else True,
            canonical,
            request_set_id,
            lkg_fallback,
            json.dumps(metadata or {}),
        ],
    )
    return obs_id


def latest_canonical_event_ts(
    conn: duckdb.DuckDBPyConnection,
    instrument_id: InstrumentId,
) -> datetime | None:
    row = conn.execute(
        """
        SELECT event_ts FROM live_observations
        WHERE instrument_id = ? AND is_canonical = TRUE AND event_ts IS NOT NULL
        ORDER BY event_ts DESC
        LIMIT 1
        """,
        [instrument_id],
    ).fetchone()
    return row[0] if row else None


def latest_trusted_receive_ts(
    conn: duckdb.DuckDBPyConnection,
    instrument_id: InstrumentId,
) -> datetime | None:
    """Receive time of the last non-LKG trusted observation (for age display)."""
    row = conn.execute(
        """
        SELECT receive_ts FROM live_observations
        WHERE instrument_id = ? AND price IS NOT NULL AND lkg_fallback = FALSE
        ORDER BY receive_ts DESC
        LIMIT 1
        """,
        [instrument_id],
    ).fetchone()
    return row[0] if row else None


def latest_lkg_quote(
    conn: duckdb.DuckDBPyConnection,
    instrument_id: InstrumentId,
) -> tuple[float | None, DataStatus, datetime | None, bool]:
    """Return (price, status, trusted_receive_ts, is_lkg) for display."""
    row = conn.execute(
        """
        SELECT price, data_status, receive_ts, lkg_fallback
        FROM live_observations
        WHERE instrument_id = ? AND price IS NOT NULL
        ORDER BY receive_ts DESC
        LIMIT 1
        """,
        [instrument_id],
    ).fetchone()
    if row is None:
        return None, DataStatus.MISSING, None, False
    price, status, receive_ts, lkg = row
    trusted_at = latest_trusted_receive_ts(conn, instrument_id) or receive_ts
    return float(price), DataStatus(status), trusted_at, bool(lkg)


def coverage_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    request_set_id: RequestSetId | None = None,
) -> dict[str, int]:
    if request_set_id:
        rows = conn.execute(
            """
            SELECT status, COUNT(*) FROM market_coverage
            WHERE request_set_id = ?
            GROUP BY status
            """,
            [request_set_id],
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT status, COUNT(*) FROM market_coverage
            GROUP BY status
            """
        ).fetchall()
    summary = {"requested": 0, "succeeded": 0, "missing": 0, "error": 0, "quarantined": 0}
    for status, count in rows:
        key = status.lower()
        if key in summary:
            summary[key] = int(count)
        summary["requested"] += int(count)
    return summary


def upsert_provider_state(
    conn: duckdb.DuckDBPyConnection,
    *,
    provider_key: str,
    role: str,
    status: DataStatus,
    last_success_at: datetime | None = None,
    last_error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO provider_state
        (provider_key, role, status, last_success_at, last_error, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [provider_key, role, status.value, last_success_at, last_error, _utcnow()],
    )


def replace_live_watchlist(
    conn: duckdb.DuckDBPyConnection,
    entries: list[tuple[InstrumentId, str, int]],
) -> None:
    conn.execute("DELETE FROM live_watchlist")
    now = _utcnow()
    for instrument_id, reason, priority in entries:
        conn.execute(
            """
            INSERT INTO live_watchlist (instrument_id, reason, priority, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            [instrument_id, reason, priority, now],
        )
