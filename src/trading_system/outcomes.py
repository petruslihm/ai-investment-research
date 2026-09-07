"""Matured recommendation outcomes and quant_only vs llm_final override scoring."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import duckdb

from trading_system.ids import new_outcome_snapshot_id
from trading_system.market.calendar import add_sessions
from trading_system.sec_llm import score_overrides


def _valid_px(px: float | None) -> bool:
    return px is not None and px == px and px > 0 and abs(px) != float("inf")


def _close_asof(conn: duckdb.DuckDBPyConnection, instrument_id: str, as_of: date) -> float | None:
    if "btc" in instrument_id:
        row = conn.execute(
            """
            SELECT close FROM btc_daily_bars
            WHERE session_date <= ? ORDER BY session_date DESC LIMIT 1
            """,
            [as_of],
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT close FROM equity_daily_bars
            WHERE instrument_id = ? AND session_date <= ? AND finality = 'final'
            ORDER BY session_date DESC LIMIT 1
            """,
            [instrument_id, as_of],
        ).fetchone()
    if not row:
        return None
    px = float(row[0])
    return px if _valid_px(px) else None


def record_matured_outcomes(
    conn: duckdb.DuckDBPyConnection,
    *,
    last_equity: date,
    last_btc: date,
    feature_snapshot_id: str,
    provider: str,
    adjustment_revision: str,
) -> dict:
    """Persist realized returns for recommendations whose horizon window has matured."""
    oid = str(new_outcome_snapshot_id())
    now = datetime.now(timezone.utc)
    conn.execute(
        """
        INSERT INTO outcome_snapshots
        (outcome_snapshot_id, feature_snapshot_id, price_basis, adjustment_revision,
         evaluation_basis, provider, published_at)
        VALUES (?, ?, 'split_adjusted', ?, 'adjusted_revisioned', ?, ?)
        """,
        [oid, feature_snapshot_id, adjustment_revision, provider, now],
    )
    recs = conn.execute(
        "SELECT tick_id, recommendation_id, source, payload_json FROM tick_recommendations"
    ).fetchall()
    n = 0
    matured_by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    for tick_id, rec_id, source, payload_json in recs:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            continue
        inner = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
        inst = str(inner.get("instrument_id") or payload.get("instrument_id") or "")
        if not inst:
            continue
        wm = conn.execute(
            "SELECT value FROM watermarks WHERE tick_id = ? AND watermark_key = 'cycle'",
            [tick_id],
        ).fetchone()
        if not wm:
            continue
        try:
            as_of = date.fromisoformat(str(wm[0]))
        except ValueError:
            continue
        is_btc = "btc" in inst
        last = last_btc if is_btc else last_equity
        p0 = _close_asof(conn, inst, as_of)
        if p0 is None:
            continue
        horizons = inner.get("horizons") or []
        hs = [int(h.get("horizon")) for h in horizons if isinstance(h, dict) and h.get("horizon") is not None]
        if not hs:
            hs = [5, 10, 20]
        rets: dict[str, float] = {}
        for h in hs:
            end = (as_of + timedelta(days=h)) if is_btc else add_sessions(as_of, h)
            if end > last:
                continue
            exists = conn.execute(
                "SELECT 1 FROM recommendation_outcomes WHERE recommendation_id = ? AND horizon = ?",
                [rec_id, h],
            ).fetchone()
            if exists:
                row = conn.execute(
                    """
                    SELECT realized_return FROM recommendation_outcomes
                    WHERE recommendation_id = ? AND horizon = ?
                    """,
                    [rec_id, h],
                ).fetchone()
                if row and row[0] is not None:
                    rets[str(h)] = float(row[0])
                continue
            p1 = _close_asof(conn, inst, end)
            if p1 is None:
                continue
            realized = (p1 / p0) - 1.0
            conn.execute(
                """
                INSERT INTO recommendation_outcomes
                (outcome_id, recommendation_id, tick_id, source, instrument_id, horizon,
                 realized_return, outcome_snapshot_id, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [f"ro_{uuid4().hex[:16]}", rec_id, tick_id, source, inst, h, realized, oid, now],
            )
            rets[str(h)] = realized
            n += 1
        if rets:
            matured_by_key[(str(tick_id), inst, str(source))] = rets

    scored = 0
    pairs: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for (tick_id, inst, source), rets in matured_by_key.items():
        pairs.setdefault((tick_id, inst), {})[source] = rets
    for (tick_id, inst), by_src in pairs.items():
        q = by_src.get("quant_only")
        l = by_src.get("llm_final")
        if not q or not l:
            continue
        q_ret = float(sum(q.values()) / len(q))
        l_ret = float(sum(l.values()) / len(l))
        payload = score_overrides(q_ret, l_ret)
        conn.execute(
            """
            INSERT INTO override_score_rows
            (score_id, outcome_snapshot_id, tick_id, instrument_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [f"ov_{uuid4().hex[:16]}", oid, tick_id, inst, json.dumps(payload), now],
        )
        scored += 1
    return {"outcome_snapshot_id": oid, "n_outcomes": n, "n_override_scores": scored}
