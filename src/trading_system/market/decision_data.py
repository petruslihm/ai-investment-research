"""Frozen price provenance and completed-daily input identity. No provider calls."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime


DAILY_BASIS = "completed_daily_bars_only_v1"


def content_id(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def daily_input_fingerprints(conn, *, equity_end: date, btc_end: date) -> dict[str, str]:
    """Fingerprint values/revisions, not download timestamps; identical re-fetches are cheap."""
    out = {}
    for asset, table, end in [("us_equity", "equity_daily_bars", equity_end), ("btc", "btc_daily_bars", btc_end)]:
        ident = "instrument_id," if asset == "us_equity" else ""
        row = conn.execute(
            f"SELECT count(*), bit_xor(hash({ident} session_date, open, high, low, close, volume, "
            f"adjustment_revision, provider)) FROM {table} WHERE finality='final' AND session_date<=?", [end]
        ).fetchone()
        out[asset] = content_id([DAILY_BASIS, str(end), row])
    return out


def latest_price_snapshot(conn, instrument_id: str, *, as_of: datetime) -> dict:
    """A partial daily close is an observation, never a timestamped live quote.

    Providers in the existing daily table do not retain the last trade's event time.
    Keep that unknown instead of relabeling receipt time as trade time.
    """
    btc = "btc" in instrument_id.lower()
    table = "btc_daily_bars" if btc else "equity_daily_bars"
    clause = "" if btc else "AND instrument_id=?"
    params = [as_of] if btc else [as_of, instrument_id]
    row = conn.execute(
        f"SELECT close, session_date, finality, provider, receive_ts, adjustment_revision FROM {table} "
        f"WHERE receive_ts<=? {clause} ORDER BY session_date DESC, receive_ts DESC LIMIT 1", params
    ).fetchone()
    if not row:
        return {"kind": "missing", "price": None, "input_as_of": as_of.isoformat(), "provider": None}
    price, session, finality, provider, received, revision = row
    if not math.isfinite(float(price)) or float(price) <= 0:
        return {"kind": "missing", "price": None, "input_as_of": as_of.isoformat(), "provider": provider,
                "reason": "INVALID_PRICE", "received_at": received.isoformat()}
    return {
        "price": float(price), "session_date": session.isoformat(),
        "kind": "final_close" if finality == "final" else "partial_daily_bar",
        "price_at": None, "received_at": received.isoformat(), "provider": provider,
        "adjustment_revision": revision, "input_as_of": as_of.isoformat(),
        "daily_feature_basis": DAILY_BASIS,
    }


def price_description(snapshot: dict | None) -> str:
    if not snapshot or snapshot.get("price") is None:
        return "가격 미확보 · 기준 시각/출처 확인 필요"
    kind = "확정 일봉 종가" if snapshot.get("kind") == "final_close" else "장중 미완성 일봉 관측"
    return (
        f"{kind} {snapshot['price']} · 세션 {snapshot.get('session_date')} · "
        f"출처 {snapshot.get('provider')} · 수집 {snapshot.get('received_at')} · "
        "정확한 체결 시각 미제공"
    )
