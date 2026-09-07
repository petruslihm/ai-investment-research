"""Live watchlist: held positions + top candidates."""

from __future__ import annotations

from datetime import date

import duckdb

from trading_system.config import Settings
from trading_system.ids import InstrumentId
from trading_system.market.registry import symbol_for_instrument


def build_live_watchlist(
    conn: duckdb.DuckDBPyConnection,
    settings: Settings,
    symbol_map: dict[str, InstrumentId],
    *,
    provider: str,
    candidate_scores: dict[InstrumentId, float] | None = None,
    as_of: date | None = None,
) -> list[tuple[InstrumentId, str, int]]:
    """Return (instrument_id, reason, priority) entries capped by max_live_equity_symbols."""
    as_of = as_of or date.today()
    entries: dict[InstrumentId, tuple[str, int]] = {}

    # Manual holdings always included
    held_rows = conn.execute(
        "SELECT DISTINCT instrument_id FROM portfolio_lots"
    ).fetchall()
    for (inst_id,) in held_rows:
        iid = InstrumentId(inst_id)
        sym = symbol_for_instrument(conn, iid, provider=provider, as_of=as_of) or inst_id
        entries[iid] = (f"held:{sym}", 0)

    # Strongest candidates when available (placeholder: scored or SPY-first universe)
    scores = candidate_scores or {}
    if not scores:
        for sym, inst_id in symbol_map.items():
            if sym == settings.btc_symbol:
                continue
            priority = 1 if sym == "SPY" else 10
            if inst_id not in entries:
                entries[inst_id] = (f"candidate:{sym}", priority)
    else:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (inst_id, score) in enumerate(ranked):
            sym = symbol_for_instrument(conn, inst_id, provider=provider, as_of=as_of) or inst_id
            entries[inst_id] = (f"score:{score:.4f}:{sym}", rank + 1)

    # SPY benchmark always present for relative features
    spy_id = symbol_map.get("SPY")
    if spy_id is not None:
        entries[spy_id] = entries.get(spy_id, ("benchmark:SPY", 0))

    ordered = sorted(entries.items(), key=lambda kv: (kv[1][1], kv[0]))
    cap = settings.max_live_equity_symbols
    return [(iid, reason, pri) for iid, (reason, pri) in ordered[:cap]]
