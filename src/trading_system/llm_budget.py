"""Soft daily budget threshold for automatic Gemini + GPT-5.6 Sol calls.

The threshold is checked before a call using estimated usage. Actual usage is
known afterward, so this is not a strict billing ceiling. Manual scans are not
counted toward the threshold.

Never places orders.
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timezone
from typing import Any
from uuid import uuid4

import duckdb

from trading_system.config import Settings

_log = logging.getLogger("trading_system.llm_budget")

AUTO_KIND_PREFIX = "auto:"

# In-process fail-closed latch: if an *automatic* ledger write fails, its cost is
# lost and every later daily_auto_spend_usd() read for that UTC day would otherwise
# under-count real spend, silently weakening the auto-scan threshold. Once set,
# every automatic call for that exact UTC day is blocked (daily_auto_spend_usd
# returns infinity) regardless of what the DB read itself reports. A manual
# (non-toward_daily_cap) write failure must never set this -- manual scans are
# uncapped by design and must not consume/block the automatic budget. Comparing
# against the current UTC day on every read means day rollover needs no explicit
# reset: yesterday's latch simply stops matching today's date.
_auto_ledger_lock = threading.Lock()
_auto_ledger_write_failed_day: date | None = None


def _mark_auto_ledger_write_failed(day: date) -> None:
    global _auto_ledger_write_failed_day
    with _auto_ledger_lock:
        _auto_ledger_write_failed_day = day


def _auto_ledger_write_failed_for(day: date) -> bool:
    with _auto_ledger_lock:
        return _auto_ledger_write_failed_day == day


# Official list prices used only to stop overspend on automatic scans. Not invoices.
_PRICE_PER_M: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (4.0, 20.0),
    "gpt-5.6": (4.0, 20.0),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3-flash": (0.75, 3.75),
}


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def estimate_usd(model: str, prompt_tokens: int | None, completion_tokens: int | None) -> float:
    key = (model or "").strip().lower()
    inp_rate, out_rate = _PRICE_PER_M.get(key, (4.0, 20.0) if "gpt" in key else (0.75, 3.75))
    prompt_n = max(0, int(prompt_tokens or 0))
    completion_n = max(0, int(completion_tokens or 0))
    return (prompt_n / 1_000_000.0) * inp_rate + (completion_n / 1_000_000.0) * out_rate


def _stored_kind(kind: str, *, toward_daily_cap: bool) -> str:
    raw = str(kind or "call")
    if toward_daily_cap and not raw.startswith(AUTO_KIND_PREFIX):
        return f"{AUTO_KIND_PREFIX}{raw}"
    return raw


def record_usage(
    conn: duckdb.DuckDBPyConnection | None,
    *,
    provider: str,
    model: str,
    kind: str,
    ticker: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    toward_daily_cap: bool = False,
) -> float:
    usd = estimate_usd(model, prompt_tokens, completion_tokens)
    stored_kind = _stored_kind(kind, toward_daily_cap=toward_daily_cap)
    is_auto = stored_kind.startswith(AUTO_KIND_PREFIX)
    today = utc_today()
    if conn is None:
        if is_auto:
            _mark_auto_ledger_write_failed(today)
            _log.error(
                "record_usage: no ledger connection for automatic usage on %s; "
                "automatic LLM budget is closed for the rest of the day",
                today,
            )
        return usd
    try:
        conn.execute(
            """
            INSERT INTO llm_cost_ledger (
                entry_id, utc_date, provider, model, kind, ticker,
                prompt_tokens, completion_tokens, estimated_usd, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"cost_{uuid4().hex}",
                today,
                provider,
                model,
                stored_kind,
                ticker,
                prompt_tokens,
                completion_tokens,
                usd,
                datetime.now(timezone.utc),
            ],
        )
    except Exception:  # noqa: BLE001 — cost ledger must never abort a scan
        # Swallowed on purpose (see above), but a lost row silently under-counts every
        # later daily-threshold check for the rest of the day -- must not vanish without a trace.
        _log.warning("record_usage: failed to persist ledger row (provider=%s model=%s)", provider, model, exc_info=True)
        if is_auto:
            # This call's real spend is now unaccounted for. Fail the whole day
            # closed rather than let a silently-undercounted ledger keep authorizing
            # more automatic spend. A manual write failure never reaches here.
            _mark_auto_ledger_write_failed(today)
            _log.error(
                "record_usage: automatic ledger write lost for %s; automatic LLM budget is closed for the rest of the day",
                today,
            )
    return usd


def daily_spend_usd(conn: duckdb.DuckDBPyConnection | None, *, day: date | None = None) -> float:
    if conn is None:
        return 0.0
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(estimated_usd), 0) FROM llm_cost_ledger WHERE utc_date = ?",
            [day or utc_today()],
        ).fetchone()
    except Exception:  # noqa: BLE001
        return 0.0
    return float(row[0]) if row else 0.0


def daily_auto_spend_usd(conn: duckdb.DuckDBPyConnection | None, *, day: date | None = None) -> float:
    query_day = day or utc_today()
    if _auto_ledger_write_failed_for(query_day):
        # A prior automatic write for this exact UTC day was lost -- the ledger sum
        # below can no longer be trusted to reflect real spend, so every automatic
        # call is blocked regardless of what the DB currently reports.
        return float("inf")
    if conn is None:
        # Automatic calls must not be authorized when their usage cannot be
        # accounted for. Infinity makes callers fail closed without aborting the
        # rest of a scan.
        return float("inf")
    try:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(estimated_usd), 0) FROM llm_cost_ledger
            WHERE utc_date = ? AND kind LIKE ?
            """,
            [query_day, f"{AUTO_KIND_PREFIX}%"],
        ).fetchone()
    except Exception:  # noqa: BLE001
        _log.error(
            "daily_auto_spend_usd: ledger unavailable; automatic LLM budget is closed",
            exc_info=True,
        )
        return float("inf")
    return float(row[0]) if row else 0.0


def budget_remaining(conn: duckdb.DuckDBPyConnection | None, settings: Settings) -> float:
    cap = float(settings.llm_daily_budget_usd or 0.0)
    if cap <= 0:
        return 0.0
    return max(0.0, cap - daily_auto_spend_usd(conn))


def would_exceed_budget(
    conn: duckdb.DuckDBPyConnection | None,
    settings: Settings,
    *,
    model: str,
    prompt_tokens: int = 8_000,
    completion_tokens: int = 2_000,
    extra_usd: float = 0.0,
) -> bool:
    """True when the next automatic call would pass the daily soft threshold.

    This is a pre-call estimate, not a strict upper bound on the final invoice.

    extra_usd covers spend already incurred by the in-flight run but not yet
    written to the ledger -- a multi-turn judgement only records usage once, at
    the end, so without this every turn of the same conversation would see the
    same (stale) daily total and the threshold could be exceeded several times over
    inside one run. Measured 2026-09-04: one 9-turn judgement was 2.13M prompt
    tokens, while the old single pre-flight estimate assumed 80K.
    """
    cap = float(settings.llm_daily_budget_usd or 0.0)
    if cap <= 0:
        return True
    projected = (
        daily_auto_spend_usd(conn)
        + max(0.0, float(extra_usd))
        + estimate_usd(model, prompt_tokens, completion_tokens)
    )
    return projected > cap + 1e-9


def usage_from_payload(payload: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not isinstance(payload, dict):
        return None, None
    exchange = payload.get("_exchange") if isinstance(payload.get("_exchange"), dict) else payload
    prompt = exchange.get("prompt_tokens")
    completion = exchange.get("completion_tokens")
    try:
        prompt_n = int(prompt) if prompt is not None else None
    except (TypeError, ValueError):
        prompt_n = None
    try:
        completion_n = int(completion) if completion is not None else None
    except (TypeError, ValueError):
        completion_n = None
    return prompt_n, completion_n
