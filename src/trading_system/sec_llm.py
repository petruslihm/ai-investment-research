"""SEC filing passages + LLM desk (Gemini research, GPT-5.6 Sol judge). Never executes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import duckdb

from trading_system.config import Settings
from trading_system.filing_chunks import FILING_BODY_CHARS, select_filing_passages
from trading_system.llm_client import (
    EXTRACT_PROMPT_VERSION,
    STATUS_AVAILABLE,
    STATUS_DEGRADED,
    STATUS_NOT_CONFIGURED,
    STATUS_UNAVAILABLE,
    openai_configured,
)
from trading_system.recommendations import RecommendationAction, RecommendationRecord, llm_final_record
from trading_system.sec_edgar import fetch_filing_text, last_sec_error, lookup_cik, recent_filings

# Caller should already pass buy-candidate + holding tickers. Cap is a safety rail.
MAX_SEC_TICKERS = 25
SMOKE_SEC_TICKERS = MAX_SEC_TICKERS
SMOKE_FILINGS_PER_TICKER = 1


def cache_extract(
    conn: duckdb.DuckDBPyConnection,
    *,
    accession: str,
    accepted_at: datetime | None,
    payload: dict,
    provider_model: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO sec_extracts
        (accession, provider_model, prompt_version, accepted_at, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            accession,
            provider_model,
            EXTRACT_PROMPT_VERSION,
            accepted_at,
            json.dumps(payload),
        ],
    )


def _cached_extracts(conn: duckdb.DuckDBPyConnection, ticker: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT payload_json FROM sec_extracts ORDER BY accepted_at DESC"
    ).fetchall()
    out: list[dict[str, Any]] = []
    needle = ticker.upper()
    for (raw,) in rows:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("ticker", "")).upper() != needle:
            continue
        payload["source"] = "lkg"
        out.append(payload)
        if len(out) >= SMOKE_FILINGS_PER_TICKER:
            break
    return out


def ingest_sec_extracts(
    conn: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    tickers: list[str],
    tick_id: str | None = None,
) -> dict[str, Any]:
    """Fetch real EDGAR 8-K/10-Q metadata and section-aware passages. No OpenAI mini extract."""
    conn.execute("DELETE FROM sec_extracts WHERE accession = 'stub-000' OR provider_model = 'stub'")
    equities = [t for t in tickers if t.upper() not in {"BTC/USD", "BTCUSD", "SPY"}][:MAX_SEC_TICKERS]
    if not equities:
        equities = [t for t in tickers if t.upper() not in {"BTC/USD", "BTCUSD"}][:MAX_SEC_TICKERS]
    configured = openai_configured(settings)
    provider_model = "filing_passages_v1"
    stored: list[dict[str, Any]] = []
    errors: list[str] = []
    used_lkg = False

    for ticker in equities:
        cik = lookup_cik(ticker)
        if not cik:
            errors.append(f"{ticker}: CIK UNAVAILABLE")
            cached = _cached_extracts(conn, ticker)
            if cached:
                used_lkg = True
                stored.extend(cached)
            continue
        filings = recent_filings(cik, limit=SMOKE_FILINGS_PER_TICKER)
        if not filings:
            cached = _cached_extracts(conn, ticker)
            if cached:
                used_lkg = True
                stored.extend(cached)
                errors.append(f"{ticker}: live EDGAR UNAVAILABLE; using LKG metadata")
            else:
                errors.append(f"{ticker}: no recent 8-K/10-Q")
            continue
        for filing in filings:
            if filing.get("source") == "lkg":
                used_lkg = True
            payload: dict[str, Any] = {
                "status": STATUS_UNAVAILABLE,
                "ticker": ticker,
                "cik": cik,
                "form": filing["form"],
                "accession": filing["accession"],
                "accepted_at": filing["accepted_at"].isoformat()
                if hasattr(filing["accepted_at"], "isoformat")
                else str(filing["accepted_at"]),
                "note": None,
                "source": filing.get("source") or "live",
                "passages": [],
            }
            if filing.get("source") == "lkg":
                cached = _cached_extracts(conn, ticker)
                match = next((p for p in cached if p.get("accession") == filing["accession"]), None)
                if match:
                    match["source"] = "lkg"
                    stored.append(match)
                    continue
                payload["note"] = "LKG filing metadata; live EDGAR UNAVAILABLE. Extract not invented."
                payload["status"] = STATUS_DEGRADED
            elif filing.get("source") != "lkg":
                text = fetch_filing_text(
                    cik,
                    filing["accession"],
                    primary_document=filing.get("primary_document"),
                    max_chars=FILING_BODY_CHARS,
                )
                if not text:
                    payload["status"] = STATUS_UNAVAILABLE
                    payload["note"] = "Filing body UNAVAILABLE"
                    provider_model = "unavailable"
                else:
                    passages = select_filing_passages(text, ticker=ticker)
                    payload["passages"] = passages
                    payload["status"] = STATUS_AVAILABLE if passages else STATUS_UNAVAILABLE
                    payload["note"] = (
                        f"Section-aware passages n={len(passages)}"
                        if passages
                        else "Filing body present but no usable passages"
                    )
            cache_extract(
                conn,
                accession=filing["accession"],
                accepted_at=filing["accepted_at"] if hasattr(filing["accepted_at"], "isoformat") else None,
                payload=payload,
                provider_model=provider_model if filing.get("source") != "lkg" else "lkg",
            )
            stored.append(payload)

    if not stored and not errors:
        errors.append("SEC ingest produced no filings")
    sec_err = last_sec_error()
    if sec_err and sec_err not in errors:
        errors.append(sec_err)

    live_ok = stored and any(p.get("source") != "lkg" for p in stored)
    if stored and all(p.get("status") == STATUS_NOT_CONFIGURED for p in stored):
        sec_status = STATUS_DEGRADED if used_lkg and not live_ok else STATUS_AVAILABLE
        extract_status = STATUS_NOT_CONFIGURED
    elif stored and any(p.get("status") == STATUS_AVAILABLE for p in stored):
        sec_status = STATUS_DEGRADED if used_lkg else STATUS_AVAILABLE
        extract_status = STATUS_AVAILABLE
    elif stored:
        sec_status = STATUS_DEGRADED if used_lkg else STATUS_AVAILABLE
        extract_status = STATUS_UNAVAILABLE
    else:
        sec_status = STATUS_UNAVAILABLE
        extract_status = STATUS_NOT_CONFIGURED if not configured else STATUS_UNAVAILABLE
    if used_lkg and stored and sec_status == STATUS_UNAVAILABLE:
        sec_status = STATUS_DEGRADED

    return {
        "sec_status": sec_status,
        "extract_status": extract_status,
        "llm_configured": configured,
        "filings": stored,
        "errors": errors,
        "used_lkg": used_lkg,
    }


def skipped_llm_record(quant: RecommendationRecord, *, quant_status: str) -> RecommendationRecord:
    """No OpenAI call. Quant already produced this name; it is not a portfolio candidate.

    Kept as a record-shaping helper (no API call) so llm_log.py's history reconstruction
    can render a "skipped" tick the same way whether or not a per-name judge ever ran.
    """
    rec = llm_final_record(
        tick_id=quant.tick_id,
        decision_epoch_id=quant.decision_epoch_id,
        feature_snapshot_id=quant.feature_snapshot_id,
        instrument_id=quant.instrument_id,
        action=RecommendationAction.NO_ACTION,
        override_of_recommendation_id=quant.recommendation_id,
        override_reasons=["NOT_A_CANDIDATE"],
        thesis=(
            "LLM 판단을 건너뛰었습니다. 이 종목은 이번 비중/보유 후보가 아닙니다. "
            f"Quant 결과는 {quant_status}로 따로 유지됩니다."
        ),
        confidence=quant.confidence,
        horizons=quant.horizons,
        current_units=quant.current_units,
        recommended_units=quant.recommended_units,
        delta_units=quant.delta_units,
        actionable=False,
    )
    rec.llm_tainted = False
    rec.contrary_evidence = "llm_judge_status=NOT_A_CANDIDATE"
    return rec


def evidence_for_instrument(filings: list[dict[str, Any]], instrument_id: str) -> dict[str, Any] | None:
    needle = instrument_id.removeprefix("inst_").upper().replace("_", "")
    for row in filings:
        ticker = str(row.get("ticker", "")).upper().replace("/", "").replace(".", "")
        if ticker and (ticker in needle or needle.endswith(ticker)):
            return row
    return None


def score_overrides(quant_return: float, llm_return: float) -> dict:
    """Whether overrides added 5/10/20 value; used to reduce reliance."""
    delta = llm_return - quant_return
    rely = 1.0 if delta > 0 else 0.4
    return {
        "quant_return": quant_return,
        "llm_return": llm_return,
        "override_value": delta,
        "llm_reliance": rely,
        "scored_as": "live_or_oos_matured_not_backtest",
    }
