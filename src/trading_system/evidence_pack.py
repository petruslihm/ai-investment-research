"""Evidence Pack: Gemini research artifact passed to the Sol judge. Not a sentiment stub."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import duckdb


def clamp_score_0_100(value: object) -> float | None:
    """Coerce a research pack's self-reported 0-100 score field. None (not 0, not 50)
    when the field is missing/non-numeric -- callers (research_sizing_multiplier) must
    be able to tell "the model didn't answer" apart from "the model said 0"."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return max(0.0, min(100.0, v))


def empty_pack(
    ticker: str,
    *,
    note: str | None = None,
    open_questions: list[str] | None = None,
) -> dict[str, Any]:
    questions = list(open_questions or [])
    if note:
        questions.append(note)
    return {
        "ticker": ticker.upper(),
        "as_of": datetime.now(timezone.utc).date().isoformat(),
        "material_change": True,
        "web_search_used": False,
        "rerating_score": None,
        "valuation_support_score": None,
        "cash_relative_score": None,
        "data_quality_score": None,
        "bearish_severity_score": None,
        "filings": [],
        "news_events": [],
        "industry": {"sector": "", "competitors": [], "notes": ""},
        "macro": {"regime": "", "notes": ""},
        "price_events": [],
        "portfolio_context": {},
        "supporting_evidence": [],
        "contrary_evidence": [],
        "adversarial_review": {"main_objections": [], "what_would_change_the_call": ""},
        "sources": [],
        "open_questions": questions,
        "data_quality": note or "incomplete",
        "summary_ko": "",
        "status": "UNAVAILABLE",
        "researched_at": None,
        "freshness": {},
        "freshness_state": "unknown",
        "grounding_urls": [],
    }


def _as_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}):
        return []
    return [value]


def _claim_rows(value: object) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in _as_list(value):
        if isinstance(item, str) and item.strip():
            rows.append({"claim": item.strip(), "source_url": ""})
            continue
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or item.get("text") or item.get("summary") or "").strip()
        url = str(item.get("source_url") or item.get("url") or "").strip()
        if claim:
            rows.append({"claim": claim, "source_url": url})
    return rows


def normalize_pack(raw: object, *, ticker: str) -> dict[str, Any]:
    base = empty_pack(ticker)
    if not isinstance(raw, dict):
        base["open_questions"].append("Research model did not return a JSON object.")
        return base
    out = dict(base)
    out["ticker"] = str(raw.get("ticker") or ticker).upper()
    out["as_of"] = str(raw.get("as_of") or base["as_of"])
    out["material_change"] = bool(raw.get("material_change", True))
    out["web_search_used"] = bool(raw.get("web_search_used", False))
    out["summary_ko"] = str(raw.get("summary_ko") or "").strip()
    out["filings"] = _as_list(raw.get("filings"))
    out["news_events"] = _as_list(raw.get("news_events"))
    industry = raw.get("industry") if isinstance(raw.get("industry"), dict) else {}
    out["industry"] = {
        "sector": str(industry.get("sector") or ""),
        "competitors": [str(x) for x in _as_list(industry.get("competitors"))],
        "notes": str(industry.get("notes") or ""),
    }
    macro = raw.get("macro") if isinstance(raw.get("macro"), dict) else {}
    out["macro"] = {
        "regime": str(macro.get("regime") or ""),
        "notes": str(macro.get("notes") or ""),
    }
    out["price_events"] = _as_list(raw.get("price_events"))
    out["portfolio_context"] = raw.get("portfolio_context") if isinstance(raw.get("portfolio_context"), dict) else {}
    out["supporting_evidence"] = _claim_rows(raw.get("supporting_evidence"))
    out["contrary_evidence"] = _claim_rows(raw.get("contrary_evidence"))
    adv = raw.get("adversarial_review") if isinstance(raw.get("adversarial_review"), dict) else {}
    out["adversarial_review"] = {
        "main_objections": [str(x) for x in _as_list(adv.get("main_objections"))],
        "what_would_change_the_call": str(adv.get("what_would_change_the_call") or ""),
    }
    sources: list[dict[str, str]] = []
    for item in _as_list(raw.get("sources")):
        if isinstance(item, str) and item.strip():
            sources.append({"title": "", "url": item.strip()})
        elif isinstance(item, dict):
            url = str(item.get("url") or item.get("uri") or "").strip()
            title = str(item.get("title") or "").strip()
            if url or title:
                sources.append({"title": title, "url": url})
    out["sources"] = sources
    out["open_questions"] = [str(x) for x in _as_list(raw.get("open_questions")) if str(x).strip()]
    out["data_quality"] = str(raw.get("data_quality") or base["data_quality"])
    out["rerating_score"] = clamp_score_0_100(raw.get("rerating_score"))
    out["valuation_support_score"] = clamp_score_0_100(raw.get("valuation_support_score"))
    out["cash_relative_score"] = clamp_score_0_100(raw.get("cash_relative_score"))
    out["data_quality_score"] = clamp_score_0_100(raw.get("data_quality_score"))
    out["bearish_severity_score"] = clamp_score_0_100(raw.get("bearish_severity_score"))
    out["status"] = str(raw.get("status") or "AVAILABLE")
    out["researched_at"] = raw.get("researched_at")
    out["freshness"] = raw.get("freshness") if isinstance(raw.get("freshness"), dict) else {}
    out["freshness_state"] = str(raw.get("freshness_state") or "unknown")
    grounding = []
    for item in _as_list(raw.get("grounding_urls")):
        if isinstance(item, str) and item.strip():
            grounding.append({"title": "", "url": item.strip()})
        elif isinstance(item, dict):
            url = str(item.get("url") or item.get("uri") or "").strip()
            title = str(item.get("title") or "").strip()
            if url or title:
                grounding.append({"title": title, "url": url})
    out["grounding_urls"] = grounding
    skip = {"_exchange", "prompt_tokens", "completion_tokens", "total_tokens", "status"}
    for key, value in raw.items():
        if key not in out and key not in skip:
            out[key] = value
    return out


RESEARCH_TTL_MINUTES = 45
MATERIAL_PRICE_MOVE = 0.03


def cache_key(*, ticker: str, accession: str, quant_action: str, units_bucket: str) -> str:
    return "|".join(
        [
            ticker.upper(),
            accession or "no-accession",
            (quant_action or "").upper() or "NONE",
            units_bucket,
        ]
    )


def units_bucket(value: object) -> str:
    try:
        n = float(value or 0)
    except (TypeError, ValueError):
        return "0"
    if abs(n) < 1e-9:
        return "0"
    return str(int(round(n)))


def _parse_ts(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def pack_is_fresh(
    pack: dict[str, Any],
    *,
    now: datetime | None = None,
    last_price: float | None = None,
    portfolio_sig: str | None = None,
    regime: str | None = None,
    created_at: object | None = None,
    ttl_minutes: int = RESEARCH_TTL_MINUTES,
) -> bool:
    """TTL plus early invalidation on price, portfolio, or regime change."""
    stamp = _parse_ts(pack.get("researched_at")) or _parse_ts(created_at)
    current = now or datetime.now(timezone.utc)
    if stamp is None:
        return False
    if current - stamp > timedelta(minutes=max(1, int(ttl_minutes))):
        return False
    freshness = pack.get("freshness") if isinstance(pack.get("freshness"), dict) else {}
    cached_price = freshness.get("price")
    if last_price is not None and cached_price not in (None, ""):
        try:
            prev = float(cached_price)
            cur = float(last_price)
        except (TypeError, ValueError):
            prev = 0.0
            cur = 0.0
        if prev > 0 and cur > 0 and abs(cur / prev - 1.0) > MATERIAL_PRICE_MOVE:
            return False
    cached_sig = str(freshness.get("portfolio_sig") or "")
    if portfolio_sig and cached_sig and portfolio_sig != cached_sig:
        return False
    cached_regime = str(freshness.get("regime") or "")
    if regime and cached_regime and regime != cached_regime:
        return False
    return True


def load_cached_pack(
    conn: duckdb.DuckDBPyConnection | None,
    key: str,
    *,
    now: datetime | None = None,
    last_price: float | None = None,
    portfolio_sig: str | None = None,
    regime: str | None = None,
) -> dict[str, Any] | None:
    if conn is None or not key:
        return None
    try:
        row = conn.execute(
            "SELECT payload_json, created_at FROM research_evidence_packs WHERE cache_key = ? LIMIT 1",
            [key],
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    if not row or not row[0]:
        return None
    try:
        payload = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not pack_is_fresh(
        payload,
        now=now,
        last_price=last_price,
        portfolio_sig=portfolio_sig,
        regime=regime,
        created_at=row[1],
    ):
        return None
    payload["cached"] = True
    payload["freshness_state"] = "cached"
    return payload


def persist_pack(
    conn: duckdb.DuckDBPyConnection | None,
    pack: dict[str, Any],
    *,
    cache_key_value: str,
    ticker: str,
    instrument_id: str | None = None,
    tick_id: str | None = None,
    quant_action: str | None = None,
    accession: str | None = None,
) -> None:
    if conn is None:
        return
    now = datetime.now(timezone.utc)
    day = now.date()
    try:
        conn.execute("DELETE FROM research_evidence_packs WHERE cache_key = ?", [cache_key_value])
        conn.execute(
            """
            INSERT INTO research_evidence_packs (
                pack_id, ticker, utc_date, cache_key, instrument_id, tick_id,
                quant_action, accession, material_change, web_search_used,
                payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"pack_{uuid4().hex}",
                ticker.upper(),
                day,
                cache_key_value,
                instrument_id,
                tick_id,
                quant_action,
                accession,
                bool(pack.get("material_change", True)),
                bool(pack.get("web_search_used", False)),
                json.dumps(pack, ensure_ascii=False),
                now,
            ],
        )
    except Exception:  # noqa: BLE001
        pass


def format_contrary(pack: dict[str, Any] | None) -> str:
    if not isinstance(pack, dict):
        return ""
    rows = pack.get("contrary_evidence") or []
    parts: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            claim = str(row.get("claim") or "").strip()
            url = str(row.get("source_url") or "").strip()
            if claim and url:
                parts.append(f"{claim} ({url})")
            elif claim:
                parts.append(claim)
        elif str(row).strip():
            parts.append(str(row).strip())
    adv = pack.get("adversarial_review") if isinstance(pack.get("adversarial_review"), dict) else {}
    for obj in adv.get("main_objections") or []:
        if str(obj).strip():
            parts.append(str(obj).strip())
    return " | ".join(parts)
