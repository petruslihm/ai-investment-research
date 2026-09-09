"""Gemini research (two passes) + shortlist for GPT-5.6 Sol. No broker orders."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from trading_system.config import Settings
from trading_system.evidence_pack import (
    RESEARCH_TTL_MINUTES,
    cache_key,
    clamp_score_0_100,
    empty_pack,
    load_cached_pack,
    normalize_pack,
    persist_pack,
    units_bucket,
)
from trading_system.filing_chunks import passages_as_text
from trading_system.gemini_client import generate_json, gemini_configured, research_model
from trading_system.judge_package import (
    compact_allocation_trace,
    compact_evidence_pack,
    dumps_complete,
    verify_pack_sources,
)
from trading_system.llm_budget import record_usage, usage_from_payload, would_exceed_budget
from trading_system.llm_client import (
    STATUS_AVAILABLE,
    STATUS_DEGRADED,
    STATUS_NOT_CONFIGURED,
    STATUS_QUOTA_EXCEEDED,
    STATUS_RATE_LIMITED,
    STATUS_UNAVAILABLE,
)
from trading_system.llm_log import persist_exchange, ticker_from_instrument
from trading_system.recommendations import RecommendationRecord

RESEARCH_PROMPT_VERSION = "research_legacy_b_v4"
ADVERSARIAL_PROMPT_VERSION = "research_adversarial_legacy_b_v4"

RESEARCH_SYSTEM = (
    "You are the evidence investigator for a personal US-stock + BTC portfolio.\n\n"
    "[ANALYSIS PURPOSE]\n"
    "Follow the legacy-b rerating-candidate format. Look for a stock that can keep rerating even after a "
    "large rise because estimates, an industry bottleneck, a durable theme, or institutional reassessment "
    "continue to improve. Do not confuse 'cheap now' with 'expensive but able to rerate further'. Quant "
    "screens a large universe but is only background evidence, not a direction forecast or final verdict.\n\n"
    "[Q1] Verify the latest earnings date, EPS/revenue beat or miss, post-earnings consensus revisions, "
    "guidance changes, and whether the move is genuine rerating or simple momentum.\n"
    "[Q2] Test whether next-12-month EPS/revenue growth supports the price move. Compare relevant forward "
    "valuation with the company's history and peers and separate priced-in expectations from revision headroom.\n"
    "[Q3] Judge whether the TAM/cycle catalyst lasts one quarter or one to two years; identify early, middle, "
    "late, or unknown stage and compare with a relevant prior cycle when possible.\n"
    "[Q4] Compare peers and the sector. State global competitive position, approximate peer rank, margin quality, "
    "and whether the move is broadly confirmed or isolated. Flag a temporary theme, liquidity/manipulation risk, "
    "or a defensible company-specific narrative.\n"
    "[Q5] Check the most recent one-to-three months of target-price changes, new coverage, estimate revisions, "
    "and credible institutional reassessment. Distinguish early catch-up from already-completed repricing.\n\n"
    "Also investigate the latest supplied or searchable macro/rate/liquidity/geopolitical/theme-rotation issues. "
    "Keep three conclusions separate: rerating potential, present undervaluation, and attractiveness versus CASH. "
    "For a held name discuss add/hold/sell evidence; for a new name discuss buy/watch evidence. You are not the "
    "final decision-maker and must not place orders. Do not invent URLs, filings, prices, headlines, flows, "
    "targets, or estimates. If a fact cannot be verified, say '확인 불가' and add it to open_questions. "
    "Return only the required structured JSON. All narrative fields must be concise Korean."
)

ADVERSARIAL_SYSTEM = (
    "You are the legacy-b-style adversarial reviewer of an investment evidence pack. "
    "Actively test whether apparent rerating is merely late momentum, whether earnings revisions lag the price, "
    "whether the cycle is near a peak, whether peers fail to confirm, whether targets already price in the upside, "
    "and whether CASH is superior after macro and event risk. For a held name, find evidence that supports selling "
    "rather than automatically retaining it; for a new name, find what makes buying now a mistake. Do not invent "
    "facts or URLs. Return only the required structured JSON, with Korean narrative fields."
)

_CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"claim": {"type": "string"}, "source_url": {"type": "string"}},
    "required": ["claim", "source_url"],
}
_SOURCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"title": {"type": "string"}, "url": {"type": "string"}},
    "required": ["title", "url"],
}
RESEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ticker": {"type": "string"},
        "material_change": {"type": "boolean"},
        "summary_ko": {"type": "string"},
        "q1_consensus_revision": {"type": "string"},
        "q2_earnings_support": {"type": "string"},
        "q3_cycle_duration_and_stage": {"type": "string"},
        "q4_peer_confirmation": {"type": "string"},
        "q5_institutional_repricing": {"type": "string"},
        "competitive_position_and_rank": {"type": "string"},
        "margin_quality": {"type": "string"},
        "rerating_potential": {"type": "string"},
        "present_undervaluation": {"type": "string"},
        "cash_relative_attractiveness": {"type": "string"},
        "latest_market_issues": {"type": "array", "items": {"type": "string"}},
        "news_events": {"type": "array", "items": _CLAIM_SCHEMA},
        "industry": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "sector": {"type": "string"},
                "competitors": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
            "required": ["sector", "competitors", "notes"],
        },
        "macro": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"regime": {"type": "string"}, "notes": {"type": "string"}},
            "required": ["regime", "notes"],
        },
        "price_events": {"type": "array", "items": _CLAIM_SCHEMA},
        "portfolio_context": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "position_status": {"type": "string", "enum": ["NEW", "HELD"]},
                "implication": {"type": "string"},
            },
            "required": ["position_status", "implication"],
        },
        "supporting_evidence": {"type": "array", "items": _CLAIM_SCHEMA},
        "contrary_evidence": {"type": "array", "items": _CLAIM_SCHEMA},
        "sources": {"type": "array", "items": _SOURCE_SCHEMA},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "data_quality": {"type": "string"},
        "rerating_score": {
            "type": "number",
            "description": "0-100 numeric version of rerating_potential. 50 = no clear signal either way.",
        },
        "valuation_support_score": {
            "type": "number",
            "description": "0-100 numeric version of present_undervaluation. 50 = fairly valued / unclear.",
        },
        "cash_relative_score": {
            "type": "number",
            "description": "0-100 numeric version of cash_relative_attractiveness. 50 = no edge over CASH.",
        },
        "data_quality_score": {
            "type": "number",
            "description": (
                "0-100 self-rated confidence in this pack, considering how much was verifiable via "
                "live search vs '확인 불가'. Score honestly low when key facts could not be confirmed -- "
                "this directly gates whether the pack is allowed to move real position sizing."
            ),
        },
    },
    "required": [
        "ticker",
        "material_change",
        "summary_ko",
        "q1_consensus_revision",
        "q2_earnings_support",
        "q3_cycle_duration_and_stage",
        "q4_peer_confirmation",
        "q5_institutional_repricing",
        "competitive_position_and_rank",
        "margin_quality",
        "rerating_potential",
        "present_undervaluation",
        "cash_relative_attractiveness",
        "latest_market_issues",
        "news_events",
        "industry",
        "macro",
        "price_events",
        "portfolio_context",
        "supporting_evidence",
        "contrary_evidence",
        "sources",
        "open_questions",
        "data_quality",
        "rerating_score",
        "valuation_support_score",
        "cash_relative_score",
        "data_quality_score",
    ],
}
ADVERSARIAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary_ko": {"type": "string"},
        "main_objections": {"type": "array", "items": {"type": "string"}},
        "contrary_evidence": {"type": "array", "items": _CLAIM_SCHEMA},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "what_would_change_the_call": {"type": "string"},
        "data_quality": {"type": "string"},
        "bearish_severity_score": {
            "type": "number",
            "description": (
                "0-100: how strong are your objections? 0 = found nothing that changes the bull case, "
                "100 = objections are severe enough to override it. This discounts the research pack's "
                "bullish scores before they can move real position sizing."
            ),
        },
    },
    "required": [
        "summary_ko",
        "main_objections",
        "contrary_evidence",
        "missing_evidence",
        "what_would_change_the_call",
        "data_quality",
        "bearish_severity_score",
    ],
}


def _quant_slice(quant: RecommendationRecord) -> dict[str, Any]:
    return {
        "instrument_id": str(quant.instrument_id),
        "action": str(getattr(quant.action, "value", quant.action)),
        "current_units": quant.current_units,
        "recommended_units": quant.recommended_units,
        "delta_units": quant.delta_units,
        "acquisition_units": getattr(quant, "acquisition_units", None),
        "marked_units_final": getattr(quant, "marked_units_final", None),
        "confidence": quant.confidence,
        "thesis": quant.thesis,
        "horizons": [
            {
                "horizon": h.horizon,
                "expected_return": h.expected_return,
                "rank_score": h.rank_score,
                "confidence": h.confidence,
            }
            for h in (quant.horizons or [])
        ],
    }


def _compact_research_portfolio(portfolio: dict[str, Any] | None) -> dict[str, Any]:
    """Keep decision context while removing the hundreds-row allocation dump.

    The full trace was previously repeated in every Gemini call, obscuring the
    legacy-b questions and needlessly inflating prompts.
    """
    if not isinstance(portfolio, dict) or not portfolio:
        return {}
    holdings: list[dict[str, Any]] = []
    for row in list(portfolio.get("holdings") or [])[:25]:
        if not isinstance(row, dict):
            continue
        holdings.append(
            {
                key: row.get(key)
                for key in (
                    "instrument_id",
                    "acquisition_units",
                    "marked_units_final",
                    "quant_recommended_units",
                    "weight",
                )
                if row.get(key) is not None
            }
        )
    trace = portfolio.get("allocation_trace")
    return {
        "total_base_units": portfolio.get("total_base_units"),
        "cash_units": portfolio.get("cash_units"),
        "cash_floor_weight": portfolio.get("cash_floor_weight"),
        "holdings": holdings,
        "unpriced_holdings": list(portfolio.get("unpriced_holdings") or [])[:25],
        "concentration": portfolio.get("concentration") or {},
        "btc_sleeve": portfolio.get("btc_sleeve") or {},
        "constraints": portfolio.get("constraints") or {},
        "units_basis": portfolio.get("units_basis"),
        "allocation_trace": compact_allocation_trace(trace if isinstance(trace, dict) else None),
    }


def select_final_judge_recs(
    recs: list[RecommendationRecord],
    *,
    max_names: int,
    always_include: set[str] | None = None,
) -> list[str]:
    """Holdings first (never dropped), then equity buys up to max_names.

    BTC is appended separately so a large BTC sleeve cannot crowd out equity names.
    Order is stable so the scan UI can show 3/12 with the current ticker.
    """
    always = {str(x) for x in (always_include or set())}
    holdings: list[RecommendationRecord] = []
    buys: list[RecommendationRecord] = []
    btc: list[RecommendationRecord] = []
    for rec in recs:
        inst = str(rec.instrument_id)
        if "btc" in inst.lower():
            btc.append(rec)
            continue
        held = (
            inst in always
            or float(rec.current_units or 0) > 1e-12
            or float(getattr(rec, "acquisition_units", None) or 0) > 1e-12
        )
        if held:
            holdings.append(rec)
            continue
        action = str(getattr(rec.action, "value", rec.action))
        if action in {"BUY", "ENTER", "ADD"} and float(rec.recommended_units or 0) > 1e-12:
            buys.append(rec)
    buys.sort(key=lambda r: float(r.recommended_units or 0), reverse=True)
    chosen: list[str] = []
    seen: set[str] = set()
    for rec in holdings:
        inst = str(rec.instrument_id)
        if inst in seen:
            continue
        seen.add(inst)
        chosen.append(inst)
    extra = 0
    for rec in buys:
        inst = str(rec.instrument_id)
        if inst in seen:
            continue
        if extra >= max_names:
            break
        seen.add(inst)
        chosen.append(inst)
        extra += 1
    for rec in btc:
        inst = str(rec.instrument_id)
        if inst in seen:
            continue
        seen.add(inst)
        chosen.append(inst)
    return chosen


def _local_pack(
    ticker: str,
    *,
    filings: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    note: str,
    status: str = STATUS_NOT_CONFIGURED,
) -> dict[str, Any]:
    pack = empty_pack(ticker, note=note)
    pack["filings"] = filings
    if passages:
        pack["filings"] = [
            {
                **(filings[0] if filings else {"ticker": ticker}),
                "passages": passages,
            }
        ]
    pack["status"] = status
    pack["data_quality"] = note
    pack["material_change"] = bool(passages or filings)
    return pack


_FAIL_RESEARCH = {
    STATUS_NOT_CONFIGURED,
    STATUS_UNAVAILABLE,
    STATUS_QUOTA_EXCEEDED,
    STATUS_RATE_LIMITED,
}


def _stamp_freshness(
    pack: dict[str, Any],
    *,
    last_price: float | None,
    portfolio_sig: str | None,
    regime: str | None,
    state: str = "fresh",
) -> dict[str, Any]:
    pack["researched_at"] = datetime.now(timezone.utc).isoformat()
    pack["freshness"] = {
        "ttl_minutes": RESEARCH_TTL_MINUTES,
        "price": last_price,
        "portfolio_sig": portfolio_sig or "",
        "regime": regime or "",
    }
    pack["freshness_state"] = state
    return pack


def _merge_sources(pack: dict[str, Any], incoming: dict[str, Any] | None) -> None:
    extra = []
    if isinstance(incoming, dict):
        extra.extend(incoming.get("sources") or [])
        extra.extend(incoming.get("grounding_urls") or [])
    existing = {str(s.get("url") if isinstance(s, dict) else s) for s in (pack.get("sources") or [])}
    sources = list(pack.get("sources") or [])
    grounding = list(pack.get("grounding_urls") or [])
    for item in extra:
        url = item if isinstance(item, str) else str((item or {}).get("url") or "")
        if url and url not in existing:
            row = item if isinstance(item, dict) else {"title": "", "url": url}
            sources.append(row)
            grounding.append(row)
            existing.add(url)
    pack["sources"] = sources
    pack["grounding_urls"] = grounding



def research_ticker(
    settings: Settings,
    quant: RecommendationRecord,
    *,
    filings: list[dict[str, Any]] | None = None,
    conn=None,
    tick_id: str | None = None,
    portfolio: dict[str, Any] | None = None,
    last_price: float | None = None,
    price_snapshot: dict[str, Any] | None = None,
    portfolio_sig: str | None = None,
    regime: str | None = None,
    enforce_llm_budget: bool = False,
) -> dict[str, Any]:
    ticker = ticker_from_instrument(quant.instrument_id)
    filing_rows = [f for f in (filings or []) if isinstance(f, dict)]
    accession = str((filing_rows[0] or {}).get("accession") or "") if filing_rows else ""
    passages: list[dict[str, Any]] = []
    for row in filing_rows:
        passages.extend(row.get("passages") or [])
    key = cache_key(
        ticker=ticker,
        accession=accession,
        quant_action=str(getattr(quant.action, "value", quant.action)),
        units_bucket=units_bucket(quant.recommended_units),
    ) + f"|{RESEARCH_PROMPT_VERSION}"
    cached = load_cached_pack(
        conn,
        key,
        last_price=last_price,
        portfolio_sig=portfolio_sig,
        regime=regime,
    )
    if cached:
        cached["cached"] = True
        return cached

    if not gemini_configured(settings):
        pack = _local_pack(
            ticker,
            filings=filing_rows,
            passages=passages,
            note="Gemini NOT_CONFIGURED. Evidence pack is filings + Quant only; no web/news research.",
        )
        persist_pack(
            conn,
            _stamp_freshness(pack, last_price=last_price, portfolio_sig=portfolio_sig, regime=regime),
            cache_key_value=key,
            ticker=ticker,
            instrument_id=str(quant.instrument_id),
            tick_id=tick_id,
            quant_action=str(getattr(quant.action, "value", quant.action)),
            accession=accession or None,
        )
        return pack

    if enforce_llm_budget and would_exceed_budget(
        conn, settings, model=research_model(settings), prompt_tokens=6_000, completion_tokens=2_000
    ):
        pack = _local_pack(
            ticker,
            filings=filing_rows,
            passages=passages,
            note="Automatic-scan daily LLM budget reached before Gemini research. Pack is filings + Quant only.",
            status="UNAVAILABLE",
        )
        pack["open_questions"].append("BUDGET_EXCEEDED")
        return _stamp_freshness(pack, last_price=last_price, portfolio_sig=portfolio_sig, regime=regime)

    filing_text = passages_as_text(passages) if passages else json.dumps(
        [{k: v for k, v in row.items() if k != "passages"} for row in filing_rows],
        ensure_ascii=False,
    )[:8000]
    user = dumps_complete(
        {
            "ticker": ticker,
            "quant": _quant_slice(quant),
            "price_snapshot": price_snapshot,
            "daily_feature_basis": "completed_daily_bars_only_v1",
            "filing_passages": filing_text,
            "filing_metadata": [
                {
                    "form": row.get("form"),
                    "accession": row.get("accession"),
                    "accepted_at": row.get("accepted_at"),
                    "note": row.get("note"),
                }
                for row in filing_rows
            ],
            "portfolio": _compact_research_portfolio(portfolio)
            if isinstance(portfolio, dict) and portfolio
            else {
                "current_units": quant.current_units,
                "recommended_units": quant.recommended_units,
            },
            "ask": "Answer legacy-b Q1-Q5 and the three separate attractiveness judgments.",
        }
    )
    first = generate_json(
        settings,
        system=RESEARCH_SYSTEM,
        user=user,
        use_search=True,
        response_schema=RESEARCH_SCHEMA,
    )
    prompt_n, completion_n = usage_from_payload(first)
    record_usage(
        conn,
        provider="gemini",
        model=research_model(settings),
        kind="research",
        ticker=ticker,
        prompt_tokens=prompt_n,
        completion_tokens=completion_n,
        toward_daily_cap=enforce_llm_budget,
    )
    persist_exchange(
        conn,
        first,
        tick_id=tick_id,
        kind="research",
        instrument_id=str(quant.instrument_id),
        ticker=ticker,
        prompt_version=RESEARCH_PROMPT_VERSION,
        default_system=RESEARCH_SYSTEM,
        default_user=user,
    )
    status = str(first.get("status") or STATUS_AVAILABLE)
    pack = normalize_pack(first, ticker=ticker)
    pack["filings"] = filing_rows or pack.get("filings") or []
    pack["status"] = status
    pack["web_search_used"] = bool(first.get("web_search_used"))
    _merge_sources(pack, first)
    if status in _FAIL_RESEARCH:
        pack["open_questions"] = list(pack.get("open_questions") or [])
        pack["open_questions"].append(str(first.get("error") or first.get("note") or status))
        persist_pack(
            conn,
            _stamp_freshness(pack, last_price=last_price, portfolio_sig=portfolio_sig, regime=regime),
            cache_key_value=key,
            ticker=ticker,
            instrument_id=str(quant.instrument_id),
            tick_id=tick_id,
            quant_action=str(getattr(quant.action, "value", quant.action)),
            accession=accession or None,
        )
        return pack

    attack_user = dumps_complete(
        {"ticker": ticker, "evidence_pack": compact_evidence_pack(pack), "goal": "find missing bearish facts"}
    )
    second = generate_json(
        settings,
        system=ADVERSARIAL_SYSTEM,
        user=attack_user,
        use_search=True,
        response_schema=ADVERSARIAL_SCHEMA,
    )
    p2, c2 = usage_from_payload(second)
    record_usage(
        conn,
        provider="gemini",
        model=research_model(settings),
        kind="adversarial",
        ticker=ticker,
        prompt_tokens=p2,
        completion_tokens=c2,
        toward_daily_cap=enforce_llm_budget,
    )
    persist_exchange(
        conn,
        second,
        tick_id=tick_id,
        kind="adversarial",
        instrument_id=str(quant.instrument_id),
        ticker=ticker,
        prompt_version=ADVERSARIAL_PROMPT_VERSION,
        default_system=ADVERSARIAL_SYSTEM,
        default_user=attack_user,
    )
    second_status = str(second.get("status") or STATUS_UNAVAILABLE)
    if second_status in {STATUS_AVAILABLE, STATUS_DEGRADED}:
        adv = second.get("adversarial_review") if isinstance(second.get("adversarial_review"), dict) else second
        objections = None
        if isinstance(adv, dict):
            objections = adv.get("main_objections") or adv.get("objections") or adv.get("bearish_points")
        if isinstance(objections, str) and objections.strip():
            objections = [objections]
        pack["adversarial_review"] = {
            "main_objections": [str(x) for x in (objections or [])],
            "what_would_change_the_call": str(
                (adv or {}).get("what_would_change_the_call") or second.get("what_would_change_the_call") or ""
            ),
        }
        bearish_score = clamp_score_0_100(
            (adv or {}).get("bearish_severity_score")
            if isinstance(adv, dict)
            else None
        )
        if bearish_score is None:
            bearish_score = clamp_score_0_100(second.get("bearish_severity_score"))
        pack["bearish_severity_score"] = bearish_score
        if not pack.get("summary_ko"):
            pack["summary_ko"] = str(
                (adv or {}).get("summary_ko") or second.get("summary_ko") or ""
            ).strip()
        extra = second.get("contrary_evidence") or (adv or {}).get("contrary_evidence") or []
        if extra:
            pack["contrary_evidence"] = list(pack.get("contrary_evidence") or []) + list(
                normalize_pack({"contrary_evidence": extra}, ticker=ticker)["contrary_evidence"]
            )
        missing = second.get("missing_evidence") or (adv or {}).get("missing_evidence")
        if missing:
            pack["open_questions"] = list(pack.get("open_questions") or []) + [
                str(x) for x in (missing if isinstance(missing, list) else [missing])
            ]
        _merge_sources(pack, second)
        if second.get("web_search_used"):
            pack["web_search_used"] = True
        if second_status == STATUS_DEGRADED:
            pack["open_questions"] = list(pack.get("open_questions") or [])
            if not any("Search grounding" in str(q) or "web search" in str(q).lower() for q in pack["open_questions"]):
                pack["open_questions"].append(str(second.get("note") or "Adversarial web search was unavailable."))
    else:
        pack["open_questions"] = list(pack.get("open_questions") or [])
        pack["open_questions"].append("Adversarial Gemini pass failed; judge still receives the research pack.")

    if pack.get("status") not in _FAIL_RESEARCH:
        pack["status"] = STATUS_AVAILABLE
    pack["web_search_used"] = bool(pack.get("web_search_used"))
    if not pack["web_search_used"]:
        questions = list(pack.get("open_questions") or [])
        if not any(
            "Search grounding" in str(q) or "web search" in str(q).lower() or "검색" in str(q)
            for q in questions
        ):
            questions.append(
                "Live Google Search grounding was not used. Pack is Gemini + filings/quant only."
            )
            pack["open_questions"] = questions
    pack = verify_pack_sources(pack)
    persist_pack(
        conn,
        _stamp_freshness(pack, last_price=last_price, portfolio_sig=portfolio_sig, regime=regime),
        cache_key_value=key,
        ticker=ticker,
        instrument_id=str(quant.instrument_id),
        tick_id=tick_id,
        quant_action=str(getattr(quant.action, "value", quant.action)),
        accession=accession or None,
    )
    return pack


def reuse_prior_judge(conn, instrument_id: str) -> dict[str, Any] | None:
    """Last stored Sol JSON for this name, used when the evidence pack did not change."""
    if conn is None:
        return None
    try:
        row = conn.execute(
            """
            SELECT payload_json FROM tick_recommendations
            WHERE source = 'llm_final' AND payload_json LIKE ?
            ORDER BY tick_id DESC
            LIMIT 40
            """,
            [f"%{instrument_id}%"],
        ).fetchall()
    except Exception:  # noqa: BLE001
        return None
    for (raw,) in row:
        try:
            wrapper = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        body = wrapper.get("payload") if isinstance(wrapper, dict) else None
        rec = body if isinstance(body, dict) else wrapper
        if not isinstance(rec, dict):
            continue
        if str(rec.get("instrument_id")) != str(instrument_id):
            continue
        reasons = {str(x) for x in (rec.get("override_reasons") or [])}
        if reasons & {
            "NOT_A_CANDIDATE",
            "NOT_SELECTED_FOR_FINAL_JUDGE",
            "NOT_CONFIGURED",
            "UNAVAILABLE",
            "RATE_LIMITED",
            "QUOTA_EXCEEDED",
            "BUDGET_EXCEEDED",
        }:
            continue
        if rec.get("thesis"):
            return rec
    return None


__all__ = [
    "research_ticker",
    "reuse_prior_judge",
    "select_final_judge_recs",
]
