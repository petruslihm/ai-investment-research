"""Structured Evidence Pack / judge payload compaction and source verification."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

JUDGE_PACKAGE_MAX_CHARS = 22_000
_SEC_HOSTS = frozenset({"sec.gov", "www.sec.gov", "data.sec.gov", "efts.sec.gov"})


def dumps_complete(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _trim_list(value: object, n: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return list(value[: max(0, n)])


def _normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = (parsed.path or "").rstrip("/")
    return f"{host}{path}"


def _is_sec_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host in _SEC_HOSTS:
        return True
    return host.endswith(".sec.gov")


def verified_url_set(pack: dict[str, Any] | None) -> set[str]:
    out: set[str] = set()
    if not isinstance(pack, dict):
        return out
    for key in ("sources", "grounding_urls", "verified_sources"):
        for item in pack.get(key) or []:
            url = ""
            if isinstance(item, str):
                url = item
            elif isinstance(item, dict):
                url = str(item.get("url") or item.get("uri") or item.get("source_url") or "")
            if url:
                out.add(_normalize_url(url))
                if _is_sec_url(url):
                    out.add(_normalize_url(url))
    for row in pack.get("filings") or []:
        if not isinstance(row, dict):
            continue
        for key in ("url", "source_url", "document_url"):
            url = str(row.get(key) or "")
            if url:
                out.add(_normalize_url(url))
        acc = str(row.get("accession") or "")
        if acc:
            out.add(acc.replace("-", "").lower())
    return {x for x in out if x}


def _mark_claim(row: Any, verified: set[str]) -> dict[str, Any]:
    if isinstance(row, str):
        return {"claim": row, "source_url": "", "source_status": "UNVERIFIED"}
    if not isinstance(row, dict):
        return {"claim": str(row), "source_url": "", "source_status": "UNVERIFIED"}
    out = dict(row)
    url = str(out.get("source_url") or out.get("url") or "").strip()
    out["source_url"] = url
    if not url:
        out["source_status"] = "UNVERIFIED"
        return out
    if _is_sec_url(url) or _normalize_url(url) in verified:
        out["source_status"] = "VERIFIED_SOURCE"
    else:
        out["source_status"] = "UNVERIFIED"
    return out


def verify_pack_sources(pack: dict[str, Any] | None) -> dict[str, Any]:
    base = dict(pack or {})
    verified = verified_url_set(base)
    for key in ("supporting_evidence", "contrary_evidence"):
        base[key] = [_mark_claim(row, verified) for row in (base.get(key) or [])]
    news = []
    for row in base.get("news_events") or []:
        if isinstance(row, dict):
            item = dict(row)
            url = str(item.get("url") or item.get("source_url") or "").strip()
            item["source_url"] = url
            item["source_status"] = (
                "VERIFIED_SOURCE"
                if url and (_is_sec_url(url) or _normalize_url(url) in verified)
                else "UNVERIFIED"
            )
            news.append(item)
        else:
            news.append(row)
    base["news_events"] = news
    return base


def _is_empty(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        out = {k: _drop_empty(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if not _is_empty(v)}
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.append(_drop_empty(item) if isinstance(item, (dict, list)) else item)
        return [row for row in rows if not _is_empty(row)]
    return value


def compact_evidence_pack(pack: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(pack or {})
    known = {
        "ticker": raw.get("ticker"),
        "as_of": raw.get("as_of"),
        "status": raw.get("status"),
        "material_change": raw.get("material_change"),
        "web_search_used": raw.get("web_search_used"),
        "data_quality": raw.get("data_quality"),
        "summary_ko": raw.get("summary_ko"),
        "open_questions": _trim_list(raw.get("open_questions"), 12),
        "macro": raw.get("macro") if isinstance(raw.get("macro"), dict) else {},
        "industry": raw.get("industry") if isinstance(raw.get("industry"), dict) else {},
        "portfolio_context": raw.get("portfolio_context") if isinstance(raw.get("portfolio_context"), dict) else {},
        "adversarial_review": raw.get("adversarial_review")
        if isinstance(raw.get("adversarial_review"), dict)
        else {},
        "news_events": _trim_list(raw.get("news_events"), 10),
        "filings": _trim_list(raw.get("filings"), 6),
        "supporting_evidence": _trim_list(raw.get("supporting_evidence"), 8),
        "contrary_evidence": _trim_list(raw.get("contrary_evidence"), 8),
        "price_events": _trim_list(raw.get("price_events"), 8),
        "sources": _trim_list(raw.get("sources"), 12),
        "grounding_urls": _trim_list(raw.get("grounding_urls") or raw.get("sources"), 12),
    }
    skip = {
        "status",
        "researched_at",
        "freshness",
        "freshness_state",
        "_exchange",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    for key, value in raw.items():
        if key not in known and key not in skip:
            known[key] = value
    return _drop_empty(verify_pack_sources(known))


def _quant_compact(quant: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(quant, dict):
        return {}
    horizons = []
    for row in quant.get("horizons") or []:
        if not isinstance(row, dict):
            continue
        horizons.append(
            {
                "horizon": row.get("horizon"),
                "expected_return": row.get("expected_return"),
                "rank_score": row.get("rank_score"),
                "confidence": row.get("confidence"),
            }
        )
    return _drop_empty(
        {
            "instrument_id": quant.get("instrument_id"),
            "action": quant.get("action"),
            "current_units": quant.get("current_units"),
            "recommended_units": quant.get("recommended_units"),
            "delta_units": quant.get("delta_units"),
            "acquisition_units": quant.get("acquisition_units"),
            "marked_units_final": quant.get("marked_units_final"),
            "confidence": quant.get("confidence"),
            "horizons": horizons,
            "thesis": str(quant.get("thesis") or "")[:400],
        }
    )


def compact_allocation_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    """Keep the book-size diagnosis for the judge; drop the 400-row dump."""
    if not isinstance(trace, dict):
        return {}
    top: list[dict[str, Any]] = []
    for row in list(trace.get("rows_passed") or [])[:8]:
        if not isinstance(row, dict):
            continue
        top.append(
            {
                "ticker": row.get("ticker"),
                "pred_5d": row.get("pred_5d"),
                "pred_10d": row.get("pred_10d"),
                "pred_20d": row.get("pred_20d"),
                "vol": row.get("vol"),
                "opp_score": row.get("opp_score"),
                "conviction": row.get("conviction"),
                "allocation_score": row.get("allocation_score"),
                "final_weight": row.get("final_weight"),
            }
        )
    diag = trace.get("diagnosis") if isinstance(trace.get("diagnosis"), dict) else {}
    return _drop_empty(
        {
            "risk_appetite": trace.get("risk_appetite"),
            "sizing_alpha": trace.get("sizing_alpha"),
            "vol_unit": trace.get("vol_unit"),
            "universe": trace.get("universe"),
            "passed_cutoff": trace.get("passed_cutoff"),
            "cutoff": trace.get("cutoff"),
            "strong_score": trace.get("strong_score"),
            "score_percentiles": trace.get("score_percentiles"),
            "mean_return_percentiles": trace.get("mean_return_percentiles"),
            "aggregate_conviction": trace.get("aggregate_conviction"),
            "equivalent_names": trace.get("equivalent_names"),
            "deployment_factor": trace.get("deployment_factor"),
            "equity_budget": trace.get("equity_budget"),
            "stock_weight": trace.get("stock_weight"),
            "held_keep_weight": trace.get("held_keep_weight"),
            "new_name_budget": trace.get("new_name_budget"),
            "btc_weight": trace.get("btc_weight"),
            "cash_weight": trace.get("cash_weight"),
            "residual_cash_reason": trace.get("residual_cash_reason"),
            "diagnosis": {
                "model_expected_returns": diag.get("model_expected_returns"),
                "cutoff_vs_distribution": diag.get("cutoff_vs_distribution"),
                "sizing_compression": diag.get("sizing_compression"),
                "notes": _trim_list(diag.get("notes"), 6),
            },
            "top_passed": top,
            "note": (
                "Quant book size and cross-section. "
                "min_opportunity_score is a buy gate, not a sell gate: "
                "held names below cutoff stay unless 5/10/20 mean expected return is negative. "
                "Judge whether this is warranted caution or calibration; you may size larger than Quant."
            ),
        }
    )


def compact_judge_package(package: dict[str, Any], *, max_chars: int = JUDGE_PACKAGE_MAX_CHARS) -> dict[str, Any]:
    """Always return a complete JSON object. Never slice serialized text."""
    port = dict(package.get("portfolio") if isinstance(package.get("portfolio"), dict) else {})
    raw_trace = package.get("allocation_trace")
    if not isinstance(raw_trace, dict):
        raw_trace = port.pop("allocation_trace", None)
    else:
        port.pop("allocation_trace", None)
    out = {
        "dq_level": package.get("dq_level"),
        "portfolio": port,
        "quant": _quant_compact(package.get("quant") if isinstance(package.get("quant"), dict) else {}),
        "evidence_pack": compact_evidence_pack(
            package.get("evidence_pack") if isinstance(package.get("evidence_pack"), dict) else {}
        ),
        "allocation_trace": compact_allocation_trace(raw_trace if isinstance(raw_trace, dict) else None),
    }
    out = {k: v for k, v in out.items() if not _is_empty(v)}
    budgets = [
        ("news_events", 10, 4),
        ("filings", 6, 2),
        ("supporting_evidence", 8, 3),
        ("contrary_evidence", 8, 3),
        ("sources", 12, 4),
        ("price_events", 8, 2),
        ("open_questions", 12, 4),
    ]
    pack = out.get("evidence_pack") if isinstance(out.get("evidence_pack"), dict) else {}
    if pack:
        out["evidence_pack"] = pack
    while len(dumps_complete(out)) > max_chars:
        shrunk = False
        for key, _hi, lo in budgets:
            rows = pack.get(key) if isinstance(pack.get(key), list) else []
            if len(rows) > lo:
                pack[key] = rows[: max(lo, len(rows) // 2)]
                shrunk = True
                break
        if not shrunk:
            holdings = out.get("portfolio", {}).get("holdings") if isinstance(out.get("portfolio"), dict) else None
            if isinstance(holdings, list) and len(holdings) > 8:
                out["portfolio"]["holdings"] = holdings[:8]
                continue
            break
    out = _drop_empty(out)
    dumped = dumps_complete(out)
    if not dumped.startswith("{") or not dumped.endswith("}"):
        raise ValueError("compact_judge_package produced incomplete JSON")
    return out
