"""Local FastAPI UI: Dashboard through Settings. API keys at first start / Settings."""

from __future__ import annotations

import html
import math
import os
import secrets
import signal
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import markdown as _markdown

from fastapi import FastAPI, Form, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from trading_system.config import Settings, discover_project_root, get_settings
from trading_system.credentials import clear_env_keys, env_path, missing_provider_keys, read_env_map, upsert_env
from trading_system.kakao import (
    DEFAULT_REDIRECT_URI,
    KAKAO_TOKEN_KEYS,
    KakaoStatus,
    authorize_url,
    connection_status,
    exchange_code,
    notify_kakao,
    oauth_state_path,
)
from trading_system.actionability import actionable_view
from trading_system.llm_client import STATUS_AVAILABLE
from trading_system.openai_judge import resume_session_bounds_utc
from trading_system.allocation import resolve_allocation_trace
from trading_system.market.decision_data import price_description
from trading_system.alerts import AlertKind, AlertRecord
from trading_system.portfolio import Lot
from trading_system.marked_units import marked_units_from_lots
from trading_system.portfolio_service import (
    add_lot,
    delete_lot,
    get_total_base_units,
    latest_final_close,
    list_lots,
    set_total_base_units,
    update_lot,
)
from trading_system.connectivity import STATUS_OK, ConnectionReport, check_all
from trading_system.storage import Store, WriterLeaseBusy
from trading_system.ui.llm_view import render_llm_log_html
from trading_system.ui.models_view import (
    CHANGE_GROUPS,
    HORIZON_CHOICES,
    collect_models_view,
    format_ts,
    render_models_page,
)
from trading_system.universe import (
    PRESET_CUSTOM,
    PRESET_LABELS,
    PRESET_ORDER,
    add_custom_symbols,
    add_watchlist_symbols,
    coverage_stats,
    custom_symbols,
    held_symbols,
    parse_symbol_input,
    prepare_scan_universe,
    remove_custom_symbol,
    remove_watchlist_symbol,
    resolve_scan_universe,
    universe_state,
    watchlist_symbols,
)
from trading_system.llm_log import list_llm_sessions, load_llm_session, ticker_from_instrument
from trading_system.ticker_lookup import lookup_ticker
from trading_system.v1_cycle import load_last_ui_snapshot, run_v1_cycle

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"
PROJECT_ROOT = discover_project_root()

NAV_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("/", "대시보드", "⊞"),
    ("/portfolio", "포트폴리오", "◈"),
    ("/recommendations", "종목 추천", "✦"),
    ("/llm", "LLM 기록", "▣"),
    ("/btc", "BTC 신호", "₿"),
    ("/models", "모델/학습", "◎"),
    ("/runtime", "런타임", "◉"),
    ("/data-health", "데이터 상태", "◇"),
    ("/settings", "설정", "◌"),
)

_STATUS_KO = {
    "AVAILABLE": ("정상", "badge-ok"),
    "NOT_CONFIGURED": ("미설정", "badge-warn"),
    "UNAVAILABLE": ("사용 불가", "badge-err"),
    "DEGRADED": ("저하", "badge-warn"),
    "RATE_LIMITED": ("요청 한도", "badge-warn"),
    "QUOTA_EXCEEDED": ("쿼터 초과", "badge-err"),
    "UNKNOWN": ("알 수 없음", "badge-neutral"),
    "OK": ("정상", "badge-ok"),
    "CRITICAL": ("심각", "badge-err"),
    "SUCCEEDED": ("성공", "badge-ok"),
    "FAILED": ("실패", "badge-err"),
    "CONNECTED": ("연결됨", "badge-ok"),
    "AUTH_REQUIRED": ("인증 필요", "badge-warn"),
    "TOKEN_EXPIRED": ("토큰 만료", "badge-err"),
}
_ACTION_KO = {
    "BUY": "매수",
    "SELL": "매도",
    "HOLD": "유지",
    "REDUCE": "축소",
    "ENTER": "매수",
    "ADD": "추가매수",
    "EXIT": "매도",
    "NO_ACTION": "관망",
}
_CAP_KO = {
    "market_data": "시장 데이터",
    "quant": "Quant 모델",
    "sec": "SEC 분석",
    "llm_extract": "공시 구간",
    "llm_research": "Gemini 리서치",
    "llm_final_judge": "GPT-5.6 Sol 판단",
}
_CAP_HINT = {
    ("market_data", "NOT_CONFIGURED"): "Alpaca 키가 없습니다. 설정에서 넣으면 실시간 시세를 씁니다.",
    ("market_data", "UNAVAILABLE"): "Alpaca 호출에 실패했습니다. 직전 시세가 있으면 그걸로 계속합니다.",
    ("market_data", "AVAILABLE"): "시세를 받고 있습니다.",
    ("quant", "NOT_CONFIGURED"): "실시간 시세가 없어 연구용 추천만 만듭니다.",
    ("quant", "UNAVAILABLE"): "실시간 시세가 없어 추천을 실행 가능으로 표시하지 않습니다.",
    ("quant", "AVAILABLE"): "Quant 추천이 준비되었습니다.",
    ("sec", "NOT_CONFIGURED"): "SEC 공시를 아직 쓰지 않습니다.",
    ("sec", "UNAVAILABLE"): "SEC EDGAR가 이 네트워크에서 차단(403)됐거나 공시를 받지 못했습니다. 앱은 계속 동작합니다.",
    ("sec", "DEGRADED"): "실시간 EDGAR가 실패해 직전 공시 메타데이터를 씁니다. 가짜 공시는 만들지 않습니다.",
    ("sec", "AVAILABLE"): "SEC 공시를 받았습니다.",
    ("llm_extract", "NOT_CONFIGURED"): "공시 본문 구간을 아직 준비하지 못했습니다.",
    ("llm_extract", "UNAVAILABLE"): "공시 본문을 받지 못했거나 구간을 고르지 못했습니다.",
    ("llm_extract", "AVAILABLE"): "공시에서 관련 구간을 골랐습니다. 앞부분만 자르지 않습니다.",
    ("llm_research", "NOT_CONFIGURED"): "Gemini 키가 없습니다. 설정에서 넣으면 리서치 팩을 만듭니다.",
    ("llm_research", "UNAVAILABLE"): "Gemini 리서치가 실패했습니다. 공시·Quant만 최종 판단에 넘어갑니다.",
    ("llm_research", "DEGRADED"): "이번 스캔에서 실패한 Gemini 종목이 더 많습니다. 429 요청 한도를 의심하세요.",
    ("llm_research", "AVAILABLE"): "Gemini가 근거 팩을 만들었습니다.",
    ("llm_final_judge", "NOT_CONFIGURED"): "OpenAI 키가 없습니다. 설정에서 넣으면 GPT-5.6 Sol이 최종 판단을 합니다.",
    ("llm_final_judge", "UNAVAILABLE"): "최종 판단 호출에 실패했습니다. Quant 결과는 그대로 둡니다.",
    ("llm_final_judge", "RATE_LIMITED"): "OpenAI 요청 한도(429)입니다. Quant 결과는 유지되고, 최종 판단은 만들지 않습니다.",
    ("llm_final_judge", "QUOTA_EXCEEDED"): "OpenAI 쿼터/결제 한도입니다. Quant 결과는 유지되고, 최종 판단은 만들지 않습니다.",
    ("llm_final_judge", "DEGRADED"): "최종 판단이 일부만 성공했습니다. Quant 결과는 그대로 둡니다.",
    ("llm_final_judge", "BUDGET_EXCEEDED"): "자동 실행의 UTC 일일 LLM 예산 기준값(soft limit)에 닿았습니다. 실제 비용의 절대 상한은 아니며, 직접 실행한 스캔은 이 기준에 포함되지 않습니다.",
    ("llm_final_judge", "AVAILABLE"): "GPT-5.6 Sol 최종 판단이 동작합니다.",
}


def format_elapsed_ko(seconds: float) -> str:
    """Render elapsed seconds as '12초' / '3분 05초' / '1시간 02분 03초'."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}시간 {minutes:02d}분 {secs:02d}초"
    if minutes:
        return f"{minutes}분 {secs:02d}초"
    return f"{secs}초"


def _store(*, writer: bool = True, stale_seconds: int | None = None) -> Store:
    settings = get_settings()
    path = settings.duckdb_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    s = Store(path)
    s.open(
        acquire_writer=writer,
        stale_seconds=stale_seconds if stale_seconds is not None else settings.writer_lease_stale_seconds,
    )
    return s


def _esc(v: object) -> str:
    return html.escape("" if v is None else str(v))


def _render_markdown_safe(text: str) -> str:
    """GPT's final-report prose (headers/tables/bold/lists) rendered as real HTML
    instead of a raw pre-wrap dump of '#'/'**'/'|' characters. Escaping every HTML
    special char BEFORE handing the text to the markdown parser -- rather than
    trusting the parser's own HTML handling or a post-hoc sanitizer -- means any
    literal '<script>' the model echoes (e.g. from a prompt-injection attempt inside
    a page it web-searched) becomes inert '&lt;script&gt;' text, since it no longer
    looks like an HTML tag by the time the parser sees it. Markdown syntax characters
    ('#', '*', '|', '-') are untouched by html.escape, so real formatting still works.
    """
    escaped = html.escape("" if text is None else str(text))
    return _markdown.markdown(escaped, extensions=["tables", "sane_lists"])


def lot_row_html(conn, settings: Settings, lot: Lot) -> str:
    """One holding row with ticker confirmation (✓ 확인됨 / 미확인)."""
    got = lookup_ticker(conn, settings, str(lot.instrument_id))
    symbol = (got.symbol or "").strip() or ticker_from_instrument(lot.instrument_id)
    initials = "".join(ch for ch in symbol.upper() if ch.isalnum())[:2] or "??"
    acquired = lot.acquired_on.isoformat() if hasattr(lot.acquired_on, "isoformat") else str(lot.acquired_on)
    if got.ok:
        badge = (
            f'<span class="lot-verified ok" title="{_esc(got.message)}">'
            f'<span class="tick" aria-hidden="true">✓</span>확인됨</span>'
        )
    else:
        badge = (
            f'<span class="lot-verified bad" title="{_esc(got.message)}">'
            f'<span class="tick" aria-hidden="true">!</span>미확인</span>'
        )
    units_s = f"{lot.acquisition_units:g}"
    price_s = f"{lot.acquisition_price:g}"
    close = latest_final_close(conn, lot.instrument_id)
    marked = marked_units_from_lots([lot], close)
    if marked is not None:
        mark_line = f"종가 환산 {marked:g} 단위"
    else:
        mark_line = "종가 환산 없음"
    return (
        f"<div class='lot'><div class='avatar'>{_esc(initials)}</div>"
        f"<div class='lot-identity'>"
        f"<div class='lot-name'><span class='lot-symbol'>{_esc(symbol)}</span>{badge}</div>"
        f"<p class='hint lot-meta'>취득일 {acquired} · {mark_line}</p></div>"
        f"<div class='lot-actions'>"
        f"<form method='post' action='/portfolio/update' class='lot-edit'>"
        f"<input type='hidden' name='lot_id' value='{_esc(lot.lot_id)}'/>"
        f"<label class='lot-edit-field'>취득 단위"
        f"<input name='units' type='number' min='0.0001' step='any' value='{_esc(units_s)}' required/></label>"
        f"<label class='lot-edit-field'>취득가"
        f"<input name='price' type='number' min='0.0001' step='any' value='{_esc(price_s)}' required/></label>"
        f"<button class='btn' type='submit'>저장</button></form>"
        f"<form method='post' action='/portfolio/delete' class='lot-delete'>"
        f"<input type='hidden' name='lot_id' value='{_esc(lot.lot_id)}'/>"
        f"<button class='btn btn-ghost' type='submit'>삭제</button></form></div></div>"
    )


def _text_ko(v: object) -> str:
    s = "" if v is None else str(v)
    s = s.replace(
        "LLM FINAL JUDGE: NOT_CONFIGURED. Quant recommendation is NOT_CONFIGURED separately. "
        "Enter an OpenAI key in Settings to run the real judge.",
        "LLM 최종 판단이 미설정입니다. Quant 추천도 별도로 미설정입니다. 실제 판단을 쓰려면 설정에서 OpenAI 키를 입력하세요.",
    )
    s = s.replace("LLM FINAL JUDGE: UNAVAILABLE.", "LLM 최종 판단 사용 불가.")
    s = s.replace("LLM FINAL JUDGE:", "LLM 최종 판단:")
    s = s.replace("Quant recommendation is", "Quant 추천은")
    s = s.replace("Quant recommendation:", "Quant 추천:")
    s = s.replace("No synthesized LLM action.", "LLM 액션을 만들지 않았습니다.")
    s = s.replace("agreement=conflict", "합의=충돌")
    s = s.replace("agreement=agree", "합의=일치")
    s = s.replace("formula=quant_alloc_v6", "공식=quant_alloc_v6")
    s = s.replace("formula=quant_alloc_v5", "공식=quant_alloc_v5")
    s = s.replace("formula=quant_alloc_v4", "공식=quant_alloc_v4")
    s = s.replace("formula=quant_alloc_v3", "공식=quant_alloc_v3")
    s = s.replace("formula=quant_alloc_v2", "공식=quant_alloc_v2")
    s = s.replace("NOT_CONFIGURED", "미설정")
    s = s.replace("UNAVAILABLE", "사용 불가")
    s = s.replace("AVAILABLE", "정상")
    return s


def _badge(status: object) -> str:
    key = str(status).upper().replace("-", "_")
    label, cls = _STATUS_KO.get(key, (str(status), "badge-neutral"))
    return f'<span class="badge {cls}">{_esc(label)}</span>'


def _action(status: object) -> str:
    s = str(status)
    return _ACTION_KO.get(s, s)


def _pct(v: object) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _ticker(instrument_id: object) -> str:
    s = str(instrument_id)
    if s.startswith("inst_"):
        s = s[5:]
    return s.upper().replace("/", "/")


def _action_badge(action: object, label: str | None = None) -> str:
    s = str(action)
    cls = "badge-neutral"
    if s in {"BUY", "ENTER", "ADD"}:
        cls = "badge-ok"
    elif s in {"REDUCE", "EXIT", "SELL"}:
        cls = "badge-err"
    return f'<span class="badge {cls}">{_esc(label or _action(s))}</span>'


def _horizon_chips(horizons: object) -> str:
    if not isinstance(horizons, list):
        return ""
    bits: list[str] = []
    for h in horizons:
        if not isinstance(h, dict):
            continue
        hz = h.get("horizon")
        er = h.get("expected_return")
        bits.append(f'<span class="chip-static">{_esc(hz)}일 {_esc(_pct(er))}</span>')
    return "".join(bits)


def _fmt_signed_pct(value: object) -> str | None:
    try:
        pct = float(value) * 100.0
    except (TypeError, ValueError):
        return None
    if abs(pct) < 0.05:
        return "0.0%"
    return f"{pct:+.1f}%"


def _score(rec: dict) -> str:
    """Show 5-day expected return. Do not round LambdaRank to an integer (-0)."""
    hs = rec.get("horizons") or []
    if isinstance(hs, list) and hs and isinstance(hs[0], dict):
        shown = _fmt_signed_pct(hs[0].get("expected_return"))
        if shown is not None:
            return shown
    return "—"


_BUYISH = {"BUY", "ENTER", "ADD"}
_FAIL_REASONS = {
    "NOT_CONFIGURED",
    "UNAVAILABLE",
    "RATE_LIMITED",
    "QUOTA_EXCEEDED",
    "DEGRADED",
    "NOT_A_CANDIDATE",
    "NOT_SELECTED_FOR_FINAL_JUDGE",
    "BUDGET_EXCEEDED",
    "INVALID_OUTPUT",
}


def _num_or_zero(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _snap_base(snap: dict | None) -> float:
    alloc = (snap or {}).get("allocation") if isinstance(snap, dict) else None
    if isinstance(alloc, dict) and alloc.get("total_base_units") is not None:
        try:
            n = float(alloc["total_base_units"])
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    return 1000.0


def _overlay_for_display(rec: dict, llm: dict | None = None) -> dict:
    row = dict(rec)
    if not _real_llm_judge(llm):
        return row
    assert llm is not None
    for key in (
        "action",
        "recommended_units",
        "delta_units",
        "marked_units_final",
        "acquisition_units",
        "requested_units",
        "constrained_units",
        "current_units",
    ):
        if llm.get(key) is not None:
            row[key] = llm.get(key)
    return row


def _view_for(rec: dict, llm: dict | None, *, settings: Settings, total_base_units: float):
    return actionable_view(_overlay_for_display(rec, llm), settings=settings, total_base_units=total_base_units)


def _is_holding(rec: dict) -> bool:
    return (
        _num_or_zero(rec.get("current_units")) > 1e-12
        or _num_or_zero(rec.get("acquisition_units")) > 1e-12
        or _num_or_zero(rec.get("marked_units_final")) > 1e-12
    )


def _is_buy_candidate(
    rec: dict,
    *,
    llm: dict | None = None,
    settings: Settings | None = None,
    total_base_units: float | None = None,
) -> bool:
    """True only when the user-facing action is still a material add."""
    cfg = settings or get_settings()
    base = float(total_base_units) if total_base_units is not None else 1000.0
    return _view_for(rec, llm, settings=cfg, total_base_units=base).is_buy


def _llm_map(rows: list) -> dict[str, dict]:
    return {str(r.get("instrument_id")): r for r in rows if isinstance(r, dict)}


def _fmt_units(value: object) -> str:
    if value is None or value == "":
        return "—"
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(n):
        return "—"
    if abs(n - round(n)) < 1e-9:
        return f"{n:.0f}"
    return f"{n:.2f}"


def _real_llm_judge(llm: dict | None) -> bool:
    if not llm:
        return False
    reasons = {str(x) for x in (llm.get("override_reasons") or [])}
    skip = {
        "NOT_A_CANDIDATE",
        "NOT_SELECTED_FOR_FINAL_JUDGE",
        "NOT_CONFIGURED",
        "UNAVAILABLE",
        "RATE_LIMITED",
        "QUOTA_EXCEEDED",
        "BUDGET_EXCEEDED",
    }
    return not bool(reasons & skip)


def _units_panel(
    rec: dict,
    llm: dict | None = None,
    *,
    settings: Settings,
    total_base_units: float,
) -> str:
    row = _overlay_for_display(rec, llm)
    view = actionable_view(row, settings=settings, total_base_units=total_base_units)
    acq = row.get("acquisition_units")
    marked = row.get("marked_units_final")
    preview = rec.get("marked_units_intraday_preview")
    requested = row.get("requested_units")
    constrained = row.get("constrained_units")
    rec_label = ("GPT 최종 추천" if rec.get("input_id") else "AI 추천") if _real_llm_judge(llm) else ("Gemini 반영 대체 결과" if rec.get("source") == "research_adjusted" else "Quant 추천")
    rec_badge = (
        f'<span class="badge badge-neutral">{rec_label} —</span>'
        if view.suppressed
        else f'<span class="badge badge-ok">{rec_label} {_esc(_fmt_units(view.display_recommended_units))} u</span>'
    )
    delta_badge = (
        f'<span class="badge badge-neutral">증감 —</span>'
        if view.suppressed
        else f'<span class="badge badge-neutral">증감 {_esc(_fmt_units(view.display_delta_units))} u</span>'
    )
    html = (
        f'<div class="chips units-row" style="margin:0.45rem 0 0">'
        f'<span class="badge badge-neutral">현재 보유 {_esc(_fmt_units(acq))} u</span>'
        f'<span class="badge badge-neutral">현재 평가 {_esc(_fmt_units(marked))} u</span>'
        f"{rec_badge}"
        f"{delta_badge}"
        f"</div>"
        f'<p class="hint units-basis" style="margin:0.25rem 0 0">공식 기준: 직전 FINAL close (lot별 매수가 대비)</p>'
    )
    if view.note:
        html += f'<p class="hint" style="margin:0.2rem 0 0">{_esc(view.note)}</p>'
    raw_bits: list[str] = []
    if rec.get("opportunity_score") is not None:
        try:
            raw_bits.append(f"기회점수 {float(rec.get('opportunity_score')):.4f}")
        except (TypeError, ValueError):
            pass
    if rec.get("excess_score") is not None:
        try:
            raw_bits.append(f"초과점수 {float(rec.get('excess_score')):.4f}")
        except (TypeError, ValueError):
            pass
    if rec.get("conviction") is not None:
        try:
            raw_bits.append(f"확신도 {float(rec.get('conviction')):.2f}")
        except (TypeError, ValueError):
            pass
    if rec.get("equity_budget") is not None:
        try:
            raw_bits.append(f"주식 예산 {float(rec.get('equity_budget')) * 100:.1f}%")
        except (TypeError, ValueError):
            pass
    if rec.get("initial_units") is not None:
        raw_bits.append(f"초기 목표 {_fmt_units(rec.get('initial_units'))} u")
    if rec.get("pre_floor_units") is not None:
        raw_bits.append(f"하한/상한 전 {_fmt_units(rec.get('pre_floor_units'))} u")
    if view.raw_recommended_units is not None:
        raw_bits.append(f"최종 목표 {_fmt_units(view.raw_recommended_units)} u")
    if rec.get("exclusion_reason"):
        raw_bits.append(f"제외 {rec.get('exclusion_reason')}")
    if view.raw_delta_units is not None:
        raw_bits.append(f"연속 증감 {_fmt_units(view.raw_delta_units)} u")
    show_raw = bool(raw_bits) and (
        view.suppressed
        or rec.get("exclusion_reason")
        or rec.get("opportunity_score") is not None
        or view.display_recommended_units != view.raw_recommended_units
        or view.display_delta_units != view.raw_delta_units
    )
    if show_raw:
        html += (
            f'<details class="tech-details"><summary>기술 상세</summary>'
            f'<p class="hint" style="margin:0.25rem 0 0">{_esc(" · ".join(raw_bits))}</p></details>'
        )
    if preview is not None:
        html += (
            f'<p class="hint units-intraday" style="margin:0.15rem 0 0;opacity:0.75">'
            f"장중 미리보기 (비공식): {_esc(_fmt_units(preview))} u</p>"
        )
    if (
        requested is not None
        and constrained is not None
        and abs(float(requested) - float(constrained)) > 1e-9
    ):
        html += (
            f'<p class="hint" style="margin:0.15rem 0 0">'
            f"요청 {_esc(_fmt_units(requested))} u → 포트폴리오 제약 후 {_esc(_fmt_units(constrained))} u</p>"
        )
    return html


def _split_long_rationale(text: str) -> tuple[str, str]:
    raw = " ".join(str(text or "").split())
    if len(raw) <= 160:
        return raw, ""
    cutoff = 0
    for sep in (". ", "! ", "? ", "다. ", "요. "):
        idx = raw.find(sep, 40)
        if 40 <= idx <= 180:
            cutoff = idx + len(sep)
            break
    if cutoff == 0:
        sp = raw.rfind(" ", 0, 140)
        cutoff = sp if sp > 40 else 140
    short = raw[:cutoff].strip()
    rest = raw[cutoff:].strip()
    if not rest:
        return raw, ""
    return short, rest


def _llm_rationale(llm: dict | None) -> tuple[str, str]:
    if not llm:
        return "기록된 근거 없음", ""
    reasons = [str(x) for x in (llm.get("override_reasons") or [])]
    if "NOT_A_CANDIDATE" in reasons:
        return "이 종목은 이번 매수 후보가 아니라 LLM 판단을 건너뛰었습니다.", ""
    if "NOT_SELECTED_FOR_FINAL_JUDGE" in reasons:
        return (
            "Quant 매수 후보이지만 이번 Sol 최종 판단 한도에는 못 들어갔습니다. "
            "표시된 단위는 Quant 숫자입니다."
        ), str(llm.get("rationale_detail") or "").strip()
    thesis = (llm.get("thesis") or "").strip()
    detail = str(llm.get("rationale_detail") or "").strip()
    if not thesis:
        return "기록된 근거 없음", detail
    short = _text_ko(thesis)
    if detail:
        return short, detail
    return _split_long_rationale(short)


def _rationale_html(llm: dict, *, quant_detail: str = "") -> str:
    short, detail = _llm_rationale(llm)
    reasons = [str(x) for x in (llm.get("override_reasons") or []) if str(x) not in {
        "NOT_A_CANDIDATE",
        "NOT_SELECTED_FOR_FINAL_JUDGE",
    }]
    html = f'<p class="hint" style="margin:0.45rem 0 0"><strong>최종 판단 근거</strong> {_esc(short)}</p>'
    bits = []
    if detail:
        bits.append(_esc(_text_ko(detail)))
    if quant_detail:
        bits.append(_esc(_text_ko(quant_detail)))
    if reasons:
        bits.append(_esc(" · ".join(_text_ko(r) for r in reasons)))
    if bits:
        body = "</p><p class='hint'>".join(bits)
        html += (
            '<details class="rationale-details">'
            "<summary>상세보기</summary>"
            f"<p class='hint' style='margin:0.4rem 0 0'>{body}</p>"
            "</details>"
        )
    return html


def _today_priority(
    quant: dict,
    llm: dict | None,
    *,
    settings: Settings | None = None,
    total_base_units: float | None = None,
) -> bool:
    """Only when live data is actionable, Quant wants to add, and LLM also wants to add.

    Does not invent a buy if LLM is missing, skipped, or failed.
    """
    if not quant.get("actionable"):
        return False
    cfg = settings or get_settings()
    base = float(total_base_units) if total_base_units is not None else 1000.0
    view = _view_for(quant, llm, settings=cfg, total_base_units=base)
    if not view.is_buy or (view.display_delta_units or 0) <= 0:
        return False
    if not llm:
        return False
    reasons = {str(x) for x in (llm.get("override_reasons") or [])}
    if reasons & _FAIL_REASONS:
        return False
    return str(llm.get("action")) in _BUYISH


def _rec_card(
    rec: dict,
    source: str,
    *,
    rank: int | None = None,
    llm: dict | None = None,
    today: bool = False,
    settings: Settings | None = None,
    total_base_units: float = 1000.0,
) -> str:
    cfg = settings or get_settings()
    view = _view_for(rec, llm, settings=cfg, total_base_units=total_base_units)
    act = view.display_action
    buyish = view.is_buy
    holdish = act in {"HOLD", "NO_ACTION"}
    if rank is not None:
        filt = "buy"
    else:
        filt = "buy" if buyish else ("neutral" if holdish else "other")
    score = _score(rec)
    ticker = _ticker(rec.get("instrument_id"))
    actionable = bool(rec.get("actionable")) and not view.suppressed
    if view.suppressed:
        act_label = "실행하지 않음"
    elif rec.get("actionable"):
        act_label = "실행 가능"
    else:
        act_label = "연구용"
    extra = ""
    if rec.get("input_id"):
        extra += (
            f'<p class="hint price-provenance">{_esc(price_description(rec.get("price_snapshot")))}<br>'
            f'입력 기준 {_esc(rec.get("input_as_of"))} · 일봉 피처·거래량: 완료 일봉만 사용</p>'
        )
        if not llm:
            extra += '<p class="hint">GPT 최종 판단 미적용 · Gemini 반영 결과</p>'
    quant_bits = ""
    if rec.get("allocation_note"):
        extra += f'<p class="hint" style="margin:0.4rem 0 0">{_esc(rec.get("allocation_note"))}</p>'
    elif rec.get("thesis"):
        quant_bits = str(rec.get("thesis") or "").strip()
    if llm is not None:
        extra += _rationale_html(llm, quant_detail=quant_bits)
        llm_act = str(llm.get("action") or "")
        skip_act = {"NOT_A_CANDIDATE", "NOT_SELECTED_FOR_FINAL_JUDGE"}
        if llm_act and not (set(str(x) for x in (llm.get("override_reasons") or [])) & skip_act):
            extra += (
                f'<p class="hint" style="margin:0.2rem 0 0">최종 액션: {_esc(view.display_label)}</p>'
            )
        contrary = str(llm.get("contrary_evidence") or "").strip()
        if contrary and not contrary.startswith("llm_judge_status="):
            extra += (
                f'<p class="hint" style="margin:0.2rem 0 0"><strong>반대 증거</strong> '
                f"{_esc(_text_ko(contrary))}</p>"
            )
    elif quant_bits:
        extra += f'<p class="hint" style="margin:0.4rem 0 0">{_esc(_text_ko(quant_bits))}</p>'
    today_html = (
        '<span class="badge badge-ok" data-today="1">오늘 우선</span>' if today else ""
    )
    if rank is not None:
        rank_html = (
            f'<div class="score"><b>{rank}</b><span>순위</span>'
            f'<span class="score-sub">{_esc(score)}</span></div>'
        )
    else:
        rank_html = f'<div class="score"><b>{_esc(score)}</b><span>점수</span></div>'
    today_attr = ' data-today="1"' if today else ""
    return (
        f'<div class="rec-card" data-filter="{filt}" data-ticker="{_esc(ticker.lower())}"{today_attr}>'
        f"{rank_html}"
        f'<div class="vline"></div>'
        f'<div class="grow">'
        f'<div class="row" style="border:0;padding:0">'
        f'<span style="font-weight:700">{_esc(ticker)}</span>'
        f'<span class="badge badge-neutral">{_esc(source)}</span>'
        f"{today_html}"
        f"</div>"
        f'<div class="chips" style="margin:0.45rem 0 0">'
        f"{_action_badge(act, view.display_label)}"
        f'<span class="badge {"badge-ok" if actionable else "badge-warn"}">{act_label}</span>'
        f"{_horizon_chips(rec.get('horizons'))}"
        f"</div>"
        f"{_units_panel(rec, llm, settings=cfg, total_base_units=total_base_units)}"
        f"{_technical_factor_html(rec)}"
        f"{_research_summary_html(rec)}"
        f"{extra}"
        f"</div></div>"
    )


def _rec_row_compact(
    rec: dict,
    *,
    llm: dict | None = None,
    settings: Settings | None = None,
    total_base_units: float = 1000.0,
) -> str:
    """One <tr> per scored ticker for the '전체 종목 점수' table.

    2026-09-05: this used to be a full _rec_card() (badges, units panel, technical
    chips, rationale text) repeated for every one of the ~500 scanned names, which
    put ~500 large divs in the DOM at once -- on a real scan this rendered a
    ~116,000px-tall page (~470 cards x ~213px each), most of it names nobody was
    looking at. A dense table row carries the same filter/search data attributes the
    existing chip/search JS already relies on, at a fraction of the DOM weight.
    """
    cfg = settings or get_settings()
    view = _view_for(rec, llm, settings=cfg, total_base_units=total_base_units)
    act = view.display_action
    buyish = view.is_buy
    holdish = act in {"HOLD", "NO_ACTION"}
    filt = "buy" if buyish else ("neutral" if holdish else "other")
    ticker = _ticker(rec.get("instrument_id"))
    conf = rec.get("confidence")
    try:
        conf_label = f"{float(conf):.2f}" if conf is not None else "—"
    except (TypeError, ValueError):
        conf_label = "—"
    track = str(rec.get("technical_track") or "").strip()
    # allocation_note (the boilerplate "why not") isn't worth a visible column across
    # ~500 rows, but it shouldn't vanish either -- a title attribute keeps it a hover
    # away without adding height.
    note = str(rec.get("allocation_note") or "").strip()
    note_attr = f' title="{_esc(note)}"' if note else ""
    return (
        f'<tr class="rec-row" data-filter="{filt}" data-ticker="{_esc(ticker.lower())}"{note_attr}>'
        f'<td class="trace-ticker">{_esc(ticker)}</td>'
        f"<td>{_action_badge(act, view.display_label)}</td>"
        f'<td style="text-align:right">{_esc(_score(rec))}</td>'
        f'<td style="text-align:right">{_esc(conf_label)}</td>'
        f"<td>{_esc(track) if track else '—'}</td>"
        f"</tr>"
    )


def _technical_factor_html(rec: dict) -> str:
    """legacy-b style chart-shape read (technical_factors.py): moving averages,
    52-week proximity, breakout/candle patterns, MDD -- reduced to a handful of
    composite scores. None when there wasn't enough price history to compute it."""
    track = rec.get("technical_track")
    if not track:
        return ""
    track_cls = "badge-warn" if track == "Avoid / Weak" else "badge-neutral"
    chips = [f'<span class="badge {track_cls}">{_esc(track)}</span>']

    def _pt(label: str, key: str) -> str:
        val = rec.get(key)
        if val is None:
            return ""
        try:
            return f'<span class="badge badge-neutral">{label} {float(val):.0f}</span>'
        except (TypeError, ValueError):
            return ""

    chips.append(_pt("모멘텀", "momentum_score"))
    chips.append(_pt("리더", "leader_score"))
    chips.append(_pt("매수적기", "buyable_score"))
    chips.append(_pt("돌파", "breakout_score"))
    chips.append(_pt("과열위험", "top_risk_score"))
    chips = [c for c in chips if c]
    return f'<div class="chips" style="margin:0.35rem 0 0">{"".join(chips)}</div>'


def _research_summary_html(rec: dict) -> str:
    """Gemini research + adversarial pass (research_agent.research_ticker), revived
    2026-09-05. Only set for NEW-entry candidates that were actually researched (see
    v1_cycle.py's research loop) -- held names and BTC never get this."""
    summary = str(rec.get("research_summary_ko") or "").strip()
    mult = rec.get("research_sizing_multiplier")
    scores: list[str] = []
    for label, key in (
        ("리레이팅", "research_rerating_score"),
        ("밸류에이션지지", "research_valuation_support_score"),
        ("현금대비매력", "research_cash_relative_score"),
        ("데이터품질", "research_data_quality_score"),
    ):
        val = rec.get(key)
        if isinstance(val, (int, float)):
            scores.append(f'<span class="badge badge-neutral">{label} {float(val):.0f}</span>')
    if not summary and not scores:
        return ""
    mult_bit = ""
    try:
        if mult is not None and abs(float(mult) - 1.0) > 1e-6:
            cls = "badge-ok" if float(mult) > 1.0 else "badge-warn"
            mult_bit = f'<span class="badge {cls}">사이징 {float(mult):.2f}x</span>'
    except (TypeError, ValueError):
        pass
    parts = [
        '<div style="margin:0.5rem 0 0">',
        '<p class="hint" style="margin:0 0 0.3rem"><strong>Gemini 리서치</strong></p>',
    ]
    if summary:
        parts.append(f'<p class="hint" style="margin:0 0 0.3rem">{_esc(summary)}</p>')
    if scores or mult_bit:
        parts.append(f'<div class="chips">{"".join(scores)}{mult_bit}</div>')
    parts.append("</div>")
    return "".join(parts)


_CONN_KO: dict[str, tuple[str, str]] = {
    STATUS_OK: ("연결완료", "badge-ok"),
    "NOT_CONFIGURED": ("미설정", "badge-neutral"),
    "AUTH_REQUIRED": ("인증 실패", "badge-err"),
    "RATE_LIMITED": ("요청 한도", "badge-warn"),
    "QUOTA_EXCEEDED": ("쿼터 초과", "badge-err"),
    "UNAVAILABLE": ("연결 실패", "badge-err"),
}


def _allocation_weight_text(alloc: dict, bucket: str) -> str:
    if alloc.get("weights_confirmed") is False:
        return "미확정"
    value = sum((alloc.get("stock_weights") or {}).values()) if bucket == "stock" else alloc.get(f"{bucket}_weight")
    return _pct(value)


def _allocation_summary(alloc: dict) -> str:
    text = " · ".join(f"{label} {_allocation_weight_text(alloc, key)}" for key, label in
                      (("stock", "주식"), ("btc", "BTC"), ("cash", "현금")))
    if alloc.get("weights_confirmed") is False:
        known = alloc.get("known_stock_units") or {}
        text += " · 전체 비중·합계 미확정"
        if known:
            text += f" · 평가 확인된 주식 {len(known)}종목 합계 {sum(known.values()):g}단위 (전체 합계 아님)"
    return f'<p class="hint">적용 배분: {_esc(text)}</p>'


def _deployment_card(alloc: dict) -> str:
    if alloc.get("weights_confirmed") is False:
        return _allocation_summary(alloc)
    note = str(alloc.get("deployment_note") or "").strip()
    if not note and not str(alloc.get("formula_version") or "").startswith("quant_alloc_v"):
        return ""
    if not note:
        return ""
    max_w = alloc.get("max_total_stock_weight")
    used = alloc.get("equity_budget_used")
    budget = alloc.get("equity_budget")
    factor = alloc.get("deployment_factor")
    cash_label = str(alloc.get("residual_cash_label") or "").strip()
    strong = alloc.get("strong_opportunity") if isinstance(alloc.get("strong_opportunity"), dict) else {}
    bits: list[str] = []
    appetite = str(alloc.get("risk_appetite") or "").strip()
    if appetite:
        bits.append(
            {"aggressive": "성향 공격적", "balanced": "성향 균형", "conservative": "성향 보수적"}.get(
                appetite, f"성향 {appetite}"
            )
        )
    alpha = alloc.get("sizing_alpha")
    if alpha is not None:
        try:
            bits.append(f"비중지수 {float(alpha):.1f}")
        except (TypeError, ValueError):
            pass
    if factor is not None:
        try:
            bits.append(f"투입계수 {float(factor):.2f}")
        except (TypeError, ValueError):
            pass
    if max_w is not None:
        try:
            bits.append(f"주식 한도 {float(max_w) * 100:.0f}%")
        except (TypeError, ValueError):
            pass
    if budget is not None:
        try:
            bits.append(f"주식 목표 {float(budget) * 100:.1f}%")
        except (TypeError, ValueError):
            pass
    if used is not None:
        try:
            bits.append(f"실제 주식 {float(used) * 100:.1f}%")
        except (TypeError, ValueError):
            pass
    kept = alloc.get("held_keep_weight")
    if kept is not None:
        try:
            if float(kept) > 1e-9:
                bits.append(f"기존 보유 유지 {float(kept) * 100:.1f}%")
        except (TypeError, ValueError):
            pass
    why = str(strong.get("why") or "")
    src = str(strong.get("source") or "")
    used_score = strong.get("used_score")
    mean_ret = strong.get("strong_horizon_mean_return")
    strong_bits: list[str] = []
    if mean_ret is not None:
        try:
            strong_bits.append(f"강한 기회 기준 기대수익 {float(mean_ret) * 100:.0f}%")
        except (TypeError, ValueError):
            pass
    if used_score is not None:
        try:
            strong_bits.append(f"강한 기회점수 {float(used_score):.4f}")
        except (TypeError, ValueError):
            pass
    if src:
        strong_bits.append(src)
    html = (
        f'<div class="card"><p style="margin:0;font-weight:600">{_esc(note)}</p>'
    )
    if cash_label:
        html += f'<p class="hint" style="margin:0.35rem 0 0">{_esc(cash_label)}</p>'
    if bits:
        html += (
            f'<p class="hint" style="margin:0.35rem 0 0">{_esc(" · ".join(bits))}</p>'
        )
    if strong_bits:
        html += (
            f'<p class="hint" style="margin:0.25rem 0 0">{_esc(" · ".join(strong_bits))}</p>'
        )
    if why:
        html += f'<p class="hint" style="margin:0.25rem 0 0">{_esc(why)}</p>'
    html += "</div>"
    return html


def _fmt_trace_ret(v: object) -> str:
    try:
        return f"{float(v) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_trace_num(v: object, digits: int = 4) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _trace_table_row(row: dict) -> str:
    passed = bool(row.get("pass_cutoff"))
    return (
        f'<tr class="{"trace-pass" if passed else "trace-fail"}">'
        f'<td class="trace-ticker">{_esc(row.get("ticker") or "")}</td>'
        f'<td>{_esc(_fmt_trace_ret(row.get("pred_5d")))}</td>'
        f'<td>{_esc(_fmt_trace_ret(row.get("pred_10d")))}</td>'
        f'<td>{_esc(_fmt_trace_ret(row.get("pred_20d")))}</td>'
        f'<td>{_esc(_fmt_trace_num(row.get("vol"), 3))}</td>'
        f'<td>{_esc(_fmt_trace_num(row.get("opp_score")))}</td>'
        f'<td>{_esc(_fmt_trace_num(row.get("conviction"), 2))}</td>'
        f'<td>{"Y" if passed else "N"}</td>'
        f'<td>{_esc(_pct(row.get("final_weight")))}</td>'
        f"</tr>"
    )


def _allocation_trace_html(alloc: dict, recs: list[dict]) -> str:
    if alloc.get("weights_confirmed") is False:
        return ""
    trace = resolve_allocation_trace(alloc, recs)
    if not trace.get("universe") and not trace.get("rows"):
        return ""
    score_p = trace.get("score_percentiles") if isinstance(trace.get("score_percentiles"), dict) else {}
    mean_p = (
        trace.get("mean_return_percentiles")
        if isinstance(trace.get("mean_return_percentiles"), dict)
        else {}
    )
    diag = trace.get("diagnosis") if isinstance(trace.get("diagnosis"), dict) else {}
    notes = diag.get("notes") if isinstance(diag.get("notes"), list) else []
    cutoff = trace.get("cutoff")
    rows = [r for r in (trace.get("rows") or []) if isinstance(r, dict)]
    passed = [r for r in rows if r.get("pass_cutoff")]
    note_html = "".join(f"<li>{_esc(n)}</li>" for n in notes if n)
    passed_body = "".join(_trace_table_row(r) for r in passed)
    all_body = "".join(_trace_table_row(r) for r in rows)
    log = str(trace.get("log_text") or "")
    equiv = _esc(_fmt_trace_num(trace.get("equivalent_names"), 2))
    budget_label = _esc(_pct(trace.get("equity_budget")))
    return f"""
        <details class="card allocation-trace">
          <summary>배분 추적 — 예측값에서 주식 {budget_label}까지</summary>
          <p class="hint" style="margin:0.45rem 0 0.7rem">
            <code>pred_5d/10d/20d</code>는 기대수익률(소수, +3%면 0.03)입니다. 확률이나 z-score가 아닙니다.
            기회점수 = 세 기간 평균 × 신뢰도 × 합의배수(동의 1.15 / 충돌 0.7). vol은 더 이상 나누지 않습니다
            (2026-09-04, 실사용 스캔에서 변동성 페널티가 강한 모멘텀 종목의 97%를 부당하게 걸러냄).
            컷오프 {_esc(_fmt_trace_num(cutoff))} 통과 뒤,
            확신도 = clip((점수 − 컷오프) / (강한점수 − 컷오프), 0, 1) 이고
            주식 예산 = 최대주식여유 × min(1, 숏리스트 확신도 합 / {equiv}).
            종목 비중은 기회점수가 아니라 컷오프 초과분^{_esc(_fmt_trace_num(trace.get("sizing_alpha"), 1))} 입니다.
            성향 {_esc(str(trace.get("risk_appetite") or ""))}. 통과 종목 수와 실제 투자 비중은 다릅니다.
            매수 컷오프 미달 보유는 5/10/20 평균 기대수익이 음수일 때만 청산합니다.
          </p>
          <div class="trace-stats">
            <div class="row"><span>유니버스</span><span>{int(trace.get("universe") or 0)}</span></div>
            <div class="row"><span>원시 기대수익 중앙값</span><span>{_esc(_fmt_trace_ret(mean_p.get("p50")))}</span></div>
            <div class="row"><span>원시 기대수익 p90 / max</span>
              <span>{_esc(_fmt_trace_ret(mean_p.get("p90")))} / {_esc(_fmt_trace_ret(mean_p.get("max")))}</span></div>
            <div class="row"><span>기회점수 중앙값</span><span>{_esc(_fmt_trace_num(score_p.get("p50")))}</span></div>
            <div class="row"><span>기회점수 p75 / p90 / p95 / max</span>
              <span>{_esc(_fmt_trace_num(score_p.get("p75")))} / {_esc(_fmt_trace_num(score_p.get("p90")))} /
              {_esc(_fmt_trace_num(score_p.get("p95")))} / {_esc(_fmt_trace_num(score_p.get("max")))}</span></div>
            <div class="row"><span>컷오프</span><span>{_esc(_fmt_trace_num(cutoff))}</span></div>
            <div class="row"><span>컷오프 통과</span><span>{int(trace.get("passed_cutoff") or 0)}</span></div>
            <div class="row"><span>확신도 합 / 강한 4개 기준</span>
              <span>{_esc(_fmt_trace_num(trace.get("aggregate_conviction")))} / {equiv}</span></div>
            <div class="row"><span>상위 4개 확신도 합</span>
              <span>{_esc(_fmt_trace_num(trace.get("top4_conviction_sum")))}</span></div>
            <div class="row"><span>투입계수</span>
              <span>{_esc(_fmt_trace_num(trace.get("deployment_factor")))}</span></div>
            <div class="row"><span>기존 보유 유지 / 새 매수 예산</span>
              <span>{_esc(_pct(trace.get("held_keep_weight")))} / {_esc(_pct(trace.get("new_name_budget")))}</span></div>
            <div class="row"><span>주식 / BTC / 현금</span>
              <span>{_esc(_pct(trace.get("stock_weight")))} / {_esc(_pct(trace.get("btc_weight")))} /
              {_esc(_pct(trace.get("cash_weight")))}</span></div>
          </div>
          {"<ul class='hint' style='margin:0.7rem 0 0;padding-left:1.1rem'>" + note_html + "</ul>" if note_html else ""}
          <h3 style="margin:1rem 0 0.4rem;font-size:0.92rem">컷오프 통과 종목</h3>
          <div class="trace-scroll">
            <table class="trace-table">
              <thead>
                <tr>
                  <th>ticker</th><th>pred_5d</th><th>pred_10d</th><th>pred_20d</th>
                  <th>vol</th><th>opp_score</th><th>conv</th><th>pass_0.01</th><th>weight</th>
                </tr>
              </thead>
              <tbody>{passed_body or '<tr><td colspan="9">통과 종목 없음</td></tr>'}</tbody>
            </table>
          </div>
          <details class="tech-details" style="margin-top:0.85rem">
            <summary>유니버스 전체 ({len(rows)})</summary>
            <div class="trace-scroll trace-scroll-tall">
              <table class="trace-table">
                <thead>
                  <tr>
                    <th>ticker</th><th>pred_5d</th><th>pred_10d</th><th>pred_20d</th>
                    <th>vol</th><th>opp_score</th><th>conv</th><th>pass_0.01</th><th>weight</th>
                  </tr>
                </thead>
                <tbody>{all_body}</tbody>
              </table>
            </div>
          </details>
          <details class="tech-details" style="margin-top:0.5rem">
            <summary>텍스트 로그</summary>
            <pre class="trace-log">{_esc(log)}</pre>
          </details>
        </details>
    """


def _portfolio_committee_html(snap: dict) -> str:
    committee = snap.get("portfolio_committee")
    if not isinstance(committee, dict):
        return ""
    status = str(committee.get("status") or "NOT_RUN")
    if status == "NOT_RUN":
        return ""
    if status != "AVAILABLE":
        note = committee.get("note") or committee.get("error") or "종목별 최종 판단을 유지합니다."
        return (
            '<div class="card">'
            f'<div class="row" style="padding-top:0"><strong>포트폴리오 최종 심사</strong>{_badge(status)}</div>'
            f'<p class="hint" style="margin:0.45rem 0 0">{_esc(_text_ko(note))}</p>'
            "</div>"
        )

    final_text = str(committee.get("final_text") or "").strip()
    if not final_text:
        return (
            '<div class="card">'
            f'<div class="row" style="padding-top:0"><strong>포트폴리오 최종 심사</strong>{_badge(status)}</div>'
            '<p class="hint" style="margin:0.45rem 0 0">GPT 응답이 비어 있습니다.</p>'
            "</div>"
        )
    report_basis = (
        "보유 평가 미확정으로 전체 배분 제약 검증과 실행은 불가합니다. 아래 수치는 실행 가능한 배분이 아닙니다."
        if (snap.get("allocation") or {}).get("weights_confirmed") is False else
        "검증된 구조화 판단과 배분 제약을 적용한 결과입니다. 아래 카드와 같은 저장 결과를 사용합니다."
        if committee.get("validated") else
        "과거 서술형 응답입니다. 아래 수치 추천과 연결되지 않은 기록일 수 있습니다."
    )
    return f"""
        <div class="card portfolio-committee">
          <div class="row" style="padding-top:0">
            <strong>GPT 최종 투자 판단</strong>{_badge(status)}
          </div>
          <p class="hint" style="margin:0.45rem 0 0">
            {report_basis}
          </p>
          <div class="trace-scroll gpt-report-scroll">
            <div class="gpt-report">{_render_markdown_safe(final_text)}</div>
          </div>
        </div>
    """


def render_recommendations_html(snap: dict, *, settings: Settings | None = None) -> str:
    cfg = settings or get_settings()
    is_demo = snap.get("mode") == "DEMO / SYNTHETIC"
    lead = ("DEMO: 가상 종목 3개의 예제 응답을 실제 검증·저장 코드로 처리했습니다. 실제 모델 예측이나 API 응답이 아닙니다."
            if is_demo else "Quant는 후보와 숫자를 산출합니다. GPT의 검증된 최종 판단을 적용하며, 미적용 결과는 별도로 표시합니다. 주문은 넣지 않습니다.")
    base = _snap_base(snap)
    q_rows = [
        r for r in (snap.get("effective") if snap.get("decision_contract") else snap.get("quant")) or []
        if "btc" not in str(r.get("instrument_id")).lower()
    ]
    l_map = _llm_map(
        [r for r in snap.get("llm_final") or [] if "btc" not in str(r.get("instrument_id")).lower()]
    )
    buy = [
        r
        for r in q_rows
        if _is_buy_candidate(
            r,
            llm=l_map.get(str(r.get("instrument_id"))),
            settings=cfg,
            total_base_units=base,
        )
    ]
    def final_sort_key(r):
        judged = l_map.get(str(r.get("instrument_id"))) or {}
        rank = judged.get("final_rank")
        target = _overlay_for_display(r, judged).get("recommended_units")
        return (rank is None, float(rank or 0), -_num_or_zero(target), -_num_or_zero(r.get("confidence")))
    buy.sort(key=final_sort_key)
    def source_label(r):
        if snap.get("decision_contract"):
            return "GPT 최종" if str(r.get("instrument_id")) in l_map else "Gemini 반영 · GPT 미적용"
        return "Quant"
    held_ids = {str(r.get("instrument_id")) for r in buy}
    held = [
        r
        for r in q_rows
        if _is_holding(r) and str(r.get("instrument_id")) not in held_ids
    ]
    buy_cards = []
    today_n = 0
    for i, rec in enumerate(buy, start=1):
        llm = l_map.get(str(rec.get("instrument_id")))
        today = _today_priority(rec, llm, settings=cfg, total_base_units=base)
        if today:
            today_n += 1
        buy_cards.append(
            _rec_card(rec, source_label(rec), rank=(llm or {}).get("final_rank") or i, llm=llm, today=today, settings=cfg, total_base_units=base)
        )
    held_cards = "".join(
        _rec_card(
            r,
            source_label(r),
            llm=l_map.get(str(r.get("instrument_id"))),
            settings=cfg,
            total_base_units=base,
        )
        for r in held
    )
    # Names genuinely worth a full card: sized (held or newly recommended), judged
    # by the LLM, researched by Gemini, or a real near-miss (BELOW_MIN_POSITION means
    # it cleared the opportunity-score cutoff and got a nonzero pre-floor weight
    # before the position floor zeroed it -- a real "almost made it", not a blanket
    # rejection). Everything else collapses into a table row.
    #
    # Two fields turned out NOT to be usable signals here, despite looking like
    # "this name has something to say": `thesis` is set on every single scored name
    # to an internal diagnostic string (allocation.py's "agreement=agree
    # formula=quant_alloc_v8 lambdarank_tiebreak=0.1234"), and `allocation_note` is
    # set for EVERY exclusion reason via _note_for_exclusion (allocation.py), so most
    # of the ~500 names -- overwhelmingly BELOW_MIN_OPPORTUNITY, which just means "not
    # close at all" -- get one too. Treating either as "interesting" put every name
    # back in the full card and defeated the whole split (~500 cards, ~700,000px
    # page). See _rec_row_compact's docstring for the DOM-weight problem this fixes.
    all_detail_cards_list: list[str] = []
    all_rows_list: list[str] = []
    for r in q_rows:
        llm = l_map.get(str(r.get("instrument_id")))
        has_size = (
            _num_or_zero(r.get("recommended_units")) > 1e-9 or _num_or_zero(r.get("current_units")) > 1e-9
        )
        near_miss = r.get("exclusion_reason") == "BELOW_MIN_POSITION"
        if r.get("research_summary_ko") or llm is not None or has_size or near_miss:
            all_detail_cards_list.append(_rec_card(r, source_label(r), llm=llm, settings=cfg, total_base_units=base))
        else:
            all_rows_list.append(_rec_row_compact(r, llm=llm, settings=cfg, total_base_units=base))
    all_detail_cards = "".join(all_detail_cards_list)
    all_rows = "".join(all_rows_list)
    judge = snap.get("llm_judge_status", "UNKNOWN")
    quant_st = snap.get("quant_status", "UNKNOWN")
    alloc = snap.get("allocation") if isinstance(snap.get("allocation"), dict) else {}
    deploy_html = _deployment_card(alloc)
    trace_html = _allocation_trace_html(alloc, q_rows)
    if snap.get("decision_contract"):
        # Every current-contract view uses the effective allocation, including fallback.
        deploy_html = _allocation_summary(alloc)
        trace_html = ""
    today_hint = (
        "오늘 우선은 검증된 GPT 판단이 매수·추가이며 실행 가능한 종목입니다. 주문이 나가지는 않습니다."
        if snap.get("decision_contract") else
        "오늘 우선은 실행 가능하고, Quant와 LLM이 모두 매수·추가인 종목입니다. 주문이 나가지는 않습니다."
    )
    committee_html = _portfolio_committee_html(snap)
    buy_block = (
        "".join(buy_cards)
        if buy_cards
        else '<p class="muted">이번 스캔에서 비중을 늘리라고 나온 종목이 없습니다.</p>'
    )
    held_block = (
        f'<h2 style="margin:1.25rem 0 0.6rem">보유 종목</h2>{held_cards}' if held_cards else ""
    )
    return f"""
        <h1>종목 추천</h1>
        <p class="lead">{lead}</p>
        <div class="card">
          <div class="row" style="padding-top:0"><span>Quant</span>{_badge(quant_st)}</div>
          <div class="row"><span>{'GPT 예제 형식 검증 · API 호출 없음' if is_demo else 'GPT 최종 판단'}</span>{_badge(judge)}</div>
          <div class="row"><span>오늘 우선</span><span>{today_n}개</span></div>
        </div>
        {committee_html}
        {deploy_html}
        {trace_html}
        <div class="field">
          <label for="ticker-search">종목 검색</label>
          <input id="ticker-search" type="search" placeholder="티커 일부 (A → AA → AAPL)" autocomplete="off"/>
          <p class="hint">입력한 글자가 티커에 포함된 종목만 남습니다. A, AA, AAO처럼 글자를 늘리면 더 좁혀집니다.</p>
        </div>
        <div class="chips" id="rec-filters">
          <button type="button" class="chip on" data-filter="all">전체</button>
          <button type="button" class="chip" data-filter="buy">매수 후보</button>
          <button type="button" class="chip" data-filter="neutral">중립</button>
        </div>
        <h2 style="margin:0.4rem 0 0.6rem">매수 후보</h2>
        <p class="hint" style="margin:0 0 0.75rem">{today_hint}</p>
        <div id="buy-list">{buy_block}</div>
        {held_block}
        <h2 style="margin:1.25rem 0 0.6rem">전체 종목 점수</h2>
        <p class="hint" style="margin:0 0 0.6rem">Quant가 스캔한 전체 종목입니다. 위 검색·필터가 여기도 함께 적용됩니다.</p>
        {all_detail_cards}
        <div id="rec-list">{
            f'''<div class="trace-scroll trace-scroll-tall">
              <table class="trace-table rec-table">
                <thead><tr>
                  <th>티커</th><th>액션</th><th style="text-align:right">점수</th>
                  <th style="text-align:right">신뢰도</th><th>기술 트랙</th>
                </tr></thead>
                <tbody>{all_rows}</tbody>
              </table>
            </div>''' if all_rows else '' if all_detail_cards else '<p class="muted">아직 추천이 없습니다. 스캔을 실행하세요.</p>'
        }</div>
        <script>
          const chips = document.querySelectorAll("#rec-filters .chip");
          const search = document.getElementById("ticker-search");
          function applyFilters() {{
            const f = document.querySelector("#rec-filters .chip.on")?.dataset.filter || "all";
            const q = (search.value || "").trim().toLowerCase();
            document.querySelectorAll(".rec-card, .rec-row").forEach((el) => {{
              const ticker = el.dataset.ticker || "";
              const okType = (f === "all" || el.dataset.filter === f);
              const okSearch = !q || ticker.includes(q);
              el.style.display = (okType && okSearch) ? "" : "none";
            }});
          }}
          chips.forEach((chip) => chip.addEventListener("click", () => {{
            chips.forEach((c) => c.classList.remove("on"));
            chip.classList.add("on");
            applyFilters();
          }}));
          search.addEventListener("input", applyFilters);
        </script>
        """


def _btc_effective_cards(snap: dict, *, settings: Settings | None = None) -> str:
    """Render one authoritative BTC stage. Other stages are stored, not shown in a comparison UI."""
    rows = [
        r for r in (snap.get("effective") if snap.get("decision_contract") else snap.get("quant")) or []
        if "btc" in str(r.get("instrument_id")).lower()
    ]
    return "".join(
        _rec_card(
            r,
            "GPT 최종" if r.get("source") == "llm_final" else (
                "Gemini 반영 · GPT 미적용" if snap.get("decision_contract") else "Quant"
            ),
            llm=r if r.get("source") == "llm_final" else None,
            settings=settings,
            total_base_units=_snap_base(snap),
        )
        for r in rows
    )


def _conn_badge(key: str, status: str | None) -> str:
    label, cls = _CONN_KO.get(status or "", ("미확인", "badge-neutral"))
    return f'<span class="badge {cls}" data-conn="{_esc(key)}">{_esc(label)}</span>'


def _conn_rows_html(report: ConnectionReport | None) -> str:
    if report is None:
        return (
            '<p class="muted" style="margin:0">아래 버튼을 누르면 모든 API 연결을 한 번에 확인합니다.</p>'
        )
    rows = []
    for r in report.results:
        label, cls = _CONN_KO.get(r.status, ("미확인", "badge-neutral"))
        usage = f'<p class="hint" style="margin:0.15rem 0 0">{_esc(r.usage)}</p>' if r.usage else ""
        rows.append(
            f'<div class="row"><div><p style="margin:0;font-size:0.875rem;font-weight:500">{_esc(r.label)}</p>'
            f'<p class="hint" style="margin:0.15rem 0 0">{_esc(r.detail)}</p>{usage}</div>'
            f'<span class="badge {cls}">{_esc(label)}</span></div>'
        )
    return "".join(rows)


def _secret_field(
    name: str,
    label: str,
    env: dict[str, str],
    hint: str = "",
    *,
    conn_key: str | None = None,
    conn_status: str | None = None,
) -> str:
    hint_html = f'<p class="hint">{hint}</p>' if hint else ""
    if env.get(name):
        conn_html = _conn_badge(conn_key, conn_status) if conn_key else ""
        return (
            f'<div class="field" data-secret-field>'
            f"<label>{_esc(label)}</label>"
            f'<div class="secret-row">'
            f'<span class="badge badge-ok">입력완료</span>'
            f"{conn_html}"
            f'<button type="button" class="btn-ghost js-reveal-secret">수정</button>'
            f"</div>"
            f'<input name="{_esc(name)}" type="password" class="js-secret-input" '
            f'style="display:none" autocomplete="off" placeholder="새 값만 입력하면 덮어씁니다"/>'
            f"{hint_html}</div>"
        )
    return (
        f'<div class="field"><label>{_esc(label)}</label>'
        f'<input name="{_esc(name)}" type="password" autocomplete="off"/>'
        f"{hint_html}</div>"
    )


def create_app() -> FastAPI:
    server_started_at = datetime.now(timezone.utc)
    app = FastAPI(title="Investment Assistant", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def reject_cross_site_mutations(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Keep untrusted web pages from submitting forms to the local service.

        CORS protects JavaScript from reading responses, but it does not prevent a
        normal cross-origin HTML form from sending a POST to localhost.  The UI has
        several POST-only controls, so reject browser-originated writes unless they
        come from this server or the supported Vite development server.  Requests
        without browser origin metadata remain available to local CLI/test clients.
        """
        if request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
            fetch_site = (request.headers.get("sec-fetch-site") or "").strip().lower()
            if fetch_site == "cross-site":
                return JSONResponse({"detail": "cross-site request rejected"}, status_code=403)

            origin = (request.headers.get("origin") or "").strip()
            if origin:
                parsed = urlsplit(origin)
                request_host = (request.headers.get("host") or "").strip().lower()
                same_origin = (
                    parsed.scheme.lower() in {"http", "https"}
                    and parsed.netloc.lower() == request_host
                )
                vite_origin = origin.rstrip("/").lower() in {
                    "http://127.0.0.1:5173",
                    "http://localhost:5173",
                }
                if not (same_origin or vite_origin):
                    return JSONResponse({"detail": "untrusted origin"}, status_code=403)

        return await call_next(request)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    cache: dict[str, object] = {"last": None}
    scan: dict[str, object] = {
        "status": "idle",
        "error": None,
        "latest": "",
        "mode": "scan",
        "started_at": None,
        "step_started_at": None,
        "step_key": "",
    }
    scan_lock = threading.Lock()
    conn_state: dict[str, ConnectionReport | None] = {"report": None}

    def _conn_status_map() -> dict[str, str]:
        report = conn_state.get("report")
        return {r.key: r.status for r in report.results} if report else {}

    def _preflight() -> tuple[str, list[str]]:
        """(kind, messages) for the run button. Empty kind means nothing to warn about."""
        report = conn_state.get("report")
        env = read_env_map(env_path(PROJECT_ROOT))
        configured = [k for k in ("ALPACA_API_KEY", "OPENAI_API_KEY") if (env.get(k) or "").strip()]
        if report is None:
            if not configured:
                return "", []
            return "unchecked", ["API 연결을 아직 확인하지 않았습니다. 실행 전에 설정에서 연결 확인을 한 번 눌러 주세요."]
        failures = [f"{r.label}: {r.detail}" for r in report.failures]
        return ("failed", failures) if failures else ("", [])

    def live_settings() -> Settings:
        return get_settings()

    def _prepare_universe(conn, cfg: Settings):
        return prepare_scan_universe(conn, cfg)

    def _scan_copy() -> dict[str, object]:
        with scan_lock:
            return dict(scan)

    def _begin_scan(*, mode: str, latest: str) -> bool:
        """Mark the scan as running. Returns False if another run is already in flight."""
        now = time.monotonic()
        with scan_lock:
            if scan["status"] == "running":
                return False
            scan["status"] = "running"
            scan["error"] = None
            scan["mode"] = mode
            scan["latest"] = latest
            scan["started_at"] = now
            scan["step_started_at"] = now
            scan["step_key"] = latest
        return True

    def _elapsed_fields(*, running: bool, latest: str | None = None) -> dict[str, object]:
        now = time.monotonic()
        with scan_lock:
            if latest is not None and running and latest != scan.get("step_key"):
                scan["step_key"] = latest
                scan["step_started_at"] = now
                scan["latest"] = latest
            started = scan.get("started_at")
            step = scan.get("step_started_at")
        if not running or started is None:
            total = 0.0
            step_s = 0.0
        else:
            total = max(0.0, now - float(started))
            step_s = max(0.0, now - float(step if step is not None else started))
        return {
            "elapsed_total_s": round(total, 1),
            "elapsed_step_s": round(step_s, 1),
            "elapsed_total": format_elapsed_ko(total),
            "elapsed_step": format_elapsed_ko(step_s),
        }

    def _humanize_scan_hint(message: str, *, running: bool) -> str:
        msg = (message or "").strip()
        if msg.startswith("Alpaca bars="):
            n = msg.split("=", 1)[1]
            if running:
                return (
                    f"시세 저장은 끝났습니다 ({n}봉). "
                    "지금은 피처 계산·학습 중이라 이 문구가 한동안 그대로일 수 있습니다."
                )
            return f"시세 저장 완료 ({n}봉)"
        if msg == "Alpaca history backfill":
            return "Alpaca에서 시세를 받는 중입니다."
        if msg.startswith("Gemini research n="):
            n = msg.split("=", 1)[1]
            return f"Gemini 리서치 대상 {n}종목"
        return msg or "실행 중…"

    def _latest_runtime_hint(*, running: bool = False) -> str:
        try:
            store = _store(writer=False)
            try:
                row = store.conn.execute(
                    "SELECT message FROM runtime_events ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            finally:
                store.close()
            raw = str(row[0]) if row and row[0] else "실행 중…"
        except Exception:  # noqa: BLE001 — UI hint only
            raw = "실행 중…"
        return _humanize_scan_hint(raw, running=running)

    def _resumable_scan_hint() -> str | None:
        """When the last GPT judge attempt ended mid-conversation (e.g. an OpenAI
        429) and is still inside the current ET session's resume window, tell
        the user up front that the next scan will pick up from there instead of
        paying for the whole conversation again -- see resume_session_bounds_utc /
        _load_resumable_turns in openai_judge.py, which this only mirrors for
        display (the real resume decision is made turn-by-turn during the scan)."""
        try:
            store = _store(writer=False)
        except Exception:  # noqa: BLE001 — hint only
            return None
        try:
            session_start, session_end = resume_session_bounds_utc()
            last = store.conn.execute(
                "SELECT status, created_at FROM llm_transcripts "
                "WHERE kind = 'portfolio_judge' AND created_at >= ? AND created_at < ? "
                "ORDER BY created_at DESC LIMIT 1",
                [session_start, session_end],
            ).fetchone()
            if not last or str(last[0] or "") == STATUS_AVAILABLE:
                return None
            # Distinct STAGE names with at least one success in the window, across
            # possibly several retries -- an upper bound on what could be reused, not
            # a promise. The real check (_load_resumable_turns) requires the exact
            # same prompt text stage-by-stage from one specific attempt, which needs
            # the actual candidate list to compare against -- too expensive to
            # recompute just to render this hint, so this stays a rough "at most N".
            done = store.conn.execute(
                "SELECT COUNT(DISTINCT kind) FROM llm_transcripts WHERE kind LIKE 'legacy_b_%' "
                "AND ticker = 'PORTFOLIO' AND status = ? AND created_at >= ? AND created_at < ?",
                [STATUS_AVAILABLE, session_start, session_end],
            ).fetchone()
            n = int(done[0]) if done else 0
            if n <= 0:
                return None
            return (
                f"직전 GPT 대화가 도중에 중단됐습니다({_esc(str(last[0]))}), 최대 {n}턴은 이미 끝난 적 있습니다. "
                "지금 스캔을 실행하면 후보 목록이 그때와 같은 부분까지는 다시 호출하지 않고 이어서 진행합니다 "
                "(후보가 바뀌었으면 그 지점부터는 자동으로 처음부터 다시 진행됩니다)."
            )
        except Exception:  # noqa: BLE001 — hint only
            return None
        finally:
            store.close()

    def _auto_slot_path() -> Path:
        cfg = live_settings()
        return cfg.duckdb_path if cfg.duckdb_path.is_absolute() else PROJECT_ROOT / cfg.duckdb_path

    def _refund_auto_slot(*, train: bool) -> None:
        """Give today's auto slot back after an attempt that did no work.

        The slot is claimed before the run starts (so a crash mid-LLM-call can't
        loop), which on 2026-09-04 meant one hung scan silently cost the whole
        day. Refunding on failure re-opens it, still bounded by
        MAX_AUTO_ATTEMPTS so a consistently failing job can't retry forever.
        """
        from trading_system.daily_scan import (
            MARKET_TZ,
            release_daily_train,
            release_session_run,
        )

        try:
            db = _auto_slot_path()
            session = datetime.now(timezone.utc).astimezone(MARKET_TZ).date()
            if train:
                release_daily_train(db, session)
            else:
                release_session_run(db, session)
        except Exception:  # noqa: BLE001 — refund is best effort, never break the run
            pass

    def _run_scan_worker(
        *,
        retrain: bool = False,
        train_only: bool = False,
        enforce_llm_budget: bool = False,
        auto: bool = False,
    ) -> None:
        err: str | None = None
        try:
            cfg = live_settings()
            store = _store(writer=True, stale_seconds=max(cfg.writer_lease_stale_seconds, 900))
            try:
                out = run_v1_cycle(
                    store,
                    cfg,
                    artifacts_dir=PROJECT_ROOT / "data" / "artifacts",
                    universe=_prepare_universe(store.conn, cfg),
                    retrain=retrain,
                    train_only=train_only,
                    enforce_llm_budget=enforce_llm_budget,
                )
                # train_only never produces a new recommendation set (see run_v1_cycle's
                # early return) -- leave the last real scan's cache exactly as it was,
                # instead of blanking the recommendations page for a run that has none.
                if not train_only:
                    cache["last"] = out
            finally:
                store.close()
        except WriterLeaseBusy:
            err = "다른 실행이 데이터베이스를 사용 중입니다. 잠시 후 다시 시도하세요."
        except Exception as exc:  # noqa: BLE001 — never 500 the browser
            traceback.print_exc()
            err = f"스캔 중 오류가 발생했습니다 ({type(exc).__name__}). 런타임 페이지를 확인하세요."
        if err and auto:
            _refund_auto_slot(train=train_only)
        with scan_lock:
            scan["status"] = "error" if err else "done"
            scan["error"] = err
            if not err:
                if train_only:
                    scan["latest"] = "모델 학습이 끝났습니다 (API 호출 없음)."
                elif retrain:
                    scan["latest"] = "학습이 끝났습니다."
                else:
                    scan["latest"] = "스캔이 끝났습니다."

    def snapshot(*, compute: bool = False) -> dict:
        if cache.get("last"):
            return cache["last"]  # type: ignore[return-value]
        empty = {
            "allocation": {},
            "capabilities": {},
            "quant": [],
            "llm_final": [],
        }
        if not compute:
            try:
                store = _store(writer=False)
                try:
                    loaded = load_last_ui_snapshot(store.conn)
                finally:
                    store.close()
            except Exception:  # noqa: BLE001 — empty board is better than a 500
                loaded = None
            if loaded:
                cache["last"] = loaded
                return loaded
            return empty
        cfg = live_settings()
        store = _store(writer=True, stale_seconds=max(cfg.writer_lease_stale_seconds, 900))
        try:
            cache["last"] = run_v1_cycle(
                store,
                cfg,
                artifacts_dir=PROJECT_ROOT / "data" / "artifacts",
                universe=_prepare_universe(store.conn, cfg),
            )
        finally:
            store.close()
        return cache["last"]  # type: ignore[return-value]

    def page(request: Request, title: str, body: str) -> HTMLResponse:
        env = read_env_map(env_path(PROJECT_ROOT))
        missing = missing_provider_keys(env)
        banner = ""
        if missing:
            banner = (
                "<div class='banner'>시작 시 API 키는 선택 사항입니다. "
                "실시간 Alpaca/LLM을 쓰려면 <a href='/settings'>설정</a>에서 입력하세요. "
                f"현재 없음: {_esc(', '.join(missing))}. "
                "그때까지 해당 기능은 미설정으로 표시되며, 가짜 LLM/SEC 추출을 만들지 않습니다.</div>"
            )
        return templates.TemplateResponse(
            request,
            "shell.html",
            {
                "title": title,
                "body": banner + body,
                "nav": NAV_ROUTES,
                "path": request.url.path,
                "ui_stack": live_settings().ui_stack,
                "server_started_label": format_ts(server_started_at),
            },
        )

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        snap = snapshot()
        alloc = snap["allocation"]
        caps = snap.get("capabilities") or {}
        stocks = alloc.get("stock_weights") or {}
        stock_w = sum(stocks.values()) if isinstance(stocks, dict) else 0.0
        n_names = len(stocks) if isinstance(stocks, dict) else 0
        names_label = "평가 미확정" if alloc.get("weights_confirmed") is False else f"{n_names}개 종목"
        n_buy = sum(
            1
            for r in (snap.get("effective") if snap.get("decision_contract") else snap.get("quant")) or []
            if "btc" not in str(r.get("instrument_id")).lower()
            and _is_buy_candidate(
                r,
                settings=live_settings(),
                total_base_units=_snap_base(snap),
            )
        )
        cap_rows = "".join(
            f"<div class='row'><div><p style='margin:0;font-size:0.875rem;font-weight:500'>{_esc(_CAP_KO.get(k, k))}</p>"
            f"<p class='hint' style='margin:0.15rem 0 0'>{_esc(_CAP_HINT.get((k, str(v)), v))}</p></div>{_badge(v)}</div>"
            for k, v in caps.items()
        )
        st = _scan_copy()
        running = st.get("status") == "running"
        st.update(_elapsed_fields(running=bool(running), latest=str(st.get("latest") or "") if running else None))
        err_html = ""
        if st.get("status") == "error" and st.get("error"):
            err_html = f"<div class='banner'>{_esc(st.get('error'))}</div>"
        if running:
            mode = st.get("mode") or "scan"
            heading = "학습 실행 중" if mode == "train" else "스캔 실행 중"
            hint = (
                "최신 시세로 모델만 다시 학습합니다. API 호출 없음, 보통 1분 이내. 이 페이지를 닫지 마세요."
                if mode == "train"
                else "시세 보강 후 저장된 모델로 추천을 만듭니다. 학습은 하지 않습니다. 이 페이지를 닫지 마세요."
            )
            scan_ui = (
                "<div class='scan-progress' id='scan-box'>"
                f"<strong>{heading}</strong>"
                f"<p class='hint' style='margin:0.35rem 0 0'>{hint}</p>"
                f"<p id='scan-msg' style='margin:0.55rem 0 0;font-size:0.85rem'>{_esc(st.get('latest') or '시작 중…')}</p>"
                f"<p id='scan-clock' class='scan-clock'>현재 단계 {_esc(st.get('elapsed_step') or '0초')} · 전체 {_esc(st.get('elapsed_total') or '0초')}</p>"
                "</div>"
                "<button class='btn' type='button' disabled>실행 중…</button>"
                """<script>
                function fmtElapsed(s) {
                  s = Math.max(0, Math.floor(Number(s) || 0));
                  const m = Math.floor(s / 60);
                  const h = Math.floor(m / 60);
                  const mm = m % 60;
                  const r = s % 60;
                  if (h) return h + "시간 " + String(mm).padStart(2, "0") + "분 " + String(r).padStart(2, "0") + "초";
                  if (m) return m + "분 " + String(r).padStart(2, "0") + "초";
                  return r + "초";
                }
                const clock = { t: Date.now(), total: 0, step: 0, running: true };
                function paintClock() {
                  const el = document.getElementById("scan-clock");
                  if (!el) return;
                  const extra = clock.running ? (Date.now() - clock.t) / 1000 : 0;
                  el.textContent = "현재 단계 " + fmtElapsed(clock.step + extra) + " · 전체 " + fmtElapsed(clock.total + extra);
                }
                async function pollScan() {
                  try {
                    const r = await fetch("/api/v1/scan-status");
                    const j = await r.json();
                    const el = document.getElementById("scan-msg");
                    if (el) el.textContent = j.latest || "실행 중…";
                    clock.t = Date.now();
                    clock.total = Number(j.elapsed_total_s) || 0;
                    clock.step = Number(j.elapsed_step_s) || 0;
                    clock.running = j.status === "running";
                    paintClock();
                    if (j.status === "running") { setTimeout(pollScan, 2000); return; }
                    window.location.href = "/";
                  } catch (e) { setTimeout(pollScan, 3000); }
                }
                paintClock();
                setInterval(paintClock, 1000);
                pollScan();
                </script>"""
            )
        else:
            kind, notes = _preflight()
            notice = ""
            guard = ""
            if kind == "unchecked":
                notice = (
                    "<div class='banner' style='margin:1rem 0 0'>"
                    "연결 확인 버튼은 선택입니다. 설정에 키가 저장돼 있으면 스캔 때 "
                    "legacy-b 방식의 GPT 연속 분석과 최종 판단이 바로 돌아갑니다. "
                    "확인은 키가 살아 있는지 보고 싶을 때만 누르면 됩니다. "
                    "<a href='/settings'>설정</a></div>"
                )
            elif kind:
                heading = "연결되지 않은 API가 있습니다."
                detail = "".join(f"<li>{_esc(n)}</li>" for n in notes)
                notice = (
                    "<div class='banner' style='margin:1rem 0 0'>"
                    f"<strong>{_esc(heading)}</strong>"
                    f"<ul style='margin:0.4rem 0 0.6rem 1.1rem;padding:0'>{detail}</ul>"
                    "그래도 실행할 수 있습니다. 실패한 단계는 사용 불가로 남고 Quant는 계속됩니다. "
                    "<a href='/settings'>설정에서 연결 확인</a></div>"
                )
                message = "연결되지 않은 API가 있습니다. 그래도 스캔을 실행할까요?"
                guard = f"if (!confirm('{message}')) return false; "
            resume_hint = _resumable_scan_hint()
            resume_notice = (
                f"<div class='banner' style='margin:1rem 0 0'>{_esc(resume_hint)}</div>" if resume_hint else ""
            )
            scan_label = "이어서 스캔 실행" if resume_hint else "지금 스캔 실행"
            scan_ui = (
                f"{notice}{resume_notice}"
                "<div style='display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:1rem;align-items:center'>"
                "<form method='post' action='/run-once' "
                f"onsubmit=\"{guard}const b=this.querySelector('button'); "
                "b.disabled=true; b.textContent='스캔 실행 중…';\">"
                f"<button class='btn' type='submit'>{_esc(scan_label)}</button></form>"
                "<form method='post' action='/train-once' "
                "onsubmit=\"if (!confirm('최신 시세로 모델만 다시 학습합니다. API 호출 없이 보통 1분 이내 끝납니다. 계속할까요?')) return false; "
                "const b=this.querySelector('button'); b.disabled=true; b.textContent='학습 중…';\">"
                "<button class='btn btn-ghost' type='submit'>모델 학습</button></form>"
                "</div>"
                "<p class='hint' style='margin:0.6rem 0 0'>스캔은 시세와 추천만 합니다. 이미 학습된 모델을 씁니다. "
                "모델 학습은 시세만 다시 받아 모델을 갱신할 뿐 SEC·Gemini·GPT를 전혀 호출하지 않습니다(API 비용 $0). "
                "자동 스캔·자동 학습 모두 거래일마다 1회씩 시도합니다. 중간에 끊겨도 같은 날 자동으로 다시 돌리지 않습니다. "
                "UTC 일일 자동 실행 LLM 예산 기준값은 기본 $15입니다. 다음 호출 전에 예상 비용을 검사하는 soft limit이며, "
                "실제 비용의 절대 상한은 아닙니다. 학습에는 LLM 호출이 없고 직접 실행한 스캔은 이 기준에 포함되지 않습니다.</p>"
            )
        body = f"""
        {err_html}
        <h1>대시보드</h1>
        <p class="lead">미국 주식 + BTC + USD 현금 개인 투자 비서</p>
        <div class="grid-stats">
          <div class="stat"><span class="stat-label">주식 비중</span><span class="stat-value">{_esc(_allocation_weight_text(alloc, "stock"))}</span><span class="stat-sub">{names_label}</span></div>
          <div class="stat"><span class="stat-label">BTC 비중</span><span class="stat-value">{_esc(_allocation_weight_text(alloc, "btc"))}</span><span class="stat-sub">슬리브 분리</span></div>
          <div class="stat"><span class="stat-label">USD 현금</span><span class="stat-value">{_esc(_allocation_weight_text(alloc, "cash"))}</span><span class="stat-sub">100% 현금 허용</span></div>
          <div class="stat"><span class="stat-label">매수 신호</span><span class="stat-value">{n_buy}개</span><span class="stat-sub">Quant 추천</span></div>
        </div>
        <div class="card">
          <h2>시스템 상태</h2>
          <p class="hint">빨간 <strong>사용 불가</strong>는 연결/호출 실패, 노란 <strong>미설정</strong>은 API 키를 아직 안 넣은 상태입니다. 앱이 꺼진 것은 아닙니다.</p>
          {cap_rows or '<p class="muted">아직 스캔 기록이 없습니다. 아래 버튼으로 스캔하세요.</p>'}
          {scan_ui}
        </div>
        <div class="card">
          <p class="hint" style="margin:0">⚠ 이 앱은 실제 거래를 실행하지 않습니다. 스모크 유니버스는 전체 S&amp;P 500이 아니며 생존편향이 있을 수 있습니다.</p>
        </div>
        """
        return page(request, "대시보드", body)

    def _is_auto(source: str) -> bool:
        return source.strip().lower() in {"auto", "daily", "scheduler"}

    @app.post("/run-once")
    def run_once(source: str = Query("")) -> RedirectResponse:
        auto = _is_auto(source)
        if not _begin_scan(mode="scan", latest="스캔을 시작했습니다."):
            # Another run already holds the in-process lock, so this attempt does
            # nothing -- hand the auto slot back instead of burning the day on a
            # run that never happened (see _refund_auto_slot).
            if auto:
                _refund_auto_slot(train=False)
            return RedirectResponse("/", status_code=303)
        cache["last"] = None
        threading.Thread(
            target=_run_scan_worker,
            kwargs={"retrain": False, "enforce_llm_budget": auto, "auto": auto},
            name="v1-scan",
            daemon=True,
        ).start()
        return RedirectResponse("/", status_code=303)

    @app.post("/train-once")
    def train_once(source: str = Query("")) -> RedirectResponse:
        """Model refit only -- no SEC/Gemini/GPT calls, $0. See run_v1_cycle(train_only=True).
        Does not touch the recommendations cache; the last real scan stays on screen."""
        auto = _is_auto(source)
        if not _begin_scan(mode="train", latest="모델 학습을 시작했습니다 (API 호출 없음)."):
            if auto:
                _refund_auto_slot(train=True)
            return RedirectResponse("/", status_code=303)
        threading.Thread(
            target=_run_scan_worker,
            kwargs={"retrain": True, "train_only": True, "enforce_llm_budget": False, "auto": auto},
            name="v1-train",
            daemon=True,
        ).start()
        return RedirectResponse("/", status_code=303)

    @app.get("/api/v1/scan-status")
    def api_scan_status() -> dict[str, object]:
        st = _scan_copy()
        running = st.get("status") == "running"
        latest = _latest_runtime_hint(running=True) if running else st.get("latest")
        if running:
            st["latest"] = latest
        st.update(_elapsed_fields(running=bool(running), latest=str(latest or "") if running else None))
        return st

    @app.get("/api/v1/ticker-lookup")
    def api_ticker_lookup(q: str = Query("")) -> dict[str, object]:
        store = _store(writer=False)
        try:
            return lookup_ticker(store.conn, live_settings(), q).as_dict()
        except Exception as exc:  # noqa: BLE001 — lookup must not 500 the form
            return {
                "ok": False,
                "query": q,
                "symbol": (q or "").strip().upper(),
                "instrument_id": "",
                "name": None,
                "in_universe": False,
                "has_bars": False,
                "source": "none",
                "message": f"확인 중 오류가 났습니다 ({type(exc).__name__}).",
                "suggestions": [],
            }
        finally:
            store.close()

    @app.get("/portfolio", response_class=HTMLResponse)
    def portfolio(request: Request) -> HTMLResponse:
        store = _store(writer=False)
        try:
            base_units = get_total_base_units(store.conn)
            lots = list_lots(store.conn)
            cfg = live_settings()
            rows = "".join(lot_row_html(store.conn, cfg, l) for l in lots)
        finally:
            store.close()
        today = live_settings().today_in_user_tz().isoformat()
        err = request.query_params.get("err")
        err_html = ""
        if err == "lot":
            err_html = "<div class='banner'>단위와 취득가는 0보다 커야 합니다. 전량 매도는 삭제를 쓰세요.</div>"
        elif err == "meta":
            err_html = "<div class='banner'>총 기준 단위는 0보다 큰 유한한 숫자여야 합니다.</div>"
        elif err == "busy":
            err_html = "<div class='banner'>다른 스캔이 저장소를 사용 중입니다. 잠시 후 다시 저장하세요.</div>"
        body = f"""
        <h1>포트폴리오</h1>
        <p class="lead">단위 기준입니다. 실제 계좌 잔고는 필요 없고, AI가 보유를 바꾸지 않습니다.</p>
        {err_html}
        <div class="card">
          <form method="post" action="/portfolio/meta" class="inline-form">
            <div class="field" style="margin:0;flex:1"><label>총 기준 단위</label>
            <input name="total_base_units" type="number" min="0.0001" step="any" value="{_esc(f'{base_units:g}')}"/></div>
            <button class="btn" type="submit" style="width:auto">저장</button>
          </form>
        </div>
        <div class="card">
          <h2>로트 추가</h2>
          <form method="post" action="/portfolio/add" id="lot-add-form">
            <div class="field">
              <label>티커</label>
              <div class="inline-form" style="margin:0">
                <input id="lot-ticker" name="instrument_id" placeholder="AAPL" autocomplete="off" spellcheck="false"/>
                <button type="button" class="btn btn-ghost" id="lot-ticker-check">확인</button>
              </div>
              <p id="lot-ticker-status" class="ticker-status">종목을 입력한 뒤 확인을 눌러 티커가 맞는지 보세요.</p>
              <p id="lot-ticker-suggestions" class="ticker-suggestions"></p>
            </div>
            <div class="inline-form">
              <div class="field" style="margin:0;flex:1"><label>단위</label><input name="units" value="1"/></div>
              <div class="field" style="margin:0;flex:1"><label>가격</label><input name="price" value="1"/></div>
              <div class="field" style="margin:0;flex:1"><label>일자</label><input name="acquired_on" type="date" value="{today}"/></div>
            </div>
            <button class="btn" type="submit">로트 추가</button>
          </form>
        </div>
        <div class="card">
          <h2>보유 로트</h2>
          <p class="hint" style="margin:0 0 0.7rem">취득 단위는 추가매수·부분매도할 때 직접 고칩니다. 종가 환산 단위는 스캔 때 종가÷취득가로 다시 계산되며, 추천 비중은 환산 단위를 씁니다. 전량 매도는 삭제입니다.</p>
          {rows or '<p class="muted">아직 로트가 없습니다.</p>'}
        </div>
        """
        body += """
        <script>
        (function () {
          const input = document.getElementById("lot-ticker");
          const status = document.getElementById("lot-ticker-status");
          const hints = document.getElementById("lot-ticker-suggestions");
          const checkBtn = document.getElementById("lot-ticker-check");
          const form = document.getElementById("lot-add-form");
          if (!input || !status || !checkBtn || !form) return;
          let lastOk = "";
          function setStatus(text, kind) {
            status.textContent = text;
            status.className = "ticker-status" + (kind ? " " + kind : "");
          }
          function setSuggestions(items) {
            if (!hints) return;
            hints.innerHTML = "";
            (items || []).forEach(function (sym) {
              const b = document.createElement("button");
              b.type = "button";
              b.className = "chip";
              b.textContent = String(sym);
              b.addEventListener("click", function () {
                input.value = String(sym);
                checkTicker();
              });
              hints.appendChild(b);
            });
          }
          async function checkTicker() {
            const q = (input.value || "").trim();
            setSuggestions([]);
            if (!q) {
              lastOk = "";
              setStatus("티커를 입력하세요.", "bad");
              return false;
            }
            setStatus("확인 중…", "wait");
            try {
              const r = await fetch("/api/v1/ticker-lookup?q=" + encodeURIComponent(q));
              const j = await r.json();
              if (j.ok) {
                lastOk = String(j.symbol || q).toUpperCase();
                input.value = lastOk;
                setStatus(j.message || ("확인됨: " + lastOk), "ok");
                return true;
              }
              lastOk = "";
              setStatus(j.message || "확인할 수 없습니다.", "bad");
              setSuggestions(j.suggestions || []);
              return false;
            } catch (e) {
              lastOk = "";
              setStatus("확인 요청에 실패했습니다. 네트워크를 보세요.", "bad");
              return false;
            }
          }
          checkBtn.addEventListener("click", function () { checkTicker(); });
          input.addEventListener("change", function () { if ((input.value || "").trim()) checkTicker(); });
          input.addEventListener("input", function () {
            const cur = (input.value || "").trim().toUpperCase();
            if (lastOk && cur !== lastOk) {
              lastOk = "";
              setSuggestions([]);
              setStatus("티커가 바뀌었습니다. 다시 확인하세요.", "wait");
            }
          });
          form.addEventListener("submit", function (ev) {
            const cur = (input.value || "").trim().toUpperCase();
            if (cur && lastOk && cur === lastOk) return;
            if (!confirm("티커를 확인하지 않았거나 확인에 실패했습니다. 그래도 추가할까요?")) {
              ev.preventDefault();
            }
          });
        })();
        </script>
        """
        return page(request, "포트폴리오", body)

    @app.post("/portfolio/meta")
    def portfolio_meta(total_base_units: float = Form(...)) -> RedirectResponse:
        try:
            store = _store()
        except WriterLeaseBusy:
            return RedirectResponse("/portfolio?err=busy", status_code=303)
        try:
            set_total_base_units(store.conn, total_base_units)
        except ValueError:
            return RedirectResponse("/portfolio?err=meta", status_code=303)
        finally:
            store.close()
        return RedirectResponse("/portfolio", status_code=303)

    @app.post("/portfolio/add")
    def portfolio_add(
        instrument_id: str = Form(...),
        units: float = Form(...),
        price: float = Form(...),
        acquired_on: str = Form(...),
    ) -> RedirectResponse:
        from datetime import date as date_cls

        try:
            store = _store()
        except WriterLeaseBusy:
            return RedirectResponse("/portfolio?err=busy", status_code=303)
        try:
            add_lot(
                store.conn,
                Lot(
                    instrument_id=instrument_id,  # type: ignore[arg-type]
                    acquisition_units=units,
                    acquisition_price=price,
                    acquired_on=date_cls.fromisoformat(acquired_on),
                ),
            )
        except ValueError:
            return RedirectResponse("/portfolio?err=lot", status_code=303)
        finally:
            store.close()
        return RedirectResponse("/portfolio", status_code=303)

    @app.post("/portfolio/update")
    def portfolio_update(
        lot_id: str = Form(...),
        units: float = Form(...),
        price: float = Form(...),
    ) -> RedirectResponse:
        try:
            store = _store()
        except WriterLeaseBusy:
            return RedirectResponse("/portfolio?err=busy", status_code=303)
        try:
            update_lot(store.conn, lot_id, units=units, price=price)
        except (ValueError, KeyError):
            return RedirectResponse("/portfolio?err=lot", status_code=303)
        finally:
            store.close()
        return RedirectResponse("/portfolio", status_code=303)

    @app.post("/portfolio/delete")
    def portfolio_delete(lot_id: str = Form(...)) -> RedirectResponse:
        try:
            store = _store()
        except WriterLeaseBusy:
            return RedirectResponse("/portfolio?err=busy", status_code=303)
        try:
            delete_lot(store.conn, lot_id)
        finally:
            store.close()
        return RedirectResponse("/portfolio", status_code=303)

    @app.get("/recommendations", response_class=HTMLResponse)
    def recommendations(request: Request) -> HTMLResponse:
        return page(request, "종목 추천", render_recommendations_html(snapshot()))

    @app.get("/llm", response_class=HTMLResponse)
    def llm_log_index(request: Request) -> HTMLResponse:
        return _llm_log_page(request, None)

    @app.get("/llm/{tick_id}", response_class=HTMLResponse)
    def llm_log_detail(request: Request, tick_id: str) -> HTMLResponse:
        return _llm_log_page(request, tick_id)

    def _llm_log_page(request: Request, tick_id: str | None) -> HTMLResponse:
        error = None
        sessions: list = []
        detail = None
        selected = tick_id
        try:
            store = _store(writer=False)
            try:
                sessions = list_llm_sessions(store.conn)
                if selected is None and sessions:
                    selected = str(sessions[0]["tick_id"])
                if selected:
                    detail = load_llm_session(store.conn, selected)
            finally:
                store.close()
        except WriterLeaseBusy:
            error = "다른 실행이 데이터베이스를 사용 중입니다. 잠시 후 다시 열어 주세요."
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            error = "LLM 기록을 읽지 못했습니다. 스캔이 끝나 있으면 잠시 후 다시 열어 주세요."
        return page(
            request,
            "LLM 기록",
            render_llm_log_html(
                sessions,
                detail,
                selected_id=selected,
                error=error,
                current_model=(
                    f"{live_settings().llm_judge_model or 'gpt-5.6-sol'} · legacy-b 연속 대화"
                ),
            ),
        )

    @app.get("/btc", response_class=HTMLResponse)
    def btc(request: Request) -> HTMLResponse:
        snap = snapshot()
        cards = _btc_effective_cards(snap, settings=live_settings())
        alloc = snap["allocation"]
        body = f"""
        <h1>BTC 신호</h1>
        <p class="lead">BTC/USD만 다룹니다. 즉시 현금이 아니며 이체 마찰(available / unsettled / transfer_pending / unavailable)을 반영합니다.</p>
        <div class="grid-stats">
          <div class="stat"><span class="stat-label">BTC 비중</span><span class="stat-value">{_esc(_allocation_weight_text(alloc, "btc"))}</span><span class="stat-sub">슬리브 분리</span></div>
          <div class="stat"><span class="stat-label">현금 비중</span><span class="stat-value">{_esc(_allocation_weight_text(alloc, "cash"))}</span><span class="stat-sub">히스테리시스 적용</span></div>
        </div>
        {cards or '<div class="card"><p class="muted" style="margin:0">아직 BTC 신호가 없습니다. 스캔을 실행하세요.</p></div>'}
        """
        return page(request, "BTC 신호", body)

    @app.get("/models", response_class=HTMLResponse)
    def models(request: Request) -> HTMLResponse:
        horizon = request.query_params.get("h") or "all"
        change = request.query_params.get("c") or "all"
        family = request.query_params.get("f") or "all"
        if horizon not in HORIZON_CHOICES:
            horizon = "all"
        if change not in {key for key, _label, _kinds in CHANGE_GROUPS}:
            change = "all"
        store = _store(writer=False)
        try:
            view = collect_models_view(store.conn, horizon=horizon, change=change, family=family)
        finally:
            store.close()
        if family != "all" and family not in view.families_seen:
            family = "all"
        body = render_models_page(view, horizon=horizon, change=change, family=family)
        return page(request, "모델/학습", body)

    @app.get("/runtime", response_class=HTMLResponse)
    def runtime(request: Request) -> HTMLResponse:
        store = _store(writer=False)
        try:
            rows = store.conn.execute(
                "SELECT created_at, kind, status, message FROM runtime_events ORDER BY created_at DESC LIMIT 30"
            ).fetchall()
        finally:
            store.close()
        items = "".join(
            f'<div class="log"><span class="tag {"tag-err" if str(c).lower() in {"failed","error","unavailable"} else ""}">{_esc(b)}</span>'
            f'<div class="grow"><p style="margin:0;font-size:0.875rem">{_esc(_text_ko(d))}</p>'
            f'<p class="hint" style="margin:0.2rem 0 0">{_esc(a)} · {_esc(c)}</p></div></div>'
            for a, b, c, d in rows
        )
        body = f"""
        <h1>런타임</h1>
        <p class="lead">시스템 로그 및 실행 이력</p>
        <div class="card">{items or '<p class="muted" style="margin:0">아직 이벤트가 없습니다. 스캔을 실행하세요.</p>'}</div>
        """
        return page(request, "런타임", body)

    @app.get("/data-health", response_class=HTMLResponse)
    def data_health(request: Request) -> HTMLResponse:
        store = _store(writer=False)
        try:
            row = store.conn.execute(
                "SELECT overall, equity_feed, btc_feed, coverage_json, created_at FROM data_health_snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            ev = store.conn.execute(
                "SELECT kind, status, message FROM runtime_events WHERE kind IN ('llm_judge','sec_ingest','fetch_us_daily') ORDER BY created_at DESC LIMIT 8"
            ).fetchall()
        finally:
            store.close()
        caps = "".join(
            f'<div class="row"><div><p style="margin:0;font-weight:500">{_esc(a)}</p>'
            f'<p class="hint" style="margin:0.15rem 0 0">{_esc(_text_ko(c))}</p></div>{_badge(b)}</div>'
            for a, b, c in ev
        )
        if not row:
            body = """
            <h1>데이터 상태</h1>
            <p class="lead">키가 없으면 미설정, 실시간 호출 실패는 사용 불가로 표시합니다. 둘 다 정상 커버리지가 아닙니다.</p>
            <div class="card"><p class="muted" style="margin:0">아직 스냅샷이 없습니다. 스캔을 실행하세요.</p></div>
            """
        else:
            body = f"""
            <h1>데이터 상태</h1>
            <p class="lead">키가 없으면 미설정, 실시간 호출 실패는 사용 불가로 표시합니다. 둘 다 정상 커버리지가 아닙니다.</p>
            <div class="grid-stats">
              <div class="stat"><span class="stat-label">전체</span><span class="stat-value">{_badge(row[0])}</span></div>
              <div class="stat"><span class="stat-label">주식 피드</span><span class="stat-value">{_badge(row[1])}</span></div>
              <div class="stat"><span class="stat-label">BTC 피드</span><span class="stat-value">{_badge(row[2])}</span></div>
              <div class="stat"><span class="stat-label">시각</span><span class="stat-value" style="font-size:0.85rem">{_esc(row[4])}</span></div>
            </div>
            <div class="card">
              <h2>커버리지</h2>
              <p class="hint" style="white-space:pre-wrap">{_esc(row[3])}</p>
            </div>
            <div class="card">
              <h2>최근 기능 이벤트</h2>
              {caps or '<p class="muted">없음</p>'}
            </div>
            """
        return page(request, "데이터 상태", body)

    def _universe_card(cfg: Settings) -> str:
        store = _store(writer=False)
        try:
            conn = store.conn
            resolved = resolve_scan_universe(conn, cfg)
            state = universe_state(conn, resolved.preset)
            stats = coverage_stats(conn, resolved)
            held = held_symbols(conn)
            watch = watchlist_symbols(conn)
            custom = custom_symbols(conn)
        finally:
            store.close()

        options = ""
        for preset in PRESET_ORDER:
            checked = " checked" if preset == resolved.preset else ""
            recommended = " (권장)" if preset == PRESET_ORDER[0] else ""
            options += (
                f'<label class="row" style="cursor:pointer">'
                f'<span><input type="radio" name="preset" value="{_esc(preset)}"{checked}/> '
                f"{_esc(PRESET_LABELS[preset])}{recommended}</span></label>"
            )

        refreshed = state["last_refresh_at"]
        refreshed_txt = _esc(str(refreshed)[:19]) if refreshed else "없음"
        err = state.get("last_error")
        err_html = f'<p class="hint">최근 갱신 메모: {_esc(str(err)[:160])}</p>' if err else ""

        def _chips(symbols: list[str], action: str) -> str:
            if not symbols:
                return '<p class="hint">없음</p>'
            out = '<div class="chips">'
            for sym in symbols:
                out += (
                    f'<form method="post" action="{action}" style="display:inline">'
                    f'<input type="hidden" name="symbol" value="{_esc(sym)}"/>'
                    f'<button class="chip" type="submit" title="제거">{_esc(sym)} ×</button>'
                    f"</form>"
                )
            return out + "</div>"

        custom_block = ""
        if resolved.preset == PRESET_CUSTOM:
            custom_block = f"""
          <h2 style="margin-top:1.2rem">직접 지정한 종목</h2>
          {_chips(custom, "/universe/custom/remove")}
          <form method="post" action="/universe/custom/add">
            <div class="field"><label>티커 추가</label>
              <input name="symbols" placeholder="예: AAPL, MSFT NVDA"/>
              <p class="hint">쉼표나 공백으로 구분해 여러 개를 넣을 수 있습니다.</p></div>
            <button class="btn" type="submit">추가</button>
          </form>"""

        return f"""
        <div class="card">
          <h2>스캔 유니버스</h2>
          <p class="help">추천 후보를 찾을 종목 범위입니다. SPY는 비교 기준으로만 쓰고 추천 후보로는 넣지 않습니다.
          BTC/USD는 별도 BTC 슬리브에서 다룹니다.</p>
          <div class="chips"><span class="chip-static">보유/관심 종목은 항상 포함</span></div>
          <form method="post" action="/universe/preset">
            {options}
            <button class="btn" type="submit" style="margin-top:0.85rem">이 범위로 저장</button>
          </form>
          {custom_block}
          <h2 style="margin-top:1.2rem">현재 상태</h2>
          <div class="row"><span class="muted">대상 종목 수</span><span>{stats["total"]}개</span></div>
          <div class="row"><span class="muted">데이터 확보</span><span>{stats["ready"]}개</span></div>
          <div class="row"><span class="muted">오래된 데이터</span><span>{stats["stale"]}개</span></div>
          <div class="row"><span class="muted">데이터 없음</span><span>{stats["unavailable"]}개</span></div>
          <div class="row"><span class="muted">커버리지</span><span>{stats["coverage_pct"]}%</span></div>
          <div class="row"><span class="muted">목록 상태</span>{_badge(state["status"])}</div>
          <div class="row"><span class="muted">목록 출처</span><span>{_esc(state["source"])}</span></div>
          <div class="row"><span class="muted">마지막 목록 갱신</span><span>{refreshed_txt}</span></div>
          {err_html}
          <form method="post" action="/universe/refresh" style="margin-top:0.85rem">
            <button class="btn btn-ghost" type="submit">종목 목록 새로 가져오기</button>
          </form>
          <h2 style="margin-top:1.2rem">관심 종목 (항상 스캔)</h2>
          {_chips(watch, "/universe/watchlist/remove")}
          <form method="post" action="/universe/watchlist/add">
            <div class="field"><label>관심 종목 추가</label>
              <input name="symbols" placeholder="예: TSLA, COST"/></div>
            <button class="btn" type="submit">추가</button>
          </form>
          <h2 style="margin-top:1.2rem">보유 종목 (항상 스캔)</h2>
          <p class="hint">{_esc(", ".join(held)) if held else "보유 종목이 없습니다."}</p>
        </div>
        """

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request) -> HTMLResponse:
        env = read_env_map(env_path(PROJECT_ROOT))
        cfg = live_settings()
        universe_card = _universe_card(cfg)
        kakao_st = connection_status(env)
        redirect_uri = env.get("KAKAO_REDIRECT_URI") or DEFAULT_REDIRECT_URI
        nick = env.get("KAKAO_NICKNAME") or ""
        flash = request.query_params.get("kakao")
        flash_html = ""
        if flash == "connected":
            flash_html = '<div class="banner" style="background:#f0faf3;border-color:#b7eb8f">카카오 연결이 완료되었습니다.</div>'
        elif flash == "disconnected":
            flash_html = '<div class="banner">카카오 연결을 해제했습니다.</div>'
        elif flash == "sent":
            flash_html = '<div class="banner" style="background:#f0faf3;border-color:#b7eb8f">테스트 메시지를 보냈습니다. 나와의 채팅을 확인하세요.</div>'
        elif flash == "need_key":
            flash_html = '<div class="banner">먼저 Kakao REST API 키를 저장하세요.</div>'
        elif flash == "denied":
            flash_html = '<div class="banner">카카오 권한 요청이 거부되었습니다.</div>'
        elif flash == "auth_failed":
            flash_html = '<div class="banner">카카오 인증에 실패했습니다. 다시 연결해 보세요.</div>'
        elif flash == "not_connected":
            flash_html = '<div class="banner">카카오가 연결되지 않았습니다.</div>'
        elif flash == "send_failed":
            flash_html = '<div class="banner">테스트 전송에 실패했습니다. 연결 상태를 확인하세요.</div>'
        universe_flash = request.query_params.get("universe")
        if universe_flash == "saved":
            flash_html += '<div class="banner" style="background:#f0faf3;border-color:#b7eb8f">스캔 범위를 저장했습니다.</div>'
        elif universe_flash == "save_failed":
            flash_html += '<div class="banner">스캔 범위를 저장하지 못했습니다. 다시 시도해 주세요.</div>'
        appetite_flash = request.query_params.get("appetite")
        if appetite_flash == "saved":
            flash_html += '<div class="banner" style="background:#f0faf3;border-color:#b7eb8f">투자 성향을 저장했습니다. 다음 스캔부터 적용됩니다.</div>'
        elif appetite_flash == "save_failed":
            flash_html += '<div class="banner">투자 성향을 저장하지 못했습니다.</div>'
        nick_row = (
            f'<div class="row"><span class="muted">카카오 계정</span><span>{_esc(nick)}</span></div>'
            if nick
            else ""
        )
        secret_script = """
        <script>
          document.querySelectorAll(".js-reveal-secret").forEach((btn) => {
            btn.addEventListener("click", () => {
              const field = btn.closest("[data-secret-field]");
              const input = field.querySelector(".js-secret-input");
              input.style.display = "block";
              input.focus();
              btn.disabled = true;
            });
          });
          const CONN_KO = {
            AVAILABLE: ["연결완료", "badge-ok"],
            NOT_CONFIGURED: ["미설정", "badge-neutral"],
            AUTH_REQUIRED: ["인증 실패", "badge-err"],
            RATE_LIMITED: ["요청 한도", "badge-warn"],
            QUOTA_EXCEEDED: ["쿼터 초과", "badge-err"],
            UNAVAILABLE: ["연결 실패", "badge-err"]
          };
          const connBtn = document.getElementById("conn-check-btn");
          if (connBtn) {
            connBtn.addEventListener("click", async () => {
              connBtn.disabled = true;
              connBtn.textContent = "확인 중…";
              try {
                const res = await fetch("/api/v1/connection-check", { method: "POST" });
                const data = await res.json();
                const box = document.getElementById("conn-rows");
                box.innerHTML = data.results.map((r) => {
                  const [label, cls] = CONN_KO[r.status] || ["미확인", "badge-neutral"];
                  const usage = r.usage
                    ? '<p class="hint" style="margin:0.15rem 0 0">' + r.usage + "</p>"
                    : "";
                  return '<div class="row"><div><p style="margin:0;font-size:0.875rem;font-weight:500">'
                    + r.label + '</p><p class="hint" style="margin:0.15rem 0 0">' + r.detail + "</p>"
                    + usage + '</div><span class="badge ' + cls + '">' + label + "</span></div>";
                }).join("");
                data.results.forEach((r) => {
                  const [label, cls] = CONN_KO[r.status] || ["미확인", "badge-neutral"];
                  document.querySelectorAll('[data-conn="' + r.key + '"]').forEach((el) => {
                    el.textContent = label;
                    el.className = "badge " + cls;
                  });
                });
                const stamp = document.getElementById("conn-checked");
                if (stamp) stamp.textContent = "마지막 확인 " + (data.checked_at_label || "방금");
              } catch (e) {
                const stamp = document.getElementById("conn-checked");
                if (stamp) stamp.textContent = "확인에 실패했습니다. 잠시 후 다시 시도하세요.";
              }
              connBtn.disabled = false;
              connBtn.textContent = "전체 연결 확인";
            });
          }
        </script>
        """
        conn_map = _conn_status_map()
        report = conn_state.get("report")
        checked_line = (
            f"마지막 확인 {_esc(format_ts(report.checked_at))}"
            if report
            else "아직 확인하지 않았습니다."
        )
        body = f"""
        {flash_html}
        <h1>설정</h1>
        <p class="lead">실시간 데이터·LLM·카카오 알림은 필요할 때만 설정하면 됩니다. 키 없이도 앱은 시작됩니다.</p>
        {universe_card}
        <div class="card">
          <h2>투자 성향</h2>
          <p class="hint">Quant가 주식 예산을 얼마나 채울지입니다. 자격 컷오프(0.01)와 강한 기회 기준은 아직 성과 데이터로 맞춘 값이 아닙니다.
          공격적은 강한 기회 1개만 있어도 한도까지 채울 수 있고, 종목 비중은 컷오프를 얼마나 넘겼는지에 따릅니다. 주문은 내지 않습니다. 다음 스캔부터 적용됩니다.</p>
          <form method="post" action="/settings/appetite">
            <div class="chips" style="margin:0.5rem 0 0.75rem">
              <label class="chip {'on' if cfg.risk_appetite=='aggressive' else ''}">
                <input type="radio" name="risk_appetite" value="aggressive" {'checked' if cfg.risk_appetite=='aggressive' else ''} style="margin-right:0.35rem"/>공격적
              </label>
              <label class="chip {'on' if cfg.risk_appetite=='balanced' else ''}">
                <input type="radio" name="risk_appetite" value="balanced" {'checked' if cfg.risk_appetite=='balanced' else ''} style="margin-right:0.35rem"/>균형
              </label>
              <label class="chip {'on' if cfg.risk_appetite=='conservative' else ''}">
                <input type="radio" name="risk_appetite" value="conservative" {'checked' if cfg.risk_appetite=='conservative' else ''} style="margin-right:0.35rem"/>보수적
              </label>
            </div>
            <p class="hint">공격적: 강한 종목 1개면 주식 한도. 균형: 2개. 보수적: 종목 상한/단일 상한 비율(보통 4개). 비중은 모두 초과분 기준입니다.</p>
            <button class="btn" type="submit">성향 저장</button>
          </form>
        </div>
        <div class="card">
          <h2>API 키</h2>
          <form method="post" action="/settings">
            {_secret_field("ALPACA_API_KEY", "Alpaca API Key", env, "시장 데이터", conn_key="alpaca", conn_status=conn_map.get("alpaca"))}
            {_secret_field("ALPACA_SECRET_KEY", "Alpaca Secret Key", env, conn_key="alpaca", conn_status=conn_map.get("alpaca"))}
            {_secret_field("OPENAI_API_KEY", "OpenAI API Key", env, "GPT-5.6 Sol 최종 판단에 씁니다. Quant 점수를 그대로 따르지 않습니다.", conn_key="openai", conn_status=conn_map.get("openai"))}
            {_secret_field("GEMINI_API_KEY", "Gemini API Key", env, "현재 legacy-b 방식의 연속 GPT 스캔에서는 사용하지 않습니다.", conn_key="gemini", conn_status=conn_map.get("gemini"))}
            {_secret_field("ANTHROPIC_API_KEY", "Anthropic API Key", env, "저장만 됩니다. 지금은 스캔에 쓰지 않습니다. console.anthropic.com의 API Keys에서 sk-ant- 키를 넣으세요.", conn_key="anthropic", conn_status=conn_map.get("anthropic"))}
            <button class="btn" type="submit">로컬 .env에 저장</button>
          </form>
        </div>
        <div class="card">
          <h2>연결 확인</h2>
          <p class="hint">한 번에 모든 API로 가장 저렴한 요청을 하나씩 동시에 보냅니다.
          스캔/학습이 이 버튼을 기다리지 않습니다. 설정에 Gemini·OpenAI 키가 있으면 그대로 호출합니다.
          OpenAI만 1토큰짜리 실제 호출이고, Anthropic은 모델 목록으로 확인합니다(목록이 막히면 Haiku 1토큰).
          남은 잔액은 API로 조회할 수 없어 표시하지 않습니다.</p>
          <p class="hint" style="margin-top:0.45rem">SEC EDGAR는 미국 증권거래위원회 공시(8-K, 10-Q)입니다. 가입/키가 없고,
          한국 인터넷에서 자주 차단됩니다. 막혀도 앱은 돌아가며, Gemini/Sol은 공시 없이 Quant와 가능한 근거만 봅니다.</p>
          <div id="conn-rows">{_conn_rows_html(report)}</div>
          <p class="hint" id="conn-checked" style="margin-top:0.6rem">{checked_line}</p>
          <div class="btn-row">
            <button class="btn" type="button" id="conn-check-btn">전체 연결 확인</button>
          </div>
        </div>
        <div class="card">
          <h2>카카오톡 알림</h2>
          <p class="help">개인 카카오 개발자 앱의 <strong>나에게 보내기</strong>입니다. 비즈메시지/알림톡이 아닙니다.
          스캔이 끝나면 최종 매수(추가 포함)·매도 종목만 짧은 한 통으로 보냅니다. 주문은 내지 않습니다.<br/>
          Redirect URI는 카카오 개발자 콘솔에 아래와 동일하게 등록하세요:<br/>
          <code>{_esc(DEFAULT_REDIRECT_URI)}</code></p>
          <div class="row"><span>연결 상태</span>{_badge(kakao_st.value)}</div>
          {nick_row}
          <form method="post" action="/settings/kakao">
            {_secret_field("KAKAO_REST_API_KEY", "Kakao REST API Key", env)}
            {_secret_field("KAKAO_CLIENT_SECRET", "Kakao Client Secret", env, "콘솔에서 Client Secret을 쓰는 앱만")}
            <div class="field"><label>Redirect URI</label>
              <input name="KAKAO_REDIRECT_URI" value="{_esc(redirect_uri)}"/>
              <p class="hint">기본값 {_esc(DEFAULT_REDIRECT_URI)}</p></div>
            <button class="btn" type="submit">카카오 앱 설정 저장</button>
          </form>
          <div class="btn-row" style="margin-top:0.85rem">
            <a class="btn" href="/auth/kakao/connect">카카오 연결</a>
            <a class="btn btn-ghost" href="/auth/kakao/connect">다시 연결 / 권한 갱신</a>
          </div>
          <div class="btn-row">
            <form method="post" action="/auth/kakao/test" style="flex:1;margin:0">
              <button class="btn btn-ghost" type="submit" style="width:100%">테스트 메시지 보내기</button>
            </form>
            <form method="post" action="/auth/kakao/disconnect" style="flex:1;margin:0">
              <button class="btn btn-ghost" type="submit" style="width:100%">카카오 연결 해제</button>
            </form>
          </div>
        </div>
        <div class="card">
          <h2>앱 정보</h2>
          <div class="row"><span class="muted">버전</span><span>1.0.0</span></div>
          <div class="row"><span class="muted">거래 기능</span><span>비활성 (주문 없음)</span></div>
        </div>
        <div class="card">
          <h2>서버 종료</h2>
          <p class="hint">브라우저 창을 닫아도 서버는 계속 실행됩니다(별개 프로세스). 여기서 끄면
          이 서버 창(cmd)도 함께 종료되며, 다시 쓰려면 START_STOCK_AI.bat로 새로 띄워야 합니다.</p>
          <form method="post" action="/shutdown"
            onsubmit="if (!confirm('서버를 종료할까요? 다시 쓰려면 START_STOCK_AI.bat로 새로 띄워야 합니다.')) return false;
              const b=this.querySelector('button'); b.disabled=true; b.textContent='종료 중…';">
            <button class="btn btn-ghost" type="submit" style="width:100%;color:var(--negative)">서버 종료</button>
          </form>
        </div>
        {secret_script}
        """
        return page(request, "설정", body)

    @app.post("/shutdown", response_class=HTMLResponse)
    def shutdown(request: Request) -> HTMLResponse:
        """Graceful self-stop: same signal a Ctrl+C in the server's own console would
        send, just triggered from the browser instead of that console window."""
        def _stop() -> None:
            time.sleep(0.3)  # let this response reach the browser first
            os.kill(os.getpid(), signal.SIGINT)

        threading.Thread(target=_stop, name="shutdown", daemon=True).start()
        body = """
        <h1>서버 종료</h1>
        <p class="lead">서버를 종료했습니다. 이 탭은 이제 닫으셔도 됩니다.</p>
        <div class="card"><p class="hint" style="margin:0">
          다시 쓰려면 START_STOCK_AI.bat를 실행해 새로 띄우세요.</p></div>
        """
        return page(request, "서버 종료", body)

    @app.post("/api/v1/connection-check")
    def api_connection_check() -> dict[str, object]:
        report = check_all(read_env_map(env_path(PROJECT_ROOT)))
        conn_state["report"] = report
        payload = report.as_dict()
        payload["checked_at_label"] = format_ts(report.checked_at)
        return payload

    @app.get("/api/v1/connection-state")
    def api_connection_state() -> dict[str, object]:
        report = conn_state.get("report")
        if report is None:
            return {"checked": False, "ok": False, "results": []}
        payload = report.as_dict()
        payload["checked"] = True
        payload["checked_at_label"] = format_ts(report.checked_at)
        return payload

    @app.post("/settings")
    def settings_save(
        ALPACA_API_KEY: str = Form(""),
        ALPACA_SECRET_KEY: str = Form(""),
        OPENAI_API_KEY: str = Form(""),
        GEMINI_API_KEY: str = Form(""),
        ANTHROPIC_API_KEY: str = Form(""),
    ) -> RedirectResponse:
        try:
            upsert_env(
                env_path(PROJECT_ROOT),
                {
                    "ALPACA_API_KEY": ALPACA_API_KEY,
                    "ALPACA_SECRET_KEY": ALPACA_SECRET_KEY,
                    "OPENAI_API_KEY": OPENAI_API_KEY,
                    "GEMINI_API_KEY": GEMINI_API_KEY,
                    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
                },
            )
        except ValueError:
            return RedirectResponse("/settings?err=invalid_value", status_code=303)
        conn_state["report"] = None
        return RedirectResponse("/settings", status_code=303)

    @app.post("/settings/appetite")
    def settings_appetite_save(risk_appetite: str = Form("aggressive")) -> RedirectResponse:
        choice = str(risk_appetite or "").strip().lower()
        if choice not in {"aggressive", "balanced", "conservative"}:
            choice = "aggressive"
        try:
            upsert_env(env_path(PROJECT_ROOT), {"RISK_APPETITE": choice})
        except Exception:
            return RedirectResponse("/settings?appetite=save_failed", status_code=303)
        return RedirectResponse("/settings?appetite=saved", status_code=303)

    def _universe_write(fn) -> RedirectResponse:
        try:
            store = _store(writer=True)
        except WriterLeaseBusy:
            return RedirectResponse("/settings?universe=busy", status_code=303)
        try:
            fn(store.conn)
        except Exception:  # noqa: BLE001 — keep settings controls usable during provider/storage failure
            return RedirectResponse("/settings?universe=save_failed", status_code=303)
        finally:
            store.close()
        return RedirectResponse("/settings", status_code=303)

    @app.post("/universe/preset")
    def universe_preset(preset: str = Form(...)) -> RedirectResponse:
        choice = preset if preset in PRESET_ORDER else PRESET_ORDER[0]
        try:
            upsert_env(env_path(PROJECT_ROOT), {"SCAN_UNIVERSE_PRESET": choice})
        except Exception:
            return RedirectResponse("/settings?universe=save_failed", status_code=303)
        return RedirectResponse("/settings?universe=saved", status_code=303)

    @app.post("/universe/refresh")
    def universe_refresh() -> RedirectResponse:
        cfg = live_settings()
        return _universe_write(lambda conn: prepare_scan_universe(conn, cfg, force_refresh=True))

    @app.post("/universe/custom/add")
    def universe_custom_add(symbols: str = Form("")) -> RedirectResponse:
        picked = parse_symbol_input(symbols)
        return _universe_write(lambda conn: add_custom_symbols(conn, picked))

    @app.post("/universe/custom/remove")
    def universe_custom_remove(symbol: str = Form(...)) -> RedirectResponse:
        return _universe_write(lambda conn: remove_custom_symbol(conn, symbol))

    @app.post("/universe/watchlist/add")
    def universe_watchlist_add(symbols: str = Form("")) -> RedirectResponse:
        picked = parse_symbol_input(symbols)
        return _universe_write(lambda conn: add_watchlist_symbols(conn, picked))

    @app.post("/universe/watchlist/remove")
    def universe_watchlist_remove(symbol: str = Form(...)) -> RedirectResponse:
        return _universe_write(lambda conn: remove_watchlist_symbol(conn, symbol))

    @app.post("/settings/kakao")
    def settings_kakao_save(
        KAKAO_REST_API_KEY: str = Form(""),
        KAKAO_CLIENT_SECRET: str = Form(""),
        KAKAO_REDIRECT_URI: str = Form(""),
    ) -> RedirectResponse:
        try:
            upsert_env(
                env_path(PROJECT_ROOT),
                {
                    "KAKAO_REST_API_KEY": KAKAO_REST_API_KEY,
                    "KAKAO_CLIENT_SECRET": KAKAO_CLIENT_SECRET,
                    "KAKAO_REDIRECT_URI": KAKAO_REDIRECT_URI,
                },
            )
        except ValueError:
            return RedirectResponse("/settings?err=invalid_value", status_code=303)
        return RedirectResponse("/settings", status_code=303)

    @app.get("/auth/kakao/connect")
    def kakao_connect() -> RedirectResponse:
        env = read_env_map(env_path(PROJECT_ROOT))
        if not (env.get("KAKAO_REST_API_KEY") or "").strip():
            return RedirectResponse("/settings?kakao=need_key", status_code=303)
        state = secrets.token_urlsafe(24)
        oauth_state_path(PROJECT_ROOT).write_text(state, encoding="utf-8")
        return RedirectResponse(authorize_url(env, state), status_code=303)

    @app.get("/auth/kakao/callback")
    def kakao_callback(request: Request) -> RedirectResponse:
        env_file = env_path(PROJECT_ROOT)
        env = read_env_map(env_file)
        if request.query_params.get("error"):
            return RedirectResponse("/settings?kakao=denied", status_code=303)
        code = request.query_params.get("code") or ""
        state = request.query_params.get("state") or ""
        expected = ""
        sp = oauth_state_path(PROJECT_ROOT)
        if sp.exists():
            expected = sp.read_text(encoding="utf-8").strip()
            sp.unlink(missing_ok=True)
        if not code or not state or state != expected:
            return RedirectResponse("/settings?kakao=auth_failed", status_code=303)
        try:
            exchange_code(env, code, env_file=env_file)
        except Exception:  # noqa: BLE001 — never leak token errors to the UI
            return RedirectResponse("/settings?kakao=auth_failed", status_code=303)
        return RedirectResponse("/settings?kakao=connected", status_code=303)

    @app.post("/auth/kakao/disconnect")
    def kakao_disconnect() -> RedirectResponse:
        clear_env_keys(env_path(PROJECT_ROOT), KAKAO_TOKEN_KEYS)
        return RedirectResponse("/settings?kakao=disconnected", status_code=303)

    @app.post("/auth/kakao/test")
    def kakao_test() -> RedirectResponse:
        env = read_env_map(env_path(PROJECT_ROOT))
        st = connection_status(env)
        if st is KakaoStatus.NOT_CONFIGURED:
            return RedirectResponse("/settings?kakao=need_key", status_code=303)
        if st is not KakaoStatus.CONNECTED:
            return RedirectResponse("/settings?kakao=not_connected", status_code=303)
        alert = AlertRecord(
            kind=AlertKind.RUNTIME,
            message="테스트 알림입니다. 브로커 주문은 실행되지 않습니다.",
        )
        result = notify_kakao(alert, env_file=env_path(PROJECT_ROOT))
        if not result.ok:
            return RedirectResponse("/settings?kakao=send_failed", status_code=303)
        return RedirectResponse("/settings?kakao=sent", status_code=303)

    @app.get("/api/health")
    def api_health() -> dict[str, object]:
        cfg = live_settings()
        return {
            "ok": True,
            "ui_stack": cfg.ui_stack,
            "btc_symbol": cfg.btc_symbol,
            "horizons": list(cfg.horizons),
            "trading": False,
        }

    @app.get("/api/v1/snapshot")
    def api_snapshot() -> JSONResponse:
        return JSONResponse(snapshot(compute=True))

    @app.get("/api/v1/runtime")
    def api_runtime() -> dict[str, object]:
        store = _store(writer=False)
        try:
            rows = store.conn.execute(
                "SELECT created_at, kind, status, message FROM runtime_events ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
            return {"events": [{"at": str(a), "kind": b, "status": c, "message": d} for a, b, c, d in rows]}
        finally:
            store.close()

    @app.get("/api/v1/journal")
    def api_journal() -> dict[str, object]:
        store = _store(writer=False)
        try:
            rows = store.conn.execute(
                "SELECT created_at, kind, payload_json FROM model_change_journal ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
            return {"events": [{"at": str(a), "kind": b, "payload": c} for a, b, c in rows]}
        finally:
            store.close()

    @app.get("/api/v1/data-health")
    def api_data_health() -> dict[str, object]:
        store = _store(writer=False)
        try:
            row = store.conn.execute(
                "SELECT overall, equity_feed, btc_feed, coverage_json, created_at FROM data_health_snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return {"overall": "unknown", "note": "Run a scan."}
            return {
                "overall": row[0],
                "equity_feed": row[1],
                "btc_feed": row[2],
                "coverage": row[3],
                "created_at": str(row[4]),
            }
        finally:
            store.close()

    spa = STATIC_DIR / "app"
    if spa.exists():
        app.mount("/app", StaticFiles(directory=str(spa), html=True), name="spa")

    if os.environ.get("INVESTASSIST_DAILY_SCAN") == "1":
        _start_daily_scan_thread()

    return app


_daily_scan_started = False


def _start_daily_scan_thread() -> None:
    """Weekday NYSE pre-open: POST /run-once. Skips if today already scanned."""
    global _daily_scan_started
    if _daily_scan_started:
        return
    _daily_scan_started = True

    def loop() -> None:
        import httpx

        from trading_system.daily_scan import claim_due_auto_job

        while True:
            time.sleep(30)
            try:
                cfg = get_settings()
                db = cfg.duckdb_path if cfg.duckdb_path.is_absolute() else PROJECT_ROOT / cfg.duckdb_path
                job = claim_due_auto_job(
                    db,
                    datetime.now(timezone.utc),
                    scan_enabled=cfg.daily_scan_enabled,
                    train_enabled=cfg.daily_train_enabled,
                    minutes_before=cfg.daily_scan_minutes_before_open,
                    minutes_after_close=cfg.daily_train_minutes_after_close,
                )
                if job == "scan":
                    httpx.post(
                        "http://127.0.0.1:8743/run-once?source=auto",
                        timeout=5.0,
                        follow_redirects=False,
                    )
                elif job == "train":
                    httpx.post(
                        "http://127.0.0.1:8743/train-once?source=auto",
                        timeout=5.0,
                        follow_redirects=False,
                    )
            except Exception:  # noqa: BLE001 — scheduler must never kill the UI
                pass

    threading.Thread(target=loop, name="daily-preopen-scan", daemon=True).start()


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Investment assistant (no auto-trading)")
    parser.add_argument("--init", action="store_true", help="Seed/train one cycle then exit")
    parser.add_argument("--live", action="store_true", help="Start background live refresh loop")
    args = parser.parse_args()
    settings = get_settings()
    print("API keys: open http://127.0.0.1:8743/settings if you want live Alpaca/LLM.")
    print("App starts without keys. Missing Alpaca/LLM is NOT_CONFIGURED, not a stub success.")
    if args.init:
        store = _store()
        try:
            out = run_v1_cycle(store, settings, artifacts_dir=PROJECT_ROOT / "data" / "artifacts")
            print("init tick", out["tick_id"])
        finally:
            store.close()
        return
    if args.live:
        from trading_system.runtime_loop import LiveRuntime

        LiveRuntime(settings, project_root=PROJECT_ROOT).start()
        print("Live runtime loop started (never-block).")
    os.environ["INVESTASSIST_DAILY_SCAN"] = "1"
    uvicorn.run(
        "trading_system.ui.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=8743,
        reload=False,
    )


app = create_app()


if __name__ == "__main__":
    main()
