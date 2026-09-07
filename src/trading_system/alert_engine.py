"""Stop / take-profit / disagreement alerts. Never place orders."""

from __future__ import annotations

import math

from trading_system.actionability import actionable_view
from trading_system.alerts import AlertChannel, AlertKind, AlertOutboxEntry, AlertRecord
from trading_system.btc_sleeve import BtcSleeveState
from trading_system.config import Settings
from trading_system.ids import InstrumentId
from trading_system.kakao import KAKAO_TEXT_MAX
from trading_system.llm_log import ticker_from_instrument
from trading_system.recommendations import RecommendationAction, RecommendationRecord

_MATERIAL_ACTIONS = {
    RecommendationAction.BUY,
    RecommendationAction.SELL,
    RecommendationAction.ENTER,
    RecommendationAction.ADD,
    RecommendationAction.REDUCE,
    RecommendationAction.EXIT,
}
_ACTION_KO = {
    "BUY": "매수",
    "SELL": "매도",
    "HOLD": "보유",
    "REDUCE": "축소",
    "ENTER": "진입",
    "ADD": "추가",
    "EXIT": "청산",
    "NO_ACTION": "관망",
}


STOP_FORMULA = "stop_atr_v1"
TP_FORMULA = "tp_horizon_v1"


def stop_level(price: float, vol: float, confidence: float) -> float:
    atr = max(price * max(vol, 0.01), price * 0.02)
    return price - atr * (1.2 - 0.4 * min(1.0, confidence))


def take_profit_level(price: float, expected: float) -> float:
    return price * (1.0 + max(0.01, expected))


def _valid_price(price: float) -> bool:
    return math.isfinite(price) and price > 0


def evaluate_alerts(
    *,
    price: float,
    expected: float,
    vol: float,
    confidence: float,
    agreement: str,
    instrument_id: InstrumentId,
    tick_id: str,
    btc_blocked: bool = False,
) -> list[AlertRecord]:
    if not _valid_price(price):
        return []
    out: list[AlertRecord] = []
    stop = stop_level(price, vol, confidence)
    tp = take_profit_level(price, expected)
    if price <= stop * 1.01:
        out.append(
            AlertRecord(
                kind=AlertKind.STOP_LOSS,
                instrument_id=instrument_id,
                tick_id=tick_id,  # type: ignore[arg-type]
                message=f"Stop approaching ({STOP_FORMULA}) stop={stop:.4f} px={price:.4f}",
                level="warning",
            )
        )
    if price >= tp * 0.99:
        out.append(
            AlertRecord(
                kind=AlertKind.TAKE_PROFIT,
                instrument_id=instrument_id,
                tick_id=tick_id,  # type: ignore[arg-type]
                message=f"Take-profit approaching ({TP_FORMULA}) tp={tp:.4f} px={price:.4f}",
                level="info",
            )
        )
    if agreement == "conflict":
        out.append(
            AlertRecord(
                kind=AlertKind.EXIT_DOWNGRADE,
                instrument_id=instrument_id,
                tick_id=tick_id,  # type: ignore[arg-type]
                message="Horizon conflict spike",
                level="warning",
            )
        )
    if btc_blocked:
        out.append(
            AlertRecord(
                kind=AlertKind.DATA_HEALTH,
                instrument_id=instrument_id,
                tick_id=tick_id,  # type: ignore[arg-type]
                message="BTC sleeve transfer/liquidity blocks immediate action",
                level="info",
            )
        )
    for a in out:
        assert a.places_orders is False
    return out


def console_notify(alert: AlertRecord) -> AlertOutboxEntry:
    print(f"[ALERT:{alert.kind}] {alert.message}")
    return AlertOutboxEntry(
        alert_id=alert.alert_id,
        channel=AlertChannel.CONSOLE,
        idempotency_key=f"{alert.alert_id}:console",
    )


def _ticker(instrument_id: object) -> str:
    s = str(instrument_id or "")
    if "btc" in s.lower():
        return "BTC"
    t = ticker_from_instrument(s)
    return t or s.upper()


def _pct(v: object) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def format_recommendation_message(
    rec: RecommendationRecord,
    *,
    data_status: str,
    llm_status: str,
) -> str:
    action = rec.action.value
    lines = [
        f"{_ticker(rec.instrument_id)} — {_ACTION_KO.get(action, action)}",
        "",
    ]
    if rec.current_units is not None:
        lines.append(f"현재 단위: {rec.current_units:g}")
    if rec.recommended_units is not None:
        lines.append(f"권장 단위: {rec.recommended_units:g}")
    if rec.delta_units is not None:
        sign = "+" if rec.delta_units > 0 else ""
        lines.append(f"변화: {sign}{rec.delta_units:g}")
    if rec.confidence is not None:
        lines.append(f"신뢰도: {_pct(rec.confidence)}")
    for h in rec.horizons:
        lines.append(f"{h.horizon}일: {_pct(h.expected_return)}")
    lines.append(f"데이터: {data_status}")
    lines.append(f"LLM: {llm_status}")
    if rec.thesis:
        lines.append(f"이유: {rec.thesis}")
    return "\n".join(lines)


def _rec_action(rec: object) -> str:
    raw = rec.action if hasattr(rec, "action") else (rec or {}).get("action")  # type: ignore[union-attr]
    return str(getattr(raw, "value", raw) or "").upper()


def _rec_instrument(rec: object) -> str:
    raw = rec.instrument_id if hasattr(rec, "instrument_id") else (rec or {}).get("instrument_id")  # type: ignore[union-attr]
    return str(raw or "")


def _unique(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _fit_line(prefix: str, names: list[str], budget: int) -> str:
    if budget < len(prefix) + 2:
        return prefix[: max(0, budget)]
    if not names:
        line = f"{prefix}없음"
        return line if len(line) <= budget else prefix[:budget]
    for n in range(len(names), 0, -1):
        shown = names[:n]
        extra = len(names) - n
        body = ", ".join(shown)
        if extra:
            body += f" 외{extra}"
        line = f"{prefix}{body}"
        if len(line) <= budget:
            return line
    fallback = f"{prefix}외{len(names)}"
    return fallback if len(fallback) <= budget else prefix[:budget]


def format_scan_kakao_digest(
    *,
    final_recs: list[object],
    quant_recs: list[object] | None = None,
    llm_status: str | None = None,
    final_text: str | None = None,
) -> str:
    """One Kakao-sized scan summary: final buy/add and sell/exit tickers.

    Kakao text templates cap at 200 characters; longer per-name alerts were truncated
    into unreadable fragments.
    """
    raw_final = " ".join(str(final_text or "").split())
    if raw_final:
        prefix = "[Stock AI] GPT 최종 판단\n"
        return (prefix + raw_final[: KAKAO_TEXT_MAX - len(prefix)])[:KAKAO_TEXT_MAX]

    judged = bool(final_recs)
    recs = list(final_recs) if judged else list(quant_recs or [])
    _ = llm_status
    buys: list[str] = []
    sells: list[str] = []
    for rec in recs:
        action = _rec_action(rec)
        ticker = _ticker(_rec_instrument(rec))
        if not ticker:
            continue
        if action in {"ENTER", "BUY"}:
            buys.append(ticker)
        elif action == "ADD":
            buys.append(f"{ticker}(추가)")
        elif action in {"EXIT", "SELL"}:
            sells.append(ticker)
        elif action == "REDUCE":
            sells.append(f"{ticker}(축소)")
    buys = _unique(buys)
    sells = _unique(sells)
    head = "[Stock AI] 스캔완료"
    if not judged:
        head += "(Quant)"
    rest = KAKAO_TEXT_MAX - len(head) - 2
    if rest < 8:
        return head[:KAKAO_TEXT_MAX]
    sell_budget = min(max(8, rest // 2), rest - 8)
    buy_budget = rest - sell_budget
    buy_line = _fit_line("매수:", buys, buy_budget)
    sell_line = _fit_line("매도:", sells, sell_budget)
    text = f"{head}\n{buy_line}\n{sell_line}"
    return text[:KAKAO_TEXT_MAX]


def collect_cycle_notifications(
    *,
    recommendations: list[RecommendationRecord],
    tick_id: str,
    data_status: str,
    llm_status: str,
    market_status: str,
    alpaca_configured: bool,
    health_overall: str | None = None,
    max_recs: int = 5,
    settings: Settings | None = None,
    total_base_units: float = 1000.0,
) -> list[AlertRecord]:
    """Material rec / health / runtime alerts. Skip HOLD and non-actionable recs."""
    out: list[AlertRecord] = []
    n = 0
    cfg = settings or Settings(_env_file=None)
    for rec in recommendations:
        if not rec.actionable:
            continue
        view = actionable_view(rec, settings=cfg, total_base_units=total_base_units)
        if view.suppressed or view.display_action not in {a.value for a in _MATERIAL_ACTIONS}:
            continue
        if rec.action not in _MATERIAL_ACTIONS:
            continue
        if rec.delta_units is not None and abs(rec.delta_units) < 1e-9:
            continue
        cur = rec.current_units or 0.0
        recu = rec.recommended_units or 0.0
        delta = rec.delta_units if rec.delta_units is not None else recu - cur
        basis = max(abs(cur), abs(recu), 1.0)
        if abs(delta) / basis < 0.05 and abs(delta) < 1.0:
            continue
        n += 1
        if n > max_recs:
            break
        out.append(
            AlertRecord(
                kind=AlertKind.RECOMMENDATION,
                instrument_id=rec.instrument_id,
                tick_id=tick_id,  # type: ignore[arg-type]
                message=format_recommendation_message(
                    rec, data_status=data_status, llm_status=llm_status
                ),
                level="info",
            )
        )
    overall = (health_overall or "").lower()
    if alpaca_configured and overall in {"unavailable", "critical"}:
        out.append(
            AlertRecord(
                kind=AlertKind.DATA_HEALTH,
                tick_id=tick_id,  # type: ignore[arg-type]
                message=f"데이터 상태 저하: {overall}",
                level="warning",
            )
        )
    if alpaca_configured and market_status == "UNAVAILABLE":
        out.append(
            AlertRecord(
                kind=AlertKind.RUNTIME,
                tick_id=tick_id,  # type: ignore[arg-type]
                message="시장 데이터 사용 불가 — 확인이 필요합니다.",
                level="warning",
            )
        )
    for a in out:
        assert a.places_orders is False
    return out
