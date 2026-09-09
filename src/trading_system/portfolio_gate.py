"""Deterministic post-LLM portfolio safety gate. Does not place orders.

GPT remains the investment decision-maker. This only clamps mathematically
impossible unit recommendations onto the existing Quant constraints.
"""

from __future__ import annotations

import math
from typing import Any

from trading_system.btc_sleeve import BtcSleeveState
from trading_system.config import Settings
from trading_system.recommendations import RecommendationRecord, normalize_position


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def _is_btc(instrument_id: object) -> bool:
    return "btc" in str(instrument_id).lower()


def _finite_nonneg(value: object) -> float:
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(n) or n < 0:
        return 0.0
    return n


def unit_cap(
    *,
    instrument_id: str,
    settings: Settings,
    total_base_units: float,
    other_stock_units: float,
    btc_units: float,
    btc_blocked: bool,
) -> float:
    base = max(1e-9, float(total_base_units))
    min_cash = _clamp01(settings.min_cash_weight)
    max_stock = _clamp01(settings.max_total_stock_weight)
    max_single = min(max_stock, _clamp01(settings.max_single_stock_weight))
    max_btc = _clamp01(settings.max_btc_weight)
    investable = max(0.0, (1.0 - min_cash) * base)
    other = max(0.0, float(other_stock_units))
    btc = max(0.0, float(btc_units))
    if _is_btc(instrument_id):
        if btc_blocked:
            return btc
        room = max(0.0, investable - other)
        return min(max_btc * base, room)
    room_total = max(0.0, max_stock * base - other)
    room_cash = max(0.0, investable - other - (0.0 if btc_blocked else btc))
    if btc_blocked:
        room_cash = max(0.0, investable - other - btc)
    return min(max_single * base, room_total, room_cash)


def constrain_units(
    *,
    instrument_id: str,
    requested: float | None,
    current: float | None,
    settings: Settings,
    total_base_units: float,
    other_stock_units: float,
    btc_units: float,
    btc_blocked: bool,
) -> tuple[float, float | None, list[str]]:
    """Return (constrained, requested_or_none, reason codes)."""
    reasons: list[str] = []
    raw = requested
    if raw is None:
        req = 0.0
    else:
        try:
            req = float(raw)
        except (TypeError, ValueError):
            req = 0.0
            reasons.append("UNITS_CLAMPED_NONNEGATIVE")
        if not math.isfinite(req) or req < 0:
            req = 0.0
            reasons.append("UNITS_CLAMPED_NONNEGATIVE")
    if _is_btc(instrument_id) and btc_blocked:
        kept = _finite_nonneg(current)
        if abs(kept - req) > 1e-9:
            reasons.append("BTC_TRANSFER_BLOCK")
        return kept, requested, reasons
    cap = unit_cap(
        instrument_id=instrument_id,
        settings=settings,
        total_base_units=total_base_units,
        other_stock_units=other_stock_units,
        btc_units=btc_units,
        btc_blocked=btc_blocked,
    )
    constrained = min(req, cap)
    if constrained < req - 1e-9:
        reasons.append("UNITS_CONSTRAINED")
    return constrained, requested, reasons


def apply_portfolio_gate(
    recs: list[RecommendationRecord],
    *,
    settings: Settings,
    total_base_units: float,
    stock_marked: dict[str, float],
    btc_state: BtcSleeveState,
    skip_reasons: set[str] | None = None,
) -> list[RecommendationRecord]:
    """Clamp successful GPT unit calls. Skip records are left unchanged."""
    skip = skip_reasons or {
        "NOT_A_CANDIDATE",
        "NOT_SELECTED_FOR_FINAL_JUDGE",
        "NOT_CONFIGURED",
        "UNAVAILABLE",
        "RATE_LIMITED",
        "QUOTA_EXCEEDED",
        "BUDGET_EXCEEDED",
    }
    book = {str(k): float(v) for k, v in stock_marked.items()}
    btc_u = float(btc_state.current_units or 0.0)
    blocked = btc_state.transfer_blocks_immediate_rebalance()
    out: list[RecommendationRecord] = []
    for rec in recs:
        reasons = {str(x) for x in (rec.override_reasons or [])}
        if reasons & skip:
            out.append(rec)
            continue
        if rec.recommended_units is None:
            out.append(rec)
            continue
        inst = str(rec.instrument_id)
        is_btc = _is_btc(inst)
        current = float(rec.current_units or 0.0)
        others = sum(v for k, v in book.items() if k != inst)
        constrained, requested, extra = constrain_units(
            instrument_id=inst,
            requested=rec.recommended_units,
            current=rec.current_units if rec.current_units is not None else (btc_u if is_btc else book.get(inst, 0.0)),
            settings=settings,
            total_base_units=total_base_units,
            other_stock_units=others,
            btc_units=btc_u,
            btc_blocked=blocked,
        )
        if not (is_btc and blocked):
            if float(rec.recommended_units) <= current:
                # A HOLD/WATCH must not become a forced reduction just because an
                # existing position exceeds today's cap. Respect explicit reductions.
                constrained = float(rec.recommended_units)
                extra = []
            else:
                # Lack of room can block an increase, but cannot turn BUY into SELL.
                constrained = max(current, constrained)
        merged = list(rec.override_reasons or []) + extra
        cur = rec.current_units
        delta = None if cur is None else constrained - float(cur)
        out.append(
            rec.model_copy(
                update={
                    "requested_units": requested,
                    "constrained_units": constrained,
                    "recommended_units": constrained,
                    "delta_units": delta,
                    "override_reasons": merged,
                }
            )
        )
        if is_btc:
            btc_u = constrained
        else:
            book[inst] = constrained
    return out


def build_portfolio_context(
    *,
    total_base_units: float,
    settings: Settings,
    stock_marked: dict[str, float],
    stock_acquisition: dict[str, float],
    quant_targets: dict[str, float | None],
    btc_state: BtcSleeveState,
    unpriced: list[str],
    cash_units: float | None,
    allocation_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unit/weight snapshot for the final judge. No account-money amounts."""
    base = max(1e-9, float(total_base_units))
    holdings: list[dict[str, Any]] = []
    for inst in sorted(set(stock_marked) | set(stock_acquisition)):
        marked = stock_marked.get(inst)
        acquisition = float(stock_acquisition.get(inst, 0.0) or 0.0)
        position = normalize_position({"current_units": marked, "acquisition_units": acquisition})
        marked = position["current_units"]
        valuation_available = marked is not None
        holdings.append(
            {
                "instrument_id": inst,
                "position_held": position["position_held"],
                "acquisition_units": acquisition,
                "marked_units_final": marked,
                "quant_recommended_units": quant_targets.get(inst),
                "valuation_status": position["valuation_status"],
                "weight": (float(marked) / base) if valuation_available else None,
            }
        )
    btc_w = float(btc_state.current_units or 0.0) / base
    stock_w = sum(stock_marked.values()) / base
    valued = [h for h in holdings if h["weight"] is not None]
    largest = max(valued, key=lambda h: float(h["weight"]), default=None)
    return {
        "total_base_units": base,
        "cash_units": cash_units,
        "cash_floor_weight": float(settings.min_cash_weight),
        "holdings": holdings,
        "unpriced_holdings": list(unpriced),
        "valuation_status": "INCOMPLETE" if unpriced else "COMPLETE",
        "concentration": {
            "largest_instrument_id": (largest or {}).get("instrument_id"),
            "largest_weight": (largest or {}).get("weight"),
            "stock_weight": None if unpriced else stock_w,
            "btc_weight": None if unpriced else btc_w,
        },
        "btc_sleeve": {
            "liquidity": str(getattr(btc_state.liquidity, "value", btc_state.liquidity)),
            "current_units": float(btc_state.current_units or 0.0),
            "blocked": btc_state.transfer_blocks_immediate_rebalance(),
            "available": not btc_state.transfer_blocks_immediate_rebalance(),
            "unsettled_or_transfer_pending": btc_state.transfer_blocks_immediate_rebalance(),
        },
        "constraints": {
            "max_single_stock_weight": settings.max_single_stock_weight,
            "max_total_stock_weight": settings.max_total_stock_weight,
            "min_cash_weight": settings.min_cash_weight,
            "max_btc_weight": settings.max_btc_weight,
        },
        "units_basis": "final_close",
        "note": "Units and weights only. No account currency amounts.",
        "allocation_trace": allocation_trace or {},
    }


def portfolio_signature(*, stock_marked: dict[str, float], btc_units: float) -> str:
    parts = [f"{k}:{round(float(v), 4)}" for k, v in sorted(stock_marked.items())]
    parts.append(f"btc:{round(float(btc_units or 0.0), 4)}")
    return "|".join(parts)
