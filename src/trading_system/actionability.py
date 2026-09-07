"""User-facing actionability / deadband. Does not change model predictions.

Raw recommended_units stay on the record for diagnostics and scoring.
This layer only decides what a person should see as an action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from trading_system.config import Settings
from trading_system.recommendations import RecommendationAction

FORMULA_VERSION = "actionability_v1"

REASON_SMALL_TARGET = "SMALL_TARGET"
REASON_SMALL_DELTA = "SMALL_DELTA"

NOTE_SMALL_TARGET = "추천 규모가 작아 실행하지 않습니다."
NOTE_SMALL_DELTA = "변화 폭이 작아 실행하지 않습니다."

_BUYISH = {
    RecommendationAction.BUY.value,
    RecommendationAction.ENTER.value,
    RecommendationAction.ADD.value,
}


@dataclass(frozen=True)
class ActionableView:
    raw_action: str
    display_action: str
    display_label: str
    display_recommended_units: float | None
    display_delta_units: float | None
    raw_recommended_units: float | None
    raw_delta_units: float | None
    suppressed: bool
    reason_code: str | None
    note: str | None
    quantum: float
    min_target_units: float
    min_delta_units: float
    formula_version: str = FORMULA_VERSION

    @property
    def is_buy(self) -> bool:
        return (not self.suppressed) and self.display_action in _BUYISH


def _f(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(n):
        return None
    return n


def min_target_units(settings: Settings, total_base_units: float) -> float:
    base = max(1e-9, float(total_base_units))
    return max(float(settings.min_actionable_units_floor), float(settings.min_actionable_weight) * base)


def min_delta_units(settings: Settings, total_base_units: float) -> float:
    base = max(1e-9, float(total_base_units))
    return max(float(settings.min_actionable_units_floor), float(settings.min_delta_weight) * base)


def display_unit_quantum(settings: Settings, total_base_units: float) -> float:
    """Practical increment: ~0.1% of the book, at least 1u when the book is ≥100u."""
    base = max(1e-9, float(total_base_units))
    step = max(0.0, float(settings.display_unit_step_weight)) * base
    if base >= 100:
        return max(1.0, float(round(step)) if step >= 0.5 else 1.0)
    return max(0.1, step if step > 0 else 0.1)


def round_units(value: float, quantum: float) -> float:
    if quantum <= 0:
        return value
    return round(value / quantum) * quantum


def _label(action: str, *, reason_code: str | None) -> str:
    if reason_code == REASON_SMALL_TARGET or action == RecommendationAction.NO_ACTION.value:
        return "관망"
    if reason_code == REASON_SMALL_DELTA:
        return "변화 없음"
    return {
        "BUY": "매수",
        "SELL": "매도",
        "HOLD": "보유",
        "REDUCE": "축소",
        "ENTER": "진입",
        "ADD": "추가",
        "EXIT": "청산",
        "NO_ACTION": "관망",
    }.get(action, action)


def _as_mapping(rec: object) -> Mapping[str, Any]:
    if isinstance(rec, Mapping):
        return rec
    dump = getattr(rec, "model_dump", None)
    if callable(dump):
        return dump()
    return {
        "action": getattr(rec, "action", None),
        "current_units": getattr(rec, "current_units", None),
        "recommended_units": getattr(rec, "recommended_units", None),
        "delta_units": getattr(rec, "delta_units", None),
        "marked_units_final": getattr(rec, "marked_units_final", None),
        "acquisition_units": getattr(rec, "acquisition_units", None),
    }


def actionable_view(
    rec: object,
    *,
    settings: Settings,
    total_base_units: float,
) -> ActionableView:
    """Map a raw recommendation onto a human-actionable view."""
    row = _as_mapping(rec)
    raw_action = str(getattr(row.get("action"), "value", row.get("action") or "") or "")
    raw_target = _f(row.get("recommended_units"))
    current = _f(row.get("current_units"))
    if current is None:
        current = _f(row.get("marked_units_final"))
    if current is None:
        current = _f(row.get("acquisition_units")) or 0.0
    raw_delta = _f(row.get("delta_units"))
    if raw_delta is None and raw_target is not None:
        raw_delta = raw_target - current

    base = max(1e-9, float(total_base_units))
    min_t = min_target_units(settings, base)
    min_d = min_delta_units(settings, base)
    quantum = display_unit_quantum(settings, base)

    disp_target = None if raw_target is None else round_units(raw_target, quantum)
    disp_delta = None if raw_delta is None else round_units(raw_delta, quantum)

    display_action = raw_action or RecommendationAction.NO_ACTION.value
    reason: str | None = None
    note: str | None = None
    suppressed = False

    has_position = abs(current) > 1e-12
    target_mag = 0.0 if raw_target is None else abs(raw_target)
    delta_mag = 0.0 if raw_delta is None else abs(raw_delta)
    disp_target_mag = 0.0 if disp_target is None else abs(disp_target)
    disp_delta_mag = 0.0 if disp_delta is None else abs(disp_delta)
    would_act = raw_action in _BUYISH or raw_action in {"SELL", "REDUCE", "EXIT"}

    if not has_position and (target_mag < min_t or disp_target_mag < min_t):
        display_action = RecommendationAction.NO_ACTION.value
        if would_act:
            reason = REASON_SMALL_TARGET
            note = NOTE_SMALL_TARGET
            suppressed = True
        if raw_target is not None:
            disp_target = 0.0
        if raw_delta is not None:
            disp_delta = 0.0
    elif has_position and (delta_mag < min_d or disp_delta_mag < min_d):
        display_action = RecommendationAction.HOLD.value
        reason = REASON_SMALL_DELTA
        if would_act:
            note = NOTE_SMALL_DELTA
            suppressed = True
        disp_target = round_units(current, quantum)
        disp_delta = 0.0

    return ActionableView(
        raw_action=raw_action,
        display_action=display_action,
        display_label=_label(display_action, reason_code=reason),
        display_recommended_units=disp_target,
        display_delta_units=disp_delta,
        raw_recommended_units=raw_target,
        raw_delta_units=raw_delta,
        suppressed=suppressed,
        reason_code=reason,
        note=note,
        quantum=quantum,
        min_target_units=min_t,
        min_delta_units=min_d,
    )
