"""Recommendation records: quant_only and llm_final (separate persistence/scoring)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import math
from uuid import uuid4

from pydantic import BaseModel, Field

from trading_system.ids import DecisionEpochId, FeatureSnapshotId, InstrumentId, TickId


class RecommendationSource(StrEnum):
    QUANT_ONLY = "quant_only"
    RESEARCH_ADJUSTED = "research_adjusted"
    LLM_FINAL = "llm_final"


class RecommendationAction(StrEnum):
    NO_ACTION = "NO_ACTION"
    HOLD = "HOLD"
    ENTER = "ENTER"
    ADD = "ADD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    BUY = "BUY"
    SELL = "SELL"


def normalize_position(data: dict) -> dict:
    """Interpret v1/v2 unit records without treating acquisition units as a mark.

    In v1 acquisition_units is the sum of lot acquisition units, not shares or
    current value. A positive sum proves ownership only. Missing evidence or
    contradictory fields requires re-evaluation, never an assumed empty position.
    Returns a new mapping; historical JSON is never mutated.
    """
    out = dict(data)
    values = {}
    for key in ("current_units", "acquisition_units", "marked_units_final"):
        value = data.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError("REEVALUATION_REQUIRED:INVALID_POSITION_UNITS")
        values[key] = value
    current, acquisition = values["current_units"], values["acquisition_units"]
    explicit = data.get("position_held")
    if explicit is not None and not isinstance(explicit, bool):
        raise ValueError("REEVALUATION_REQUIRED:INVALID_POSITION_HELD")
    positive = any(v is not None and v > 1e-12 for v in values.values())
    if explicit is False and positive:
        raise ValueError("REEVALUATION_REQUIRED:CONFLICTING_POSITION")
    held = explicit if explicit is not None else (True if positive else False if current == 0 or acquisition == 0 else None)
    if held is None:
        raise ValueError("REEVALUATION_REQUIRED:UNKNOWN_POSITION")
    if not held:
        out.update(position_held=False, current_units=0.0, valuation_status="NOT_HELD")
        return out
    if data.get("valuation_status") == "NOT_HELD":
        raise ValueError("REEVALUATION_REQUIRED:CONFLICTING_POSITION")
    if current is None:
        if values["marked_units_final"] is not None or data.get("valuation_status") == "FINAL":
            raise ValueError("REEVALUATION_REQUIRED:CONFLICTING_VALUATION")
        out.update(position_held=True, valuation_status="UNAVAILABLE", current_units=None,
                   recommended_units=None, delta_units=None, requested_units=None,
                   constrained_units=None, actionable=False, action="HOLD")
    else:
        if data.get("valuation_status") == "UNAVAILABLE":
            raise ValueError("REEVALUATION_REQUIRED:CONFLICTING_VALUATION")
        out.update(position_held=True, valuation_status="FINAL")
    return out


class HorizonOutlook(BaseModel):
    horizon: int
    expected_return: float | None = None
    rank_score: float | None = None
    confidence: float | None = None


class RecommendationRecord(BaseModel):
    recommendation_id: str = Field(default_factory=lambda: f"rec_{uuid4().hex}")
    source: RecommendationSource
    tick_id: TickId
    decision_epoch_id: DecisionEpochId
    feature_snapshot_id: FeatureSnapshotId
    instrument_id: InstrumentId
    action: RecommendationAction
    current_units: float | None = None
    recommended_units: float | None = None
    delta_units: float | None = None
    confidence: float | None = None
    urgency: str | None = None
    horizons: list[HorizonOutlook] = Field(default_factory=list)
    thesis: str | None = None
    contrary_evidence: str | None = None
    override_of_recommendation_id: str | None = None  # llm_final -> quant_only
    override_reasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # False when market data is synthetic/degraded — display/research only, not actionable.
    actionable: bool = True
    # Explicit: LLM fields never silently enter quant training/allocation
    llm_tainted: bool = False
    acquisition_units: float | None = None
    marked_units_final: float | None = None
    marked_units_intraday_preview: float | None = None
    # Position existence and valuation availability are separate. In particular,
    # an owned name whose FINAL-close mark is unavailable must never become a
    # zero-unit non-holding merely because current_units is None.
    position_held: bool | None = None
    valuation_status: str | None = None
    requested_units: float | None = None
    constrained_units: float | None = None
    units_basis: str | None = None
    opportunity_score: float | None = None
    excess_score: float | None = None
    conviction: float | None = None
    equity_budget: float | None = None
    initial_units: float | None = None
    pre_floor_units: float | None = None
    exclusion_reason: str | None = None
    allocation_note: str | None = None
    rationale_detail: str | None = None
    # legacy-b style technical/chart-shape read (technical_factors.py). None when
    # there wasn't enough price history (< MIN_SESSIONS_FOR_FACTORS) to compute it.
    technical_track: str | None = None
    leader_score: float | None = None
    momentum_score: float | None = None
    volume_score: float | None = None
    buyable_score: float | None = None
    breakout_score: float | None = None
    top_risk_score: float | None = None
    quality_of_trend_score: float | None = None
    catalyst_score: float | None = None
    # Gemini research + adversarial pass (research_agent.research_ticker). None
    # unless this was a NEW-entry candidate that actually got researched (held
    # positions and BTC are never researched -- see v1_cycle.py).
    research_summary_ko: str | None = None
    research_rerating_score: float | None = None
    research_valuation_support_score: float | None = None
    research_cash_relative_score: float | None = None
    research_data_quality_score: float | None = None
    research_sizing_multiplier: float | None = None
    # Additive JSON fields: old persisted records remain readable and unmodified.
    input_id: str | None = None
    input_as_of: datetime | None = None
    price_snapshot: dict | None = None
    final_rank: int | None = None
    decision_status: str | None = None
    requested_action: str | None = None
    change_conditions: str | None = None
    previous_stage_recommendation_id: str | None = None


def quant_only_record(
    *,
    tick_id: TickId,
    decision_epoch_id: DecisionEpochId,
    feature_snapshot_id: FeatureSnapshotId,
    instrument_id: InstrumentId,
    action: RecommendationAction,
    **extra: object,
) -> RecommendationRecord:
    data = {
        "source": RecommendationSource.QUANT_ONLY,
        "llm_tainted": False,
        "tick_id": tick_id,
        "decision_epoch_id": decision_epoch_id,
        "feature_snapshot_id": feature_snapshot_id,
        "instrument_id": instrument_id,
        "action": action,
        **extra,
    }
    return RecommendationRecord.model_validate(data)


def llm_final_record(
    *,
    tick_id: TickId,
    decision_epoch_id: DecisionEpochId,
    feature_snapshot_id: FeatureSnapshotId,
    instrument_id: InstrumentId,
    action: RecommendationAction,
    **extra: object,
) -> RecommendationRecord:
    data = {
        "source": RecommendationSource.LLM_FINAL,
        "llm_tainted": True,
        "tick_id": tick_id,
        "decision_epoch_id": decision_epoch_id,
        "feature_snapshot_id": feature_snapshot_id,
        "instrument_id": instrument_id,
        "action": action,
        **extra,
    }
    return RecommendationRecord.model_validate(data)
