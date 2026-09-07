"""Portfolio units / lots — actual account money not required."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from trading_system.ids import InstrumentId, USD_CASH_INSTRUMENT_ID, canonicalize_instrument_id


class ValuationState(StrEnum):
    COMPLETE = "complete"
    PARTIAL_VALUATION = "partial_valuation"
    INCOMPLETE = "incomplete"


class Lot(BaseModel):
    """One acquisition lot in user-defined units (not required to be currency)."""

    lot_id: str = Field(default_factory=lambda: f"lot_{uuid4().hex[:12]}")
    instrument_id: InstrumentId
    acquisition_units: float
    acquisition_price: float  # price per unit in same unit basis
    acquired_on: date
    notes: str | None = None

    @field_validator("instrument_id", mode="before")
    @classmethod
    def _canonical_instrument_id(cls, v: object) -> InstrumentId:
        return canonicalize_instrument_id(v)

    @field_validator("acquisition_units")
    @classmethod
    def _positive_units(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("acquisition_units must be a positive finite number")
        return v

    @field_validator("acquisition_price")
    @classmethod
    def _positive_price(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("acquisition_price must be a positive finite number")
        return v


class PositionEventKind(StrEnum):
    ADD_LOT = "add_lot"
    REDUCE_LOT = "reduce_lot"
    CLOSE_LOT = "close_lot"
    ADJUST_UNITS = "adjust_units"


class PositionEvent(BaseModel):
    """Manual holdings change — user-entered only; AI never mutates holdings."""

    event_id: str = Field(default_factory=lambda: f"pe_{uuid4().hex}")
    kind: PositionEventKind
    instrument_id: InstrumentId
    lot_id: str | None = None
    units: float | None = None
    price: float | None = None
    as_of: date
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    note: str | None = None


class InstrumentPosition(BaseModel):
    instrument_id: InstrumentId
    lots: list[Lot] = Field(default_factory=list)
    marked_units_final: float | None = None  # FINAL-close official
    marked_units_intraday_preview: float | None = None  # never overwrites final
    price_available: bool = False

    @property
    def acquisition_units_total(self) -> float:
        return sum(lot.acquisition_units for lot in self.lots)


class PortfolioUnits(BaseModel):
    """Unit-first portfolio privacy model."""

    total_base_units: float = 1000.0
    positions: list[InstrumentPosition] = Field(default_factory=list)
    cash_instrument_id: InstrumentId = USD_CASH_INSTRUMENT_ID
    cash_units: float = 0.0
    valuation_state: ValuationState = ValuationState.INCOMPLETE
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("total_base_units")
    @classmethod
    def _positive_base(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("total_base_units must be a positive finite number")
        return v

    def recompute_valuation_state(self) -> ValuationState:
        if not self.positions:
            self.valuation_state = ValuationState.COMPLETE
            return self.valuation_state
        priced = sum(1 for p in self.positions if p.price_available)
        if priced == len(self.positions):
            self.valuation_state = ValuationState.COMPLETE
        elif priced == 0:
            self.valuation_state = ValuationState.INCOMPLETE
        else:
            self.valuation_state = ValuationState.PARTIAL_VALUATION
        return self.valuation_state

    def actionable_for_recommendations(self) -> bool:
        """Incomplete / partial valuation must not feed rec/alert math."""
        return self.valuation_state == ValuationState.COMPLETE
