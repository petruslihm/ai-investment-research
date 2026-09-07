"""Market-data provider interfaces only (no real ingest in foundation)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ValidationInfo, field_validator, model_validator

from trading_system.ids import InstrumentId, RequestSetId
from trading_system.providers.never_block import NeverBlockResult, RetryPolicy


class BarFinality(StrEnum):
    PRELIMINARY = "preliminary"
    FINAL = "final"


class DailyBar(BaseModel):
    instrument_id: InstrumentId
    session_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    finality: BarFinality = BarFinality.FINAL
    adjustment_revision: str
    provider: str
    receive_ts: datetime

    @field_validator("open", "high", "low", "close", "volume")
    @classmethod
    def _finite_nonneg(cls, v: float, info: ValidationInfo) -> float:
        if not math.isfinite(v):
            raise ValueError("bar fields must be finite")
        name = info.field_name
        if name != "volume" and v <= 0:
            raise ValueError("prices must be positive")
        if name == "volume" and v < 0:
            raise ValueError("volume must be non-negative")
        return v

    @model_validator(mode="after")
    def _ohlc_ok(self) -> DailyBar:
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if self.close > self.high or self.close < self.low:
            raise ValueError("close must be within [low, high]")
        if self.open > self.high or self.open < self.low:
            raise ValueError("open must be within [low, high]")
        return self


class LatestQuote(BaseModel):
    instrument_id: InstrumentId
    price: float
    event_ts: datetime | None
    receive_ts: datetime
    volume: float | None = None
    provider: str
    degraded: bool = False


class UniverseMember(BaseModel):
    instrument_id: InstrumentId
    symbol: str
    as_of: date


class CoverageStatus(StrEnum):
    REQUESTED = "requested"
    SUCCEEDED = "succeeded"
    MISSING = "missing"
    ERROR = "error"
    QUARANTINED = "quarantined"


class CoverageRow(BaseModel):
    provider: str
    adjustment: str
    adjustment_revision: str
    instrument_id: InstrumentId
    session: date
    request_set_id: RequestSetId
    status: CoverageStatus


@runtime_checkable
class HistoricalDailyBarsProvider(Protocol):
    """Sole historical daily-bar network writer interface (impl in later phase)."""

    def fetch_daily_bars(
        self,
        instrument_ids: list[InstrumentId],
        start: date,
        end: date,
        *,
        request_set_id: RequestSetId,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult: ...


@runtime_checkable
class LatestQuoteProvider(Protocol):
    def fetch_quotes(
        self,
        instrument_ids: list[InstrumentId],
        *,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult: ...


@runtime_checkable
class EquityUniverseProvider(Protocol):
    def fetch_universe(
        self,
        as_of: date,
        *,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult: ...


@runtime_checkable
class BtcMarketDataProvider(Protocol):
    def fetch_btc_bars(
        self,
        start: date,
        end: date,
        *,
        request_set_id: RequestSetId,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult: ...

    def fetch_btc_quote(self, *, policy: RetryPolicy | None = None) -> NeverBlockResult: ...


class ProviderNotImplemented(ABC):
    """Marker base for adapters that are not wired in this process."""

    @abstractmethod
    def provider_name(self) -> str: ...


class UnimplementedProvider(ProviderNotImplemented):
    def provider_name(self) -> str:
        return "unimplemented"

    def fetch_daily_bars(self, *args, **kwargs) -> NeverBlockResult:  # noqa: ANN002, ANN003
        return NeverBlockResult(
            ok=False,
            error="HistoricalDailyBarsProvider not implemented in foundation",
            degraded=True,
        )
