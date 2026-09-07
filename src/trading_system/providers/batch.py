"""Structured provider batch payloads."""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_system.ids import InstrumentId
from trading_system.providers.interfaces import CoverageRow, DailyBar, LatestQuote


@dataclass
class DailyBarsBatch:
    bars: list[DailyBar] = field(default_factory=list)
    coverage: list[CoverageRow] = field(default_factory=list)
    failed_instruments: list[InstrumentId] = field(default_factory=list)


@dataclass
class QuotesBatch:
    quotes: list[LatestQuote] = field(default_factory=list)
    failed_instruments: list[InstrumentId] = field(default_factory=list)
    coverage: list[CoverageRow] = field(default_factory=list)
