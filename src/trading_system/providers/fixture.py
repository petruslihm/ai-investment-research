"""Deterministic fixture provider for development and tests."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from trading_system.ids import InstrumentId, RequestSetId
from trading_system.providers.batch import DailyBarsBatch, QuotesBatch
from trading_system.providers.interfaces import (
    BarFinality,
    CoverageRow,
    CoverageStatus,
    DailyBar,
    LatestQuote,
)
from trading_system.providers.never_block import NeverBlockResult, RetryKind, RetryPolicy, run_with_deadline


@dataclass
class FixtureBarSpec:
    session_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 1_000_000.0
    finality: BarFinality = BarFinality.FINAL


@dataclass
class FixtureQuoteSpec:
    price: float
    event_ts: datetime | None
    volume: float | None = 100.0
    degraded: bool = False


@dataclass
class FixtureProviderConfig:
    provider_name: str = "fixture"
    adjustment_revision: str = "fixture_rev_1"
    equity_bars: dict[str, list[FixtureBarSpec]] = field(default_factory=dict)
    btc_bars: list[FixtureBarSpec] = field(default_factory=list)
    quotes: dict[str, FixtureQuoteSpec] = field(default_factory=dict)
    fail_symbols: set[str] = field(default_factory=set)
    hang_seconds: float | None = None
    always_fail: bool = False
    http_status: int | None = None


class FixtureMarketDataProvider:
    """In-memory provider implementing historical + live interfaces."""

    def __init__(
        self,
        config: FixtureProviderConfig,
        *,
        symbol_resolver: Callable[[InstrumentId], str | None] | None = None,
    ) -> None:
        self.config = config
        self._symbol_resolver = symbol_resolver or (lambda _i: None)

    def provider_name(self) -> str:
        return self.config.provider_name

    def set_symbol_resolver(self, fn: Callable[[InstrumentId], str | None]) -> None:
        self._symbol_resolver = fn

    def fetch_daily_bars(
        self,
        instrument_ids: list[InstrumentId],
        start: date,
        end: date,
        *,
        request_set_id: RequestSetId,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult:
        return run_with_deadline(
            lambda: self._fetch_daily_bars(instrument_ids, start, end, request_set_id),
            policy=policy or RetryPolicy(max_attempts=1, total_deadline_seconds=5.0),
            classify=lambda _e: RetryKind.NON_RETRYABLE,
            sleep=lambda _s: None,
        )

    def _fetch_daily_bars(
        self,
        instrument_ids: list[InstrumentId],
        start: date,
        end: date,
        request_set_id: RequestSetId,
    ) -> DailyBarsBatch:
        self._maybe_fail()
        now = datetime.now(timezone.utc)
        batch = DailyBarsBatch()
        for inst_id in instrument_ids:
            symbol = self._symbol_resolver(inst_id)
            if symbol is None or symbol in self.config.fail_symbols:
                batch.failed_instruments.append(inst_id)
                batch.coverage.append(
                    CoverageRow(
                        provider=self.provider_name(),
                        adjustment=self.config.adjustment_revision,
                        adjustment_revision=self.config.adjustment_revision,
                        instrument_id=inst_id,
                        session=end,
                        request_set_id=request_set_id,
                        status=CoverageStatus.ERROR,
                    )
                )
                continue
            specs = self._bars_for_range(self.config.equity_bars.get(symbol, []), start, end)
            for spec in specs:
                if start <= spec.session_date <= end:
                    batch.bars.append(
                        DailyBar(
                            instrument_id=inst_id,
                            session_date=spec.session_date,
                            open=spec.open,
                            high=spec.high,
                            low=spec.low,
                            close=spec.close,
                            volume=spec.volume,
                            finality=spec.finality,
                            adjustment_revision=self.config.adjustment_revision,
                            provider=self.provider_name(),
                            receive_ts=now,
                        )
                    )
                    batch.coverage.append(
                        CoverageRow(
                            provider=self.provider_name(),
                            adjustment=self.config.adjustment_revision,
                            adjustment_revision=self.config.adjustment_revision,
                            instrument_id=inst_id,
                            session=spec.session_date,
                            request_set_id=request_set_id,
                            status=CoverageStatus.SUCCEEDED,
                        )
                    )
        return batch

    def fetch_btc_bars(
        self,
        start: date,
        end: date,
        *,
        request_set_id: RequestSetId,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult:
        return run_with_deadline(
            lambda: self._fetch_btc_bars(start, end, request_set_id),
            policy=policy or RetryPolicy(max_attempts=1, total_deadline_seconds=5.0),
            classify=lambda _e: RetryKind.NON_RETRYABLE,
            sleep=lambda _s: None,
        )

    def _fetch_btc_bars(
        self,
        start: date,
        end: date,
        request_set_id: RequestSetId,
    ) -> DailyBarsBatch:
        self._maybe_fail()
        now = datetime.now(timezone.utc)
        batch = DailyBarsBatch()
        btc_inst = InstrumentId("inst_btc_usd")
        for spec in self._bars_for_range(self.config.btc_bars, start, end):
            if start <= spec.session_date <= end:
                batch.bars.append(
                    DailyBar(
                        instrument_id=btc_inst,
                        session_date=spec.session_date,
                        open=spec.open,
                        high=spec.high,
                        low=spec.low,
                        close=spec.close,
                        volume=spec.volume,
                        finality=spec.finality,
                        adjustment_revision=self.config.adjustment_revision,
                        provider=self.provider_name(),
                        receive_ts=now,
                    )
                )
                batch.coverage.append(
                    CoverageRow(
                        provider=self.provider_name(),
                        adjustment=self.config.adjustment_revision,
                        adjustment_revision=self.config.adjustment_revision,
                        instrument_id=btc_inst,
                        session=spec.session_date,
                        request_set_id=request_set_id,
                        status=CoverageStatus.SUCCEEDED,
                    )
                )
        return batch

    def fetch_quotes(
        self,
        instrument_ids: list[InstrumentId],
        *,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult:
        return run_with_deadline(
            lambda: self._fetch_quotes(instrument_ids),
            policy=policy or RetryPolicy(max_attempts=1, total_deadline_seconds=5.0),
            classify=lambda _e: RetryKind.NON_RETRYABLE,
            sleep=lambda _s: None,
        )

    def _fetch_quotes(self, instrument_ids: list[InstrumentId]) -> QuotesBatch:
        self._maybe_fail()
        now = datetime.now(timezone.utc)
        batch = QuotesBatch()
        for inst_id in instrument_ids:
            symbol = self._symbol_resolver(inst_id)
            if symbol is None or symbol in self.config.fail_symbols:
                batch.failed_instruments.append(inst_id)
                continue
            spec = self.config.quotes.get(symbol)
            if spec is None:
                batch.failed_instruments.append(inst_id)
                continue
            batch.quotes.append(
                LatestQuote(
                    instrument_id=inst_id,
                    price=spec.price,
                    event_ts=spec.event_ts,
                    receive_ts=now,
                    volume=spec.volume,
                    provider=self.provider_name(),
                    degraded=spec.degraded or spec.event_ts is None,
                )
            )
        return batch

    def fetch_btc_quote(self, *, policy: RetryPolicy | None = None) -> NeverBlockResult:
        btc_inst = InstrumentId("inst_btc_usd")
        result = self.fetch_quotes([btc_inst], policy=policy)
        if result.ok and result.value:
            batch: QuotesBatch = result.value  # type: ignore[assignment]
            if batch.quotes:
                return NeverBlockResult(ok=True, value=batch.quotes[0], attempts=result.attempts)
            return NeverBlockResult(
                ok=False,
                error="btc quote missing",
                degraded=True,
                attempts=result.attempts,
            )
        return result

    def _bars_for_range(
        self,
        specs: list[FixtureBarSpec],
        start: date,
        end: date,
    ) -> list[FixtureBarSpec]:
        """Return configured bars in range, or synthesize from the latest template."""
        if not specs:
            return []
        matching = [spec for spec in specs if start <= spec.session_date <= end]
        if matching:
            return matching
        template = specs[-1]
        out: list[FixtureBarSpec] = []
        cur = start
        offset = 0
        while cur <= end:
            close = template.close + offset
            out.append(
                FixtureBarSpec(
                    session_date=cur,
                    open=close - 0.5,
                    high=close + 1.0,
                    low=close - 1.0,
                    close=close,
                    volume=template.volume,
                    finality=template.finality,
                )
            )
            cur += timedelta(days=1)
            offset += 1
        return out

    def _maybe_fail(self) -> None:
        if self.config.hang_seconds is not None:
            time.sleep(self.config.hang_seconds)
        if self.config.always_fail:
            raise TimeoutError("fixture provider forced failure")
        if self.config.http_status == 429:
            raise RuntimeError("rate limited")
        if self.config.http_status and self.config.http_status >= 500:
            raise RuntimeError(f"server error {self.config.http_status}")


def make_smoke_fixture_bars(
    symbols: tuple[str, ...],
    *,
    end: date,
    days: int = 5,
) -> dict[str, list[FixtureBarSpec]]:
    out: dict[str, list[FixtureBarSpec]] = {}
    for i, symbol in enumerate(symbols):
        specs: list[FixtureBarSpec] = []
        base = 100.0 + i * 10.0
        for d in range(days):
            session = end - timedelta(days=days - 1 - d)
            close = base + d
            specs.append(
                FixtureBarSpec(
                    session_date=session,
                    open=close - 0.5,
                    high=close + 1.0,
                    low=close - 1.0,
                    close=close,
                )
            )
        out[symbol] = specs
    return out
