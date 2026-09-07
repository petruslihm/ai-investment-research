"""Primary → fallback provider chain with bounded never-block semantics."""

from __future__ import annotations

from datetime import date
from typing import Any

from trading_system.ids import InstrumentId, RequestSetId
from trading_system.providers.never_block import NeverBlockResult, RetryPolicy


class FallbackChainProvider:
    """Try primary, then fallback; surface degraded metadata on partial success."""

    def __init__(
        self,
        *,
        primary: Any,
        fallback: Any | None = None,
        name: str = "fallback_chain",
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = name

    def set_symbol_resolver(self, fn: Any) -> None:
        if hasattr(self.primary, "set_symbol_resolver"):
            self.primary.set_symbol_resolver(fn)
        if self.fallback is not None and hasattr(self.fallback, "set_symbol_resolver"):
            self.fallback.set_symbol_resolver(fn)

    def fetch_daily_bars(
        self,
        instrument_ids: list[InstrumentId],
        start: date,
        end: date,
        *,
        request_set_id: RequestSetId,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult:
        primary_result = self.primary.fetch_daily_bars(
            instrument_ids, start, end, request_set_id=request_set_id, policy=policy
        )
        if primary_result.ok or self.fallback is None:
            primary_result.metadata.setdefault("provider_chain", self.name)
            primary_result.metadata.setdefault("used_fallback", False)
            return primary_result

        fallback_result = self.fallback.fetch_daily_bars(
            instrument_ids, start, end, request_set_id=request_set_id, policy=policy
        )
        fallback_result.metadata["provider_chain"] = self.name
        fallback_result.metadata["used_fallback"] = True
        fallback_result.metadata["primary_error"] = primary_result.error
        return fallback_result

    def fetch_quotes(
        self,
        instrument_ids: list[InstrumentId],
        *,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult:
        primary_result = self.primary.fetch_quotes(instrument_ids, policy=policy)
        if primary_result.ok or self.fallback is None:
            primary_result.metadata.setdefault("provider_chain", self.name)
            primary_result.metadata.setdefault("used_fallback", False)
            return primary_result

        fallback_result = self.fallback.fetch_quotes(instrument_ids, policy=policy)
        fallback_result.metadata["provider_chain"] = self.name
        fallback_result.metadata["used_fallback"] = True
        fallback_result.metadata["primary_error"] = primary_result.error
        return fallback_result

    def fetch_btc_bars(
        self,
        start: date,
        end: date,
        *,
        request_set_id: RequestSetId,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult:
        primary_result = self.primary.fetch_btc_bars(
            start, end, request_set_id=request_set_id, policy=policy
        )
        if primary_result.ok or self.fallback is None:
            primary_result.metadata.setdefault("used_fallback", False)
            return primary_result
        fallback_result = self.fallback.fetch_btc_bars(
            start, end, request_set_id=request_set_id, policy=policy
        )
        fallback_result.metadata["used_fallback"] = True
        fallback_result.metadata["primary_error"] = primary_result.error
        return fallback_result

    def fetch_btc_quote(self, *, policy: RetryPolicy | None = None) -> NeverBlockResult:
        primary_result = self.primary.fetch_btc_quote(policy=policy)
        if primary_result.ok or self.fallback is None:
            primary_result.metadata.setdefault("used_fallback", False)
            return primary_result
        fallback_result = self.fallback.fetch_btc_quote(policy=policy)
        fallback_result.metadata["used_fallback"] = True
        fallback_result.metadata["primary_error"] = primary_result.error
        return fallback_result
