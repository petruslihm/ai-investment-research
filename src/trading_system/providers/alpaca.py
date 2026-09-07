"""Optional Alpaca market-data adapter (no trading endpoints)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable

import httpx

from trading_system.config import Settings
from trading_system.ids import InstrumentId, RequestSetId
from trading_system.market.calendar import (
    btc_day_finality,
    equity_bar_finality,
    is_trading_day,
)
from trading_system.providers.batch import DailyBarsBatch, QuotesBatch
from trading_system.providers.http_client import request_json
from trading_system.providers.interfaces import (
    BarFinality,
    CoverageRow,
    CoverageStatus,
    DailyBar,
    LatestQuote,
)
from trading_system.providers.never_block import Deadline, NeverBlockResult, RetryPolicy

# Alpaca accepts multi-symbol bar requests; batching keeps a 500-name scan practical.
_BAR_SYMBOL_CHUNK = 100
# Defense in depth, independent of `deadline`: a caller with no deadline (or a
# provider that always returns a unique, never-repeating next_page_token) would
# otherwise page forever. No real request needs anywhere near this many pages.
_MAX_PAGES = 500


class AlpacaMarketDataProvider:
    """Alpaca Market Data v2 (IEX) + crypto — credentials required."""

    ADJUSTMENT_REVISION = "alpaca_split_adjusted_v1"

    def __init__(
        self,
        settings: Settings,
        *,
        symbol_resolver: Callable[[InstrumentId], str | None] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise ValueError("Alpaca credentials required")
        self.settings = settings
        self._symbol_resolver = symbol_resolver or (lambda inst: None)
        self._headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }
        self._client = httpx.Client(
            base_url="https://data.alpaca.markets",
            timeout=timeout_seconds,
            headers=self._headers,
        )

    def set_symbol_resolver(self, fn: Callable[[InstrumentId], str | None]) -> None:
        self._symbol_resolver = fn

    def provider_name(self) -> str:
        return "alpaca"

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _as_bar_rows(raw: object) -> list:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            rows: list = []
            for value in raw.values():
                if isinstance(value, list):
                    rows.extend(value)
            return rows
        return []

    def _crypto_symbol(self) -> str:
        symbol = (self.settings.btc_symbol or "BTC/USD").strip()
        if "/" not in symbol:
            return f"{symbol[:-3]}/{symbol[-3:]}" if len(symbol) > 3 else symbol
        return symbol

    def _shared_deadline(self, policy: RetryPolicy | None) -> Deadline | None:
        policy = policy or RetryPolicy()
        if policy.total_deadline_seconds is None:
            return None
        return Deadline.after_seconds(policy.total_deadline_seconds)

    def _parse_equity_bar(
        self,
        row: dict,
        *,
        inst_id: InstrumentId,
        now: datetime,
    ) -> DailyBar | None:
        try:
            ts = datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00"))
            session = ts.date()
            if not is_trading_day(session):
                return None
            finality = equity_bar_finality(session, now)
            return DailyBar(
                instrument_id=inst_id,
                session_date=session,
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row["v"]),
                finality=finality,
                adjustment_revision=self.ADJUSTMENT_REVISION,
                provider=self.provider_name(),
                receive_ts=now,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _parse_quote(
        self,
        payload: dict,
        *,
        inst_id: InstrumentId,
        now: datetime,
    ) -> LatestQuote | None:
        try:
            quote = payload.get("quote")
            if not isinstance(quote, dict):
                return None
            event_raw = quote.get("t")
            event_ts = (
                datetime.fromisoformat(str(event_raw).replace("Z", "+00:00"))
                if event_raw
                else None
            )
            price = quote.get("ap") or quote.get("bp")
            if price is None:
                return None
            return LatestQuote(
                instrument_id=inst_id,
                price=float(price),
                event_ts=event_ts,
                receive_ts=now,
                volume=None,
                provider=self.provider_name(),
                degraded=event_ts is None,
            )
        except (TypeError, ValueError):
            return None

    def _fetch_bar_pages(
        self,
        symbols: list[str],
        start: date,
        end: date,
        policy: RetryPolicy | None,
        deadline: Deadline | None,
    ) -> tuple[dict[str, list] | None, str | None]:
        """Multi-symbol daily bars. Returns (rows_by_symbol, error)."""
        params = {
            "symbols": ",".join(symbols),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": "1Day",
            "adjustment": "split",
            "feed": "iex",
            "limit": "10000",
        }
        out: dict[str, list] = {}
        page_token: str | None = None
        requested_tokens: set[str] = set()
        while True:
            if deadline is not None and deadline.expired():
                return (out or None), "deadline expired"
            if len(requested_tokens) >= _MAX_PAGES:
                return (out or None), "bars pagination exceeded max page count"
            page_params = dict(params)
            if page_token:
                if page_token in requested_tokens:
                    return (out or None), "bars pagination token repeated"
                requested_tokens.add(page_token)
                page_params["page_token"] = page_token
            result = request_json(
                self._client,
                "GET",
                "/v2/stocks/bars",
                params=page_params,
                policy=policy,
                deadline=deadline,
            )
            if not result.ok:
                return (out or None), result.error
            payload = result.value
            if not isinstance(payload, dict):
                return (out or None), "bars payload malformed"
            raw = payload.get("bars")
            if isinstance(raw, dict):
                for symbol, rows in raw.items():
                    if isinstance(rows, list):
                        out.setdefault(str(symbol).upper(), []).extend(rows)
            elif isinstance(raw, list) and len(symbols) == 1:
                out.setdefault(symbols[0].upper(), []).extend(raw)
            elif raw is not None:
                return (out or None), "bars payload malformed"
            page_token = payload.get("next_page_token")
            if not page_token:
                return out, None

    def _collect_bars(
        self,
        pairs: list[tuple[InstrumentId, str]],
        start: date,
        end: date,
        policy: RetryPolicy | None,
        deadline: Deadline | None,
    ) -> tuple[dict[str, list], str | None]:
        """Batch symbols per request; retry a failed batch symbol-by-symbol.

        One bad symbol must not cost the whole universe its history.
        """
        collected: dict[str, list] = {}
        error: str | None = None
        for begin in range(0, len(pairs), _BAR_SYMBOL_CHUNK):
            chunk = pairs[begin : begin + _BAR_SYMBOL_CHUNK]
            symbols = [sym for _, sym in chunk]
            rows, err = self._fetch_bar_pages(symbols, start, end, policy, deadline)
            if rows:
                collected.update(rows)
            if err is None:
                continue
            error = error or err
            if len(symbols) == 1:
                continue
            for symbol in symbols:
                if deadline is not None and deadline.expired():
                    error = error or "deadline expired"
                    break
                if symbol.upper() in collected:
                    continue
                single, single_err = self._fetch_bar_pages([symbol], start, end, policy, deadline)
                if single:
                    collected.update(single)
                elif single_err:
                    error = error or single_err
        return collected, error

    def fetch_daily_bars(
        self,
        instrument_ids: list[InstrumentId],
        start: date,
        end: date,
        *,
        request_set_id: RequestSetId,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult:
        batch = DailyBarsBatch()
        deadline = self._shared_deadline(policy)

        pairs: list[tuple[InstrumentId, str]] = []
        for inst_id in instrument_ids:
            symbol = self._symbol_resolver(inst_id)
            if symbol is None:
                batch.failed_instruments.append(inst_id)
                continue
            pairs.append((inst_id, symbol))

        collected, fetch_error = self._collect_bars(pairs, start, end, policy, deadline)
        now = datetime.now(timezone.utc)

        for inst_id, symbol in pairs:
            raw_bars = collected.get(symbol.upper()) or []
            if not raw_bars:
                batch.failed_instruments.append(inst_id)
                batch.coverage.append(
                    CoverageRow(
                        provider=self.provider_name(),
                        adjustment=self.ADJUSTMENT_REVISION,
                        adjustment_revision=self.ADJUSTMENT_REVISION,
                        instrument_id=inst_id,
                        session=end,
                        request_set_id=request_set_id,
                        status=CoverageStatus.ERROR if fetch_error else CoverageStatus.MISSING,
                    )
                )
                continue
            parsed_any = False
            for row in raw_bars:
                bar = self._parse_equity_bar(row, inst_id=inst_id, now=now)
                if bar is None:
                    batch.coverage.append(
                        CoverageRow(
                            provider=self.provider_name(),
                            adjustment=self.ADJUSTMENT_REVISION,
                            adjustment_revision=self.ADJUSTMENT_REVISION,
                            instrument_id=inst_id,
                            session=end,
                            request_set_id=request_set_id,
                            status=CoverageStatus.QUARANTINED,
                        )
                    )
                    continue
                parsed_any = True
                batch.bars.append(bar)
                batch.coverage.append(
                    CoverageRow(
                        provider=self.provider_name(),
                        adjustment=self.ADJUSTMENT_REVISION,
                        adjustment_revision=self.ADJUSTMENT_REVISION,
                        instrument_id=inst_id,
                        session=bar.session_date,
                        request_set_id=request_set_id,
                        status=CoverageStatus.SUCCEEDED,
                    )
                )
            if not parsed_any:
                batch.failed_instruments.append(inst_id)

        ok = bool(batch.bars)
        degraded = bool(batch.failed_instruments) or fetch_error is not None
        return NeverBlockResult(ok=ok, value=batch, degraded=degraded, error=fetch_error)

    def fetch_quotes(
        self,
        instrument_ids: list[InstrumentId],
        *,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult:
        batch = QuotesBatch()
        deadline = self._shared_deadline(policy)
        now = datetime.now(timezone.utc)
        for inst_id in instrument_ids:
            if deadline is not None and deadline.expired():
                batch.failed_instruments.append(inst_id)
                continue
            symbol = self._symbol_resolver(inst_id)
            if symbol is None:
                batch.failed_instruments.append(inst_id)
                continue
            result = request_json(
                self._client,
                "GET",
                f"/v2/stocks/{symbol}/quotes/latest",
                params={"feed": "iex"},
                policy=policy,
                deadline=deadline,
            )
            if not result.ok:
                batch.failed_instruments.append(inst_id)
                continue
            payload: dict = result.value  # type: ignore[assignment]
            quote = self._parse_quote(payload, inst_id=inst_id, now=now)
            if quote is None:
                batch.failed_instruments.append(inst_id)
                continue
            batch.quotes.append(quote)
        ok = bool(batch.quotes)
        return NeverBlockResult(ok=ok, value=batch, degraded=False)

    def fetch_btc_bars(
        self,
        start: date,
        end: date,
        *,
        request_set_id: RequestSetId,
        policy: RetryPolicy | None = None,
    ) -> NeverBlockResult:
        deadline = self._shared_deadline(policy)
        now = datetime.now(timezone.utc)
        batch = DailyBarsBatch()
        btc_inst = InstrumentId("inst_btc_usd")
        params: dict[str, str] = {
            "symbols": self._crypto_symbol(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": "1Day",
        }
        page_token: str | None = None
        requested_tokens: set[str] = set()
        while True:
            if deadline is not None and deadline.expired():
                break
            if len(requested_tokens) >= _MAX_PAGES:
                error = "btc bars pagination exceeded max page count"
                if batch.bars:
                    return NeverBlockResult(ok=True, value=batch, degraded=True, error=error)
                return NeverBlockResult(ok=False, value=batch, degraded=True, error=error)
            page_params = dict(params)
            if page_token:
                if page_token in requested_tokens:
                    if batch.bars:
                        return NeverBlockResult(
                            ok=True,
                            value=batch,
                            degraded=True,
                            error="btc bars pagination token repeated",
                        )
                    return NeverBlockResult(
                        ok=False,
                        value=batch,
                        degraded=True,
                        error="btc bars pagination token repeated",
                    )
                requested_tokens.add(page_token)
                page_params["page_token"] = page_token
            result = request_json(
                self._client,
                "GET",
                "/v1beta3/crypto/us/bars",
                params=page_params,
                policy=policy,
                deadline=deadline,
            )
            if not result.ok:
                if batch.bars:
                    return NeverBlockResult(ok=True, value=batch, degraded=True)
                return result
            payload: dict = result.value  # type: ignore[assignment]
            bars = payload.get("bars") or {}
            if not isinstance(bars, dict):
                if batch.bars:
                    return NeverBlockResult(ok=True, value=batch, degraded=True)
                return NeverBlockResult(ok=False, error="btc bars malformed", degraded=True, value=batch)
            for rows in bars.values():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    try:
                        ts = datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00"))
                        session = ts.date()
                        finality = btc_day_finality(session, now)
                        batch.bars.append(
                            DailyBar(
                                instrument_id=btc_inst,
                                session_date=session,
                                open=float(row["o"]),
                                high=float(row["h"]),
                                low=float(row["l"]),
                                close=float(row["c"]),
                                volume=float(row["v"]),
                                finality=finality,
                                adjustment_revision=self.ADJUSTMENT_REVISION,
                                provider=self.provider_name(),
                                receive_ts=now,
                            )
                        )
                        batch.coverage.append(
                            CoverageRow(
                                provider=self.provider_name(),
                                adjustment=self.ADJUSTMENT_REVISION,
                                adjustment_revision=self.ADJUSTMENT_REVISION,
                                instrument_id=btc_inst,
                                session=session,
                                request_set_id=request_set_id,
                                status=CoverageStatus.SUCCEEDED,
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        batch.coverage.append(
                            CoverageRow(
                                provider=self.provider_name(),
                                adjustment=self.ADJUSTMENT_REVISION,
                                adjustment_revision=self.ADJUSTMENT_REVISION,
                                instrument_id=btc_inst,
                                session=end,
                                request_set_id=request_set_id,
                                status=CoverageStatus.QUARANTINED,
                            )
                        )
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        return NeverBlockResult(ok=bool(batch.bars), value=batch)

    def fetch_btc_quote(self, *, policy: RetryPolicy | None = None) -> NeverBlockResult:
        deadline = self._shared_deadline(policy)
        result = request_json(
            self._client,
            "GET",
            "/v1beta3/crypto/us/latest/quotes",
            params={"symbols": self._crypto_symbol()},
            policy=policy,
            deadline=deadline,
        )
        if not result.ok:
            return result
        payload: dict = result.value  # type: ignore[assignment]
        quotes = payload.get("quotes") or {}
        if not isinstance(quotes, dict) or not quotes:
            return NeverBlockResult(ok=False, error="no btc quote", degraded=True)
        quote = next(iter(quotes.values()))
        if not isinstance(quote, dict):
            return NeverBlockResult(ok=False, error="btc quote malformed", degraded=True)
        now = datetime.now(timezone.utc)
        try:
            event_raw = quote.get("t")
            event_ts = (
                datetime.fromisoformat(str(event_raw).replace("Z", "+00:00")) if event_raw else None
            )
            price = quote.get("ap") or quote.get("bp")
            if price is None:
                return NeverBlockResult(ok=False, error="btc quote missing price", degraded=True)
            return NeverBlockResult(
                ok=True,
                value=LatestQuote(
                    instrument_id=InstrumentId("inst_btc_usd"),
                    price=float(price),
                    event_ts=event_ts,
                    receive_ts=now,
                    volume=None,
                    provider=self.provider_name(),
                    degraded=event_ts is None,
                ),
            )
        except (TypeError, ValueError):
            return NeverBlockResult(ok=False, error="btc quote malformed", degraded=True)
