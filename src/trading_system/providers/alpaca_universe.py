"""Alpaca-backed liquid US equity universe.

Builds the scan universe from official Alpaca endpoints only:
  - https://api.alpaca.markets/v2/assets      (which symbols are real and tradable)
  - https://data.alpaca.markets/v2/stocks/snapshots  (recent volume for liquidity ranking)

Never invents symbols. A failed refresh keeps the last-known universe instead.
"""

from __future__ import annotations

import logging

import httpx

from trading_system.config import Settings
from trading_system.providers.http_client import request_json
from trading_system.providers.never_block import (
    Deadline,
    RetryKind,
    RetryPolicy,
    classify_http_status,
    run_with_deadline,
)

# Market-data keys may belong to a paper account, whose asset catalog lives on a
# different host. Try both; the endpoints are read-only asset metadata.
ASSETS_URLS = (
    "https://paper-api.alpaca.markets/v2/assets",
    "https://api.alpaca.markets/v2/assets",
)
SNAPSHOTS_URL = "https://data.alpaca.markets/v2/stocks/snapshots"

# Alpaca accepts long symbol lists; keep chunks modest so one failure is cheap.
_SNAPSHOT_CHUNK = 200
_ALLOWED_EXCHANGES = frozenset({"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS"})

# Leveraged / inverse / structured products are excluded unless the user adds them.
_EXCLUDED_NAME_TOKENS = (
    "2X", "3X", "-1X", "1.5X", "ULTRA", "ULTRASHORT", "INVERSE", "LEVERAGED",
    "DAILY BULL", "DAILY BEAR", "BULL 2", "BEAR 2", "BULL 3", "BEAR 3",
    "WARRANT", "RIGHTS", "UNIT", "PREFERRED", "DEPOSITARY", "TRUST PREFERRED",
    "VIX SHORT-TERM", "VOLATILITY INDEX",
)

# Pooled products are not individual equities. BTC-linked funds would also smuggle
# crypto exposure into the equity scan, which belongs to the BTC sleeve.
_FUND_NAME_TOKENS = (
    "ETF", "ETN", " FUND", "INDEX FUND", "SPDR", "ISHARES", "PROSHARES",
    "DIREXION", "GLOBAL X", "WISDOMTREE", "VANECK", "GRAYSCALE", "BITWISE",
    "INVESCO", "VANGUARD", "FIRST TRUST", "BITCOIN", "ETHEREUM", "CRYPTO",
)

# Share classes are legitimate (BRK.B); warrants/units/rights are not.
_SHARE_CLASS_SUFFIXES = frozenset({"A", "B", "C"})

_log = logging.getLogger("trading_system.alpaca_universe")

_UNIVERSE_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay_seconds=0.5,
    max_delay_seconds=8.0,
    jitter=0.2,
    total_deadline_seconds=180.0,
)


def _clean_symbol(symbol: str) -> bool:
    """Common shares only: reject warrants, units, rights and preferred classes."""
    s = symbol.upper()
    if not s:
        return False
    if "." in s:
        base, _, suffix = s.partition(".")
        return base.isalpha() and 1 <= len(base) <= 4 and suffix in _SHARE_CLASS_SUFFIXES
    return s.isalpha() and len(s) <= 5


def _excluded_by_name(name: str) -> bool:
    upper = (name or "").upper()
    if any(token in upper for token in _EXCLUDED_NAME_TOKENS):
        return True
    return any(token in upper for token in _FUND_NAME_TOKENS)


def _classify(exc: BaseException) -> RetryKind:
    if isinstance(exc, httpx.HTTPStatusError):
        return classify_http_status(exc.response.status_code)
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, TimeoutError)):
        return RetryKind.RETRYABLE
    return RetryKind.NON_RETRYABLE


class AlpacaUniverseProvider:
    """Liquidity-ranked US common shares. Requires Alpaca credentials."""

    source_name = "alpaca"

    def __init__(self, settings: Settings, *, timeout_seconds: float = 30.0) -> None:
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise ValueError("Alpaca credentials required")
        self.settings = settings
        self._headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }
        self._timeout = timeout_seconds

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self._timeout, headers=self._headers)

    def _tradable_symbols(self, deadline: Deadline) -> list[str]:
        last_error: str | None = None
        for url in ASSETS_URLS:

            def _call(url: str = url) -> list[str]:
                with self._client() as client:
                    resp = client.get(
                        url,
                        params={"status": "active", "asset_class": "us_equity"},
                    )
                _log.info("alpaca assets host=%s status=%s", resp.url.host, resp.status_code)
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list):
                    raise ValueError("assets payload malformed")
                out: list[str] = []
                for row in payload:
                    if not isinstance(row, dict):
                        continue
                    if not row.get("tradable") or row.get("status") != "active":
                        continue
                    if str(row.get("exchange") or "").upper() not in _ALLOWED_EXCHANGES:
                        continue
                    symbol = str(row.get("symbol") or "").upper()
                    if not _clean_symbol(symbol) or _excluded_by_name(str(row.get("name") or "")):
                        continue
                    out.append(symbol)
                return sorted(set(out))

            result = run_with_deadline(
                _call,
                policy=_UNIVERSE_POLICY,
                deadline=deadline,
                classify=_classify,
                label="alpaca_assets",
            )
            if result.ok and isinstance(result.value, list) and result.value:
                return result.value
            last_error = result.error or last_error
        raise RuntimeError(last_error or "Alpaca assets UNAVAILABLE")

    def _dollar_volume(self, symbols: list[str], deadline: Deadline) -> dict[str, float]:
        ranked: dict[str, float] = {}
        for start in range(0, len(symbols), _SNAPSHOT_CHUNK):
            if deadline.expired():
                break
            chunk = symbols[start : start + _SNAPSHOT_CHUNK]

            def _call(chunk: list[str] = chunk) -> dict:
                with self._client() as client:
                    return request_json(
                        client,
                        "GET",
                        SNAPSHOTS_URL,
                        params={"symbols": ",".join(chunk), "feed": "iex"},
                        policy=_UNIVERSE_POLICY,
                        deadline=deadline,
                    ).value or {}

            result = run_with_deadline(
                _call,
                policy=RetryPolicy(
                    max_attempts=2,
                    base_delay_seconds=0.5,
                    max_delay_seconds=4.0,
                    jitter=0.2,
                    total_deadline_seconds=None,
                ),
                deadline=deadline,
                classify=_classify,
                label="alpaca_snapshots",
            )
            payload = result.value if result.ok else None
            if not isinstance(payload, dict):
                continue
            rows = payload.get("snapshots") if isinstance(payload.get("snapshots"), dict) else payload
            if not isinstance(rows, dict):
                continue
            for symbol, snap in rows.items():
                if not isinstance(snap, dict):
                    continue
                bar = snap.get("dailyBar") or snap.get("prevDailyBar")
                if not isinstance(bar, dict):
                    continue
                try:
                    close = float(bar.get("c") or 0.0)
                    volume = float(bar.get("v") or 0.0)
                except (TypeError, ValueError):
                    continue
                if close <= 0 or volume <= 0:
                    continue
                ranked[str(symbol).upper()] = close * volume
        return ranked

    def fetch_liquid_symbols(self, limit: int) -> tuple[list[tuple[str, float]], str | None]:
        """Return [(symbol, dollar_volume)] ranked desc, plus a note when degraded."""
        deadline = Deadline.after_seconds(_UNIVERSE_POLICY.total_deadline_seconds or 180.0)
        symbols = self._tradable_symbols(deadline)
        if not symbols:
            raise RuntimeError("Alpaca returned no tradable US equities")
        ranked = self._dollar_volume(symbols, deadline)
        if not ranked:
            raise RuntimeError("Alpaca snapshots returned no volume data")
        ordered = sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        note = None
        if len(ordered) < limit:
            note = f"{len(ordered)}/{limit} symbols ranked (snapshot coverage partial)"
        return ordered, note
