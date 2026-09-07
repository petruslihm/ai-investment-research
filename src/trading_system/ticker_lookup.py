"""Confirm a portfolio ticker against the scan universe, bars, or Alpaca."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import duckdb
import httpx

from trading_system.config import Settings
from trading_system.ids import canonicalize_instrument_id
from trading_system.providers.alpaca_universe import ASSETS_URLS
from trading_system.universe import normalize_symbol

_TICKER_RE = re.compile(r"^[A-Z0-9]{1,10}([./-][A-Z0-9]{1,5})?$")


@dataclass
class TickerLookup:
    ok: bool
    query: str
    symbol: str
    instrument_id: str
    name: str | None = None
    in_universe: bool = False
    has_bars: bool = False
    source: str = "none"
    message: str = ""
    suggestions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "query": self.query,
            "symbol": self.symbol,
            "instrument_id": self.instrument_id,
            "name": self.name,
            "in_universe": self.in_universe,
            "has_bars": self.has_bars,
            "source": self.source,
            "message": self.message,
            "suggestions": self.suggestions,
        }


def _normalize_query(raw: str, settings: Settings) -> str:
    s = (raw or "").strip()
    if s.lower().startswith("inst_"):
        s = s[5:].replace("_", "/")
    symbol = normalize_symbol(s).replace(" ", "")
    if symbol in {"BTC", "BTCUSD", "XBT", "XBTUSD"}:
        return normalize_symbol(settings.btc_symbol)
    return symbol


def _symbol_variants(symbol: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cand in (symbol, symbol.replace("-", "."), symbol.replace(".", "-")):
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _in_universe(conn: duckdb.DuckDBPyConnection, symbol: str) -> bool:
    variants = _symbol_variants(symbol)
    placeholders = ", ".join("?" * len(variants))
    row = conn.execute(
        f"SELECT 1 FROM universe_members WHERE upper(symbol) IN ({placeholders}) LIMIT 1",
        variants,
    ).fetchone()
    return row is not None


def _display_name(conn: duckdb.DuckDBPyConnection, instrument_id: str) -> str | None:
    row = conn.execute(
        "SELECT display_name FROM instruments WHERE instrument_id = ?",
        [instrument_id],
    ).fetchone()
    if row and row[0]:
        return str(row[0])
    alias = conn.execute(
        """
        SELECT symbol FROM instrument_aliases
        WHERE instrument_id = ?
        ORDER BY effective_from DESC
        LIMIT 1
        """,
        [instrument_id],
    ).fetchone()
    if alias and alias[0]:
        return str(alias[0])
    return None


def _is_btc(symbol: str, instrument_id: str, settings: Settings) -> bool:
    return (
        symbol == normalize_symbol(settings.btc_symbol)
        or instrument_id.startswith("inst_btc")
        or symbol in {"BTC", "BTCUSD", "BTC/USD"}
    )


def _has_bars(conn: duckdb.DuckDBPyConnection, instrument_id: str, symbol: str, settings: Settings) -> bool:
    if _is_btc(symbol, instrument_id, settings):
        row = conn.execute("SELECT count(*) FROM btc_daily_bars").fetchone()
    else:
        row = conn.execute(
            "SELECT count(*) FROM equity_daily_bars WHERE instrument_id = ?",
            [instrument_id],
        ).fetchone()
    return bool(row and int(row[0]) > 0)


def _suggestions(conn: duckdb.DuckDBPyConnection, symbol: str) -> list[str]:
    prefix = symbol[: max(1, min(4, len(symbol)))]
    compact = symbol.replace(".", "").replace("-", "")
    rows = conn.execute(
        """
        SELECT DISTINCT symbol FROM universe_members
        WHERE upper(symbol) != ?
          AND (
            upper(symbol) LIKE ?
            OR replace(replace(upper(symbol), '.', ''), '-', '') LIKE ?
          )
        ORDER BY length(symbol), symbol
        LIMIT 5
        """,
        [symbol, f"{prefix}%", f"{compact}%"],
    ).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def fetch_alpaca_asset(settings: Settings, symbol: str) -> dict | None:
    """Look up one US equity on Alpaca. Returns None when unconfigured or unknown."""
    key = (settings.alpaca_api_key or "").strip()
    secret = (settings.alpaca_secret_key or "").strip()
    if not key or not secret:
        return None
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    for base in ASSETS_URLS:
        try:
            resp = httpx.get(f"{base}/{symbol}", headers=headers, timeout=6.0)
        except httpx.HTTPError:
            continue
        if resp.status_code == 404:
            continue
        if resp.status_code != 200:
            continue
        data = resp.json()
        return data if isinstance(data, dict) else None
    return None


def lookup_ticker(
    conn: duckdb.DuckDBPyConnection,
    settings: Settings,
    raw: str,
    *,
    fetch_remote: object | None = None,
) -> TickerLookup:
    query = (raw or "").strip()
    if not query:
        return TickerLookup(
            ok=False,
            query="",
            symbol="",
            instrument_id="",
            message="티커를 입력하세요.",
        )
    symbol = _normalize_query(query, settings)
    if not _TICKER_RE.fullmatch(symbol):
        return TickerLookup(
            ok=False,
            query=query,
            symbol=symbol,
            instrument_id="",
            message="티커 형식이 아닙니다. 예: AAPL, BRK.B, BTC/USD",
        )
    instrument_id = str(canonicalize_instrument_id(symbol))
    in_universe = _in_universe(conn, symbol)
    has_bars = _has_bars(conn, instrument_id, symbol, settings)
    name = _display_name(conn, instrument_id)
    suggestions = _suggestions(conn, symbol)

    if in_universe or has_bars:
        bits: list[str] = []
        if name and name.upper() != symbol:
            bits.append(name)
        if in_universe:
            bits.append("스캔 유니버스에 있습니다")
        elif has_bars:
            bits.append("시세 데이터가 있습니다")
        detail = " · ".join(bits)
        return TickerLookup(
            ok=True,
            query=query,
            symbol=symbol,
            instrument_id=instrument_id,
            name=name,
            in_universe=in_universe,
            has_bars=has_bars,
            source="universe" if in_universe else "bars",
            message=f"확인됨: {symbol}" + (f" · {detail}" if detail else ""),
            suggestions=suggestions,
        )

    remote = None
    if fetch_remote is None:
        remote = fetch_alpaca_asset(settings, symbol)
    elif callable(fetch_remote):
        remote = fetch_remote(symbol)
    if isinstance(remote, dict) and str(remote.get("status") or "").lower() == "active":
        remote_name = str(remote.get("name") or "") or None
        bits = [p for p in (remote_name, "Alpaca 상장 종목입니다", "이번 스캔 유니버스에는 없습니다") if p]
        return TickerLookup(
            ok=True,
            query=query,
            symbol=symbol,
            instrument_id=instrument_id,
            name=remote_name,
            in_universe=False,
            has_bars=False,
            source="alpaca",
            message=f"확인됨: {symbol} · " + " · ".join(bits),
            suggestions=suggestions,
        )

    extra = f" 비슷한 티커: {', '.join(suggestions)}" if suggestions else ""
    return TickerLookup(
        ok=False,
        query=query,
        symbol=symbol,
        instrument_id=instrument_id,
        name=name,
        in_universe=False,
        has_bars=False,
        source="none",
        message=f"{symbol} 은(는) 찾을 수 없습니다.{extra}",
        suggestions=suggestions,
    )
