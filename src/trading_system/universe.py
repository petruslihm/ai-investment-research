"""Scan universe for US equities.

Rules that the rest of the app relies on:
  * The production default is a liquidity-ranked set of large US equities,
    refreshed from the market-data provider — not the development smoke list.
  * Held positions and user watchlist symbols are ALWAYS scanned, even when they
    fall outside the preset.
  * SPY is benchmark/reference data, never a recommendation candidate.
  * BTC/USD is handled by the BTC sleeve and never mixed into the equity universe.
  * A failed refresh keeps the last-known universe. Symbols are never invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import duckdb

from trading_system.config import Settings

PRESET_LARGE_LIQUID = "large_liquid_500"
PRESET_FAST = "fast_100"
PRESET_CUSTOM = "custom"

PRESET_TARGETS: dict[str, int] = {
    PRESET_LARGE_LIQUID: 500,
    PRESET_FAST: 100,
}
PRESET_LABELS: dict[str, str] = {
    PRESET_LARGE_LIQUID: "미국 대형·고유동성 종목 (~500)",
    PRESET_FAST: "빠른 스캔 (~100)",
    PRESET_CUSTOM: "직접 지정한 종목",
}
PRESET_ORDER: tuple[str, ...] = (PRESET_LARGE_LIQUID, PRESET_FAST, PRESET_CUSTOM)

# Reference series used for relative features. Never a recommendation candidate.
BENCHMARK_SYMBOLS: tuple[str, ...] = ("SPY",)

STATUS_AVAILABLE = "AVAILABLE"
STATUS_DEGRADED = "DEGRADED"
STATUS_NOT_CONFIGURED = "NOT_CONFIGURED"
STATUS_UNAVAILABLE = "UNAVAILABLE"

SOURCE_SEED = "seed"

# Offline fallback only. Used when no provider is configured or a refresh fails
# before any universe was ever stored.
_SEED_LIQUID: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "BRK.B",
    "JPM", "V", "MA", "UNH", "XOM", "JNJ", "WMT", "PG", "HD", "COST",
    "ABBV", "MRK", "ADBE", "CRM", "PEP", "KO", "CVX", "BAC", "NFLX", "AMD",
    "LIN", "TMO", "CSCO", "ACN", "MCD", "ABT", "WFC", "DHR", "TXN", "INTC",
    "DIS", "VZ", "CMCSA", "NKE", "PM", "IBM", "QCOM", "AMGN", "HON", "LOW",
    "UNP", "RTX", "CAT", "GE", "BA", "SBUX", "GS", "BLK", "AXP", "NOW",
    "INTU", "ISRG", "BKNG", "AMAT", "LRCX", "MU", "PANW", "ADI", "SYK", "PFE",
    "T", "ORCL", "UBER", "PLTR", "SHOP", "MELI", "PYPL", "MS", "C", "SCHW",
    "BMY", "GILD", "MDT", "DE", "ELV", "LMT", "ADP", "MDLZ", "REGN", "VRTX",
    "CI", "SO", "DUK", "ZTS", "CB", "MMC", "EQIX", "APD", "KLAC", "SNPS",
)


def normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _equity_only(symbols: list[str], settings: Settings) -> list[str]:
    btc = normalize_symbol(settings.btc_symbol)
    out: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = normalize_symbol(raw)
        if not sym or sym == btc or sym in seen:
            continue
        # Crypto pairs never belong in the equity scan universe.
        if "/" in sym:
            continue
        seen.add(sym)
        out.append(sym)
    return out


@dataclass(frozen=True)
class ResolvedUniverse:
    preset: str
    candidates: tuple[str, ...]
    benchmarks: tuple[str, ...]
    forced: dict[str, str] = field(default_factory=dict)
    source: str = SOURCE_SEED
    status: str = STATUS_AVAILABLE
    last_refresh_at: datetime | None = None

    @property
    def scan_symbols(self) -> tuple[str, ...]:
        """Everything the market layer must fetch: benchmark + candidates."""
        return tuple(self.benchmarks) + tuple(self.candidates)

    def with_btc(self, settings: Settings) -> tuple[str, ...]:
        """Universe tuple for Settings.smoke_universe (BTC appended for the sleeve)."""
        return self.scan_symbols + (normalize_symbol(settings.btc_symbol),)


def seed_symbols(preset: str) -> list[str]:
    target = PRESET_TARGETS.get(preset, len(_SEED_LIQUID))
    return list(_SEED_LIQUID[:target])


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------


def store_universe(
    conn: duckdb.DuckDBPyConnection,
    *,
    preset: str,
    members: list[tuple[str, float | None]],
    source: str,
    status: str,
    error: str | None = None,
) -> None:
    now = _utcnow()
    conn.execute("DELETE FROM universe_members WHERE preset = ?", [preset])
    for rank, (symbol, dollar_volume) in enumerate(members, start=1):
        conn.execute(
            """
            INSERT OR REPLACE INTO universe_members
            (preset, symbol, member_rank, dollar_volume, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [preset, normalize_symbol(symbol), rank, dollar_volume, source, now],
        )
    conn.execute(
        """
        INSERT OR REPLACE INTO universe_state
        (preset, status, n_members, source, last_refresh_at, last_error, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [preset, status, len(members), source, now if members else None, error, now],
    )


def mark_universe_error(conn: duckdb.DuckDBPyConnection, *, preset: str, status: str, error: str) -> None:
    """Record a failed refresh without discarding the stored universe."""
    row = conn.execute(
        "SELECT n_members, source, last_refresh_at FROM universe_state WHERE preset = ?", [preset]
    ).fetchone()
    n_members = int(row[0]) if row else 0
    source = str(row[1]) if row else SOURCE_SEED
    last_refresh = row[2] if row else None
    conn.execute(
        """
        INSERT OR REPLACE INTO universe_state
        (preset, status, n_members, source, last_refresh_at, last_error, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [preset, status, n_members, source, last_refresh, error[:300], _utcnow()],
    )


def stored_preset_symbols(conn: duckdb.DuckDBPyConnection, preset: str) -> list[str]:
    rows = conn.execute(
        "SELECT symbol FROM universe_members WHERE preset = ? ORDER BY member_rank", [preset]
    ).fetchall()
    return [str(r[0]) for r in rows]


def universe_state(conn: duckdb.DuckDBPyConnection, preset: str) -> dict[str, object]:
    row = conn.execute(
        """
        SELECT status, n_members, source, last_refresh_at, last_error
        FROM universe_state WHERE preset = ?
        """,
        [preset],
    ).fetchone()
    if not row:
        return {
            "status": STATUS_UNAVAILABLE,
            "n_members": 0,
            "source": SOURCE_SEED,
            "last_refresh_at": None,
            "last_error": None,
        }
    return {
        "status": str(row[0]),
        "n_members": int(row[1]),
        "source": str(row[2]),
        "last_refresh_at": row[3],
        "last_error": row[4],
    }


def held_symbols(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT a.symbol
        FROM portfolio_lots l
        JOIN instrument_aliases a ON a.instrument_id = l.instrument_id
        WHERE a.symbol IS NOT NULL
        """
    ).fetchall()
    return sorted({normalize_symbol(r[0]) for r in rows if r[0]})


def watchlist_symbols(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute("SELECT symbol FROM user_watchlist ORDER BY symbol").fetchall()
    return [str(r[0]) for r in rows]


def custom_symbols(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute("SELECT symbol FROM custom_universe ORDER BY symbol").fetchall()
    return [str(r[0]) for r in rows]


def add_watchlist_symbols(conn: duckdb.DuckDBPyConnection, symbols: list[str]) -> int:
    return _insert_symbols(conn, "user_watchlist", symbols, with_note=True)


def remove_watchlist_symbol(conn: duckdb.DuckDBPyConnection, symbol: str) -> None:
    conn.execute("DELETE FROM user_watchlist WHERE symbol = ?", [normalize_symbol(symbol)])


def add_custom_symbols(conn: duckdb.DuckDBPyConnection, symbols: list[str]) -> int:
    return _insert_symbols(conn, "custom_universe", symbols, with_note=False)


def remove_custom_symbol(conn: duckdb.DuckDBPyConnection, symbol: str) -> None:
    conn.execute("DELETE FROM custom_universe WHERE symbol = ?", [normalize_symbol(symbol)])


def parse_symbol_input(raw: str) -> list[str]:
    """Accept comma / whitespace / newline separated tickers."""
    out: list[str] = []
    for chunk in (raw or "").replace(",", " ").replace("\n", " ").split():
        sym = normalize_symbol(chunk)
        if sym and sym not in out:
            out.append(sym)
    return out


def _insert_symbols(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    symbols: list[str],
    *,
    with_note: bool,
) -> int:
    now = _utcnow()
    added = 0
    for symbol in symbols:
        sym = normalize_symbol(symbol)
        if not sym:
            continue
        if with_note:
            conn.execute(
                f"INSERT OR REPLACE INTO {table} (symbol, note, created_at) VALUES (?, NULL, ?)",
                [sym, now],
            )
        else:
            conn.execute(
                f"INSERT OR REPLACE INTO {table} (symbol, created_at) VALUES (?, ?)",
                [sym, now],
            )
        added += 1
    return added


# --------------------------------------------------------------------------
# refresh + resolution
# --------------------------------------------------------------------------


def refresh_universe(
    conn: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    preset: str,
    provider: object | None = None,
) -> dict[str, object]:
    """Rebuild a preset from the provider. Keeps the last-known list on failure."""
    if preset == PRESET_CUSTOM:
        symbols = _equity_only(custom_symbols(conn), settings)
        store_universe(
            conn,
            preset=preset,
            members=[(s, None) for s in symbols],
            source="user",
            status=STATUS_AVAILABLE if symbols else STATUS_UNAVAILABLE,
        )
        return {"preset": preset, "n": len(symbols), "source": "user", "status": STATUS_AVAILABLE}

    target = PRESET_TARGETS.get(preset, 500)

    if provider is None:
        if not (settings.alpaca_api_key and settings.alpaca_secret_key):
            existing = stored_preset_symbols(conn, preset)
            if existing:
                mark_universe_error(
                    conn,
                    preset=preset,
                    status=STATUS_NOT_CONFIGURED,
                    error="Alpaca NOT_CONFIGURED; keeping last-known universe",
                )
                return {"preset": preset, "n": len(existing), "source": "lkg", "status": STATUS_NOT_CONFIGURED}
            symbols = _equity_only(seed_symbols(preset), settings)
            store_universe(
                conn,
                preset=preset,
                members=[(s, None) for s in symbols],
                source=SOURCE_SEED,
                status=STATUS_NOT_CONFIGURED,
                error="Alpaca NOT_CONFIGURED; offline seed list in use",
            )
            return {"preset": preset, "n": len(symbols), "source": SOURCE_SEED, "status": STATUS_NOT_CONFIGURED}
        from trading_system.providers.alpaca_universe import AlpacaUniverseProvider

        provider = AlpacaUniverseProvider(settings)

    try:
        ranked, note = provider.fetch_liquid_symbols(target)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 — provider failure must not break the app
        existing = stored_preset_symbols(conn, preset)
        status = STATUS_DEGRADED if existing else STATUS_UNAVAILABLE
        mark_universe_error(conn, preset=preset, status=status, error=f"{type(exc).__name__}: {exc}")
        if not existing:
            symbols = _equity_only(seed_symbols(preset), settings)
            store_universe(
                conn,
                preset=preset,
                members=[(s, None) for s in symbols],
                source=SOURCE_SEED,
                status=STATUS_DEGRADED,
                error=f"{type(exc).__name__}: {exc}",
            )
            return {"preset": preset, "n": len(symbols), "source": SOURCE_SEED, "status": STATUS_DEGRADED}
        return {"preset": preset, "n": len(existing), "source": "lkg", "status": status}

    members = [
        (normalize_symbol(sym), float(dv))
        for sym, dv in ranked
        if normalize_symbol(sym) and "/" not in normalize_symbol(sym)
    ]
    source = getattr(provider, "source_name", "provider")
    store_universe(
        conn,
        preset=preset,
        members=members,
        source=source,
        status=STATUS_DEGRADED if note else STATUS_AVAILABLE,
        error=note,
    )
    return {
        "preset": preset,
        "n": len(members),
        "source": source,
        "status": STATUS_DEGRADED if note else STATUS_AVAILABLE,
        "note": note,
    }


def active_preset(settings: Settings) -> str:
    preset = (settings.scan_universe_preset or PRESET_LARGE_LIQUID).strip()
    return preset if preset in PRESET_TARGETS or preset == PRESET_CUSTOM else PRESET_LARGE_LIQUID


def resolve_scan_universe(
    conn: duckdb.DuckDBPyConnection,
    settings: Settings,
) -> ResolvedUniverse:
    """Preset members plus held/watchlist overrides. Benchmark kept separate."""
    preset = active_preset(settings)
    state = universe_state(conn, preset)
    members = stored_preset_symbols(conn, preset)
    source = str(state["source"])
    status = str(state["status"])

    if not members:
        if preset == PRESET_CUSTOM:
            members = custom_symbols(conn)
            source, status = "user", STATUS_AVAILABLE if members else STATUS_UNAVAILABLE
        else:
            members = seed_symbols(preset)
            source, status = SOURCE_SEED, STATUS_NOT_CONFIGURED

    forced: dict[str, str] = {}
    for sym in held_symbols(conn):
        forced[sym] = "held"
    for sym in watchlist_symbols(conn):
        forced.setdefault(sym, "watchlist")

    benchmarks = tuple(_equity_only(list(BENCHMARK_SYMBOLS), settings))
    bench_set = set(benchmarks)

    # Held/watchlist names lead so downstream caps (SEC, live quotes) reach them first.
    candidates = [s for s in _equity_only(list(forced), settings) if s not in bench_set]
    for sym in _equity_only(members, settings):
        if sym not in bench_set and sym not in candidates:
            candidates.append(sym)

    return ResolvedUniverse(
        preset=preset,
        candidates=tuple(candidates),
        benchmarks=benchmarks,
        forced={k: v for k, v in forced.items() if k not in bench_set},
        source=source,
        status=status,
        last_refresh_at=state["last_refresh_at"],  # type: ignore[arg-type]
    )


def universe_needs_refresh(conn: duckdb.DuckDBPyConnection, settings: Settings) -> bool:
    preset = active_preset(settings)
    if preset == PRESET_CUSTOM:
        return False
    state = universe_state(conn, preset)
    if not state["last_refresh_at"] or int(state["n_members"]) == 0:
        return True
    last = state["last_refresh_at"]
    if not isinstance(last, datetime):
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return _utcnow() - last > timedelta(days=max(1, settings.universe_refresh_days))


def prepare_scan_universe(
    conn: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    force_refresh: bool = False,
) -> ResolvedUniverse:
    """Refresh the preset when due, then resolve. Never raises on provider failure."""
    preset = active_preset(settings)
    if force_refresh or universe_needs_refresh(conn, settings):
        try:
            refresh_universe(conn, settings, preset=preset)
        except Exception as exc:  # noqa: BLE001 — universe refresh must not break a scan
            mark_universe_error(
                conn,
                preset=preset,
                status=STATUS_DEGRADED if stored_preset_symbols(conn, preset) else STATUS_UNAVAILABLE,
                error=f"{type(exc).__name__}: {exc}",
            )
    return resolve_scan_universe(conn, settings)


def coverage_stats(
    conn: duckdb.DuckDBPyConnection,
    resolved: ResolvedUniverse,
    *,
    stale_after_days: int = 5,
) -> dict[str, object]:
    """How many scan symbols actually have usable bars."""
    from trading_system.market.registry import stable_instrument_id

    symbols = list(resolved.scan_symbols)
    total = len(symbols)
    if total == 0:
        return {
            "total": 0, "ready": 0, "stale": 0, "unavailable": 0,
            "coverage_pct": 0.0, "last_session": None,
        }

    rows = conn.execute(
        "SELECT instrument_id, MAX(session_date) FROM equity_daily_bars GROUP BY instrument_id"
    ).fetchall()
    last_by_inst = {str(r[0]): r[1] for r in rows}
    newest = max((v for v in last_by_inst.values() if v is not None), default=None)

    ready = stale = unavailable = 0
    for sym in symbols:
        inst = str(stable_instrument_id(sym))
        last = last_by_inst.get(inst)
        if last is None:
            unavailable += 1
        elif newest is not None and (newest - last).days > stale_after_days:
            stale += 1
        else:
            ready += 1

    return {
        "total": total,
        "ready": ready,
        "stale": stale,
        "unavailable": unavailable,
        "coverage_pct": round(100.0 * ready / total, 1),
        "last_session": newest,
    }


def load_universe(settings: Settings) -> tuple[str, ...]:
    """Offline seed universe (no DB). Kept for callers without a connection."""
    symbols = _equity_only(list(BENCHMARK_SYMBOLS) + seed_symbols(active_preset(settings)), settings)
    return tuple(symbols) + (normalize_symbol(settings.btc_symbol),)
