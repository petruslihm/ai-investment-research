"""Bootstrap instruments/aliases for the smoke universe."""

from __future__ import annotations

import logging
from datetime import date

import duckdb

from trading_system.config import Settings
from trading_system.ids import (
    AssetClass,
    InstrumentAlias,
    InstrumentId,
    InstrumentRecord,
    IssuerId,
    IssuerRecord,
    USD_CASH_INSTRUMENT_ID,
    USD_CASH_ISSUER_ID,
    canonicalize_instrument_id,
)
from trading_system.storage.assets import add_alias, resolve_alias, upsert_instrument, upsert_issuer

_log = logging.getLogger("trading_system.market.registry")


def stable_instrument_id(symbol: str) -> InstrumentId:
    """Deterministic instrument id for development/smoke symbols."""
    return canonicalize_instrument_id(symbol)


def stable_issuer_id(symbol: str) -> IssuerId:
    normalized = symbol.upper().replace("/", "_").replace(".", "_")
    return IssuerId(f"iss_{normalized.lower()}")


def bootstrap_smoke_universe(
    conn: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    provider: str,
    as_of: date | None = None,
) -> dict[str, InstrumentId]:
    """Register smoke-universe symbols with effective-dated aliases."""
    as_of = as_of or date.today()
    mapping: dict[str, InstrumentId] = {}

    # Residual USD cash (not a market symbol)
    upsert_issuer(
        conn,
        IssuerRecord(issuer_id=USD_CASH_ISSUER_ID, display_name="USD Cash Residual"),
    )
    upsert_instrument(
        conn,
        InstrumentRecord(
            instrument_id=USD_CASH_INSTRUMENT_ID,
            issuer_id=USD_CASH_ISSUER_ID,
            asset_class=AssetClass.USD_CASH,
            display_name="USD Cash",
        ),
    )

    for symbol in settings.smoke_universe:
        if symbol == settings.btc_symbol:
            asset = AssetClass.BTC
            inst_id = stable_instrument_id(settings.btc_symbol)
            issuer_id = stable_issuer_id(settings.btc_symbol)
        else:
            asset = AssetClass.US_EQUITY
            inst_id = stable_instrument_id(symbol)
            issuer_id = stable_issuer_id(symbol)

        upsert_issuer(conn, IssuerRecord(issuer_id=issuer_id, display_name=symbol))
        upsert_instrument(
            conn,
            InstrumentRecord(
                instrument_id=inst_id,
                issuer_id=issuer_id,
                asset_class=asset,
                display_name=symbol,
            ),
        )
        if resolve_alias(conn, provider, symbol, as_of) is None:
            add_alias(
                conn,
                InstrumentAlias(
                    instrument_id=inst_id,
                    provider=provider,
                    symbol=symbol,
                    effective_from=date(2000, 1, 1),
                ),
            )
        mapping[symbol] = inst_id

    mapping.update(register_held_equities(conn, settings, provider=provider, as_of=as_of))
    return mapping


def _held_symbol(instrument_id: InstrumentId) -> str:
    s = str(instrument_id)
    if s.startswith("inst_"):
        s = s[5:]
    return s.upper().replace("_", "/") if s.lower().startswith("btc") else s.upper()


def register_held_equities(
    conn: duckdb.DuckDBPyConnection,
    settings: Settings,
    *,
    provider: str,
    as_of: date | None = None,
) -> dict[str, InstrumentId]:
    """Ensure manually held names are in the fetch map even if they are outside the scan universe."""
    as_of = as_of or date.today()
    extra: dict[str, InstrumentId] = {}
    try:
        rows = conn.execute("SELECT DISTINCT instrument_id FROM portfolio_lots").fetchall()
    except Exception:  # noqa: BLE001 — empty DB during tests
        return extra
    for (raw,) in rows:
        try:
            inst_id = canonicalize_instrument_id(raw)
        except ValueError:
            _log.warning("register_held_equities: dropping unparseable held instrument_id=%r", raw)
            continue
        if "btc" in str(inst_id).lower() or str(inst_id) == str(USD_CASH_INSTRUMENT_ID):
            continue
        symbol = _held_symbol(inst_id)
        issuer_id = stable_issuer_id(symbol)
        upsert_issuer(conn, IssuerRecord(issuer_id=issuer_id, display_name=symbol))
        upsert_instrument(
            conn,
            InstrumentRecord(
                instrument_id=inst_id,
                issuer_id=issuer_id,
                asset_class=AssetClass.US_EQUITY,
                display_name=symbol,
            ),
        )
        if resolve_alias(conn, provider, symbol, as_of) is None:
            add_alias(
                conn,
                InstrumentAlias(
                    instrument_id=inst_id,
                    provider=provider,
                    symbol=symbol,
                    effective_from=date(2000, 1, 1),
                ),
            )
        extra[symbol] = inst_id
    return extra


def symbol_for_instrument(
    conn: duckdb.DuckDBPyConnection,
    instrument_id: InstrumentId,
    *,
    provider: str,
    as_of: date,
) -> str | None:
    rows = conn.execute(
        """
        SELECT symbol FROM instrument_aliases
        WHERE instrument_id = ? AND provider = ?
          AND effective_from <= ?
          AND (effective_to IS NULL OR effective_to >= ?)
        ORDER BY effective_from DESC
        LIMIT 1
        """,
        [instrument_id, provider, as_of, as_of],
    ).fetchall()
    return rows[0][0] if rows else None


def equity_instrument_ids(mapping: dict[str, InstrumentId], settings: Settings) -> list[InstrumentId]:
    return [
        inst_id
        for sym, inst_id in mapping.items()
        if sym != settings.btc_symbol
    ]
