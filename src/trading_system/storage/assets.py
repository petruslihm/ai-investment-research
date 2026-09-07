"""Instrument alias registry helpers (rename / reuse safe)."""

from __future__ import annotations

from datetime import date

import duckdb

from trading_system.ids import (
    InstrumentAlias,
    InstrumentId,
    InstrumentRecord,
    IssuerRecord,
)


class AliasConflictError(ValueError):
    pass


def upsert_issuer(conn: duckdb.DuckDBPyConnection, issuer: IssuerRecord) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO issuers (issuer_id, display_name, cik, created_at)
        VALUES (?, ?, ?, ?)
        """,
        [issuer.issuer_id, issuer.display_name, issuer.cik, issuer.created_at],
    )


def upsert_instrument(conn: duckdb.DuckDBPyConnection, inst: InstrumentRecord) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO instruments
        (instrument_id, issuer_id, asset_class, display_name, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            inst.instrument_id,
            inst.issuer_id,
            inst.asset_class.value,
            inst.display_name,
            inst.created_at,
        ],
    )


def add_alias(conn: duckdb.DuckDBPyConnection, alias: InstrumentAlias) -> None:
    """Add effective-dated alias. Ticker rename/reuse must not collide open ranges."""
    # Reject overlapping open-ended or overlapping ranges for same provider+symbol
    rows = conn.execute(
        """
        SELECT instrument_id, effective_from, effective_to
        FROM instrument_aliases
        WHERE provider = ? AND symbol = ?
        """,
        [alias.provider, alias.symbol],
    ).fetchall()
    for instrument_id, eff_from, eff_to in rows:
        existing = InstrumentAlias(
            instrument_id=InstrumentId(instrument_id),
            provider=alias.provider,
            symbol=alias.symbol,
            effective_from=eff_from if not isinstance(eff_from, str) else date.fromisoformat(eff_from),
            effective_to=(
                None
                if eff_to is None
                else (eff_to if not isinstance(eff_to, str) else date.fromisoformat(eff_to))
            ),
        )
        if _ranges_overlap(existing, alias):
            if existing.instrument_id != alias.instrument_id:
                raise AliasConflictError(
                    f"symbol {alias.provider}:{alias.symbol} reuse overlaps "
                    f"{existing.instrument_id} vs {alias.instrument_id}"
                )
            raise AliasConflictError(
                f"overlapping alias window for {alias.provider}:{alias.symbol}"
            )

    conn.execute(
        """
        INSERT INTO instrument_aliases
        (instrument_id, provider, symbol, effective_from, effective_to, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            alias.instrument_id,
            alias.provider,
            alias.symbol,
            alias.effective_from,
            alias.effective_to,
            alias.notes,
        ],
    )


def resolve_alias(
    conn: duckdb.DuckDBPyConnection,
    provider: str,
    symbol: str,
    as_of: date,
) -> InstrumentId | None:
    rows = conn.execute(
        """
        SELECT instrument_id, effective_from, effective_to
        FROM instrument_aliases
        WHERE provider = ? AND symbol = ?
        ORDER BY effective_from
        """,
        [provider, symbol],
    ).fetchall()
    for instrument_id, eff_from, eff_to in rows:
        alias = InstrumentAlias(
            instrument_id=InstrumentId(instrument_id),
            provider=provider,
            symbol=symbol,
            effective_from=eff_from if not isinstance(eff_from, str) else date.fromisoformat(eff_from),
            effective_to=(
                None
                if eff_to is None
                else (eff_to if not isinstance(eff_to, str) else date.fromisoformat(eff_to))
            ),
        )
        if alias.covers(as_of):
            return alias.instrument_id
    return None


def close_alias(
    conn: duckdb.DuckDBPyConnection,
    provider: str,
    symbol: str,
    instrument_id: InstrumentId,
    effective_to: date,
) -> None:
    conn.execute(
        """
        UPDATE instrument_aliases
        SET effective_to = ?
        WHERE provider = ? AND symbol = ? AND instrument_id = ? AND effective_to IS NULL
        """,
        [effective_to, provider, symbol, instrument_id],
    )


def _ranges_overlap(a: InstrumentAlias, b: InstrumentAlias) -> bool:
    a_end = a.effective_to or date.max
    b_end = b.effective_to or date.max
    return a.effective_from <= b_end and b.effective_from <= a_end
