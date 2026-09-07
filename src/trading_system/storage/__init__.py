"""DuckDB store: open, migrate schema, writer lease helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb

from trading_system.storage.leases import WriterLease, WriterLeaseBusy
from trading_system.storage.schema import ALTER_STATEMENTS, DDL_STATEMENTS, FOUNDATION_TABLES, SCHEMA_VERSION

_log = logging.getLogger("trading_system.storage")


class Store:
    """Canonical DuckDB store with exclusive writer lease."""

    def __init__(self, path: Path | str, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None
        self.writer_lease: WriterLease | None = None

    def open(self, *, acquire_writer: bool = True, stale_seconds: int = 30) -> duckdb.DuckDBPyConnection:
        if not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path), read_only=self.read_only)
        try:
            if not self.read_only:
                schema_ready = self._schema_ready()
                if acquire_writer and schema_ready:
                    # Once the lease table exists, take ownership before any DDL.
                    # Applying idempotent migrations first would still be an
                    # unleased write racing the current owner.
                    lease = WriterLease(self._conn, stale_seconds=stale_seconds)
                    lease.acquire()
                    self.writer_lease = lease
                    self.apply_schema()
                elif acquire_writer:
                    # First-ever initialization has no lease table to acquire yet.
                    # Create the schema, then immediately establish ownership.
                    self.apply_schema()
                    lease = WriterLease(self._conn, stale_seconds=stale_seconds)
                    lease.acquire()
                    self.writer_lease = lease
                elif not schema_ready:
                    self.apply_schema()
            return self._conn
        except BaseException:
            # open() has no caller-owned connection to clean up when schema setup or
            # lease acquisition fails.  Leaving it attached leaks a DuckDB handle and
            # makes a later retry overwrite self._conn without closing the old one.
            conn = self._conn
            lease = self.writer_lease
            self._conn = None
            self.writer_lease = None
            if lease is not None and lease.held and conn is not None:
                # A lease can be acquired successfully and then apply_schema() can
                # still fail (e.g. a bad migration). Release it here, with the
                # connection still open, or the DB row keeps looking "held" by a
                # now-dead owner for the full stale_seconds window -- blocking every
                # other writer for an otherwise-instant DDL failure.
                try:
                    lease.release()
                except Exception:  # noqa: BLE001
                    _log.warning(
                        "Store.open: failed to release writer_lease after a failed open()", exc_info=True
                    )
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    # A close() failure here must not replace/mask whatever real
                    # exception (from apply_schema()/lease.acquire()) is already
                    # propagating via the `raise` below -- that's the one the caller
                    # needs to see.
                    _log.warning(
                        "Store.open: failed to close connection after a failed open()", exc_info=True
                    )
            raise

    def _schema_ready(self) -> bool:
        row = self.conn.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'writer_lease'
            """
        ).fetchone()
        return row is not None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("store not open")
        return self._conn

    def apply_schema(self) -> None:
        for ddl in DDL_STATEMENTS:
            self.conn.execute(ddl)
        for stmt in ALTER_STATEMENTS:
            # Every migration is idempotent (ADD COLUMN IF NOT EXISTS). A real
            # schema/storage error must abort open() instead of being mislabeled as
            # an already-applied migration and hidden until a later query fails.
            self.conn.execute(stmt)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)
            """,
            [str(SCHEMA_VERSION)],
        )

    def list_tables(self) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        ).fetchall()
        return [r[0] for r in rows]

    def table_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in FOUNDATION_TABLES:
            row = self.conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
            counts[name] = int(row[0]) if row else 0
        return counts

    def close(self) -> None:
        if self.writer_lease is not None and self.writer_lease.held:
            try:
                self.writer_lease.release()
            except Exception:  # noqa: BLE001
                # The DB row is never deleted here, so it will keep looking "held" by a
                # token whose connection is already gone until stale_seconds elapses --
                # must not be invisible when it happens.
                _log.warning("Store.close: writer_lease.release() failed; lease left stale", exc_info=True)
            self.writer_lease = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Store:
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def try_open_second_writer(path: Path | str, *, stale_seconds: int = 30) -> None:
    """Helper for tests: second process/connection must be rejected while lease live."""
    store = Store(path)
    store.open(acquire_writer=True, stale_seconds=stale_seconds)
    # If we got here without Busy, release and signal unexpected success
    store.close()
    raise AssertionError("second writer was not rejected")


__all__ = [
    "FOUNDATION_TABLES",
    "Store",
    "WriterLeaseBusy",
    "try_open_second_writer",
]
