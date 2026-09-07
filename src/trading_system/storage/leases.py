"""Exclusive cross-process DuckDB writer lease (owner-token fence)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import duckdb


class WriterLeaseError(RuntimeError):
    pass


class WriterLeaseBusy(WriterLeaseError):
    """Raised when another process holds a non-stale writer lease."""


LEASE_NAME = "duckdb_writer"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WriterLease:
    """Exclusive writer ownership with owner-token fencing."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        stale_seconds: int = 30,
        owner_token: str | None = None,
    ) -> None:
        self._conn = conn
        self.stale_seconds = stale_seconds
        self.owner_token = owner_token or f"owner_{uuid4().hex}"
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> str:
        """Acquire or reclaim stale lease. Rejects concurrent live owners."""
        now = _utcnow()
        expires = now + timedelta(seconds=self.stale_seconds)
        row = self._conn.execute(
            "SELECT owner_token, expires_at FROM writer_lease WHERE lease_name = ?",
            [LEASE_NAME],
        ).fetchone()

        if row is None:
            self._conn.execute(
                """
                INSERT INTO writer_lease (lease_name, owner_token, acquired_at, heartbeat_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [LEASE_NAME, self.owner_token, now, now, expires],
            )
            self._held = True
            return self.owner_token

        existing_owner, expires_at = row[0], row[1]
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if existing_owner == self.owner_token:
            self.heartbeat()
            self._held = True
            return self.owner_token

        if expires_at > now:
            raise WriterLeaseBusy(
                f"writer lease held by {existing_owner!r} until {expires_at.isoformat()}"
            )

        # Stale owner — reclaim with fence (only if still the stale token)
        self._conn.execute(
            """
            UPDATE writer_lease
            SET owner_token = ?, acquired_at = ?, heartbeat_at = ?, expires_at = ?
            WHERE lease_name = ? AND owner_token = ? AND expires_at <= ?
            """,
            [
                self.owner_token,
                now,
                now,
                expires,
                LEASE_NAME,
                existing_owner,
                now,
            ],
        )
        # Verify owner-token fence
        check = self._conn.execute(
            "SELECT owner_token FROM writer_lease WHERE lease_name = ?",
            [LEASE_NAME],
        ).fetchone()
        if check is None or check[0] != self.owner_token:
            raise WriterLeaseBusy("failed to reclaim stale writer lease (race)")
        self._held = True
        return self.owner_token

    def heartbeat(self) -> None:
        self._require_held()
        now = _utcnow()
        expires = now + timedelta(seconds=self.stale_seconds)
        self._conn.execute(
            """
            UPDATE writer_lease
            SET heartbeat_at = ?, expires_at = ?
            WHERE lease_name = ? AND owner_token = ?
            """,
            [now, expires, LEASE_NAME, self.owner_token],
        )
        row = self._conn.execute(
            "SELECT owner_token FROM writer_lease WHERE lease_name = ?",
            [LEASE_NAME],
        ).fetchone()
        if row is None or row[0] != self.owner_token:
            self._held = False
            raise WriterLeaseError("writer lease lost (owner-token fence)")

    def release(self) -> None:
        if not self._held:
            return
        self._conn.execute(
            "DELETE FROM writer_lease WHERE lease_name = ? AND owner_token = ?",
            [LEASE_NAME, self.owner_token],
        )
        self._held = False

    def _require_held(self) -> None:
        if not self._held:
            raise WriterLeaseError("writer lease not held")


class JobLeaseError(RuntimeError):
    pass


class JobLeaseBusy(JobLeaseError):
    pass


class JobLease:
    """Fenced job lease with owner-token heartbeat/finish."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        job_key: str,
        *,
        stale_seconds: int = 120,
        owner_token: str | None = None,
    ) -> None:
        self._conn = conn
        self.job_key = job_key
        self.stale_seconds = stale_seconds
        self.owner_token = owner_token or f"job_{uuid4().hex}"
        self._held = False

    def acquire(self, payload_json: str | None = None) -> str:
        now = _utcnow()
        expires = now + timedelta(seconds=self.stale_seconds)
        row = self._conn.execute(
            "SELECT owner_token, expires_at, status FROM job_lease WHERE job_key = ?",
            [self.job_key],
        ).fetchone()

        if row is None:
            self._conn.execute(
                """
                INSERT INTO job_lease
                (job_key, owner_token, status, acquired_at, heartbeat_at, expires_at, payload_json)
                VALUES (?, ?, 'running', ?, ?, ?, ?)
                """,
                [self.job_key, self.owner_token, now, now, expires, payload_json],
            )
            self._held = True
            return self.owner_token

        existing_owner, expires_at, status = row
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if existing_owner == self.owner_token and status == "running":
            self.heartbeat()
            self._held = True
            return self.owner_token

        if status == "running" and expires_at > now:
            raise JobLeaseBusy(
                f"job {self.job_key!r} held by {existing_owner!r}"
            )

        # Stale or finished — reclaim / restart with fence
        self._conn.execute(
            """
            UPDATE job_lease
            SET owner_token = ?, status = 'running', acquired_at = ?,
                heartbeat_at = ?, expires_at = ?, payload_json = COALESCE(?, payload_json)
            WHERE job_key = ? AND owner_token = ?
            """,
            [
                self.owner_token,
                now,
                now,
                expires,
                payload_json,
                self.job_key,
                existing_owner,
            ],
        )
        check = self._conn.execute(
            "SELECT owner_token FROM job_lease WHERE job_key = ?",
            [self.job_key],
        ).fetchone()
        if check is None or check[0] != self.owner_token:
            raise JobLeaseBusy(f"failed to reclaim job lease {self.job_key!r}")
        self._held = True
        return self.owner_token

    def heartbeat(self) -> None:
        if not self._held:
            raise JobLeaseError("job lease not held")
        now = _utcnow()
        expires = now + timedelta(seconds=self.stale_seconds)
        self._conn.execute(
            """
            UPDATE job_lease
            SET heartbeat_at = ?, expires_at = ?
            WHERE job_key = ? AND owner_token = ? AND status = 'running'
            """,
            [now, expires, self.job_key, self.owner_token],
        )
        row = self._conn.execute(
            "SELECT owner_token, status FROM job_lease WHERE job_key = ?",
            [self.job_key],
        ).fetchone()
        if row is None or row[0] != self.owner_token or row[1] != "running":
            self._held = False
            raise JobLeaseError("job lease lost (owner-token fence)")

    def finish(self, status: str = "succeeded") -> None:
        if not self._held:
            raise JobLeaseError("job lease not held")
        now = _utcnow()
        # Owner-token fence: only the current owner may finish
        row = self._conn.execute(
            "SELECT owner_token, status FROM job_lease WHERE job_key = ?",
            [self.job_key],
        ).fetchone()
        if row is None or row[0] != self.owner_token:
            self._held = False
            raise JobLeaseError("finish rejected — owner-token fence")
        self._conn.execute(
            """
            UPDATE job_lease
            SET status = ?, heartbeat_at = ?
            WHERE job_key = ? AND owner_token = ?
            """,
            [status, now, self.job_key, self.owner_token],
        )
        self._held = False
