"""Crash-consistent tick commit primitives (tick_id, watermark-last, alert outbox)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import duckdb

from trading_system.ids import TickId, new_tick_id


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TickCommitPayload:
    tick_id: TickId = field(default_factory=new_tick_id)
    decision_epoch_id: str | None = None
    feature_snapshot_id: str | None = None
    observations: list[dict[str, object]] = field(default_factory=list)
    predictions: list[dict[str, object]] = field(default_factory=list)
    recommendations: list[dict[str, object]] = field(default_factory=list)
    watermarks: list[dict[str, object]] = field(default_factory=list)
    alert_outbox: list[dict[str, object]] = field(default_factory=list)


class TickCommitError(RuntimeError):
    pass


def commit_tick(conn: duckdb.DuckDBPyConnection, payload: TickCommitPayload) -> TickId:
    """Commit observations/predictions/recs/alerts then watermarks last.

    Uses an explicit BEGIN/COMMIT transaction. Watermark updates are applied
    after other rows so a crash mid-commit does not advance watermarks alone.
    Alert outbox inserts are idempotent on idempotency_key.
    """
    tick_id = payload.tick_id
    now = _utcnow()

    # Idempotent: if tick already committed, return
    existing = conn.execute(
        "SELECT status FROM ticks WHERE tick_id = ?", [tick_id]
    ).fetchone()
    if existing is not None and existing[0] == "committed":
        return tick_id

    try:
        # Prefer SQL BEGIN — portable across DuckDB Python bindings
        try:
            conn.execute("BEGIN TRANSACTION")
        except Exception:  # noqa: BLE001
            conn.begin()
        conn.execute(
            """
            INSERT OR REPLACE INTO ticks
            (tick_id, decision_epoch_id, feature_snapshot_id, committed_at, status)
            VALUES (?, ?, ?, ?, 'committing')
            """,
            [
                tick_id,
                payload.decision_epoch_id,
                payload.feature_snapshot_id,
                now,
            ],
        )

        for obs in payload.observations:
            conn.execute(
                """
                INSERT OR REPLACE INTO tick_observations (tick_id, instrument_id, payload_json)
                VALUES (?, ?, ?)
                """,
                [tick_id, str(obs["instrument_id"]), json.dumps(obs)],
            )

        for pred in payload.predictions:
            conn.execute(
                """
                INSERT OR REPLACE INTO tick_predictions (tick_id, instrument_id, payload_json)
                VALUES (?, ?, ?)
                """,
                [tick_id, str(pred["instrument_id"]), json.dumps(pred)],
            )

        for rec in payload.recommendations:
            conn.execute(
                """
                INSERT OR REPLACE INTO tick_recommendations
                (tick_id, recommendation_id, source, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    tick_id,
                    str(rec["recommendation_id"]),
                    str(rec.get("source", "quant_only")),
                    json.dumps(rec),
                ],
            )

        for alert in payload.alert_outbox:
            # Idempotent on idempotency_key
            conn.execute(
                """
                INSERT INTO alert_outbox
                (outbox_id, alert_id, channel, idempotency_key, status, attempts,
                 last_error, created_at, updated_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                [
                    str(alert["outbox_id"]),
                    str(alert["alert_id"]),
                    str(alert["channel"]),
                    str(alert["idempotency_key"]),
                    str(alert.get("status", "pending")),
                    int(alert.get("attempts", 0)),
                    alert.get("last_error"),
                    now,
                    now,
                    json.dumps(alert.get("payload", {})),
                ],
            )

        # Watermark-last
        for wm in payload.watermarks:
            conn.execute(
                """
                INSERT OR REPLACE INTO watermarks
                (watermark_key, instrument_id, value, updated_at, tick_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    str(wm["watermark_key"]),
                    wm.get("instrument_id"),
                    str(wm["value"]),
                    now,
                    tick_id,
                ],
            )

        conn.execute(
            "UPDATE ticks SET status = 'committed', committed_at = ? WHERE tick_id = ?",
            [now, tick_id],
        )
        try:
            conn.execute("COMMIT")
        except Exception:  # noqa: BLE001
            conn.commit()
    except Exception as exc:
        try:
            conn.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
        raise TickCommitError(f"tick commit failed: {exc}") from exc

    return tick_id
