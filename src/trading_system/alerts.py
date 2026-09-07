"""Alert records and idempotent outbox (no order placement)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from trading_system.ids import InstrumentId, TickId


class AlertKind(StrEnum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    EXIT_DOWNGRADE = "exit_downgrade"
    DATA_HEALTH = "data_health"
    RUNTIME = "runtime"
    RECOMMENDATION = "recommendation"


class AlertChannel(StrEnum):
    CONSOLE = "console"
    UI = "ui"
    KAKAO = "kakao"


class AlertRecord(BaseModel):
    alert_id: str = Field(default_factory=lambda: f"alert_{uuid4().hex}")
    kind: AlertKind
    instrument_id: InstrumentId | None = None
    tick_id: TickId | None = None
    message: str
    level: str = "info"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Alerts never place orders — explicit product invariant
    places_orders: bool = False


class AlertOutboxStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    DEAD = "dead"


class AlertOutboxEntry(BaseModel):
    """Idempotent alert delivery outbox row."""

    outbox_id: str = Field(default_factory=lambda: f"out_{uuid4().hex}")
    alert_id: str
    channel: AlertChannel
    idempotency_key: str
    status: AlertOutboxStatus = AlertOutboxStatus.PENDING
    attempts: int = 0
    last_error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
