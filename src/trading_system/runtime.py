"""Runtime activity events persisted for the Runtime Activity view."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class RuntimeEventKind(StrEnum):
    FETCH_US_DAILY = "fetch_us_daily"
    FETCH_BTC = "fetch_btc"
    INFERENCE = "inference"
    TRAINING = "training"
    PUBLISH_EPOCH = "publish_epoch"
    LABEL_WAIT = "label_wait"
    PROVIDER_RETRY = "provider_retry"
    PROVIDER_RECOVERY = "provider_recovery"
    LLM_JUDGE = "llm_judge"
    LLM_RESEARCH = "llm_research"
    SEC_INGEST = "sec_ingest"
    TICK_COMMIT = "tick_commit"
    HEARTBEAT = "heartbeat"


class RuntimeEventStatus(StrEnum):
    STARTED = "started"
    PROGRESS = "progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEGRADED = "degraded"
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"


class RuntimeEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"rt_{uuid4().hex}")
    kind: RuntimeEventKind
    status: RuntimeEventStatus
    message: str
    progress_done: int | None = None
    progress_total: int | None = None
    duration_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, object] = Field(default_factory=dict)
