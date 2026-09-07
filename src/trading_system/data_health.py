"""Data Health / Data Quality Manifest (never report missing feeds as healthy)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from trading_system.ids import RequestSetId


class HealthLevel(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CoverageSummary(BaseModel):
    request_set_id: RequestSetId | None = None
    requested: int = 0
    succeeded: int = 0
    missing: int = 0
    error: int = 0
    quarantined: int = 0


class DataQualityManifest(BaseModel):
    """Attached to quant/LLM decision packages."""

    manifest_id: str = Field(default_factory=lambda: f"dqm_{uuid4().hex}")
    level: HealthLevel = HealthLevel.UNKNOWN
    coverage: CoverageSummary = Field(default_factory=CoverageSummary)
    lkg_in_use: bool = False
    stale_components: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DataHealthSnapshot(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: f"dh_{uuid4().hex}")
    overall: HealthLevel = HealthLevel.UNKNOWN
    equity_feed: HealthLevel = HealthLevel.UNKNOWN
    btc_feed: HealthLevel = HealthLevel.UNKNOWN
    coverage: CoverageSummary = Field(default_factory=CoverageSummary)
    last_tick_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
