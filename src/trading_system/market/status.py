"""Data availability flags for fresh, degraded, and unavailable inputs."""

from __future__ import annotations

from enum import StrEnum


class DataStatus(StrEnum):
    VALID = "valid"
    STALE = "stale"
    DEGRADED = "degraded"
    MISSING = "missing"
    RECOVERING = "recovering"
    UNAVAILABLE = "unavailable"
    SCHEMA_CHANGED = "schema_changed"


def is_displayable(status: DataStatus) -> bool:
    """Statuses that may be shown with an explicit age/warning."""
    return status in {
        DataStatus.VALID,
        DataStatus.STALE,
        DataStatus.DEGRADED,
        DataStatus.RECOVERING,
    }


def is_canonical_decision(status: DataStatus, *, has_event_ts: bool) -> bool:
    """Only VALID timestamped observations may drive canonical decisions."""
    if status != DataStatus.VALID:
        return False
    return has_event_ts
