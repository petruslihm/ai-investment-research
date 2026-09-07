"""Quote validity gates for canonical decision data."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from trading_system.providers.interfaces import LatestQuote

DEFAULT_MAX_AGE_SECONDS = 24 * 3600
DEFAULT_MAX_FUTURE_SKEW_SECONDS = 60.0


def ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("naive timestamp")
    return ts.astimezone(timezone.utc)


def validate_quote_for_decision(
    quote: LatestQuote,
    *,
    prior_event_ts: datetime | None = None,
    max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
    max_future_skew_seconds: float = DEFAULT_MAX_FUTURE_SKEW_SECONDS,
    receive_at: datetime | None = None,
) -> tuple[bool, str | None]:
    """Return (is_valid, reason_code). Invalid quotes must not be canonical."""
    receive_at = receive_at or datetime.now(timezone.utc)

    if not math.isfinite(quote.price) or quote.price <= 0:
        return False, "non_positive_or_nonfinite_price"

    if quote.event_ts is None:
        return False, "missing_event_ts"

    try:
        event_ts = ensure_utc(quote.event_ts)
    except ValueError:
        return False, "naive_event_ts"

    if event_ts > receive_at + timedelta(seconds=max_future_skew_seconds):
        return False, "future_skew"

    age = (receive_at - event_ts).total_seconds()
    if age > max_age_seconds:
        return False, "stale_event_ts"

    if prior_event_ts is not None:
        prior = ensure_utc(prior_event_ts)
        if event_ts <= prior:
            return False, "duplicate_or_out_of_order_event_ts"

    return True, None
