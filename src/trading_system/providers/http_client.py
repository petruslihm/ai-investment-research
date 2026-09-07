"""Bounded HTTP helper for market-data adapters."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from trading_system.providers.never_block import (
    Deadline,
    NeverBlockResult,
    RetryKind,
    RetryPolicy,
    classify_http_status,
    run_with_deadline,
)


def parse_retry_after(headers: httpx.Headers | dict[str, str]) -> float | None:
    raw = headers.get("Retry-After") if headers is not None else None
    if raw is None:
        raw = headers.get("retry-after") if headers is not None else None
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    policy: RetryPolicy | None = None,
    deadline: Deadline | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> NeverBlockResult:
    policy = policy or RetryPolicy()

    def _classify(exc: BaseException) -> RetryKind:
        if isinstance(exc, httpx.HTTPStatusError):
            return classify_http_status(exc.response.status_code)
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, TimeoutError)):
            return RetryKind.RETRYABLE
        return RetryKind.NON_RETRYABLE

    def _call() -> dict:
        response = client.request(method, url, headers=headers, params=params)
        if response.status_code == 429:
            # Don't sleep here and then still fail the same (unretried) response --
            # raise immediately and let run_with_deadline's own retry/backoff honor
            # Retry-After (via retry_after_seconds) before making the next attempt.
            exc = httpx.HTTPStatusError("rate limited", request=response.request, response=response)
            delay = parse_retry_after(response.headers)
            if delay is not None:
                exc.retry_after_seconds = delay  # type: ignore[attr-defined]
            raise exc
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
        return data

    return run_with_deadline(
        _call,
        policy=policy,
        deadline=deadline,
        classify=_classify,
        sleep=sleep,
    )
