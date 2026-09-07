"""Additional foundation smoke tests for never-block helpers."""

from __future__ import annotations

import threading
import time

import pytest

from trading_system.providers import never_block
from trading_system.providers.never_block import (
    Deadline,
    ProviderHttpError,
    RetryKind,
    RetryPolicy,
    classify_http_status,
    run_with_deadline,
)


def test_never_block_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return "ok"

    result = run_with_deadline(
        flaky,
        policy=RetryPolicy(max_attempts=4, base_delay_seconds=0.0, jitter=0.0, total_deadline_seconds=5.0),
        sleep=lambda _s: None,
    )
    assert result.ok is True
    assert result.value == "ok"
    assert result.attempts == 3


def test_never_block_non_retryable() -> None:
    def boom() -> None:
        raise ValueError("bad")

    result = run_with_deadline(
        boom,
        policy=RetryPolicy(max_attempts=5, base_delay_seconds=0.0, jitter=0.0),
        classify=lambda _e: RetryKind.NON_RETRYABLE,
        sleep=lambda _s: None,
    )
    assert result.ok is False
    assert result.degraded is True
    assert result.attempts == 1
    assert result.kind == RetryKind.NON_RETRYABLE


def test_retry_after_is_honored_and_capped() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def limited() -> str:
        calls["n"] += 1
        raise ProviderHttpError(
            "SEC 429 /submissions/CIK0000320193.json",
            status_code=429,
            kind=RetryKind.RATE_LIMITED,
            retry_after_seconds=30.0,
        )

    result = run_with_deadline(
        limited,
        policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.1,
            max_delay_seconds=4.0,
            jitter=0.0,
            total_deadline_seconds=20.0,
        ),
        sleep=lambda s: sleeps.append(s),
    )
    assert result.ok is False
    assert result.kind == RetryKind.RATE_LIMITED
    assert result.attempts == 3
    assert sleeps
    assert all(0 < s <= 4.0 for s in sleeps)
    assert sleeps[0] == 4.0


def test_classify_http_status() -> None:
    assert classify_http_status(429) == RetryKind.RATE_LIMITED
    assert classify_http_status(500) == RetryKind.RETRYABLE
    assert classify_http_status(401) == RetryKind.NON_RETRYABLE
    assert classify_http_status(403) == RetryKind.NON_RETRYABLE
    assert classify_http_status(402) == RetryKind.BILLING


def test_hung_worker_uses_daemon_thread() -> None:
    def hang() -> str:
        time.sleep(0.25)
        return "late"

    t0 = time.monotonic()
    result = run_with_deadline(
        hang,
        policy=RetryPolicy(max_attempts=1, total_deadline_seconds=0.03),
        label="hang",
    )
    elapsed = time.monotonic() - t0
    assert result.ok is False
    assert elapsed < 0.15
    assert any(
        thread.name.startswith("never_block:hang") and thread.daemon
        for thread in threading.enumerate()
    )


def test_hung_worker_does_not_block_shutdown() -> None:
    """Daemon worker threads must not prevent the main thread from continuing."""

    def hang_forever() -> str:
        time.sleep(60.0)
        return "late"

    result = run_with_deadline(
        hang_forever,
        policy=RetryPolicy(max_attempts=1, total_deadline_seconds=0.02),
        label="shutdown_hang",
    )
    assert result.ok is False
    assert result.degraded is True


def test_worker_capacity_is_reserved_before_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent slow calls cannot all pass the capacity check at once."""
    release = threading.Event()
    entered = threading.Event()

    monkeypatch.setattr(never_block, "_MAX_HUNG_WORKERS", 1)
    monkeypatch.setattr(never_block, "_hung_workers", 0)

    first_errors: list[BaseException] = []

    def slow() -> None:
        entered.set()
        release.wait(1.0)

    def first_call() -> None:
        try:
            never_block._run_call_with_deadline(slow, Deadline.after_seconds(0.5), "first")
        except BaseException as exc:  # noqa: BLE001
            first_errors.append(exc)

    first = threading.Thread(target=first_call)
    first.start()
    assert entered.wait(0.5)

    called: list[bool] = []
    with pytest.raises(TimeoutError, match="too many provider workers"):
        never_block._run_call_with_deadline(
            lambda: called.append(True), Deadline.after_seconds(0.2), "second"
        )
    assert called == []

    release.set()
    first.join(timeout=1.0)
    assert not first.is_alive()
    assert first_errors == []
    assert never_block._hung_workers == 0
