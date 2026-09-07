"""Never-block deadline / retry primitives for external I/O."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

T = TypeVar("T")

# Bound all in-flight deadline workers.  Reserving only after a timeout has a
# race: many concurrent calls can all observe zero hung workers and start before
# any of them times out.
_MAX_HUNG_WORKERS = 16
_hung_workers = 0
_hung_lock = threading.Lock()


class RetryKind(StrEnum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    RATE_LIMITED = "rate_limited"
    BILLING = "billing"


class ProviderHttpError(Exception):
    """HTTP failure with retry classification. Message must not include secrets."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        kind: RetryKind = RetryKind.RETRYABLE,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class Deadline:
    """Absolute wall-clock deadline (monotonic)."""

    deadline_mono: float

    @classmethod
    def after_seconds(cls, seconds: float) -> Deadline:
        if seconds <= 0:
            raise ValueError("deadline seconds must be positive")
        return cls(deadline_mono=time.monotonic() + seconds)

    def remaining(self) -> float:
        return max(0.0, self.deadline_mono - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def raise_if_expired(self, label: str = "operation") -> None:
        if self.expired():
            raise TimeoutError(f"{label} deadline expired")


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 8.0
    jitter: float = 0.2
    total_deadline_seconds: float | None = 30.0

    def backoff(self, attempt: int) -> float:
        delay = min(self.max_delay_seconds, self.base_delay_seconds * (2**attempt))
        if self.jitter > 0:
            delay *= 1.0 + random.uniform(-self.jitter, self.jitter)
        return max(0.0, delay)


@dataclass
class NeverBlockResult:
    ok: bool
    value: object | None = None
    error: str | None = None
    attempts: int = 0
    degraded: bool = False
    kind: RetryKind | None = None
    elapsed_seconds: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


def classify_http_status(status_code: int) -> RetryKind:
    if status_code == 429:
        return RetryKind.RATE_LIMITED
    if status_code in {401, 402, 403}:
        return RetryKind.BILLING if status_code == 402 else RetryKind.NON_RETRYABLE
    if 500 <= status_code <= 599:
        return RetryKind.RETRYABLE
    if status_code >= 400:
        return RetryKind.NON_RETRYABLE
    return RetryKind.RETRYABLE


def _run_call_with_deadline(
    fn: Callable[[], T],
    deadline: Deadline | None,
    label: str,
) -> T:
    """Run fn, interrupting when the shared deadline expires.

    Hung workers run as daemon threads so they never block process shutdown.
    """
    global _hung_workers
    if deadline is None:
        return fn()
    remaining = deadline.remaining()
    if remaining <= 0:
        raise TimeoutError(f"{label} deadline expired")

    with _hung_lock:
        if _hung_workers >= _MAX_HUNG_WORKERS:
            raise TimeoutError(f"{label} too many provider workers; refusing new call")
        _hung_workers += 1

    result: list[T] = []
    error: list[BaseException] = []

    def worker() -> None:
        global _hung_workers
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001 — bounded external boundary
            error.append(exc)
        finally:
            with _hung_lock:
                _hung_workers = max(0, _hung_workers - 1)

    thread = threading.Thread(target=worker, daemon=True, name=f"never_block:{label}")
    try:
        thread.start()
    except BaseException:
        with _hung_lock:
            _hung_workers = max(0, _hung_workers - 1)
        raise
    thread.join(timeout=remaining)
    if thread.is_alive():
        raise TimeoutError(f"{label} deadline expired")
    if error:
        raise error[0]
    if not result:
        raise RuntimeError(f"{label} finished without result")
    return result[0]


def run_with_deadline(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    deadline: Deadline | None = None,
    classify: Callable[[BaseException], RetryKind] | None = None,
    sleep: Callable[[float], None] | None = None,
    label: str = "external_call",
) -> NeverBlockResult:
    """Execute fn with bounded retries and an optional total deadline.

    Never loops forever. Surfaces structured failure instead of hanging.
    """
    policy = policy or RetryPolicy()
    classify = classify or (lambda _exc: RetryKind.RETRYABLE)
    sleeper = sleep or time.sleep
    if deadline is None and policy.total_deadline_seconds:
        deadline = Deadline.after_seconds(policy.total_deadline_seconds)
    started = time.monotonic()
    last_error: str | None = None
    last_kind: RetryKind | None = None
    attempts_used = 0

    for attempt in range(policy.max_attempts):
        if deadline is not None and deadline.expired():
            break
        attempts_used = attempt + 1
        try:
            value = _run_call_with_deadline(fn, deadline, label)
            if deadline is not None and deadline.expired():
                return NeverBlockResult(
                    ok=False,
                    error=f"{label} deadline expired",
                    attempts=attempts_used,
                    degraded=True,
                    kind=RetryKind.NON_RETRYABLE,
                    elapsed_seconds=time.monotonic() - started,
                )
            return NeverBlockResult(
                ok=True,
                value=value,
                attempts=attempts_used,
                elapsed_seconds=time.monotonic() - started,
            )
        except BaseException as exc:  # noqa: BLE001 — bounded external boundary
            last_kind = exc.kind if isinstance(exc, ProviderHttpError) else classify(exc)
            last_error = f"{type(exc).__name__}: {exc}"
            if last_kind in {RetryKind.NON_RETRYABLE, RetryKind.BILLING}:
                break
            if attempt + 1 >= policy.max_attempts:
                break
            delay = policy.backoff(attempt)
            retry_after = getattr(exc, "retry_after_seconds", None)
            if retry_after is not None:
                try:
                    delay = max(delay, float(retry_after))
                except (TypeError, ValueError):
                    pass
            delay = min(delay, policy.max_delay_seconds)
            if deadline is not None:
                remaining = deadline.remaining()
                if remaining <= 0:
                    break
                delay = min(delay, remaining)
            if delay > 0:
                sleeper(delay)

    return NeverBlockResult(
        ok=False,
        error=last_error
        or (f"{label} deadline expired" if deadline is not None and deadline.expired() else f"{label} failed"),
        attempts=max(1, attempts_used),
        degraded=True,
        kind=last_kind,
        elapsed_seconds=time.monotonic() - started,
    )
