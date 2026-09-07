"""request_json 429 handling: Retry-After must gate the actual retried request, not a
sleep that's immediately followed by raising on the same (still-429) response."""

from __future__ import annotations

import httpx
import pytest

from trading_system.providers.http_client import parse_retry_after, request_json
from trading_system.providers.never_block import Deadline, RetryPolicy


def _response(status: int, url: str, *, headers: dict | None = None, json: dict | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, json=json, request=httpx.Request("GET", url))


class _FakeClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.request_count = 0

    def request(self, method: str, url: str, *, headers=None, params=None) -> httpx.Response:
        self.request_count += 1
        return self._responses.pop(0)


def test_retry_after_gates_a_real_retry_not_a_wasted_sleep() -> None:
    client = _FakeClient(
        [
            _response(429, "https://x.test/a", headers={"Retry-After": "2"}),
            _response(200, "https://x.test/a", json={"ok": True}),
        ]
    )
    sleeps: list[float] = []
    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.01, max_delay_seconds=5.0, jitter=0.0, total_deadline_seconds=10.0)

    result = request_json(
        client,  # type: ignore[arg-type]
        "GET",
        "https://x.test/a",
        policy=policy,
        deadline=Deadline.after_seconds(10.0),
        sleep=lambda s: sleeps.append(s),
    )

    assert result.ok is True
    assert result.value == {"ok": True}
    assert client.request_count == 2, "the second (retried) request must actually happen"
    # Exactly one sleep, honoring the server's Retry-After -- not a wasted local sleep
    # plus a second, uncoordinated backoff sleep for the same failed attempt.
    assert sleeps == pytest.approx([2.0])


def test_retry_after_still_bounded_by_max_delay_seconds() -> None:
    client = _FakeClient(
        [
            _response(429, "https://x.test/a", headers={"Retry-After": "9999"}),
            _response(200, "https://x.test/a", json={"ok": True}),
        ]
    )
    sleeps: list[float] = []
    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.01, max_delay_seconds=3.0, jitter=0.0, total_deadline_seconds=30.0)

    result = request_json(
        client,  # type: ignore[arg-type]
        "GET",
        "https://x.test/a",
        policy=policy,
        deadline=Deadline.after_seconds(30.0),
        sleep=lambda s: sleeps.append(s),
    )

    assert result.ok is True
    assert sleeps == pytest.approx([3.0])


def test_parse_retry_after_reads_seconds_and_http_date() -> None:
    assert parse_retry_after({"Retry-After": "5"}) == pytest.approx(5.0)
    assert parse_retry_after({}) is None
    assert parse_retry_after({"Retry-After": "not-a-date"}) is None
