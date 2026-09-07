"""The optional live-data loop releases resources between bounded refreshes."""

from __future__ import annotations

from pathlib import Path

import pytest

from trading_system.config import Settings
from trading_system.runtime_loop import LiveRuntime
from trading_system.storage import Store


class _FakeResolved:
    def with_btc(self, settings: Settings) -> tuple[str, ...]:
        return settings.smoke_universe


class _FakeProvider:
    def __init__(self, closes: list[int]) -> None:
        self.closes = closes

    def close(self) -> None:
        self.closes.append(1)


class _FakeSvc:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.startup_error: Exception | None = None

    def startup_backfill(self) -> None:
        self.calls.append("startup_backfill")
        if self.startup_error is not None:
            raise self.startup_error

    def refresh_live_watchlist(self) -> None:
        self.calls.append("refresh_live_watchlist")

    def refresh_btc(self) -> None:
        self.calls.append("refresh_btc")

    def build_data_health_snapshot(self) -> None:
        self.calls.append("build_data_health_snapshot")

    def broad_universe_scan(self) -> None:
        self.calls.append("broad_universe_scan")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        smoke_universe=("AAPL", "BTC/USD"),
        duckdb_path=tmp_path / "rt.duckdb",
    )


def _patch_common(
    monkeypatch: pytest.MonkeyPatch, svc: _FakeSvc, provider_closes: list[int]
) -> None:
    monkeypatch.setattr(
        "trading_system.runtime_loop.prepare_scan_universe",
        lambda conn, settings: _FakeResolved(),
    )
    monkeypatch.setattr(
        "trading_system.runtime_loop.make_market_provider",
        lambda settings: _FakeProvider(provider_closes),
    )
    monkeypatch.setattr("trading_system.runtime_loop.MarketDataService", lambda *a, **k: svc)


def _stop_on_wait(runtime: LiveRuntime):  # type: ignore[no-untyped-def]
    def fake_wait(timeout: float | None = None) -> bool:
        runtime._stop.set()
        return True

    return fake_wait


def test_initial_backfill_failure_is_recorded_and_loop_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = _FakeSvc()
    svc.startup_error = RuntimeError("boom")
    closes: list[int] = []
    _patch_common(monkeypatch, svc, closes)

    settings = _settings(tmp_path)
    runtime = LiveRuntime(settings=settings, project_root=tmp_path)
    monkeypatch.setattr(runtime._stop, "wait", _stop_on_wait(runtime))

    runtime._loop()

    assert runtime.last_error == "boom"
    assert closes == [1]
    store = Store(settings.duckdb_path, read_only=True)
    store.open(acquire_writer=False)
    try:
        row = store.conn.execute(
            "SELECT COUNT(*) FROM runtime_events WHERE status = 'degraded' AND message LIKE '%boom%'"
        ).fetchone()
        assert row is not None and int(row[0]) >= 1
    finally:
        store.close()


def test_cadence_timers_do_not_immediately_repeat_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = _FakeSvc()
    closes: list[int] = []
    _patch_common(monkeypatch, svc, closes)
    runtime = LiveRuntime(settings=_settings(tmp_path), project_root=tmp_path)
    monkeypatch.setattr(runtime._stop, "wait", _stop_on_wait(runtime))

    runtime._loop()

    assert svc.calls == ["startup_backfill"]
    assert closes == [1]


def test_writer_lease_is_released_while_loop_is_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = _FakeSvc()
    closes: list[int] = []
    _patch_common(monkeypatch, svc, closes)
    settings = _settings(tmp_path)
    runtime = LiveRuntime(settings=settings, project_root=tmp_path)
    acquired: list[bool] = []

    def verify_idle_lease(timeout: float | None = None) -> bool:
        second_writer = Store(settings.duckdb_path)
        second_writer.open(acquire_writer=True)
        try:
            acquired.append(True)
        finally:
            second_writer.close()
        runtime._stop.set()
        return True

    monkeypatch.setattr(runtime._stop, "wait", verify_idle_lease)
    runtime._loop()

    assert acquired == [True]
    assert closes == [1]


def test_due_refreshes_share_one_scoped_store_and_close_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = _FakeSvc()
    closes: list[int] = []
    _patch_common(monkeypatch, svc, closes)
    clock = iter([100.0, 10_000.0])
    monkeypatch.setattr("trading_system.runtime_loop.time.monotonic", lambda: next(clock))
    runtime = LiveRuntime(settings=_settings(tmp_path), project_root=tmp_path)
    monkeypatch.setattr(runtime._stop, "wait", _stop_on_wait(runtime))

    runtime._loop()

    assert svc.calls == [
        "startup_backfill",
        "refresh_live_watchlist",
        "refresh_btc",
        "build_data_health_snapshot",
        "broad_universe_scan",
    ]
    assert closes == [1, 1]


def test_store_close_failure_does_not_mask_the_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """store.close() raising in _service_scope's finally must be logged and swallowed,
    not replace whatever real exception was already propagating out of the try body --
    this scope now runs on every due cycle (not just at shutdown), so it's reachable
    far more often than before."""
    import logging

    svc = _FakeSvc()
    closes: list[int] = []
    _patch_common(monkeypatch, svc, closes)
    clock = iter([100.0, 10_000.0])  # forces live_due=True on the loop's first tick
    monkeypatch.setattr("trading_system.runtime_loop.time.monotonic", lambda: next(clock))

    def boom_refresh() -> None:
        raise RuntimeError("original failure")

    svc.refresh_live_watchlist = boom_refresh  # type: ignore[method-assign]

    original_close = Store.close

    def boom_close(self: Store) -> None:
        original_close(self)  # real cleanup still happens (lease released, conn closed)
        raise RuntimeError("close failure")

    monkeypatch.setattr(Store, "close", boom_close)

    runtime = LiveRuntime(settings=_settings(tmp_path), project_root=tmp_path)
    monkeypatch.setattr(runtime._stop, "wait", _stop_on_wait(runtime))

    with caplog.at_level(logging.WARNING, logger="trading_system.runtime_loop"):
        runtime._loop()

    assert runtime.last_error == "original failure"
    assert any("store.close() failed" in r.message for r in caplog.records)
