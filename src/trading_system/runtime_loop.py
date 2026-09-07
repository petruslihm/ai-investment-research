"""Optional background market-data refresh loop. Never holds the writer lease idle."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from trading_system.config import Settings, get_settings
from trading_system.market.service import MarketDataService
from trading_system.provider_factory import make_market_provider
from trading_system.runtime import RuntimeEventKind, RuntimeEventStatus
from trading_system.storage import Store
from trading_system.universe import prepare_scan_universe

_log = logging.getLogger("trading_system.runtime_loop")


class LiveRuntime:
    def __init__(self, settings: Settings | None = None, *, project_root: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.project_root = project_root or Path.cwd()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="investassist-runtime", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        """Signal shutdown and wait briefly; hung provider I/O runs in daemon threads."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout_seconds)

    def _loop(self) -> None:
        # Startup backfill is useful in live mode, but a failed provider must not
        # terminate the background thread.  The store/provider are scoped to this
        # operation so the UI can acquire the writer lease between refreshes.
        try:
            with self._service_scope() as (_store, svc):
                svc.startup_backfill()
        except Exception as exc:  # noqa: BLE001
            self._remember_error(exc)

        now0 = time.monotonic()
        last_live = now0
        last_broad = now0
        while not self._stop.is_set():
            now = time.monotonic()
            live_due = now - last_live >= self.settings.live_refresh_seconds
            broad_due = now - last_broad >= self.settings.broad_scan_seconds
            if live_due:
                last_live = now
            if broad_due:
                last_broad = now
            if live_due or broad_due:
                try:
                    with self._service_scope() as (_store, svc):
                        if live_due:
                            svc.refresh_live_watchlist()
                            svc.refresh_btc()
                            svc.build_data_health_snapshot()
                        if broad_due:
                            svc.broad_universe_scan()
                except Exception as exc:  # noqa: BLE001
                    # Advance the cadence even after a failure. Retrying every five
                    # seconds would turn a provider outage into an API hammer loop.
                    self._remember_error(exc)
            self._stop.wait(5.0)

    @contextmanager
    def _service_scope(self) -> Iterator[tuple[Store, MarketDataService]]:
        settings = self.settings
        db = settings.duckdb_path if settings.duckdb_path.is_absolute() else self.project_root / settings.duckdb_path
        store = Store(db)
        provider: object | None = None
        try:
            store.open(acquire_writer=True)
            resolved = prepare_scan_universe(store.conn, settings)
            scoped_settings = settings.model_copy(
                update={"smoke_universe": resolved.with_btc(settings)}
            )
            provider = make_market_provider(scoped_settings)
            pname = provider.provider_name() if hasattr(provider, "provider_name") else "fixture"
            service = MarketDataService(
                store.conn,
                scoped_settings,
                provider,  # type: ignore[arg-type]
                provider_name=pname,
            )
            yield store, service
        finally:
            close_provider = getattr(provider, "close", None)
            if callable(close_provider):
                try:
                    close_provider()
                except Exception:  # noqa: BLE001 — cleanup must not hide the task result
                    pass
            try:
                store.close()
            except Exception:  # noqa: BLE001 — a close() failure here must not mask
                # whatever real exception is already propagating out of the `try`
                # body above (this scope now runs every due cycle, not just at
                # process shutdown, so this is reachable far more often now).
                _log.warning("LiveRuntime._service_scope: store.close() failed", exc_info=True)

    def _remember_error(self, exc: Exception) -> None:
        self.last_error = str(exc)
        settings = self.settings
        db = settings.duckdb_path if settings.duckdb_path.is_absolute() else self.project_root / settings.duckdb_path
        store = Store(db)
        try:
            store.open(acquire_writer=True)
            self._record_error_event(store, exc)
        except Exception:  # noqa: BLE001 — last_error still exposes the failure in memory
            pass
        finally:
            # _remember_error is called from _loop's own except blocks specifically so
            # a failure never kills the background thread -- store.close() raising here
            # unguarded would defeat that by escaping this method entirely.
            try:
                store.close()
            except Exception:  # noqa: BLE001
                _log.warning("LiveRuntime._remember_error: store.close() failed", exc_info=True)

    @staticmethod
    def _record_error_event(store: Store, exc: Exception) -> None:
        store.conn.execute(
            """
            INSERT INTO runtime_events
            (event_id, kind, status, message, progress_done, progress_total, duration_ms, created_at, metadata_json)
            VALUES (?, ?, ?, ?, NULL, NULL, NULL, CURRENT_TIMESTAMP, '{}')
            """,
            [
                f"rt_err_{uuid4().hex}",
                RuntimeEventKind.PROVIDER_RECOVERY.value,
                RuntimeEventStatus.DEGRADED.value,
                f"loop continued after {exc}",
            ],
        )
