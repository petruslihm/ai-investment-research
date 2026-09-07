"""Market data orchestration: backfill, broad scan, live watchlist, BTC refresh."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

import duckdb

from trading_system.config import Settings
from trading_system.data_health import CoverageSummary, DataHealthSnapshot, HealthLevel
from trading_system.ids import AssetClass, InstrumentId, RequestSetId, new_request_set_id
from trading_system.market.registry import (
    bootstrap_smoke_universe,
    equity_instrument_ids,
    symbol_for_instrument,
)
from trading_system.market.repository import (
    coverage_summary,
    insert_live_observation,
    latest_canonical_event_ts,
    latest_lkg_quote,
    replace_live_watchlist,
    upsert_btc_daily_bars,
    upsert_coverage_rows,
    upsert_equity_daily_bars,
    upsert_provider_state,
)
from trading_system.market.quote_validation import validate_quote_for_decision
from trading_system.market.status import DataStatus, is_canonical_decision
from trading_system.market.watchlist import build_live_watchlist
from trading_system.providers.batch import DailyBarsBatch, QuotesBatch
from trading_system.providers.interfaces import CoverageRow, CoverageStatus
from trading_system.providers.never_block import NeverBlockResult, RetryPolicy
from trading_system.runtime import RuntimeEvent, RuntimeEventKind, RuntimeEventStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _years_ago(years: int, end: date) -> date:
    return end - timedelta(days=365 * years)


# If last stored bar is newer than this, only fetch a short overlapping tail.
_INCREMENTAL_MAX_GAP_DAYS = 14
_INCREMENTAL_OVERLAP_DAYS = 10


class MarketDataService:
    """Provider-abstracted ingestion with never-block degradation."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        settings: Settings,
        provider: Any,
        *,
        provider_name: str,
    ) -> None:
        self.conn = conn
        self.settings = settings
        self.provider = provider
        self.provider_name = provider_name
        self.symbol_map = bootstrap_smoke_universe(
            conn, settings, provider=provider_name
        )
        if hasattr(provider, "set_symbol_resolver"):
            provider.set_symbol_resolver(self._resolve_symbol)

    def _resolve_symbol(self, instrument_id: InstrumentId) -> str | None:
        return symbol_for_instrument(
            self.conn,
            instrument_id,
            provider=self.provider_name,
            as_of=date.today(),
        )

    def _record_request_set(self, request_set_id: RequestSetId, notes: str) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO request_sets (request_set_id, created_at, notes)
            VALUES (?, ?, ?)
            """,
            [request_set_id, _utcnow(), notes],
        )

    def _emit_runtime(
        self,
        kind: RuntimeEventKind,
        status: RuntimeEventStatus,
        message: str,
        *,
        progress_done: int | None = None,
        progress_total: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        event = RuntimeEvent(
            kind=kind,
            status=status,
            message=message,
            progress_done=progress_done,
            progress_total=progress_total,
            metadata=metadata or {},
        )
        self.conn.execute(
            """
            INSERT INTO runtime_events
            (event_id, kind, status, message, progress_done, progress_total,
             duration_ms, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event.event_id,
                event.kind.value,
                event.status.value,
                event.message,
                event.progress_done,
                event.progress_total,
                event.duration_ms,
                event.created_at,
                json.dumps(event.metadata),
            ],
        )

    def _last_equity_dates(self) -> dict[str, date]:
        rows = self.conn.execute(
            "SELECT instrument_id, max(session_date) FROM equity_daily_bars GROUP BY instrument_id"
        ).fetchall()
        return {str(iid): d for iid, d in rows if d is not None}

    def _last_btc_date(self) -> date | None:
        row = self.conn.execute("SELECT max(session_date) FROM btc_daily_bars").fetchone()
        return row[0] if row and row[0] else None

    def _ingest_equities(
        self,
        equities: list[InstrumentId],
        *,
        end: date,
        request_set_id: RequestSetId,
    ) -> tuple[NeverBlockResult, int, int]:
        """Full history for new/stale names; short overlapping tail when yesterday's bars exist."""
        last = self._last_equity_dates()
        full_start = _years_ago(self.settings.history_years, end)
        fresh: list[InstrumentId] = []
        stale: list[InstrumentId] = []
        for iid in equities:
            prev = last.get(str(iid))
            if prev is not None and (end - prev).days <= _INCREMENTAL_MAX_GAP_DAYS:
                fresh.append(iid)
            else:
                stale.append(iid)
        policy_full = RetryPolicy(max_attempts=3, total_deadline_seconds=600.0)
        policy_inc = RetryPolicy(max_attempts=3, total_deadline_seconds=300.0)
        merged = NeverBlockResult(ok=True, value=DailyBarsBatch())
        if stale:
            result = self.provider.fetch_daily_bars(
                stale, full_start, end, request_set_id=request_set_id, policy=policy_full
            )
            n_bars = len(getattr(result.value, "bars", []) or [])
            if (not result.ok) or n_bars == 0:
                self._record_failure_coverage(stale, request_set_id, end)
                result = NeverBlockResult(
                    ok=False, value=result.value, error=result.error or "empty equity bars", degraded=True
                )
            else:
                self._persist_daily_result(result, asset_class=AssetClass.US_EQUITY)
            merged = result
        if fresh:
            inc_start = min(last[str(i)] for i in fresh) - timedelta(days=_INCREMENTAL_OVERLAP_DAYS)
            result = self.provider.fetch_daily_bars(
                fresh, inc_start, end, request_set_id=request_set_id, policy=policy_inc
            )
            n_bars = len(getattr(result.value, "bars", []) or [])
            if result.ok and n_bars > 0:
                self._persist_daily_result(result, asset_class=AssetClass.US_EQUITY)
            elif not stale:
                self._record_failure_coverage(fresh, request_set_id, end)
                result = NeverBlockResult(
                    ok=False, value=result.value, error=result.error or "empty equity bars", degraded=True
                )
            if stale:
                merged = NeverBlockResult(
                    ok=bool(merged.ok and result.ok),
                    value=result.value if result.value else merged.value,
                    error=result.error or merged.error,
                    degraded=bool(merged.degraded or result.degraded),
                )
            else:
                merged = result
        return merged, len(fresh), len(stale)

    def startup_backfill(self, *, end: date | None = None) -> RequestSetId:
        """Fetch history for the scan universe + SPY + BTC.

        Names that already have recent bars only refresh a short tail. New or stale
        names still get the full history_years window.
        """
        end = end or date.today()
        start = _years_ago(self.settings.history_years, end)
        request_set_id = new_request_set_id()
        self._record_request_set(request_set_id, "startup_backfill")
        equities = equity_instrument_ids(self.symbol_map, self.settings)
        last = self._last_equity_dates()
        n_fresh_preview = sum(
            1
            for iid in equities
            if last.get(str(iid)) is not None
            and (end - last[str(iid)]).days <= _INCREMENTAL_MAX_GAP_DAYS
        )
        n_stale_preview = len(equities) - n_fresh_preview
        if n_stale_preview == 0 and n_fresh_preview:
            started_msg = f"startup backfill incremental {n_fresh_preview} names (already stored)"
        else:
            started_msg = (
                f"startup backfill {start}..{end} incremental={n_fresh_preview} full={n_stale_preview}"
            )
        self._emit_runtime(
            RuntimeEventKind.FETCH_US_DAILY,
            RuntimeEventStatus.STARTED,
            started_msg,
            metadata={"request_set_id": request_set_id},
        )

        result, n_fresh, n_stale = self._ingest_equities(equities, end=end, request_set_id=request_set_id)

        btc_last = self._last_btc_date()
        btc_start = start
        if btc_last is not None and (end - btc_last).days <= _INCREMENTAL_MAX_GAP_DAYS:
            btc_start = btc_last - timedelta(days=_INCREMENTAL_OVERLAP_DAYS)
        btc_policy = RetryPolicy(max_attempts=3, total_deadline_seconds=120.0)
        btc_result = self.provider.fetch_btc_bars(
            btc_start, end, request_set_id=request_set_id, policy=btc_policy
        )
        if not btc_result.ok:
            self._record_failure_coverage(
                [self.symbol_map[self.settings.btc_symbol]],
                request_set_id,
                end,
            )
        self._persist_btc_daily_result(btc_result)

        status = (
            RuntimeEventStatus.SUCCEEDED
            if result.ok or btc_result.ok
            else RuntimeEventStatus.DEGRADED
        )
        extra = f" incremental={n_fresh} full={n_stale}"
        if not result.ok and result.error:
            extra += f" equity={result.error[:160]}"
        if not btc_result.ok and btc_result.error:
            extra += f" btc={btc_result.error[:160]}"
        self._emit_runtime(
            RuntimeEventKind.FETCH_US_DAILY,
            status,
            f"startup backfill complete{extra}",
            progress_done=len(equities),
            progress_total=len(equities),
            metadata={"request_set_id": request_set_id, "incremental": n_fresh, "full": n_stale},
        )
        upsert_provider_state(
            self.conn,
            provider_key=self.provider_name,
            role="primary",
            status=DataStatus.VALID if result.ok else DataStatus.DEGRADED,
            last_success_at=_utcnow() if result.ok else None,
            last_error=result.error,
        )
        return request_set_id

    def broad_universe_scan(self) -> RequestSetId:
        """Refresh daily bars. Uses a short tail when recent history is already stored."""
        request_set_id = new_request_set_id()
        self._record_request_set(request_set_id, "broad_universe_scan")
        end = date.today()
        equities = equity_instrument_ids(self.symbol_map, self.settings)
        result, n_fresh, n_stale = self._ingest_equities(equities, end=end, request_set_id=request_set_id)
        batch = result.value
        failed = list(getattr(batch, "failed_instruments", []) or []) if batch else []
        complete_success = bool(result.ok and not failed)
        self._emit_runtime(
            RuntimeEventKind.FETCH_US_DAILY,
            RuntimeEventStatus.SUCCEEDED if complete_success else RuntimeEventStatus.DEGRADED,
            f"broad scan {len(equities)} symbols incremental={n_fresh} full={n_stale}",
            progress_done=len(equities) - len(failed),
            progress_total=len(equities),
        )
        return request_set_id

    def refresh_live_watchlist(
        self,
        *,
        candidate_scores: dict[InstrumentId, float] | None = None,
    ) -> RequestSetId:
        """Fast refresh for watchlist symbols; one failed symbol must not freeze others."""
        request_set_id = new_request_set_id()
        self._record_request_set(request_set_id, "live_watchlist")
        watchlist = build_live_watchlist(
            self.conn,
            self.settings,
            self.symbol_map,
            provider=self.provider_name,
            candidate_scores=candidate_scores,
        )
        replace_live_watchlist(self.conn, watchlist)
        instrument_ids = [iid for iid, _, _ in watchlist]

        result = self.provider.fetch_quotes(
            instrument_ids,
            policy=RetryPolicy(total_deadline_seconds=15.0),
        )
        self._persist_quotes_result(
            result,
            asset_class=AssetClass.US_EQUITY,
            request_set_id=request_set_id,
            instrument_ids=instrument_ids,
        )
        self._emit_runtime(
            RuntimeEventKind.INFERENCE,
            RuntimeEventStatus.SUCCEEDED if result.ok else RuntimeEventStatus.DEGRADED,
            f"live watchlist refresh ({len(instrument_ids)} symbols)",
            progress_done=len(instrument_ids),
            progress_total=len(instrument_ids),
        )
        return request_set_id

    def refresh_btc(self) -> None:
        """BTC frequent refresh on 24/7 calendar (not equity sessions)."""
        request_set_id = new_request_set_id()
        self._record_request_set(request_set_id, "btc_refresh")
        btc_id = self.symbol_map[self.settings.btc_symbol]
        result = self.provider.fetch_btc_quote(
            policy=RetryPolicy(total_deadline_seconds=10.0),
        )
        quote = result.value if result.ok else None
        status = DataStatus.VALID
        lkg = False
        if quote is None or not result.ok:
            price, lkg_status, trusted_at, lkg = latest_lkg_quote(self.conn, btc_id)
            if price is not None:
                status = DataStatus.STALE if lkg_status == DataStatus.VALID else lkg_status
                from trading_system.providers.interfaces import LatestQuote

                lkg_quote = LatestQuote(
                    instrument_id=btc_id,
                    price=price,
                    event_ts=trusted_at,
                    receive_ts=trusted_at or _utcnow(),
                    provider=self.provider_name,
                    degraded=True,
                )
                insert_live_observation(
                    self.conn,
                    instrument_id=btc_id,
                    asset_class=AssetClass.BTC.value,
                    quote=lkg_quote,
                    data_status=status,
                    request_set_id=request_set_id,
                    lkg_fallback=True,
                    metadata={"note": "last-known-good; latest refresh failed"},
                )
            else:
                status = DataStatus.UNAVAILABLE
                insert_live_observation(
                    self.conn,
                    instrument_id=btc_id,
                    asset_class=AssetClass.BTC.value,
                    quote=None,
                    data_status=status,
                    request_set_id=request_set_id,
                )
        else:
            from trading_system.providers.interfaces import LatestQuote

            q: LatestQuote = quote  # type: ignore[assignment]
            status = DataStatus.VALID
            if q.event_ts is None:
                status = DataStatus.DEGRADED
            elif q.degraded or result.degraded:
                status = DataStatus.DEGRADED
            prior_ts = latest_canonical_event_ts(self.conn, btc_id)
            valid, reason = validate_quote_for_decision(
                q,
                prior_event_ts=prior_ts,
                receive_at=_utcnow(),
            )
            canonical = valid and is_canonical_decision(
                status, has_event_ts=q.event_ts is not None
            )
            if not valid:
                status = DataStatus.DEGRADED
                canonical = False
            insert_live_observation(
                self.conn,
                instrument_id=btc_id,
                asset_class=AssetClass.BTC.value,
                quote=q,
                data_status=status,
                request_set_id=request_set_id,
                is_canonical=canonical,
                lkg_fallback=lkg,
                metadata={"validation_reason": reason},
            )

        self._emit_runtime(
            RuntimeEventKind.FETCH_BTC,
            RuntimeEventStatus.SUCCEEDED if status in {DataStatus.VALID, DataStatus.STALE} else RuntimeEventStatus.DEGRADED,
            f"BTC refresh status={status.value}",
        )
        upsert_provider_state(
            self.conn,
            provider_key=f"{self.provider_name}:btc",
            role="btc",
            status=status,
            last_success_at=_utcnow() if status == DataStatus.VALID else None,
            last_error=result.error,
        )

    def build_data_health_snapshot(self, request_set_id: RequestSetId | None = None) -> DataHealthSnapshot:
        summary = coverage_summary(self.conn, request_set_id=request_set_id)
        cov = CoverageSummary(
            request_set_id=request_set_id,
            requested=summary.get("requested", 0),
            succeeded=summary.get("succeeded", 0),
            missing=summary.get("missing", 0),
            error=summary.get("error", 0),
            quarantined=summary.get("quarantined", 0),
        )
        equity_level = HealthLevel.UNKNOWN
        if cov.requested == 0:
            equity_level = HealthLevel.CRITICAL
        elif cov.succeeded == 0:
            equity_level = HealthLevel.CRITICAL
        elif cov.error > 0 or cov.missing > 0:
            equity_level = HealthLevel.DEGRADED
        else:
            equity_level = HealthLevel.OK

        btc_row = self.conn.execute(
            """
            SELECT data_status FROM live_observations
            WHERE asset_class = ?
            ORDER BY receive_ts DESC LIMIT 1
            """,
            [AssetClass.BTC.value],
        ).fetchone()
        btc_level = HealthLevel.UNKNOWN
        if btc_row:
            st = DataStatus(btc_row[0])
            btc_level = (
                HealthLevel.OK
                if st == DataStatus.VALID
                else HealthLevel.DEGRADED
                if st in {DataStatus.STALE, DataStatus.DEGRADED, DataStatus.RECOVERING}
                else HealthLevel.CRITICAL
            )

        overall = _worst_health_level(equity_level, btc_level)
        snap = DataHealthSnapshot(
            overall=overall,
            equity_feed=equity_level,
            btc_feed=btc_level,
            coverage=cov,
            last_tick_at=_utcnow(),
        )
        self.conn.execute(
            """
            INSERT INTO data_health_snapshots
            (snapshot_id, overall, equity_feed, btc_feed, coverage_json, last_tick_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snap.snapshot_id,
                snap.overall.value,
                snap.equity_feed.value,
                snap.btc_feed.value,
                json.dumps(snap.coverage.model_dump()),
                snap.last_tick_at,
                snap.created_at,
            ],
        )
        return snap

    def display_quote(self, instrument_id: InstrumentId) -> dict[str, object]:
        """LKG display semantics — never show '-' when a prior value exists."""
        price, status, receive_ts, lkg = latest_lkg_quote(self.conn, instrument_id)
        if price is None:
            return {
                "price": None,
                "display": "UNAVAILABLE — no prior observation",
                "status": DataStatus.UNAVAILABLE.value,
            }
        age = (_utcnow() - receive_ts).total_seconds() if receive_ts else None
        if status != DataStatus.VALID or lkg:
            suffix = f"; last trusted value {int(age)}s ago" if age is not None else ""
            return {
                "price": price,
                "display": f"{price:.2f} — latest refresh failed{suffix}; recovering",
                "status": status.value,
                "lkg": lkg or status == DataStatus.STALE,
            }
        return {"price": price, "display": f"{price:.2f}", "status": status.value, "lkg": False}

    def _record_failure_coverage(
        self,
        instrument_ids: list[InstrumentId],
        request_set_id: RequestSetId,
        session: date,
    ) -> None:
        rows = [
            CoverageRow(
                provider=self.provider_name,
                adjustment="n/a",
                adjustment_revision="n/a",
                instrument_id=inst_id,
                session=session,
                request_set_id=request_set_id,
                status=CoverageStatus.ERROR,
            )
            for inst_id in instrument_ids
        ]
        upsert_coverage_rows(self.conn, rows)

    def _persist_daily_result(
        self,
        result: NeverBlockResult,
        *,
        asset_class: AssetClass,
    ) -> None:
        if not result.value:
            return
        batch: DailyBarsBatch = result.value  # type: ignore[assignment]
        upsert_equity_daily_bars(self.conn, batch.bars)
        upsert_coverage_rows(self.conn, batch.coverage)
        # Per-symbol failures do not abort others — already reflected in batch.failed_instruments
        for inst_id in batch.failed_instruments:
            upsert_provider_state(
                self.conn,
                provider_key=f"{self.provider_name}:{inst_id}",
                role="symbol",
                status=DataStatus.UNAVAILABLE,
                last_error=result.error,
            )

    def _persist_btc_daily_result(self, result: NeverBlockResult) -> None:
        if not result.value:
            return
        batch: DailyBarsBatch = result.value  # type: ignore[assignment]
        upsert_btc_daily_bars(self.conn, batch.bars)
        upsert_coverage_rows(self.conn, batch.coverage)

    def _persist_quotes_result(
        self,
        result: NeverBlockResult,
        *,
        asset_class: AssetClass,
        request_set_id: RequestSetId,
        instrument_ids: list[InstrumentId],
    ) -> None:
        quotes_by_id: dict[InstrumentId, Any] = {}
        if result.value:
            batch: QuotesBatch = result.value  # type: ignore[assignment]
            for q in batch.quotes:
                quotes_by_id[q.instrument_id] = q
            upsert_coverage_rows(self.conn, batch.coverage)

        for inst_id in instrument_ids:
            quote = quotes_by_id.get(inst_id)
            if quote is not None:
                status = DataStatus.VALID
                if quote.event_ts is None:
                    status = DataStatus.DEGRADED
                elif quote.degraded:
                    status = DataStatus.DEGRADED
                prior_ts = latest_canonical_event_ts(self.conn, inst_id)
                valid, reason = validate_quote_for_decision(
                    quote,
                    prior_event_ts=prior_ts,
                    receive_at=_utcnow(),
                )
                canonical = valid and is_canonical_decision(
                    status, has_event_ts=quote.event_ts is not None
                )
                if not valid:
                    status = DataStatus.DEGRADED
                    canonical = False
                insert_live_observation(
                    self.conn,
                    instrument_id=inst_id,
                    asset_class=asset_class.value,
                    quote=quote,
                    data_status=status,
                    request_set_id=request_set_id,
                    is_canonical=canonical,
                    metadata={
                        "used_fallback": result.metadata.get("used_fallback", False),
                        "validation_reason": reason,
                    },
                )
            else:
                price, lkg_status, trusted_at, lkg = latest_lkg_quote(self.conn, inst_id)
                if price is not None:
                    from trading_system.providers.interfaces import LatestQuote

                    lkg_quote = LatestQuote(
                        instrument_id=inst_id,
                        price=price,
                        event_ts=trusted_at,
                        receive_ts=trusted_at or _utcnow(),
                        provider=self.provider_name,
                        degraded=True,
                    )
                    insert_live_observation(
                        self.conn,
                        instrument_id=inst_id,
                        asset_class=asset_class.value,
                        quote=lkg_quote,
                        data_status=DataStatus.STALE,
                        request_set_id=request_set_id,
                        lkg_fallback=True,
                    )
                else:
                    insert_live_observation(
                        self.conn,
                        instrument_id=inst_id,
                        asset_class=asset_class.value,
                        quote=None,
                        data_status=DataStatus.MISSING,
                        request_set_id=request_set_id,
                    )


def _worst_health_level(*levels: HealthLevel) -> HealthLevel:
    order = {
        HealthLevel.UNKNOWN: 0,
        HealthLevel.OK: 1,
        HealthLevel.DEGRADED: 2,
        HealthLevel.NOT_CONFIGURED: 3,
        HealthLevel.UNAVAILABLE: 4,
        HealthLevel.CRITICAL: 5,
    }
    return max(levels, key=lambda lv: order.get(lv, 0))
