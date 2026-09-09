"""End-to-end V1 cycle: market → features → ML → quant → SEC/LLM judge → portfolio.

API keys are never required to keep the app running. Missing keys are
NOT_CONFIGURED; failed live calls are UNAVAILABLE. Neither is reported as a
successful stub extract or fake-bullish LLM judge.
"""

from __future__ import annotations

import hashlib
import copy
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from trading_system.alert_engine import (
    collect_cycle_notifications,
    console_notify,
    evaluate_alerts,
    format_scan_kakao_digest,
)
from trading_system.alerts import AlertKind, AlertRecord
from trading_system.allocation import FORMULA_VERSION, allocate, horizon_agreement, persist_allocation
from trading_system.btc_sleeve import BtcLiquidityState, BtcSleeveState
from trading_system.config import Settings, discover_project_root
from trading_system.daily_scan import mark_session_run, mark_weekly_train, week_friday, week_train_session
from trading_system.data_health import CoverageSummary, DataHealthSnapshot, HealthLevel
from trading_system.features import (
    FEATURE_NAMES,
    LSTM_LOOKBACK,
    assert_no_future_in_features,
    build_and_persist_features,
    features_cover_latest,
    load_matrix,
    load_sequences,
)
from trading_system.ids import (
    AssetClass,
    new_feature_snapshot_id,
    new_tick_id,
)
from trading_system.llm_client import (
    LEGACY_B_RAW_PROMPT_VERSION,
    LEGACY_B_RAW_SYSTEM,
    STATUS_AVAILABLE,
    STATUS_DEGRADED,
    STATUS_NOT_CONFIGURED,
    STATUS_QUOTA_EXCEEDED,
    STATUS_RATE_LIMITED,
    STATUS_UNAVAILABLE,
    openai_configured,
)
from trading_system.market.registry import bootstrap_smoke_universe, stable_instrument_id
from trading_system.market.calendar import equity_session_date
from trading_system.market.decision_data import DAILY_BASIS, content_id, daily_input_fingerprints, latest_price_snapshot
from trading_system.final_decision import CONTRACT_VERSION, apply_final_judgment, allocation_for_records, render_final_report
from trading_system.final_decision import prepare_pending
from trading_system.recommendations import RecommendationAction, RecommendationRecord, RecommendationSource, normalize_position
from trading_system.universe import ResolvedUniverse
from trading_system.market.service import MarketDataService
from trading_system.ml_engine import (
    OnlineHorizonModel,
    adapt_ensemble_from_matured,
    adapt_horizon_influence,
    apply_online_updates,
    blend,
    bundle_hash,
    default_weights,
    family_enum,
    load_ensemble_state,
    load_fitted_bundles,
    load_online_model,
    bundles_ready,
    next_ensemble_version,
    online_path,
    persist_decision_epoch,
    persist_ensemble_state,
    persist_prediction_rows,
    save_online_model,
    train_batch_models,
    _predict,
)
from trading_system.models import DecisionEpochManifest, ModelArtifactRef, ModelChangeJournalEvent, ModelChangeKind, ModelFamily
from trading_system.ml_engine import _persist_journal
from trading_system.notifiers import notify_optional_channels
from trading_system.outcomes import record_matured_outcomes
from trading_system.marked_units import stock_exposure_for_allocation
from trading_system.portfolio_gate import build_portfolio_context
from trading_system.portfolio_service import load_portfolio, mark_positions
from trading_system.provider_factory import make_market_provider
from trading_system.runtime import RuntimeEvent, RuntimeEventKind, RuntimeEventStatus
from trading_system.llm_budget import record_usage, usage_from_payload, would_exceed_budget
from trading_system.ids import canonicalize_instrument_id
from trading_system.llm_log import insert_llm_transcript, persist_exchange, ticker_from_instrument
from trading_system.openai_judge import (
    TURN_COMPLETION_TOKENS_ESTIMATE,
    TURN_PROMPT_TOKENS_ESTIMATE,
    clear_pending_package,
    legacy_b_raw_conversation,
    load_pending_package,
    resolve_judge_model,
    save_pending_package,
)
from trading_system.research_agent import research_ticker, select_final_judge_recs
from trading_system.sec_llm import (
    evidence_for_instrument,
    ingest_sec_extracts,
)
from trading_system.seed import seed_synthetic_history
from trading_system.storage import Store
from trading_system.storage.ticks import TickCommitPayload, commit_tick
from trading_system.technical_factors import (
    TechnicalFactors,
    compute_rs_percentiles,
    compute_technical_factors,
    fetch_bars_frames,
    fetch_spy_close,
)
from trading_system.universe import BENCHMARK_SYMBOLS


def _emit(conn, kind: RuntimeEventKind, status: RuntimeEventStatus, message: str) -> None:
    ev = RuntimeEvent(kind=kind, status=status, message=message)
    conn.execute(
        """
        INSERT INTO runtime_events
        (event_id, kind, status, message, progress_done, progress_total, duration_ms, created_at, metadata_json)
        VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
        """,
        [ev.event_id, kind.value, status.value, message, ev.created_at, "{}"],
    )


def llm_progress_message(stage: str, index: int, total: int, instrument_id: object) -> str:
    """Scan-status line: 'Gemini 리서치 3/12 · PLTR'."""
    ticker = ticker_from_instrument(instrument_id)
    return f"{stage} {index}/{total} · {ticker}"


def _latest_as_of(conn) -> date | None:
    row = conn.execute("SELECT max(session_date) FROM equity_daily_bars WHERE finality='final'").fetchone()
    return row[0] if row and row[0] else None


def _latest_btc_as_of(conn) -> date | None:
    row = conn.execute("SELECT max(session_date) FROM btc_daily_bars WHERE finality='final'").fetchone()
    return row[0] if row and row[0] else None


def _valid_px(px: float | None) -> bool:
    return px is not None and math.isfinite(px) and px > 0


def _vol_map(conn) -> dict[str, float]:
    out: dict[str, float] = {}
    last = conn.execute(
        "SELECT max(as_of_date) FROM feature_rows WHERE asset_class = ?",
        [AssetClass.US_EQUITY.value],
    ).fetchone()
    last_d = last[0] if last and last[0] else None
    x, _y, keys = load_matrix(
        conn,
        asset_class=AssetClass.US_EQUITY.value,
        horizon=5,
        matured_only=False,
        as_of=last_d,
    )
    for i, (inst, _d) in enumerate(keys):
        feat_i = FEATURE_NAMES.index("vol_10")
        out[inst] = float(x[i, feat_i]) if len(x) else 0.02
    return out or {"_": 0.02}


def _last_close(conn, instrument_id: str) -> float | None:
    row = conn.execute(
        """
        SELECT close FROM equity_daily_bars
        WHERE instrument_id = ? AND finality = 'final'
        ORDER BY session_date DESC LIMIT 1
        """,
        [instrument_id],
    ).fetchone()
    if row:
        px = float(row[0])
        return px if _valid_px(px) else None
    if "btc" in instrument_id:
        brow = conn.execute("SELECT close FROM btc_daily_bars ORDER BY session_date DESC LIMIT 1").fetchone()
        if brow:
            px = float(brow[0])
            return px if _valid_px(px) else None
    return None


def _persist_health(
    conn,
    *,
    equity: HealthLevel,
    btc: HealthLevel,
    notes: list[str],
    coverage: CoverageSummary | None = None,
) -> DataHealthSnapshot:
    cov = coverage or CoverageSummary()
    payload = cov.model_dump()
    payload["notes"] = notes
    rank = {
        HealthLevel.OK: 1,
        HealthLevel.DEGRADED: 2,
        HealthLevel.NOT_CONFIGURED: 3,
        HealthLevel.UNAVAILABLE: 4,
        HealthLevel.CRITICAL: 5,
        HealthLevel.UNKNOWN: 0,
    }
    overall = max((equity, btc), key=lambda lv: rank.get(lv, 0))
    snap = DataHealthSnapshot(overall=overall, equity_feed=equity, btc_feed=btc, coverage=cov)
    conn.execute(
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
            json.dumps(payload),
            snap.last_tick_at,
            snap.created_at,
        ],
    )
    return snap


def _needs_llm_judge(rec, stock_weights: dict, *, always_include: set[str] | None = None) -> bool:
    """Research/judge live holdings and names Quant actually wants to add."""
    if always_include and str(rec.instrument_id) in always_include:
        return True
    if float(rec.current_units or 0) > 1e-12 or float(getattr(rec, "acquisition_units", None) or 0) > 1e-12:
        return True
    action = str(getattr(rec.action, "value", rec.action))
    if action not in {"BUY", "ENTER", "ADD"}:
        return False
    return float(rec.recommended_units or 0) > 1e-12


def _market_regime(conn) -> str:
    spy = stable_instrument_id("SPY")
    rows = conn.execute(
        """
        SELECT close FROM equity_daily_bars
        WHERE instrument_id = ? AND finality = 'final'
        ORDER BY session_date DESC LIMIT 2
        """,
        [str(spy)],
    ).fetchall()
    if len(rows) < 2:
        return ""
    latest, prior = float(rows[0][0]), float(rows[1][0])
    if prior <= 0:
        return ""
    chg = (latest - prior) / prior
    if chg > 0.01:
        return "risk_on"
    if chg < -0.01:
        return "risk_off"
    return "neutral"


def _annotate_units(
    recs,
    *,
    stock_acq: dict[str, float],
    stock_marked: dict[str, float],
    stock_preview: dict[str, float | None],
    unpriced: list[str],
    btc_pos,
) -> list:
    unpriced_set = set(unpriced)
    btc_unpriced = bool(
        btc_pos is not None
        and float(btc_pos.acquisition_units_total or 0) > 1e-12
        and btc_pos.marked_units_final is None
    )
    out = []
    for rec in recs:
        inst = str(rec.instrument_id)
        is_btc = "btc" in inst.lower()
        if is_btc:
            acq = float(btc_pos.acquisition_units_total) if btc_pos is not None else None
            marked = btc_pos.marked_units_final if btc_pos is not None else None
            preview = btc_pos.marked_units_intraday_preview if btc_pos is not None else None
            degrade = btc_unpriced
        else:
            acq = stock_acq.get(inst)
            marked = stock_marked.get(inst)
            preview = stock_preview.get(inst)
            degrade = inst in unpriced_set
        update: dict = {
            "acquisition_units": acq,
            "marked_units_final": marked,
            "marked_units_intraday_preview": preview,
            "position_held": bool((acq or 0) > 1e-12 or (rec.current_units or 0) > 1e-12),
            "valuation_status": "UNAVAILABLE" if degrade else ("FINAL" if marked is not None or (rec.current_units or 0) > 1e-12 else "NOT_HELD"),
            "units_basis": "final_close",
        }
        if degrade:
            update["current_units"] = None
            update["recommended_units"] = None
            update["delta_units"] = None
            update["actionable"] = False
            update["marked_units_final"] = None
            update["action"] = RecommendationAction.HOLD
            update["exclusion_reason"] = "UNPRICED_HOLDING"
            update["allocation_note"] = "확정 평가가격이 없어 보유 상태만 유지하며 목표 배분은 미확정입니다."
        out.append(RecommendationRecord.model_validate(normalize_position({**rec.model_dump(), **update})))
    return out


def _research_capability(packs: dict) -> str:
    """Dashboard badge for this scan. One 429/no-search name must not hide a working Gemini."""
    if not packs:
        return STATUS_NOT_CONFIGURED
    statuses = [str(p.get("status") or "") for p in packs.values() if isinstance(p, dict)]
    if not statuses:
        return STATUS_NOT_CONFIGURED
    if all(s == STATUS_NOT_CONFIGURED for s in statuses):
        return STATUS_NOT_CONFIGURED
    n_ok = sum(s == STATUS_AVAILABLE for s in statuses)
    n_fail = sum(
        s
        in {
            STATUS_UNAVAILABLE,
            STATUS_RATE_LIMITED,
            STATUS_QUOTA_EXCEEDED,
            STATUS_NOT_CONFIGURED,
        }
        for s in statuses
    )
    if n_ok and n_ok >= n_fail:
        return STATUS_AVAILABLE
    if n_ok or any(s == STATUS_DEGRADED for s in statuses):
        return STATUS_DEGRADED
    return STATUS_UNAVAILABLE


def _equity_ticker_for_sec(instrument_id: object) -> str | None:
    s = str(instrument_id)
    if "btc" in s.lower():
        return None
    if s.startswith("inst_"):
        s = s[5:]
    ticker = s.upper().replace("_", "")
    if ticker in {"BTCUSD", "SPY"}:
        return None
    return ticker or None


def _llm_target_tickers(recs: list, stock_weights: dict) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for rec in recs:
        if not _needs_llm_judge(rec, stock_weights):
            continue
        ticker = _equity_ticker_for_sec(rec.instrument_id)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def _alpaca_configured(settings: Settings) -> bool:
    return bool(settings.alpaca_api_key and settings.alpaca_secret_key)


def _aggregate_llm_status(*, configured: bool, recs: list) -> str:
    if not configured:
        return STATUS_NOT_CONFIGURED
    if not recs:
        return STATUS_UNAVAILABLE
    statuses: list[str] = []
    for rec in recs:
        rec_reasons = rec.override_reasons or []
        if "NOT_A_CANDIDATE" in rec_reasons or "NOT_SELECTED_FOR_FINAL_JUDGE" in rec_reasons:
            continue
        if STATUS_QUOTA_EXCEEDED in rec_reasons:
            statuses.append(STATUS_QUOTA_EXCEEDED)
        elif "BUDGET_EXCEEDED" in rec_reasons:
            statuses.append(STATUS_UNAVAILABLE)
        elif STATUS_RATE_LIMITED in rec_reasons:
            statuses.append(STATUS_RATE_LIMITED)
        elif STATUS_NOT_CONFIGURED in rec_reasons:
            statuses.append(STATUS_NOT_CONFIGURED)
        elif STATUS_UNAVAILABLE in rec_reasons:
            statuses.append(STATUS_UNAVAILABLE)
        elif STATUS_DEGRADED in rec_reasons:
            statuses.append(STATUS_DEGRADED)
        else:
            statuses.append(STATUS_AVAILABLE)
    if not statuses:
        return STATUS_UNAVAILABLE
    if any(s == STATUS_QUOTA_EXCEEDED for s in statuses):
        return STATUS_QUOTA_EXCEEDED
    if any(s == STATUS_RATE_LIMITED for s in statuses) and not any(s == STATUS_AVAILABLE for s in statuses):
        return STATUS_RATE_LIMITED
    if any(s == STATUS_AVAILABLE for s in statuses) and any(s != STATUS_AVAILABLE for s in statuses):
        return STATUS_DEGRADED
    if all(s == STATUS_AVAILABLE for s in statuses):
        return STATUS_AVAILABLE
    if any(s == STATUS_RATE_LIMITED for s in statuses):
        return STATUS_RATE_LIMITED
    if any(s == STATUS_DEGRADED for s in statuses):
        return STATUS_DEGRADED
    return STATUS_UNAVAILABLE


def _synthetic_days(settings: Settings) -> int:
    return max(90, min(504, int(settings.history_years) * 252))


def _equity_provider_counts(conn) -> dict[str, int]:
    rows = conn.execute("SELECT provider, COUNT(*) FROM equity_daily_bars GROUP BY provider").fetchall()
    return {str(provider): int(n) for provider, n in rows}


def _live_equity_bar_count(conn) -> int:
    return sum(n for provider, n in _equity_provider_counts(conn).items() if provider != "fixture")


def _drop_fixture_bars(conn) -> None:
    conn.execute("DELETE FROM equity_daily_bars WHERE provider = 'fixture'")
    conn.execute("DELETE FROM btc_daily_bars WHERE provider = 'fixture'")


def refresh_market_history(store: Store, settings: Settings) -> tuple[str, bool]:
    """Load live Alpaca history when keys exist. Do not freeze on leftover fixture bars.

    Returns (market_status, used_synthetic). Failed live calls keep last-known bars (D03).
    """
    conn = store.conn
    alpaca = _alpaca_configured(settings)
    counts = _equity_provider_counts(conn)
    n_eq = sum(counts.values())
    n_live = sum(n for provider, n in counts.items() if provider != "fixture")

    if not alpaca:
        if n_eq == 0:
            days = _synthetic_days(settings)
            _emit(
                conn,
                RuntimeEventKind.FETCH_US_DAILY,
                RuntimeEventStatus.NOT_CONFIGURED,
                f"Alpaca NOT_CONFIGURED; continuing with {days} synthetic sessions (not live market data)",
            )
            seed_synthetic_history(store, settings, days=days)
        return STATUS_NOT_CONFIGURED, True

    _emit(conn, RuntimeEventKind.FETCH_US_DAILY, RuntimeEventStatus.STARTED, "Alpaca history backfill")
    svc = None
    provider = None
    fetch_error: str | None = None
    try:
        provider = make_market_provider(settings)
        svc = MarketDataService(conn, settings, provider, provider_name=provider.provider_name())
        svc.startup_backfill()
    except Exception as exc:  # noqa: BLE001 — never-block cycle
        fetch_error = f"{type(exc).__name__}: {exc}"[:240]
    finally:
        closer = getattr(provider, "close", None) if provider is not None else None
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass

    n_live = _live_equity_bar_count(conn)
    if n_live > 0:
        _drop_fixture_bars(conn)
        n_live = _live_equity_bar_count(conn)
        if svc is not None:
            try:
                svc.build_data_health_snapshot()
            except Exception:  # noqa: BLE001
                pass
        _emit(conn, RuntimeEventKind.FETCH_US_DAILY, RuntimeEventStatus.SUCCEEDED, f"Alpaca bars={n_live}")
        return STATUS_AVAILABLE, False

    n_eq = sum(_equity_provider_counts(conn).values())
    if n_eq == 0:
        days = _synthetic_days(settings)
        _emit(
            conn,
            RuntimeEventKind.FETCH_US_DAILY,
            RuntimeEventStatus.UNAVAILABLE,
            f"Alpaca returned no bars; continuing with {days} synthetic sessions so ML can run",
        )
        seed_synthetic_history(store, settings, days=days)
        return STATUS_UNAVAILABLE, True

    note = fetch_error or "Alpaca returned no live bars; keeping last-known synthetic history"
    _emit(conn, RuntimeEventKind.FETCH_US_DAILY, RuntimeEventStatus.UNAVAILABLE, note)
    return STATUS_UNAVAILABLE, True


def run_v1_cycle(
    store: Store,
    settings: Settings,
    *,
    artifacts_dir: Path | None = None,
    universe: ResolvedUniverse | None = None,
    retrain: bool = False,
    train_only: bool = False,
    enforce_llm_budget: bool = False,
) -> dict:
    """train_only=True stops right after model refit + ensemble/horizon-influence
    adaptation + online SGD updates (see the early return below) -- no technical
    gating, allocation, SEC/Gemini/GPT calls. Unlike retrain=True alone, train_only
    still refreshes market data first, since it is meant to be a standalone daily
    job rather than a follow-up to a scan that already fetched today's bars.
    """
    conn = store.conn
    artifacts_dir = artifacts_dir or Path("data/artifacts")
    benchmark_symbols: tuple[str, ...] = ()
    if universe is not None:
        settings = settings.model_copy(update={"smoke_universe": universe.with_btc(settings)})
        benchmark_symbols = universe.benchmarks
    _emit(conn, RuntimeEventKind.HEARTBEAT, RuntimeEventStatus.STARTED, "V1 cycle start (no broker)")
    bootstrap_smoke_universe(conn, settings, provider="alpaca" if _alpaca_configured(settings) else "fixture")
    if retrain and not train_only:
        n_live = _live_equity_bar_count(conn)
        if _alpaca_configured(settings) and n_live > 0:
            market_status, used_synthetic = STATUS_AVAILABLE, False
        elif not _alpaca_configured(settings):
            market_status, used_synthetic = STATUS_NOT_CONFIGURED, True
        else:
            market_status, used_synthetic = STATUS_UNAVAILABLE, True
        _emit(
            conn,
            RuntimeEventKind.TRAINING,
            RuntimeEventStatus.STARTED,
            "학습만 실행합니다. 시세는 다시 받지 않습니다.",
        )
    else:
        market_status, used_synthetic = refresh_market_history(store, settings)
    health_notes = []
    if used_synthetic and market_status == STATUS_NOT_CONFIGURED:
        health_notes.append("Market data NOT_CONFIGURED (no Alpaca keys). App continued on synthetic smoke history.")
        _persist_health(
            conn,
            equity=HealthLevel.NOT_CONFIGURED,
            btc=HealthLevel.NOT_CONFIGURED,
            notes=health_notes,
        )
    elif used_synthetic:
        health_notes.append("Live provider UNAVAILABLE; synthetic continuation in use.")
        _persist_health(
            conn,
            equity=HealthLevel.UNAVAILABLE,
            btc=HealthLevel.UNAVAILABLE,
            notes=health_notes,
        )
    elif market_status == STATUS_AVAILABLE:
        health_notes.append("Market provider configured; bars present.")
        if conn.execute("SELECT COUNT(*) FROM data_health_snapshots").fetchone()[0] == 0:
            _persist_health(
                conn,
                equity=HealthLevel.OK,
                btc=HealthLevel.OK,
                notes=health_notes,
            )

    last = _latest_as_of(conn) or date.today()
    last_btc = _latest_btc_as_of(conn) or last
    if features_cover_latest(conn, last_available=last, last_available_btc=last_btc):
        _emit(
            conn,
            RuntimeEventKind.INFERENCE,
            RuntimeEventStatus.SUCCEEDED,
            "피처가 이미 최신입니다. 다시 계산하지 않습니다.",
        )
    else:
        _emit(
            conn,
            RuntimeEventKind.INFERENCE,
            RuntimeEventStatus.STARTED,
            "시세 저장 완료. 피처 계산 중",
        )

        def _feat_progress(done: int, total: int) -> None:
            _emit(
                conn,
                RuntimeEventKind.INFERENCE,
                RuntimeEventStatus.PROGRESS,
                f"피처 계산 {done}/{total} 종목",
            )

        n_feat = build_and_persist_features(
            conn,
            last_available=last,
            last_available_btc=last_btc,
            horizons=settings.horizons,
            progress=_feat_progress,
        )
        _emit(
            conn,
            RuntimeEventKind.INFERENCE,
            RuntimeEventStatus.SUCCEEDED,
            f"피처 계산 완료 ({n_feat}행)",
        )
    assert_no_future_in_features(conn)

    loaded = load_fitted_bundles(artifacts_dir)
    must_train = retrain or train_only or not bundles_ready(loaded)
    online: dict[tuple[str, int], OnlineHorizonModel] = {}
    if must_train:
        _emit(conn, RuntimeEventKind.TRAINING, RuntimeEventStatus.STARTED, "Train horizon-aware batch models")
        bundles = train_batch_models(conn, artifacts_dir)
        _emit(conn, RuntimeEventKind.TRAINING, RuntimeEventStatus.SUCCEEDED, f"trained {len(bundles)} artifacts")

        for asset in (AssetClass.US_EQUITY.value, AssetClass.BTC.value):
            for h in settings.horizons:
                om = load_online_model(online_path(artifacts_dir, asset, h))
                x, y, keys = load_matrix(conn, asset_class=asset, horizon=h, matured_only=True)
                om = apply_online_updates(
                    conn,
                    asset=asset,
                    horizon=h,
                    model=om,
                    x=x,
                    y=y,
                    keys=keys,
                    parent_epoch_id="pre_epoch",
                )
                save_online_model(online_path(artifacts_dir, asset, h), om)
                online[(asset, h)] = om
                if om.fitted:
                    label_dates = [k[1] for k in keys]
                    window = (
                        f"{min(label_dates).isoformat()} ~ {max(label_dates).isoformat()}"
                        if label_dates
                        else None
                    )
                    _persist_journal(
                        conn,
                        [
                            ModelChangeJournalEvent(
                                event_id=f"mcj_{uuid4().hex[:12]}",
                                kind=ModelChangeKind.ONLINE_UPDATED,
                                model_family=ModelFamily.ONLINE_SGD,
                                asset_class=asset,
                                horizon=h,
                                new_version="online_v1",
                                reason_codes=["matured_label_partial_fit"],
                                matured_label_count=int(len(y)),
                                sample_count=int(len(y)),
                                training_period=window,
                                promotion_result="applied",
                            )
                        ],
                    )
    else:
        _emit(
            conn,
            RuntimeEventKind.TRAINING,
            RuntimeEventStatus.SUCCEEDED,
            "저장된 모델을 사용합니다. 학습은 건너뜁니다.",
        )
        bundles = loaded
        for asset in (AssetClass.US_EQUITY.value, AssetClass.BTC.value):
            for h in settings.horizons:
                online[(asset, h)] = load_online_model(online_path(artifacts_dir, asset, h))

    asset_weights: dict[str, dict[str, float]] = {}
    asset_horizon_influence: dict[str, dict[str, float]] = {}
    for asset in (AssetClass.US_EQUITY.value, AssetClass.BTC.value):
        prev, horizon_inf, ver = load_ensemble_state(conn, asset)
        if not must_train:
            persist_ensemble_state(conn, asset, prev, horizon_inf, ver)
            asset_weights[asset] = prev
            asset_horizon_influence[asset] = horizon_inf
            continue
        new_w, changed, diag = adapt_ensemble_from_matured(
            conn, asset=asset, bundles=bundles, online=online, prev_weights=prev
        )
        new_horizon_inf, horizon_changed, horizon_composite_err, horizon_target = adapt_horizon_influence(
            horizon_inf,
            diag.get("errors_by_horizon") or {},
            new_w,
            naive_mae_by_horizon=diag.get("naive_mae_by_horizon") or {},
        )
        family_target = diag.get("target_weights") or {}
        events: list[ModelChangeJournalEvent] = []
        next_ver = next_ensemble_version(ver) if changed else ver
        if changed:
            errors = diag.get("mean_abs_error") or {}
            matured_n = diag.get("matured_label_count")
            eval_n = diag.get("eval_row_count")
            for fam in sorted(set(new_w) | set(prev)):
                before = prev.get(fam)
                after = new_w.get(fam)
                if before is not None and after is not None and abs(after - before) <= 1e-6:
                    continue
                err = errors.get(fam) if isinstance(errors, dict) else None
                events.append(
                    ModelChangeJournalEvent(
                        event_id=f"mcj_{uuid4().hex[:12]}",
                        kind=ModelChangeKind.ENSEMBLE_WEIGHT_CHANGED,
                        model_family=family_enum(fam),
                        asset_class=asset,
                        previous_version=ver,
                        new_version=next_ver,
                        previous_ensemble_weight=before,
                        new_ensemble_weight=after,
                        target_ensemble_weight=family_target.get(fam),
                        reason_codes=["matured_ewma"],
                        matured_label_count=int(matured_n) if matured_n is not None else None,
                        # rows actually used in this update (recent-window-restricted),
                        # not the full matured_label_count -- see RECENT_EVAL_WINDOW_DAYS.
                        sample_count=int(eval_n) if eval_n is not None else None,
                        metric_name="matured_mean_abs_error" if err is not None else None,
                        metric_after=float(err) if err is not None else None,
                        evaluation_period=diag.get("evaluation_period"),  # type: ignore[arg-type]
                        promotion_result="applied",
                    )
                )
        if horizon_changed:
            matured_n = diag.get("matured_label_count")
            for hz in sorted(set(new_horizon_inf) | set(horizon_inf)):
                before = horizon_inf.get(hz)
                after = new_horizon_inf.get(hz)
                if before is not None and after is not None and abs(after - before) <= 1e-6:
                    continue
                err = horizon_composite_err.get(hz)
                events.append(
                    ModelChangeJournalEvent(
                        event_id=f"mcj_{uuid4().hex[:12]}",
                        kind=ModelChangeKind.HORIZON_WEIGHT_CHANGED,
                        asset_class=asset,
                        horizon=int(hz) if str(hz).isdigit() else None,
                        previous_version=ver,
                        previous_ensemble_weight=before,
                        new_ensemble_weight=after,
                        target_ensemble_weight=horizon_target.get(hz),
                        reason_codes=["horizon_matured_ewma"],
                        matured_label_count=int(matured_n) if matured_n is not None else None,
                        # model_MAE / same-horizon naive(zero-forecast)_MAE, not raw MAE
                        # -- see adapt_horizon_influence's docstring for why raw MAE
                        # alone was mostly measuring horizon-label volatility.
                        metric_name="mae_over_naive_mae" if err is not None else None,
                        metric_after=float(err) if err is not None else None,
                        evaluation_period=diag.get("evaluation_period"),  # type: ignore[arg-type]
                        promotion_result="applied",
                    )
                )
        if events:
            _persist_journal(conn, events)
        asset_horizon_influence[asset] = new_horizon_inf
        if changed:
            persist_ensemble_state(conn, asset, new_w, new_horizon_inf, next_ver)
            asset_weights[asset] = new_w
        else:
            persist_ensemble_state(conn, asset, prev, new_horizon_inf, ver)
            asset_weights[asset] = prev

    if train_only:
        # Model refit + ensemble/horizon-influence adaptation + online SGD updates are
        # all done at this point -- everything below here (technical-factor gating,
        # allocation, SEC ingest, Gemini research, the GPT judge conversation) exists
        # to produce a recommendation, not to train a model, and the GPT/Gemini calls
        # are the entire cost of a cycle (~$1-8, see llm_cost_ledger). Stopping here
        # keeps training itself at $0 / no external API calls.
        _emit(
            conn,
            RuntimeEventKind.TRAINING,
            RuntimeEventStatus.SUCCEEDED,
            f"학습 전용 실행 완료 (API 호출 없음) — 모델 {len(bundles)}개",
        )
        return {
            "train_only": True,
            "market_data_basis": market_status,
            "used_synthetic_market": used_synthetic,
            "bundles_trained": len(bundles),
            "asset_weights": asset_weights,
            "asset_horizon_influence": asset_horizon_influence,
        }

    batch_refs: list[ModelArtifactRef] = []
    online_refs: list[ModelArtifactRef] = []
    hash_parts: list[str] = []
    for (asset, h, fam), b in sorted(bundles.items(), key=lambda kv: kv[0]):
        ah = bundle_hash(b)
        hash_parts.append(f"{asset}:{h}:{fam}:{ah}")
        batch_refs.append(
            ModelArtifactRef(family=family_enum(fam), horizon=h, version=b.version, artifact_hash=ah)
        )
    for (asset, h), om in sorted(online.items(), key=lambda kv: kv[0]):
        oh = hashlib.sha256(f"{asset}:{h}:{int(om.fitted)}".encode()).hexdigest()[:16]
        online_refs.append(
            ModelArtifactRef(family=ModelFamily.ONLINE_SGD, horizon=h, version="online_v1", artifact_hash=oh)
        )
        hash_parts.append(f"online:{asset}:{h}:{oh}")
    ew_hash = hashlib.sha256("|".join(hash_parts).encode()).hexdigest()[:16] if hash_parts else "ew_empty"

    epoch = DecisionEpochManifest(
        preprocess_version="feat_v1",
        ensemble_weights_hash=ew_hash,
        thresholds_hash="th_v1",
        formula_versions={"alloc": FORMULA_VERSION, "stop": "stop_atr_v1"},
        batch_models=batch_refs,
        online_models=online_refs,
    )
    persist_decision_epoch(conn, epoch)
    tick_id = new_tick_id()
    fs_id = new_feature_snapshot_id()
    live_alpaca = bool(_alpaca_configured(settings) and not used_synthetic)
    provider_label = "alpaca" if live_alpaca else "not_configured_synthetic"
    adjustment_revision = (
        "alpaca_split_adjusted_v1" if provider_label == "alpaca" else "fixture_rev_1"
    )
    conn.execute(
        """
        INSERT INTO feature_snapshots
        (feature_snapshot_id, price_basis, adjustment_revision, evaluation_basis, provider, published_at)
        VALUES (?, 'split_adjusted', ?, 'adjusted_revisioned', ?, ?)
        """,
        [fs_id, adjustment_revision, provider_label, datetime.now(timezone.utc)],
    )

    _emit(conn, RuntimeEventKind.INFERENCE, RuntimeEventStatus.STARTED, "오늘 점수 계산 중 (최근 피처만)")
    stock_scores: dict[str, dict[int, float]] = {}
    stock_rank_scores: dict[str, dict[int, float]] = {}
    btc_scores: dict[int, float] = {}
    pred_rows: list[dict] = []
    for asset in (AssetClass.US_EQUITY.value, AssetClass.BTC.value):
        weights = asset_weights.get(asset) or default_weights(asset)
        last_feat = conn.execute(
            "SELECT max(as_of_date) FROM feature_rows WHERE asset_class = ?",
            [asset],
        ).fetchone()
        last_d_all = last_feat[0] if last_feat and last_feat[0] else None
        for h in settings.horizons:
            x, _y, keys = load_matrix(
                conn,
                asset_class=asset,
                horizon=h,
                matured_only=False,
                as_of=last_d_all,
            )
            if len(keys) == 0:
                continue
            last_d = max(k[1] for k in keys)
            idx = list(range(len(keys)))
            xx = x
            seq_map: dict[tuple[str, date], object] = {}
            seq, _seq_y, seq_keys = load_sequences(
                conn,
                asset_class=asset,
                horizon=h,
                lookback=LSTM_LOOKBACK,
                matured_only=False,
                as_of=last_d,
            )
            for si, sk in enumerate(seq_keys):
                seq_map[sk] = seq[si]
            fams = ["ridge", "lightgbm_reg", "torch_sequence"]
            if asset == AssetClass.US_EQUITY.value:
                fams.append("lambdarank")
            for j, irow in enumerate(idx):
                inst = keys[irow][0]
                preds: dict[str, float] = {}
                rank_s: float | None = None
                for fam in fams:
                    b = bundles.get((asset, h, fam))
                    if b is None:
                        continue
                    if fam == "torch_sequence" and b.lookback and b.lookback > 1:
                        seq_row = seq_map.get((inst, last_d))
                        if seq_row is None:
                            continue
                        val = float(_predict(b, seq_row[None, ...])[0])
                    else:
                        val = float(_predict(b, xx[j : j + 1])[0])
                    if fam == "lambdarank":
                        rank_s = val
                    else:
                        preds[fam] = val
                    pred_rows.append(
                        {
                            "prediction_id": f"pr_{uuid4().hex[:16]}",
                            "instrument_id": inst,
                            "as_of_date": last_d,
                            "horizon": h,
                            "asset_class": asset,
                            "family": fam,
                            "value": val,
                            "decision_epoch_id": str(epoch.decision_epoch_id),
                        }
                    )
                om = online.get((asset, h))
                if om:
                    preds["online"] = float(om.predict(xx[j : j + 1])[0])
                score = blend(preds, weights)
                if asset == AssetClass.BTC.value:
                    btc_scores[h] = score
                else:
                    stock_scores.setdefault(inst, {})[h] = score
                    if rank_s is not None:
                        stock_rank_scores.setdefault(inst, {})[h] = rank_s
    if pred_rows:
        persist_prediction_rows(conn, pred_rows)

    portfolio = mark_positions(conn, load_portfolio(conn))
    held_ids = {
        str(canonicalize_instrument_id(r[0]))
        for r in conn.execute("SELECT DISTINCT instrument_id FROM portfolio_lots").fetchall()
    }
    held_ids.update(str(p.instrument_id) for p in portfolio.positions)
    # Benchmarks (SPY) stay in features for relative signals but are not candidates,
    # unless the user actually holds them.
    excluded = {str(stable_instrument_id(s)) for s in benchmark_symbols} - held_ids
    if excluded:
        stock_scores = {k: v for k, v in stock_scores.items() if k not in excluded}
        stock_rank_scores = {k: v for k, v in stock_rank_scores.items() if k not in excluded}

    # legacy-b style technical/chart-shape gate. Computed for every scored equity name
    # (held names too, for display) but only ever REMOVES a name from stock_scores when
    # it is a would-be NEW entry (not currently held) and fails a hard gate -- an
    # existing holding is never silently hidden by this check, matching how
    # held_should_exit (allocation.py) is the only thing that can recommend exiting a
    # position already owned.
    spy_inst_id = str(stable_instrument_id(BENCHMARK_SYMBOLS[0]))
    tech_factors_by_inst: dict[str, TechnicalFactors] = {}
    tech_excluded: list[dict[str, str]] = []
    equity_inst_ids = [k for k in stock_scores.keys() if "btc" not in k.lower()]
    if equity_inst_ids:
        spy_close = fetch_spy_close(conn, spy_instrument_id=spy_inst_id)
        bar_frames = fetch_bars_frames(conn, equity_inst_ids)
        rs_pct = compute_rs_percentiles(bar_frames, spy_close)
        for inst in equity_inst_ids:
            frame = bar_frames.get(inst)
            if frame is None:
                continue
            factors = compute_technical_factors(frame, spy_close, market_rs_percentile=rs_pct.get(inst))
            if factors is None:
                continue
            tech_factors_by_inst[inst] = factors
            if factors.hard_excluded and inst not in held_ids:
                tech_excluded.append({"instrument_id": inst, "reason": factors.exclusion_reason or "TECH_GATE"})
        if tech_excluded:
            excluded_ids = {row["instrument_id"] for row in tech_excluded}
            stock_scores = {k: v for k, v in stock_scores.items() if k not in excluded_ids}
            stock_rank_scores = {k: v for k, v in stock_rank_scores.items() if k not in excluded_ids}
            _emit(
                conn,
                RuntimeEventKind.INFERENCE,
                RuntimeEventStatus.PROGRESS,
                "기술적 하드게이트로 신규 후보 " + str(len(tech_excluded)) + "개 제외: "
                + ", ".join(f"{row['instrument_id']}({row['reason']})" for row in tech_excluded[:8]),
            )

    btc_state = BtcSleeveState(instrument_id=stable_instrument_id(settings.btc_symbol))
    row = conn.execute("SELECT current_units, liquidity FROM btc_sleeve_state WHERE id = 1").fetchone()
    if row:
        btc_state.current_units = float(row[0])
        btc_state.liquidity = BtcLiquidityState(str(row[1]))

    stock_marked, stock_acq, stock_preview, unpriced = stock_exposure_for_allocation(portfolio.positions)
    unpriced_all = [
        str(p.instrument_id) for p in portfolio.positions
        if float(p.acquisition_units_total or 0) > 1e-12 and not p.price_available
    ]
    btc_pos = next((p for p in portfolio.positions if "btc" in str(p.instrument_id).lower()), None)
    if btc_pos is not None and btc_pos.marked_units_final is not None:
        btc_state.current_units = btc_pos.marked_units_final

    stock_technical = {
        inst: {
            "buyable_score": f.buyable_score,
            "leader_score": f.leader_score,
            "quality_of_trend_score": f.quality_of_trend_score,
            "breakout_score": f.breakout_score,
            "catalyst_score": f.catalyst_score,
        }
        for inst, f in tech_factors_by_inst.items()
    }

    def _run_allocation(*, stock_research: dict[str, dict] | None = None) -> dict:
        result = allocate(
            stock_scores=stock_scores,
            stock_vol=_vol_map(conn),
            btc_scores=btc_scores,
            settings=settings,
            btc_state=btc_state,
            tick_id=str(tick_id),
            epoch_id=str(epoch.decision_epoch_id),
            feature_snapshot_id=str(fs_id),
            total_base_units=portfolio.total_base_units,
            stock_units={**stock_acq, **stock_marked},
            stock_rank_scores=stock_rank_scores,
            stock_technical=stock_technical,
            stock_research=stock_research,
            stock_horizon_influence=asset_horizon_influence.get(AssetClass.US_EQUITY.value),
            btc_horizon_influence=asset_horizon_influence.get(AssetClass.BTC.value),
        )
        result["recommendations"] = _annotate_units(
            result["recommendations"],
            stock_acq=stock_acq,
            stock_marked=stock_marked,
            stock_preview=stock_preview,
            unpriced=unpriced_all,
            btc_pos=btc_pos,
        )
        if tech_factors_by_inst:
            annotated: list = []
            for r in result["recommendations"]:
                factors = tech_factors_by_inst.get(str(r.instrument_id))
                if factors is None:
                    annotated.append(r)
                    continue
                annotated.append(
                    r.model_copy(
                        update={
                            "technical_track": factors.track,
                            "leader_score": factors.leader_score,
                            "momentum_score": factors.momentum_score,
                            "volume_score": factors.volume_score,
                            "buyable_score": factors.buyable_score,
                            "breakout_score": factors.breakout_score,
                            "top_risk_score": factors.top_risk_score,
                            "quality_of_trend_score": factors.quality_of_trend_score,
                            "catalyst_score": factors.catalyst_score,
                        }
                    )
                )
            result["recommendations"] = annotated
        if stock_research:
            diagnostics = (result.get("payload") or {}).get("diagnostics") or {}
            annotated_research: list = []
            for r in result["recommendations"]:
                inst = str(r.instrument_id)
                pack = stock_research.get(inst)
                if pack is None:
                    annotated_research.append(r)
                    continue
                mult = (diagnostics.get(inst) or {}).get("research_sizing_multiplier")
                annotated_research.append(
                    r.model_copy(
                        update={
                            "research_summary_ko": pack.get("summary_ko") or None,
                            "research_rerating_score": pack.get("rerating_score"),
                            "research_valuation_support_score": pack.get("valuation_support_score"),
                            "research_cash_relative_score": pack.get("cash_relative_score"),
                            "research_data_quality_score": pack.get("data_quality_score"),
                            "research_sizing_multiplier": mult,
                        }
                    )
                )
            result["recommendations"] = annotated_research
        return result

    def _derive_context(current_alloc: dict) -> tuple[float, dict]:
        btc_for_cash_ = (
            0.0
            if (btc_pos is not None and btc_pos.marked_units_final is None)
            else float(btc_state.current_units or 0.0)
        )
        cash_units_ = (
            max(0.0, float(portfolio.total_base_units) - sum(stock_marked.values()) - btc_for_cash_)
            if portfolio.actionable_for_recommendations() else None
        )
        quant_targets_ = {
            str(r.instrument_id): r.recommended_units
            for r in current_alloc["recommendations"]
            if "btc" not in str(r.instrument_id).lower()
        }
        ctx = build_portfolio_context(
            total_base_units=portfolio.total_base_units,
            settings=settings,
            stock_marked=stock_marked,
            stock_acquisition=stock_acq,
            quant_targets=quant_targets_,
            btc_state=btc_state,
            unpriced=unpriced,
            cash_units=cash_units_,
            allocation_trace=(current_alloc.get("payload") or {}).get("allocation_trace"),
        )
        return cash_units_, ctx

    # Pass 1: Quant-only sizing. Used to decide who is even eligible for the GPT
    # conversation / Gemini research below -- research needs a RecommendationRecord
    # (current/recommended units, confidence, thesis) to build its prompt, so it can
    # only run after at least one allocation pass exists.
    alloc = _run_allocation()
    quant_alloc = copy.deepcopy(alloc)
    input_as_of = datetime.now(timezone.utc)
    input_id = f"input_{uuid4().hex}"
    prices = {str(r.instrument_id): latest_price_snapshot(conn, str(r.instrument_id), as_of=input_as_of)
              for r in alloc["recommendations"]}
    daily_inputs = daily_input_fingerprints(conn, equity_end=last, btc_end=last_btc)
    resume_inputs = {
        "daily": daily_inputs,
        "prices": {k: {a: b for a, b in p.items() if a not in {"received_at", "input_as_of"}} for k, p in prices.items()},
        "holdings": stock_marked, "acquisition": stock_acq, "unpriced": unpriced_all,
        "btc": {"current_units": btc_state.current_units, "liquidity": str(btc_state.liquidity)}, "base": portfolio.total_base_units,
        "policy": {k: getattr(settings, k) for k in (
            "min_cash_weight", "max_single_stock_weight", "max_total_stock_weight", "max_btc_weight",
            "min_actionable_units_floor", "min_actionable_weight", "min_delta_weight", "display_unit_step_weight")},
        "judge_model": resolve_judge_model(settings), "contract": CONTRACT_VERSION,
    }
    resume_guard = content_id(resume_inputs)
    _, portfolio_ctx = _derive_context(alloc)

    stock_weights = alloc["payload"].get("stock_weights") or {}
    equity_tickers = _llm_target_tickers(alloc["recommendations"], stock_weights)

    # A prior GPT judge attempt that didn't finish (e.g. an OpenAI 429 mid-
    # conversation) leaves its exact candidate/portfolio package frozen on disk
    # (see save_pending_package's docstring below). Checked here, before SEC
    # ingest and Gemini research, so a retry skips every step that fed into a
    # GPT conversation input that is about to be thrown away anyway -- the
    # frozen package already has each candidate's SEC evidence and research
    # baked in from when it was first built. Reusing it verbatim is also what
    # keeps GPT turn resume itself reliable (see below): the online SGD model
    # nudges scores a little on every cycle, so a plain retry's freshly
    # re-derived candidate set (and therefore its GPT prompts) can otherwise
    # drift turn to turn.
    pending_package = load_pending_package(artifacts_dir)
    resume_issue = None
    rejected_input_id = None
    if pending_package:
        try:
            pending_package = prepare_pending(pending_package, resume_guard=resume_guard)
        except ValueError as exc:
            resume_issue = str(exc)
            rejected_input_id = pending_package.get("input_id")
            pending_package = None

    if resume_issue:
        sec_pack = {"sec_status": STATUS_DEGRADED, "extract_status": STATUS_DEGRADED,
                    "llm_configured": False, "filings": [], "errors": [resume_issue], "used_lkg": False}
    elif pending_package is None:
        _emit(conn, RuntimeEventKind.SEC_INGEST, RuntimeEventStatus.STARTED, "EDGAR 8-K/10-Q ingest")
        try:
            sec_pack = ingest_sec_extracts(
                conn, settings, tickers=equity_tickers, tick_id=str(tick_id)
            )
        except Exception as exc:  # noqa: BLE001
            sec_pack = {
                "sec_status": STATUS_UNAVAILABLE,
                "extract_status": STATUS_NOT_CONFIGURED,
                "llm_configured": False,
                "filings": [],
                "errors": [str(exc)],
                "used_lkg": False,
            }
        sec_rt = {
            STATUS_AVAILABLE: RuntimeEventStatus.SUCCEEDED,
            STATUS_NOT_CONFIGURED: RuntimeEventStatus.NOT_CONFIGURED,
            STATUS_UNAVAILABLE: RuntimeEventStatus.UNAVAILABLE,
        }.get(sec_pack["sec_status"], RuntimeEventStatus.DEGRADED)
        _emit(
            conn,
            RuntimeEventKind.SEC_INGEST,
            sec_rt,
            f"sec={sec_pack['sec_status']} extract={sec_pack['extract_status']} n={len(sec_pack['filings'])}",
        )
    else:
        sec_pack = {
            "sec_status": STATUS_AVAILABLE,
            "extract_status": STATUS_AVAILABLE,
            "llm_configured": False,
            "filings": [],
            "errors": [],
            "used_lkg": False,
        }
        _emit(
            conn,
            RuntimeEventKind.SEC_INGEST,
            RuntimeEventStatus.SUCCEEDED,
            "이어서 진행 -- 얼려둔 GPT 대화 후보를 그대로 씁니다 (SEC/Gemini 재호출 없음)",
        )

    # Same pool that goes to the GPT conversation is the pool eligible for Gemini
    # research below (bounded by llm_research_max_names, protected by the same daily
    # LLM budget as the GPT judge).
    candidates = [r for r in alloc["recommendations"] if _needs_llm_judge(r, stock_weights, always_include=held_ids)]
    conversation_ids = select_final_judge_recs(
        candidates,
        max_names=int(settings.llm_research_max_names),
        always_include=held_ids,
    )
    by_id = {str(r.instrument_id): r for r in candidates}

    # pending_package was already loaded above (before SEC ingest) -- reused here
    # to gate the Gemini research loop too, for the same reason.
    # Gemini research + adversarial pass (revived 2026-09-04). Only NEW-entry equity
    # candidates are researched for sizing purposes -- held positions are never
    # resized by this signal, matching technical_sizing_multiplier's exemption (see
    # research_sizing_multiplier's docstring in allocation.py for why), and BTC is
    # skipped because RESEARCH_SYSTEM's Q1-Q5 (earnings date, EPS/revenue, forward
    # PER/EV-EBITDA, peer comps) is an equity-rerating questionnaire that has no
    # meaningful answer for BTC, and research_sizing_multiplier is never applied to
    # the BTC sleeve's own sizing formula below -- researching it would only spend
    # budget for zero effect. The resulting pack feeds a second allocate() pass below
    # so the effect is real sizing, not just a GPT-prompt footnote.
    stock_research: dict[str, dict] = {}
    if pending_package is None and not resume_issue:
        for inst in conversation_ids:
            if inst in held_ids or "btc" in inst.lower():
                continue
            rec = by_id.get(inst)
            if rec is None:
                continue
            sec_row = evidence_for_instrument(sec_pack.get("filings") or [], inst)
            try:
                pack = research_ticker(
                    settings,
                    rec,
                    filings=[sec_row] if sec_row else [],
                    conn=conn,
                    tick_id=str(tick_id),
                    portfolio=portfolio_ctx,
                    last_price=(prices.get(inst) or {}).get("price"),
                    price_snapshot=prices.get(inst),
                    enforce_llm_budget=enforce_llm_budget,
                )
            except Exception:  # noqa: BLE001 -- one name's research must never abort the scan
                continue
            stock_research[inst] = pack

    # Pass 2: only re-run when research actually produced something -- avoids a
    # pointless identical re-allocation on days Gemini is not configured/researched
    # nothing (research_sizing_multiplier is neutral for every name in that case
    # anyway, so pass 1's numbers are already correct).
    if stock_research:
        alloc = _run_allocation(stock_research=stock_research)
        stock_weights = alloc["payload"].get("stock_weights") or {}
        candidates = [
            r for r in alloc["recommendations"] if _needs_llm_judge(r, stock_weights, always_include=held_ids)
        ]
        by_id = {str(r.instrument_id): r for r in candidates}

    cash_units, portfolio_ctx = _derive_context(alloc)
    market_actionable = bool(
        _alpaca_configured(settings)
        and not used_synthetic
        and portfolio.actionable_for_recommendations()
        and not resume_issue
    )
    alloc["payload"]["actionable"] = market_actionable
    alloc["payload"]["market_data_basis"] = provider_label
    if not market_actionable:
        alloc["recommendations"] = [
            r.model_copy(update={"actionable": False}) for r in alloc["recommendations"]
        ]
    if pending_package is not None:
        # A new attempt keeps the original input/epoch and all three stage inputs;
        # only the attempt tick and record IDs change. No historical tick is updated.
        frozen = pending_package["stages"]
        quant_alloc = {"payload": copy.deepcopy(frozen["quant_allocation"]),
                       "recommendations": [RecommendationRecord.model_validate(r) for r in frozen["quant"]]}
        alloc = {"payload": copy.deepcopy(frozen["research_allocation"]),
                 "recommendations": [RecommendationRecord.model_validate(r) for r in frozen["research_adjusted"]]}
        input_id = pending_package["input_id"]
        input_as_of = datetime.fromisoformat(pending_package["input_as_of"])
        prices = pending_package["prices"]
        portfolio_ctx = pending_package["portfolio"]
    baseline_ids = {}
    for rec in quant_alloc["recommendations"]:
        rec.recommendation_id = f"rec_{uuid4().hex}"
        rec.tick_id = tick_id
        rec.input_id, rec.input_as_of = input_id, input_as_of
        rec.price_snapshot = prices.get(str(rec.instrument_id))
        rec.actionable = bool(rec.actionable and market_actionable)
        baseline_ids[str(rec.instrument_id)] = rec.recommendation_id
    for rec in alloc["recommendations"]:
        rec.recommendation_id = f"rec_{uuid4().hex}"
        rec.tick_id = tick_id
        rec.source = RecommendationSource.RESEARCH_ADJUSTED
        rec.llm_tainted = True
        rec.input_id, rec.input_as_of = input_id, input_as_of
        rec.price_snapshot = prices.get(str(rec.instrument_id))
        rec.override_of_recommendation_id = baseline_ids.get(str(rec.instrument_id))
        rec.previous_stage_recommendation_id = baseline_ids.get(str(rec.instrument_id))
        rec.decision_status = "RESEARCH_ADJUSTED"
        rec.actionable = bool(rec.actionable and market_actionable)
    if unpriced_all:
        # The unknown exposure cannot be treated as cash or available capacity.
        # Candidate research may continue, but portfolio weights remain explicitly
        # unconfirmed and every recommendation remains non-actionable.
        for stage in (quant_alloc, alloc):
            stage["payload"].update({
                "stock_weights": {}, "btc_weight": None, "cash_weight": None,
                "equity_budget_used": None, "actionable": False,
                "weights_confirmed": False, "allocation_status": "INCOMPLETE_VALUATION",
                "unpriced_holdings": sorted(unpriced_all),
                "known_stock_units": dict(stock_marked),
            })
    by_id = {str(r.instrument_id): r for r in alloc["recommendations"]}
    research_allocation = copy.deepcopy(alloc["payload"])
    record_matured_outcomes(
        conn,
        last_equity=last,
        last_btc=last_btc,
        feature_snapshot_id=str(fs_id),
        provider=provider_label,
        adjustment_revision=adjustment_revision,
    )

    dq_level = "ok" if market_actionable else (
        "incomplete_valuation" if unpriced_all else "not_configured"
    )
    quant_status = (
        STATUS_AVAILABLE
        if market_actionable and alloc["recommendations"]
        else STATUS_DEGRADED
        if unpriced_all and _alpaca_configured(settings) and not used_synthetic
        else STATUS_NOT_CONFIGURED
        if not _alpaca_configured(settings)
        else STATUS_UNAVAILABLE
    )
    # Keep the existing multi-turn research; only its final turn becomes a validated
    # recommendation contract. Quant and research-stage records remain separate.
    llm_recs: list = []
    packs: dict[str, dict] = {}
    conversation_candidates: list[dict[str, object]] = []
    for inst in conversation_ids:
        rec = by_id.get(inst)
        if rec is None:
            continue
        rank_values = [
            float(value)
            for value in (stock_rank_scores.get(inst) or {}).values()
            if value is not None and math.isfinite(float(value))
        ]
        conversation_candidates.append(
            {
                "instrument_id": inst,
                "ticker": ticker_from_instrument(inst),
                "held": inst in held_ids or float(rec.current_units or 0) > 1e-12,
                "last_price": (prices.get(inst) or {}).get("price"),
                "price_snapshot": prices.get(inst),
                "rank_score": sum(rank_values) / len(rank_values) if rank_values else None,
                "quant": rec.model_dump(mode="json"),
                "sec_filing": evidence_for_instrument(sec_pack.get("filings") or [], inst),
                "research": stock_research.get(inst),
            }
        )

    portfolio_committee: dict[str, object] = {
        "status": "NOT_RUN",
        "note": "GPT 대화에 보낼 후보가 없습니다.",
    }
    # pending_package was already loaded above (before the Gemini research loop)
    # -- reuse it verbatim instead of this run's freshly (re-)derived
    # conversation_candidates, so the retried conversation's turns are
    # byte-identical to what was already sent and _load_resumable_turns can
    # actually skip the ones that already succeeded.
    if pending_package is not None:
        conversation_package = pending_package
        committee_candidates = pending_package.get("candidates") or []
    else:
        conversation_package = {
            "contract_version": CONTRACT_VERSION,
            "input_id": input_id,
            "input_as_of": input_as_of.isoformat(),
            "resume_guard": resume_guard,
            "daily_feature_basis": DAILY_BASIS,
            "daily_inputs": daily_inputs,
            "prices": prices,
            "portfolio": portfolio_ctx,
            "candidates": conversation_candidates,
            "stages": {
                "quant": [r.model_dump(mode="json") for r in quant_alloc["recommendations"]],
                "research_adjusted": [r.model_dump(mode="json") for r in alloc["recommendations"]],
                "quant_allocation": quant_alloc["payload"],
                "research_allocation": research_allocation,
            },
            "features": [list(r) for r in conn.execute(
                "SELECT instrument_id, CAST(as_of_date AS VARCHAR), horizon, features_json FROM feature_rows "
                "WHERE (asset_class='us_equity' AND as_of_date=?) OR (asset_class='btc' AND as_of_date=?)", [last, last_btc]
            ).fetchall()],
        }
        committee_candidates = conversation_candidates
    if resume_issue:
        portfolio_committee = {"status": "REEVALUATION_REQUIRED", "error": resume_issue,
                               "rejected_input_id": rejected_input_id}
    elif committee_candidates:
        if pending_package is None:
            save_pending_package(artifacts_dir, conversation_package)
        # One turn's worth, not the whole conversation: legacy_b_raw_conversation
        # now re-checks the soft threshold before every turn (see its enforce_llm_budget
        # arg), so this only has to decide whether starting is affordable at all.
        # The old 80K/24K estimate was ~$0.80 against measured turns of $2-8.
        committee_budget_blocked = enforce_llm_budget and would_exceed_budget(
            conn,
            settings,
            model=resolve_judge_model(settings),
            prompt_tokens=TURN_PROMPT_TOKENS_ESTIMATE,
            completion_tokens=TURN_COMPLETION_TOKENS_ESTIMATE,
        )
        if committee_budget_blocked:
            portfolio_committee = {
                "status": "BUDGET_EXCEEDED",
                "note": "자동 실행의 UTC 일일 LLM 예산 기준값에 도달했습니다(soft limit).",
            }
        else:
            _emit(
                conn,
                RuntimeEventKind.LLM_JUDGE,
                RuntimeEventStatus.PROGRESS,
                f"legacy-b 방식 GPT 연속 대화 · 종목 {len(committee_candidates)}개 + CASH"
                + (" (이어서)" if pending_package is not None else ""),
            )
            portfolio_committee = legacy_b_raw_conversation(
                settings,
                conversation_package,
                conn=conn,
                enforce_llm_budget=enforce_llm_budget,
            )
            committee_prompt_n, committee_completion_n = usage_from_payload(portfolio_committee)
            record_usage(
                conn,
                provider="openai",
                model=resolve_judge_model(settings),
                kind="portfolio_judge",
                ticker="PORTFOLIO",
                prompt_tokens=committee_prompt_n,
                completion_tokens=committee_completion_n,
                toward_daily_cap=enforce_llm_budget,
            )
            for turn in portfolio_committee.get("turns") or []:
                if not isinstance(turn, dict):
                    continue
                insert_llm_transcript(
                    conn,
                    tick_id=str(tick_id),
                    kind=f"legacy_b_{turn.get('stage') or 'turn'}",
                    status=STATUS_AVAILABLE,
                    ticker="PORTFOLIO",
                    model=str(turn.get("model") or resolve_judge_model(settings)),
                    prompt_version=LEGACY_B_RAW_PROMPT_VERSION,
                    system_prompt=LEGACY_B_RAW_SYSTEM,
                    user_prompt=str(turn.get("prompt") or ""),
                    response_text=str(turn.get("response") or ""),
                    prompt_tokens=turn.get("prompt_tokens"),
                    completion_tokens=turn.get("completion_tokens"),
                    total_tokens=turn.get("total_tokens"),
                )
    effective_recs = list(alloc["recommendations"])
    portfolio_committee = dict(portfolio_committee)
    if portfolio_committee.get("status") == STATUS_AVAILABLE:
        raw_final = str(portfolio_committee.get("raw_final_text") or "")
        if not raw_final and portfolio_committee.get("structured_final"):
            raw_final = json.dumps(portfolio_committee["structured_final"], ensure_ascii=False)
        raw_final = raw_final or str(portfolio_committee.get("final_text") or "")
        try:
            llm_recs, effective_recs = apply_final_judgment(
                raw_final, conversation_package, alloc["recommendations"], settings=settings, btc_state=btc_state
            )
        except (ValueError, KeyError, TypeError) as exc:
            portfolio_committee.update(status="INVALID_OUTPUT", error=str(exc), raw_final_text=raw_final)
        else:
            alloc["payload"] = allocation_for_records(effective_recs, research_allocation)
            portfolio_committee.update(
                validated=True, contract_version=CONTRACT_VERSION, raw_final_text=raw_final,
                final_text=render_final_report(llm_recs, settings=settings, allocation=alloc["payload"], input_as_of=input_as_of.isoformat()),
            )
            clear_pending_package(artifacts_dir)
    if not portfolio_committee.get("validated"):
        portfolio_committee.update(
            final_text="", validated=False,
            fallback="research_adjusted",
            note="GPT 최종 판단 미적용. Gemini 반영 수치 결과를 대체 표시합니다. 중간 답변·이전 실행 판단은 적용하지 않습니다.",
        )
        effective_recs = [r.model_copy(update={"decision_status": "FALLBACK_RESEARCH_ADJUSTED"}) for r in alloc["recommendations"]]
        if resume_issue:
            portfolio_committee["note"] = "이전 입력의 보유 상태를 복원할 수 없어 재평가가 필요합니다. 대체 결과는 실행 불가입니다."
            effective_recs = [r.model_copy(update={"actionable": False, "decision_status": "REEVALUATION_REQUIRED"}) for r in effective_recs]
            alloc["payload"].update(actionable=False, weights_confirmed=False,
                                     allocation_status="REEVALUATION_REQUIRED", stock_weights={}, btc_weight=None, cash_weight=None)
    # Raw turn transcripts remain separate from the applied, validated result.
    # Persist the canonical committee payload (including any explicit fallback).
    if isinstance(portfolio_committee.get("_exchange"), dict):
        portfolio_committee["_exchange"]["raw_response"] = None
    persist_exchange(
        conn, portfolio_committee, tick_id=str(tick_id), kind="portfolio_judge", ticker="PORTFOLIO",
        prompt_version=LEGACY_B_RAW_PROMPT_VERSION, default_system=LEGACY_B_RAW_SYSTEM,
        default_user="\n\n".join(str(t.get("prompt") or "") for t in portfolio_committee.get("turns", [])),
    )
    persist_allocation(conn, alloc["payload"])
    llm_judge_status = str(portfolio_committee.get("status") or STATUS_UNAVAILABLE)
    if llm_judge_status == "NOT_RUN":
        llm_judge_status = STATUS_NOT_CONFIGURED if not openai_configured(settings) else STATUS_UNAVAILABLE
    judged_n = len(committee_candidates) if llm_judge_status == STATUS_AVAILABLE else 0
    skipped_n = max(0, len(alloc["recommendations"]) - len(committee_candidates))
    researched_n = 0

    llm_rt = {
        STATUS_AVAILABLE: RuntimeEventStatus.SUCCEEDED,
        STATUS_NOT_CONFIGURED: RuntimeEventStatus.NOT_CONFIGURED,
        STATUS_UNAVAILABLE: RuntimeEventStatus.UNAVAILABLE,
        STATUS_DEGRADED: RuntimeEventStatus.DEGRADED,
        STATUS_RATE_LIMITED: RuntimeEventStatus.DEGRADED,
        STATUS_QUOTA_EXCEEDED: RuntimeEventStatus.UNAVAILABLE,
    }.get(llm_judge_status, RuntimeEventStatus.DEGRADED)
    _emit(
        conn,
        RuntimeEventKind.LLM_JUDGE,
        llm_rt,
        f"GPT-5.6 Sol FINAL JUDGE: {llm_judge_status}. Quant: {quant_status}. "
        f"judged={judged_n} skipped={skipped_n} researched={researched_n} "
        f"portfolio_committee={portfolio_committee.get('status')}",
    )

    alerts = []
    if market_actionable:
        for rec in effective_recs[:3]:
            agr, _ = horizon_agreement({h.horizon: h.expected_return or 0.0 for h in rec.horizons})
            px = _last_close(conn, str(rec.instrument_id))
            if px is None:
                continue
            alerts.extend(
                evaluate_alerts(
                    price=px,
                    expected=rec.horizons[0].expected_return or 0.0,
                    vol=0.02,
                    confidence=rec.confidence or 0.5,
                    agreement=agr,
                    instrument_id=rec.instrument_id,
                    tick_id=str(tick_id),
                    btc_blocked=btc_state.transfer_blocks_immediate_rebalance(),
                )
            )
    health_row = conn.execute(
        "SELECT overall FROM data_health_snapshots ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    alerts.extend(
        collect_cycle_notifications(
            recommendations=list(effective_recs),
            tick_id=str(tick_id),
            data_status=market_status,
            llm_status=llm_judge_status,
            market_status=market_status,
            alpaca_configured=_alpaca_configured(settings),
            health_overall=str(health_row[0]) if health_row else None,
            settings=settings,
            total_base_units=float((alloc.get("payload") or {}).get("total_base_units") or 1000.0),
        )
    )
    for a in alerts:
        try:
            console_notify(a)
        except Exception:  # noqa: BLE001 — notifications must never abort a scan
            pass
    try:
        digest = format_scan_kakao_digest(
            final_recs=list(llm_recs),
            quant_recs=list(alloc["recommendations"]),
            llm_status=llm_judge_status,
            final_text=str(portfolio_committee.get("final_text") or ""),
        )
        notify_optional_channels(
            settings,
            AlertRecord(
                kind=AlertKind.RECOMMENDATION,
                tick_id=str(tick_id),  # type: ignore[arg-type]
                message=digest,
            ),
        )
    except Exception:  # noqa: BLE001 — Kakao must never abort a scan
        pass

    decision_snapshot = {
        "tick_id": str(tick_id), "input_id": input_id, "input_as_of": input_as_of.isoformat(),
        "decision_contract": CONTRACT_VERSION, "daily_feature_basis": DAILY_BASIS,
        "allocation": alloc["payload"],
        "quant": [r.model_dump(mode="json") for r in quant_alloc["recommendations"]],
        "research_adjusted": [r.model_dump(mode="json") for r in alloc["recommendations"]],
        "llm_final": [r.model_dump(mode="json") for r in llm_recs],
        "effective": [r.model_dump(mode="json") for r in effective_recs],
        "llm_judge_status": llm_judge_status, "quant_status": quant_status,
        "portfolio_committee": {k: v for k, v in portfolio_committee.items() if k != "_exchange"},
        "capabilities": {"quant": quant_status, "llm_final_judge": llm_judge_status, "market_data": market_status,
                         "sec": sec_pack["sec_status"], "llm_extract": sec_pack["extract_status"]},
        "portfolio": portfolio.model_dump(mode="json"), "no_trading": True,
        "used_synthetic_market": used_synthetic,
    }
    rec_payloads = []
    for r in quant_alloc["recommendations"] + alloc["recommendations"] + llm_recs:
        rec_payloads.append(
            {
                "recommendation_id": r.recommendation_id,
                "source": str(getattr(r.source, "value", r.source)),
                "instrument_id": str(r.instrument_id),
                "action": str(getattr(r.action, "value", r.action)),
                "payload": json.loads(r.model_dump_json()),
            }
        )

    commit_tick(
        conn,
        TickCommitPayload(
            tick_id=tick_id,
            decision_epoch_id=str(quant_alloc["recommendations"][0].decision_epoch_id) if quant_alloc["recommendations"] else str(epoch.decision_epoch_id),
            feature_snapshot_id=str(quant_alloc["recommendations"][0].feature_snapshot_id) if quant_alloc["recommendations"] else str(fs_id),
            observations=[{"instrument_id": "cycle", "note": "v1_cycle"}, {
                "instrument_id": "decision_comparison_v1", "input": conversation_package,
                "snapshot": decision_snapshot,
                "allocations": {"quant_only": quant_alloc["payload"], "research_adjusted": research_allocation,
                                "effective": alloc["payload"]},
            }],
            predictions=[{"instrument_id": "cycle", "n_bundles": len(bundles)}],
            recommendations=rec_payloads,
            watermarks=[{"watermark_key": "cycle", "instrument_id": None, "value": last.isoformat()}],
            alert_outbox=[
                {
                    "outbox_id": f"out_{uuid4().hex[:10]}",
                    "alert_id": a.alert_id,
                    "channel": "console",
                    "idempotency_key": f"{tick_id}:{a.alert_id}:console",
                    "status": "sent",
                    "attempts": 1,
                    "payload": {"msg": a.message},
                }
                for a in alerts
            ],
        ),
    )
    _emit(conn, RuntimeEventKind.TICK_COMMIT, RuntimeEventStatus.SUCCEEDED, "tick committed")
    try:
        db = settings.duckdb_path
        if not db.is_absolute():
            db = discover_project_root() / db
        db = settings.duckdb_path
        if not db.is_absolute():
            db = discover_project_root() / db
        session = equity_session_date(datetime.now(timezone.utc))
        mark_session_run(db, session)
        if retrain:
            key = week_train_session(week_friday(session)) or week_friday(session)
            mark_weekly_train(db, key)
    except Exception:  # noqa: BLE001 — schedule watermark must not abort the tick
        pass
    portfolio = mark_positions(conn, load_portfolio(conn))
    capabilities = {
        "market_data": market_status,
        "quant": quant_status,
        "sec": sec_pack["sec_status"],
        "llm_extract": sec_pack["extract_status"],
        "llm_final_judge": llm_judge_status,
    }
    return {
        "tick_id": tick_id,
        "allocation": alloc["payload"],
        "quant": [r.model_dump(mode="json") for r in alloc["recommendations"]],
        "llm_final": [r.model_dump(mode="json") for r in llm_recs],
        "llm_judge_status": llm_judge_status,
        "portfolio_committee": {
            key: value for key, value in portfolio_committee.items() if key != "_exchange"
        },
        "quant_status": quant_status,
        "capabilities": capabilities,
        "sec": {
            "status": sec_pack["sec_status"],
            "extract_status": sec_pack["extract_status"],
            "filings": [
                {
                    "ticker": f.get("ticker"),
                    "form": f.get("form"),
                    "accession": f.get("accession"),
                    "accepted_at": f.get("accepted_at"),
                    "status": f.get("status"),
                    "note": f.get("note"),
                }
                for f in sec_pack["filings"]
            ],
            "errors": sec_pack["errors"],
        },
        "alerts": [a.model_dump(mode="json") for a in alerts],
        "portfolio": portfolio.model_dump(mode="json"),
        "historical_vs_live": {
            "historical": "walk-forward diagnostics only",
            "live": "versioned quant/research/validated final decisions with frozen input provenance",
        },
        "survivorship_warning": "Smoke universe is not the full S&P-500; survivorship bias possible.",
        "no_trading": True,
        "used_synthetic_market": used_synthetic,
        **decision_snapshot,
    }


def load_last_ui_snapshot(conn) -> dict | None:
    """Rebuild the dashboard/recs payload from the last committed tick (survives process restart)."""
    row = conn.execute(
        "SELECT tick_id FROM ticks WHERE status = 'committed' ORDER BY committed_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    tick_id = row[0]
    comparison = conn.execute(
        "SELECT payload_json FROM tick_observations WHERE tick_id=? AND instrument_id='decision_comparison_v1'", [tick_id]
    ).fetchone()
    if comparison:
        # The atomic tick snapshot owns the displayed result. Never combine a newer
        # allocation or retried transcript with an older committed recommendation.
        return json.loads(comparison[0])["snapshot"]
    rec_rows = conn.execute(
        "SELECT source, payload_json FROM tick_recommendations WHERE tick_id = ?",
        [tick_id],
    ).fetchall()
    if not rec_rows:
        return None
    quant: list[dict] = []
    llm: list[dict] = []
    for source, payload_json in rec_rows:
        try:
            wrapper = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError):
            continue
        body = wrapper.get("payload") if isinstance(wrapper, dict) else None
        rec = body if isinstance(body, dict) else wrapper
        if not isinstance(rec, dict):
            continue
        src = str(source or rec.get("source") or "")
        if src == "research_adjusted":
            continue
        if "llm" in src.lower():
            llm.append(rec)
        else:
            quant.append(rec)
    alloc_row = conn.execute(
        "SELECT payload_json FROM allocation_snapshots ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    allocation: dict = {}
    if alloc_row and alloc_row[0]:
        try:
            allocation = json.loads(alloc_row[0])
        except (TypeError, json.JSONDecodeError):
            allocation = {}
    judged = [r for r in llm if "NOT_A_CANDIDATE" not in (r.get("override_reasons") or [])]

    class _Reason:
        def __init__(self, reasons: list) -> None:
            self.override_reasons = reasons

    llm_status = (
        _aggregate_llm_status(
            configured=True,
            recs=[_Reason(r.get("override_reasons") or []) for r in judged],
        )
        if judged
        else "UNKNOWN"
    )
    actionable = any(bool(r.get("actionable")) for r in quant)
    quant_status = "AVAILABLE" if actionable else "UNKNOWN"
    portfolio_committee: dict[str, object] = {"status": "NOT_RUN"}
    try:
        committee_row = conn.execute(
            """
            SELECT status, response_text, error
            FROM llm_transcripts
            WHERE tick_id = ? AND kind = 'portfolio_judge'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [tick_id],
        ).fetchone()
    except Exception:  # noqa: BLE001 — older/read-only databases may lack transcript columns
        committee_row = None
    if committee_row:
        try:
            parsed_committee = json.loads(committee_row[1]) if committee_row[1] else {}
        except (TypeError, json.JSONDecodeError):
            parsed_committee = {"final_text": str(committee_row[1] or "").strip()}
        if isinstance(parsed_committee, dict):
            portfolio_committee = parsed_committee
        portfolio_committee.setdefault("status", str(committee_row[0] or "UNAVAILABLE"))
        if committee_row[2] and not portfolio_committee.get("error"):
            portfolio_committee["error"] = str(committee_row[2])
    if portfolio_committee.get("status"):
        llm_status = str(portfolio_committee["status"])
    return {
        "tick_id": tick_id,
        "allocation": allocation,
        "quant": quant,
        "llm_final": llm,
        "llm_judge_status": llm_status,
        "portfolio_committee": portfolio_committee,
        "quant_status": quant_status,
        "capabilities": {
            "quant": quant_status,
            "llm_final_judge": llm_status,
        },
        "no_trading": True,
        "used_synthetic_market": False,
    }
