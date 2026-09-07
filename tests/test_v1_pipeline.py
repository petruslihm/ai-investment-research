"""V1 pipeline tests: features, models, allocation, no-trading, cycle."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from trading_system.allocation import allocate
from trading_system.recommendations import RecommendationAction
from trading_system.btc_sleeve import BtcLiquidityState, BtcSleeveState
from trading_system.config import Settings
from trading_system.features import (
    assert_no_future_in_features,
    build_and_persist_features,
    features_cover_latest,
    load_matrix,
)
from trading_system.ml_engine import load_fitted_bundles, bundles_ready
from trading_system.ids import new_tick_id
from trading_system.market.calendar import add_sessions, is_trading_day
from trading_system.market.registry import stable_instrument_id
from trading_system.sec_llm import score_overrides
from trading_system.seed import seed_synthetic_history
from trading_system.storage import Store
from trading_system.v1_cycle import load_last_ui_snapshot, refresh_market_history, run_v1_cycle


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        _env_file=None,
        alpaca_api_key=None,
        alpaca_secret_key=None,
        openai_api_key=None,
        gemini_api_key=None,
        anthropic_api_key=None,
        smoke_universe=("SPY", "AAPL", "MSFT", "NVDA", "BTC/USD"),
    )


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "v1.duckdb")
    s.open(acquire_writer=True)
    yield s
    s.close()


def test_equity_and_btc_calendars_do_not_mix() -> None:
    d = date(2024, 6, 14)
    sess = add_sessions(d, 5)
    cal = d.fromordinal(d.toordinal() + 5)
    assert sess != cal or is_trading_day(d)
    # BTC maturity must not call add_sessions — calendar +5 is independent
    assert (cal - d).days == 5


def test_features_labels_and_no_immature_training(store: Store, settings: Settings) -> None:
    seed_synthetic_history(store, settings, days=80, end=date(2024, 6, 14))
    last = store.conn.execute("SELECT max(session_date) FROM equity_daily_bars").fetchone()[0]
    build_and_persist_features(store.conn, last_available=last)
    assert_no_future_in_features(store.conn)
    immature = store.conn.execute(
        "SELECT COUNT(*) FROM label_rows WHERE matured AND label_available_date > ?",
        [last],
    ).fetchone()[0]
    assert immature == 0
    btc_uses_session_helper = store.conn.execute(
        "SELECT COUNT(*) FROM label_rows WHERE asset_class = 'btc' AND horizon = 5"
    ).fetchone()[0]
    assert btc_uses_session_helper > 0
    assert features_cover_latest(store.conn, last_available=last, last_available_btc=last)
    newer = last + timedelta(days=10)
    assert not features_cover_latest(store.conn, last_available=newer, last_available_btc=newer)
    x_all, _y_all, keys_all = load_matrix(
        store.conn, asset_class="us_equity", horizon=5, matured_only=False
    )
    x_last, _y_last, keys_last = load_matrix(
        store.conn, asset_class="us_equity", horizon=5, matured_only=False, as_of=last
    )
    assert len(keys_last) < len(keys_all)
    assert keys_last
    assert all(d == last for _i, d in keys_last)


def test_vectorized_features_match_scalar() -> None:
    from trading_system.features import _feature_columns, _features_at

    series = [
        (date(2024, 1, 2) + timedelta(days=i), 100.0 + i * 0.4 + (i % 5) * 0.2, 1_000_000.0 + i * 500)
        for i in range(40)
    ]
    cols = _feature_columns(series)
    for i in range(20, len(series)):
        scalar = _features_at(series, i)
        for name, value in scalar.items():
            assert cols[name][i] == pytest.approx(value, rel=1e-12, abs=1e-12)


def test_rebuild_skips_instruments_already_at_watermark(store: Store, settings: Settings) -> None:
    seed_synthetic_history(store, settings, days=80, end=date(2024, 6, 14))
    last = store.conn.execute("SELECT max(session_date) FROM equity_daily_bars").fetchone()[0]
    first = build_and_persist_features(store.conn, last_available=last)
    rows_before = store.conn.execute("SELECT COUNT(*) FROM feature_rows").fetchone()[0]
    second = build_and_persist_features(store.conn, last_available=last)
    rows_after = store.conn.execute("SELECT COUNT(*) FROM feature_rows").fetchone()[0]
    assert first > 0
    assert second == 0
    assert rows_after == rows_before


def test_allocation_can_be_100_percent_cash(settings: Settings) -> None:
    btc = BtcSleeveState(
        instrument_id=stable_instrument_id("BTC/USD"),
        liquidity=BtcLiquidityState.UNAVAILABLE,
        current_units=0,
    )
    out = allocate(
        stock_scores={"inst_aapl": {5: -0.2, 10: -0.2, 20: -0.2}},
        stock_vol={"inst_aapl": 0.05},
        btc_scores={5: -0.1, 10: -0.1, 20: -0.1},
        settings=settings,
        btc_state=btc,
        tick_id=str(new_tick_id()),
        epoch_id="epoch_x",
        feature_snapshot_id="fs_x",
    )
    assert abs(out["payload"]["cash_weight"] - 1.0) < 1e-6
    assert abs(sum(out["payload"]["stock_weights"].values()) + out["payload"]["btc_weight"] + out["payload"]["cash_weight"] - 1.0) < 1e-6


def test_allocate_keeps_unscored_holding(settings: Settings) -> None:
    btc = BtcSleeveState(
        instrument_id=stable_instrument_id("BTC/USD"),
        liquidity=BtcLiquidityState.UNAVAILABLE,
        current_units=0,
    )
    out = allocate(
        stock_scores={"inst_other": {5: 0.2, 10: 0.2, 20: 0.2}},
        stock_vol={"inst_other": 0.02, "inst_amd": 0.02},
        btc_scores={5: -0.1, 10: -0.1, 20: -0.1},
        settings=settings,
        btc_state=btc,
        tick_id=str(new_tick_id()),
        epoch_id="epoch_x",
        feature_snapshot_id="fs_x",
        stock_units={"inst_amd": 4.0},
    )
    ids = {str(r.instrument_id) for r in out["recommendations"]}
    assert "inst_amd" in ids
    held = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_amd")
    assert held.action == RecommendationAction.HOLD
    assert held.current_units == pytest.approx(4.0)
    assert held.recommended_units == pytest.approx(4.0)


def test_override_scoring_separate_from_backtest() -> None:
    s = score_overrides(0.01, -0.02)
    assert s["scored_as"].startswith("live")
    assert s["llm_reliance"] < 1.0


def _assert_journal_records_facts(store: Store) -> None:
    """The Models page can only stay honest if these fields are actually persisted."""
    import json as _json

    rows = store.conn.execute(
        "SELECT kind, payload_json FROM model_change_journal ORDER BY created_at"
    ).fetchall()
    payloads = [(kind, _json.loads(raw)) for kind, raw in rows]
    assert payloads

    promoted = [p for kind, p in payloads if kind == "promoted"]
    assert promoted, "training must record a promotion event"
    for p in promoted:
        assert p["asset_class"] in {"us_equity", "btc"}
        assert p["horizon"] in (5, 10, 20)
        assert isinstance(p["matured_label_count"], int)
        assert p["promotion_result"] == "accepted"
        assert " ~ " in (p["training_period"] or ""), "training window must be real dates"

    for kind, p in payloads:
        version = str(p.get("new_version") or "")
        assert "_adapt" not in version, f"version chain leaked into the journal: {version}"

    online = [p for kind, p in payloads if kind == "online_updated"]
    for p in online:
        assert isinstance(p["matured_label_count"], int)
        assert p["asset_class"] in {"us_equity", "btc"}

    weights = [p for kind, p in payloads if kind == "ensemble_weight_changed"]
    for p in weights:
        assert p["previous_ensemble_weight"] is not None
        assert p["new_ensemble_weight"] is not None
        assert p["reason_codes"]


def test_full_cycle_no_broker(
    store: Store, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "trading_system.v1_cycle.ingest_sec_extracts",
        lambda *_a, **_k: {
            "sec_status": "UNAVAILABLE",
            "extract_status": "NOT_CONFIGURED",
            "llm_configured": False,
            "filings": [],
            "errors": ["test: live EDGAR skipped"],
            "used_lkg": False,
        },
    )
    out = run_v1_cycle(store, settings, artifacts_dir=tmp_path / "art")
    assert out["no_trading"] is True
    assert out["used_synthetic_market"] is True
    assert out["quant_status"] == "NOT_CONFIGURED"
    assert out["capabilities"]["quant"] == "NOT_CONFIGURED"
    assert out["allocation"]["actionable"] is False
    assert all(not r.get("actionable", True) for r in out["quant"])
    assert out["quant"]
    assert out["llm_final"] == []
    assert out["portfolio_committee"]["status"] == "NOT_CONFIGURED"
    assert "5" in str(out["quant"])
    journal = store.conn.execute("SELECT COUNT(*) FROM model_change_journal").fetchone()[0]
    assert journal >= 1
    _assert_journal_records_facts(store)
    ticks = store.conn.execute("SELECT COUNT(*) FROM ticks WHERE status='committed'").fetchone()[0]
    assert ticks >= 1
    epochs = store.conn.execute("SELECT COUNT(*) FROM decision_epochs").fetchone()[0]
    assert epochs >= 1
    epoch_hash = store.conn.execute("SELECT ensemble_weights_hash FROM decision_epochs LIMIT 1").fetchone()[0]
    assert epoch_hash != "init"
    preds = store.conn.execute("SELECT COUNT(*) FROM prediction_rows").fetchone()[0]
    assert preds >= 1
    outcomes = store.conn.execute("SELECT COUNT(*) FROM outcome_snapshots").fetchone()[0]
    assert outcomes >= 1
    fs = store.conn.execute(
        "SELECT adjustment_revision, provider FROM feature_snapshots ORDER BY published_at DESC LIMIT 1"
    ).fetchone()
    assert fs[0] == "fixture_rev_1"
    assert fs[1] == "not_configured_synthetic"
    stock_rec = next(r for r in out["quant"] if "btc" not in str(r.get("instrument_id")))
    assert stock_rec.get("recommended_units") is not None
    assert stock_rec.get("current_units") is not None
    assert stock_rec.get("delta_units") is not None
    restored = load_last_ui_snapshot(store.conn)
    assert restored is not None
    assert len(restored["quant"]) == len(out["quant"])
    assert len(restored["llm_final"]) == len(out["llm_final"])


def test_empty_artifacts_are_not_ready(tmp_path: Path) -> None:
    bundles = load_fitted_bundles(tmp_path / "missing")
    assert bundles == {}
    assert not bundles_ready(bundles)


def test_second_cycle_skips_training(
    store: Store, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "trading_system.v1_cycle.ingest_sec_extracts",
        lambda *_a, **_k: {
            "sec_status": "UNAVAILABLE",
            "extract_status": "NOT_CONFIGURED",
            "llm_configured": False,
            "filings": [],
            "errors": ["test: live EDGAR skipped"],
            "used_lkg": False,
        },
    )
    art = tmp_path / "art2"
    run_v1_cycle(store, settings, artifacts_dir=art)

    def boom(*_a, **_k):
        raise AssertionError("scan must reuse saved models")

    monkeypatch.setattr("trading_system.v1_cycle.train_batch_models", boom)
    out = run_v1_cycle(store, settings, artifacts_dir=art)
    assert out["quant"]


def test_needs_llm_judge_only_weight_or_holding() -> None:
    from trading_system.recommendations import RecommendationAction, quant_only_record
    from trading_system.v1_cycle import _needs_llm_judge

    blank = quant_only_record(
        tick_id="tick_1",
        decision_epoch_id="epoch_1",
        feature_snapshot_id="fs_1",
        instrument_id="inst_aapl",
        action=RecommendationAction.BUY,
        current_units=0,
        recommended_units=0,
    )
    assert not _needs_llm_judge(blank, {})
    assert _needs_llm_judge(blank.model_copy(update={"recommended_units": 8.0}), {})
    assert _needs_llm_judge(blank.model_copy(update={"current_units": 2.0, "action": RecommendationAction.HOLD}), {})
    assert not _needs_llm_judge(
        blank.model_copy(update={"action": RecommendationAction.HOLD, "recommended_units": 8.0}),
        {"inst_aapl": 0.05},
    )
    assert not _needs_llm_judge(blank, {"inst_aapl": 0.05})


def test_cycle_sends_only_candidates_to_legacy_b_conversation(
    store: Store, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judged: list[str] = []

    def fake_conversation(_settings, package, **_kwargs):
        judged.extend(str(row["instrument_id"]) for row in package["candidates"])
        return {"status": "AVAILABLE", "final_text": "최종 원문", "turns": []}

    monkeypatch.setattr("trading_system.v1_cycle.legacy_b_raw_conversation", fake_conversation)
    monkeypatch.setattr(
        "trading_system.v1_cycle.ingest_sec_extracts",
        lambda *_a, **_k: {
            "sec_status": "UNAVAILABLE",
            "extract_status": "NOT_CONFIGURED",
            "llm_configured": False,
            "filings": [],
            "errors": ["test: live EDGAR skipped"],
            "used_lkg": False,
        },
    )
    out = run_v1_cycle(store, settings, artifacts_dir=tmp_path / "art")
    assert judged, "expected at least one candidate in the GPT conversation"
    assert len(judged) < len(out["quant"])
    judged_set = set(judged)
    buyish = {"BUY", "ENTER", "ADD"}
    for rec in out["quant"]:
        inst = str(rec.get("instrument_id"))
        action = str(rec.get("action"))
        need = float(rec.get("current_units") or 0) > 1e-12 or (
            action in buyish and float(rec.get("recommended_units") or 0) > 1e-12
        )
        if need:
            assert inst in judged_set
        else:
            assert inst not in judged_set


def test_cycle_caps_legacy_b_conversation_to_max_names(
    store: Store, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judged: list[str] = []

    def fake_conversation(_settings, package, **_kwargs):
        judged.extend(str(row["instrument_id"]) for row in package["candidates"])
        return {"status": "AVAILABLE", "final_text": "최종 원문", "turns": []}

    monkeypatch.setattr("trading_system.v1_cycle.legacy_b_raw_conversation", fake_conversation)
    monkeypatch.setattr(
        "trading_system.v1_cycle.ingest_sec_extracts",
        lambda *_a, **_k: {
            "sec_status": "UNAVAILABLE",
            "extract_status": "NOT_CONFIGURED",
            "llm_configured": False,
            "filings": [],
            "errors": ["test: live EDGAR skipped"],
            "used_lkg": False,
        },
    )
    tight = settings.model_copy(update={"openai_api_key": "sk-test", "llm_research_max_names": 1})
    out = run_v1_cycle(store, tight, artifacts_dir=tmp_path / "art_cap")
    assert len(judged) <= 1
    candidates = [
        r
        for r in out["quant"]
        if float(r.get("current_units") or 0) > 1e-12
        or (
            str(r.get("action")) in {"BUY", "ENTER", "ADD"}
            and float(r.get("recommended_units") or 0) > 1e-12
        )
    ]
    if candidates:
        assert len(judged) == 1


def test_cycle_persists_legacy_b_raw_final_answer(
    store: Store, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_conversation(_settings, package, **_kwargs):
        assert package["candidates"]
        return {
            "status": "AVAILABLE",
            "final_text": "# 최종 판단\n\n보유: 유지\n신규: 관망\n현금: 20%",
            "turns": [],
        }

    monkeypatch.setattr("trading_system.v1_cycle.legacy_b_raw_conversation", fake_conversation)
    monkeypatch.setattr(
        "trading_system.v1_cycle.ingest_sec_extracts",
        lambda *_a, **_k: {
            "sec_status": "UNAVAILABLE",
            "extract_status": "NOT_CONFIGURED",
            "llm_configured": True,
            "filings": [],
            "errors": ["test: live EDGAR skipped"],
            "used_lkg": False,
        },
    )
    live = settings.model_copy(
        update={"openai_api_key": "sk-test", "llm_final_max_names": 3}
    )
    out = run_v1_cycle(store, live, artifacts_dir=tmp_path / "art_committee")
    assert out["portfolio_committee"]["status"] == "AVAILABLE"
    assert out["portfolio_committee"]["final_text"].startswith("# 최종 판단")
    assert out["llm_final"] == []
    restored = load_last_ui_snapshot(store.conn)
    assert restored is not None
    assert restored["portfolio_committee"]["status"] == "AVAILABLE"
    assert "현금: 20%" in restored["portfolio_committee"]["final_text"]


def test_cycle_researches_only_new_entries_and_moves_real_sizing(
    store: Store, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The revived Gemini research pass (research_agent.research_ticker) must: (1)
    never be called for a name the user already holds -- research_sizing_multiplier
    is a no-op for held names anyway, so researching them would only burn budget --
    and (2) actually reach the SECOND allocate() pass's persisted diagnostics, not
    just sit unused in a GPT-prompt footnote."""
    from datetime import datetime, timezone

    from trading_system.ids import InstrumentId
    from trading_system.market.repository import upsert_equity_daily_bars
    from trading_system.portfolio import Lot
    from trading_system.portfolio_service import add_lot
    from trading_system.providers.interfaces import BarFinality, DailyBar
    from trading_system.seed import trading_days_ending

    seed_synthetic_history(store, settings, days=80, end=date(2024, 6, 14))
    aapl = stable_instrument_id("AAPL")
    add_lot(
        store.conn,
        # acquisition_price matches the fixture's actual AAPL close range (~80) --
        # too low a price makes mark_positions scale current_units into a fantasy
        # value that alone exceeds total_base_units, starving every other name of
        # equity budget before research even gets a chance to matter.
        Lot(instrument_id=InstrumentId(str(aapl)), acquisition_units=10, acquisition_price=80.0, acquired_on=date(2024, 1, 2)),
    )
    # seed_synthetic_history gives every equity nearly the same flat slope, so none of
    # them show genuine excess return over SPY (the label is stock_return - SPY_return)
    # and nothing is ever BUY-eligible. Give MSFT a real edge so pass 1 has an actual
    # new-entry candidate for research to act on.
    msft = stable_instrument_id("MSFT")
    now = datetime.now(timezone.utc)
    steep_bars = [
        DailyBar(
            instrument_id=msft, session_date=d, open=95.0 + i * 0.9 - 0.4, high=95.0 + i * 0.9 + 0.8,
            low=95.0 + i * 0.9 - 0.9, close=95.0 + i * 0.9, volume=1_000_000 + i * 1000,
            finality=BarFinality.FINAL, adjustment_revision="fixture_rev_1", provider="fixture", receive_ts=now,
        )
        for i, d in enumerate(trading_days_ending(date(2024, 6, 14), 80))
    ]
    upsert_equity_daily_bars(store.conn, steep_bars)

    researched_insts: list[str] = []

    def fake_research(_settings, quant, **_kwargs):
        researched_insts.append(str(quant.instrument_id))
        return {
            "status": "AVAILABLE",
            "web_search_used": True,
            "data_quality_score": 90.0,
            "rerating_score": 95.0,
            "valuation_support_score": 90.0,
            "cash_relative_score": 85.0,
            "bearish_severity_score": 0.0,
            "summary_ko": "test bullish pack",
        }

    monkeypatch.setattr("trading_system.v1_cycle.research_ticker", fake_research)
    monkeypatch.setattr(
        "trading_system.v1_cycle.legacy_b_raw_conversation",
        lambda _s, _pkg, **_kw: {"status": "AVAILABLE", "final_text": "최종 원문", "turns": []},
    )
    monkeypatch.setattr(
        "trading_system.v1_cycle.ingest_sec_extracts",
        lambda *_a, **_k: {
            "sec_status": "UNAVAILABLE", "extract_status": "NOT_CONFIGURED", "llm_configured": False,
            "filings": [], "errors": ["test: live EDGAR skipped"], "used_lkg": False,
        },
    )
    live = settings.model_copy(update={"openai_api_key": "sk-test", "gemini_api_key": "test-gemini"})
    out = run_v1_cycle(store, live, artifacts_dir=tmp_path / "art_research")

    assert str(aapl) not in researched_insts, "a held position must never be sent to research"
    assert not any("btc" in inst.lower() for inst in researched_insts), "BTC has no research_sizing_multiplier effect"
    new_entries = [
        r for r in out["quant"]
        if str(r.get("action")) in {"BUY", "ENTER"}
        and float(r.get("current_units") or 0) <= 1e-12
        and float(r.get("recommended_units") or 0) > 1e-12
    ]
    assert new_entries, "test universe/settings must produce at least one new-entry BUY to exercise research"
    assert researched_insts, "expected at least one new entry to have been researched"

    diagnostics = out["allocation"].get("diagnostics") or {}
    multipliers = [
        float(diagnostics[inst]["research_sizing_multiplier"])
        for inst in researched_insts
        if inst in diagnostics
    ]
    assert multipliers, "researched names must appear in the persisted allocation diagnostics"
    assert any(m > 1.0 for m in multipliers), "a strongly bullish pack must size up, not sit neutral"


def test_train_only_makes_no_api_calls_and_still_refreshes_market_data(
    store: Store, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_only=True (the daily $0 retrain path) must: (1) never reach SEC/Gemini/GPT
    -- monkeypatched to fail loudly if called, not just left unconfigured, so a silent
    regression can't slip back in; (2) still refresh market data, unlike plain
    retrain=True which assumes a scan already did that; (3) actually retrain (new
    ensemble_state row) and return before producing any recommendation."""
    seed_synthetic_history(store, settings, days=80, end=date(2024, 6, 14))

    def _fail(*_a: object, **_k: object) -> None:
        raise AssertionError("train_only must never reach this call")

    monkeypatch.setattr("trading_system.v1_cycle.ingest_sec_extracts", _fail)
    monkeypatch.setattr("trading_system.v1_cycle.research_ticker", _fail)
    monkeypatch.setattr("trading_system.v1_cycle.legacy_b_raw_conversation", _fail)

    refreshed: list[bool] = []
    orig_refresh = refresh_market_history

    def _tracking_refresh(*a: object, **k: object):
        refreshed.append(True)
        return orig_refresh(*a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr("trading_system.v1_cycle.refresh_market_history", _tracking_refresh)

    live = settings.model_copy(update={"openai_api_key": "sk-test", "gemini_api_key": "test-gemini"})
    out = run_v1_cycle(store, live, artifacts_dir=tmp_path / "art_train_only", retrain=True, train_only=True)

    assert out["train_only"] is True
    assert out["bundles_trained"] > 0
    assert refreshed, "train_only must still fetch market data, unlike plain retrain=True"
    assert "quant" not in out and "portfolio_committee" not in out

    row = store.conn.execute("SELECT weights_json FROM ensemble_state WHERE stream='us_equity'").fetchone()
    assert row is not None


def test_no_order_routes() -> None:
    from trading_system.ui.app import create_app

    app = create_app()
    paths = {getattr(r, "path", "") for r in app.routes}
    assert not any("order" in p or "broker" in p for p in paths)
    assert "/settings" in paths
    assert "/api/health" in paths


def test_btc_units_scale_with_base_units(settings: Settings) -> None:
    btc = BtcSleeveState(
        instrument_id=stable_instrument_id("BTC/USD"),
        liquidity=BtcLiquidityState.AVAILABLE,
        current_units=180,
    )
    out = allocate(
        stock_scores={},
        stock_vol={},
        btc_scores={5: 0.2, 10: 0.2, 20: 0.2},
        settings=settings,
        btc_state=btc,
        tick_id=str(new_tick_id()),
        epoch_id="epoch_x",
        feature_snapshot_id="fs_x",
        total_base_units=1000,
    )
    rec_u = out["recommended_btc_units"]
    assert rec_u == pytest.approx(out["payload"]["btc_weight"] * 1000)
    assert rec_u != pytest.approx(out["payload"]["btc_weight"] * settings.max_btc_weight * 10)


def test_cash_floor_survives_normalize(settings: Settings) -> None:
    s = settings.model_copy(update={"min_cash_weight": 0.2, "max_total_stock_weight": 0.8, "max_btc_weight": 0.3})
    inst = str(stable_instrument_id("AAPL"))
    out = allocate(
        stock_scores={inst: {5: 0.5, 10: 0.5, 20: 0.5}},
        stock_vol={inst: 0.02},
        btc_scores={5: 0.5, 10: 0.5, 20: 0.5},
        settings=s,
        btc_state=BtcSleeveState(instrument_id=stable_instrument_id("BTC/USD")),
        tick_id=str(new_tick_id()),
        epoch_id="epoch_x",
        feature_snapshot_id="fs_x",
        total_base_units=1000,
    )
    assert out["payload"]["cash_weight"] + 1e-9 >= 0.2
    assert abs(sum(out["payload"]["stock_weights"].values()) + out["payload"]["btc_weight"] + out["payload"]["cash_weight"] - 1.0) < 1e-6


def test_transfer_pending_keeps_btc_sleeve(settings: Settings) -> None:
    btc = BtcSleeveState(
        instrument_id=stable_instrument_id("BTC/USD"),
        liquidity=BtcLiquidityState.TRANSFER_PENDING,
        current_units=200,
    )
    out = allocate(
        stock_scores={},
        stock_vol={},
        btc_scores={5: 0.4, 10: 0.4, 20: 0.4},
        settings=settings,
        btc_state=btc,
        tick_id=str(new_tick_id()),
        epoch_id="epoch_x",
        feature_snapshot_id="fs_x",
        total_base_units=1000,
    )
    assert out["payload"]["btc_weight"] == pytest.approx(0.2)
    assert out["payload"]["cash_weight"] == pytest.approx(0.8)
    assert out["recommended_btc_units"] == pytest.approx(200)
    assert out["btc_action"].value == "NO_ACTION"


def test_zero_confidence_does_not_size_stock(settings: Settings) -> None:
    inst = str(stable_instrument_id("AAPL"))
    out = allocate(
        stock_scores={inst: {5: 0.5, 10: 0.5, 20: 0.5}},
        stock_vol={inst: 0.02},
        btc_scores={5: -0.1, 10: -0.1, 20: -0.1},
        settings=settings,
        btc_state=BtcSleeveState(instrument_id=stable_instrument_id("BTC/USD")),
        tick_id=str(new_tick_id()),
        epoch_id="epoch_x",
        feature_snapshot_id="fs_x",
        stock_confidence={inst: 0.0},
    )
    assert inst not in out["payload"]["stock_weights"]
    assert out["payload"]["cash_weight"] == pytest.approx(1.0)


def test_discover_project_root_is_repo() -> None:
    from trading_system.config import discover_project_root
    from trading_system.ui.app import PROJECT_ROOT

    root = discover_project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "trading_system" / "ui" / "app.py").is_file()
    assert PROJECT_ROOT == root


def test_refresh_retries_alpaca_when_fixture_exists(store: Store, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_synthetic_history(store, settings, days=20, end=date(2024, 6, 14))
    live = settings.model_copy(update={"alpaca_api_key": "k", "alpaca_secret_key": "s"})
    calls = {"n": 0}

    class FakeProvider:
        def provider_name(self) -> str:
            return "alpaca"

        def close(self) -> None:
            return None

    class FakeSvc:
        def __init__(self, conn, _settings, _provider, provider_name="alpaca"):
            self.conn = conn

        def startup_backfill(self, *, end=None):
            calls["n"] += 1
            self.conn.execute("UPDATE equity_daily_bars SET provider = 'alpaca'")
            self.conn.execute("UPDATE btc_daily_bars SET provider = 'alpaca'")

        def build_data_health_snapshot(self):
            return None

    monkeypatch.setattr("trading_system.v1_cycle.make_market_provider", lambda _s: FakeProvider())
    monkeypatch.setattr("trading_system.v1_cycle.MarketDataService", FakeSvc)
    status, synthetic = refresh_market_history(store, live)
    assert calls["n"] == 1
    assert status == "AVAILABLE"
    assert synthetic is False
    n_fix = store.conn.execute("SELECT COUNT(*) FROM equity_daily_bars WHERE provider='fixture'").fetchone()[0]
    assert n_fix == 0
    n_live = store.conn.execute("SELECT COUNT(*) FROM equity_daily_bars WHERE provider='alpaca'").fetchone()[0]
    assert n_live > 0


def test_refresh_keeps_fixture_when_alpaca_returns_nothing(
    store: Store, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_synthetic_history(store, settings, days=20, end=date(2024, 6, 14))
    live = settings.model_copy(update={"alpaca_api_key": "k", "alpaca_secret_key": "s"})
    n_before = store.conn.execute("SELECT COUNT(*) FROM equity_daily_bars").fetchone()[0]

    class FakeProvider:
        def provider_name(self) -> str:
            return "alpaca"

        def close(self) -> None:
            return None

    class FakeSvc:
        def __init__(self, *a, **k):
            return None

        def startup_backfill(self, *, end=None):
            return None

    monkeypatch.setattr("trading_system.v1_cycle.make_market_provider", lambda _s: FakeProvider())
    monkeypatch.setattr("trading_system.v1_cycle.MarketDataService", FakeSvc)
    status, synthetic = refresh_market_history(store, live)
    assert status == "UNAVAILABLE"
    assert synthetic is True
    n_after = store.conn.execute("SELECT COUNT(*) FROM equity_daily_bars").fetchone()[0]
    assert n_after == n_before


def test_parse_sec_submissions_extracts_accession() -> None:
    from trading_system.sec_edgar import _parse_submissions

    payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-K"],
                "accessionNumber": ["0000320193-26-000123", "0000320193-26-000001"],
                "acceptanceDateTime": ["2026-08-01T20:00:00-04:00", "2026-01-15T16:00:00-05:00"],
                "primaryDocument": ["aapl-8k.htm", "aapl-10k.htm"],
            }
        }
    }
    rows = _parse_submissions(payload, "0000320193", ("8-K", "10-Q"), 8)
    assert len(rows) == 1
    assert rows[0]["accession"] == "0000320193-26-000123"
    assert rows[0]["source"] == "live"


def test_btc_maturity_uses_btc_watermark(store: Store, settings: Settings) -> None:
    seed_synthetic_history(store, settings, days=80, end=date(2024, 6, 14))
    store.conn.execute("DELETE FROM btc_daily_bars WHERE session_date > DATE '2024-05-01'")
    last_eq = store.conn.execute("SELECT max(session_date) FROM equity_daily_bars").fetchone()[0]
    last_btc = store.conn.execute("SELECT max(session_date) FROM btc_daily_bars").fetchone()[0]
    assert last_btc < last_eq
    build_and_persist_features(store.conn, last_available=last_eq, last_available_btc=last_btc)
    leaked = store.conn.execute(
        """
        SELECT COUNT(*) FROM label_rows
        WHERE asset_class = 'btc' AND matured AND label_available_date > ?
        """,
        [last_btc],
    ).fetchone()[0]
    assert leaked == 0


def test_chronological_split_is_session_not_rows() -> None:
    from datetime import timedelta

    from trading_system.ml_engine import chronological_split

    keys = []
    d0 = date(2024, 1, 2)
    for i in range(20):
        d = d0 + timedelta(days=i)
        for inst in ("a", "b", "c"):
            keys.append((inst, d))
    n = len(keys)
    tr, te = chronological_split(n, purge=5, keys=keys)
    train_dates = {keys[i][1] for i in tr}
    test_dates = {keys[i][1] for i in te}
    if train_dates and test_dates:
        gap = (min(test_dates) - max(train_dates)).days
        assert gap >= 5


def test_lstm_sequences_globally_sorted(store: Store, settings: Settings) -> None:
    from trading_system.features import load_sequences
    from trading_system.ids import AssetClass

    seed_synthetic_history(store, settings, days=80, end=date(2024, 6, 14))
    last = store.conn.execute("SELECT max(session_date) FROM equity_daily_bars").fetchone()[0]
    build_and_persist_features(store.conn, last_available=last)
    _seq, _y, keys = load_sequences(
        store.conn, asset_class=AssetClass.US_EQUITY.value, horizon=5, lookback=10, matured_only=True
    )
    dates = [k[1] for k in keys]
    assert dates == sorted(dates)


def test_mark_positions_uses_price(store: Store, settings: Settings) -> None:
    from trading_system.ids import InstrumentId
    from trading_system.portfolio import Lot
    from trading_system.portfolio_service import add_lot, load_portfolio, mark_positions

    seed_synthetic_history(store, settings, days=30, end=date(2024, 6, 14))
    aapl = stable_instrument_id("AAPL")
    add_lot(
        store.conn,
        Lot(
            instrument_id=InstrumentId(str(aapl)),
            acquisition_units=100,
            acquisition_price=1.5,
            acquired_on=date(2024, 1, 2),
        ),
    )
    add_lot(
        store.conn,
        Lot(
            instrument_id=InstrumentId(str(aapl)),
            acquisition_units=50,
            acquisition_price=1.5,
            acquired_on=date(2024, 1, 3),
        ),
    )
    marked = mark_positions(store.conn, load_portfolio(store.conn))
    pos = next(p for p in marked.positions if str(p.instrument_id) == str(aapl))
    close = store.conn.execute(
        "SELECT close FROM equity_daily_bars WHERE instrument_id = ? ORDER BY session_date DESC LIMIT 1",
        [aapl],
    ).fetchone()[0]
    assert pos.marked_units_final == pytest.approx(150 * (float(close) / 1.5))
    assert pos.marked_units_final != pytest.approx(150)


def test_invalid_price_does_not_alert() -> None:
    from trading_system.alert_engine import evaluate_alerts
    from trading_system.ids import InstrumentId

    out = evaluate_alerts(
        price=-10.0,
        expected=0.05,
        vol=0.02,
        confidence=0.5,
        agreement="agree",
        instrument_id=InstrumentId("inst_aapl"),
        tick_id="tick_x",
    )
    assert out == []


def test_lambdarank_not_blended_into_expected_return() -> None:
    from trading_system.ml_engine import blend

    mixed = blend({"ridge": 0.01, "lambdarank": 50.0}, {"ridge": 0.5, "lambdarank": 0.5})
    only = blend({"ridge": 0.01}, {"ridge": 0.5, "lambdarank": 0.5})
    assert mixed == pytest.approx(only)


def test_update_weights_moves_off_equal_split() -> None:
    from trading_system.ml_engine import default_weights, update_weights
    from trading_system.ids import AssetClass

    w0 = default_weights(AssetClass.US_EQUITY.value)
    w1 = update_weights(w0, {"ridge": 0.01, "lightgbm_reg": 0.5, "torch_sequence": 0.5, "online": 0.5})
    assert w1["ridge"] > w0["ridge"]


def test_adapt_horizon_influence_favors_the_more_accurate_horizon() -> None:
    """5d has consistently lower matured error than 10d/20d across every family that
    feeds the live blend -> horizon_influence should shift trust toward 5d, away from
    its 1/3-1/3-1/3 default (this is what was previously computed and persisted but
    never actually applied to the mean() blend -- see allocation._weighted_horizon_mean)."""
    from trading_system.ml_engine import adapt_horizon_influence

    prev = {"5": 1 / 3, "10": 1 / 3, "20": 1 / 3}
    errors_by_horizon = {
        5: {"ridge": 0.01, "lightgbm_reg": 0.012},
        10: {"ridge": 0.05, "lightgbm_reg": 0.06},
        20: {"ridge": 0.08, "lightgbm_reg": 0.09},
    }
    family_weights = {"ridge": 0.5, "lightgbm_reg": 0.5}
    new_w, changed, composite, target = adapt_horizon_influence(prev, errors_by_horizon, family_weights)
    assert changed
    assert new_w["5"] > prev["5"]
    assert new_w["5"] > new_w["10"] > new_w["20"]
    assert composite["5"] < composite["10"] < composite["20"]
    # Same EWMA clamp as family weights: no horizon can be zeroed out or dominate in one update.
    assert all(0.05 - 1e-9 <= v <= 0.5 + 1e-9 for v in new_w.values())
    # target is the un-damped destination update_weights() moved DEFAULT_EWMA_LR of
    # the way toward -- 5d should be favored there even more sharply than in new_w.
    assert target["5"] > target["10"] > target["20"]


def test_adapt_horizon_influence_ignores_pure_volatility_scale_difference() -> None:
    """Real-data audit (2026-09-05): 5d/10d/20d raw MAE was ~4.90/7.02/10.24%, which
    tracks sqrt(horizon) almost exactly (sqrt(1)/sqrt(2)/sqrt(4) = 1/1.41/2.00 vs the
    actual ratios 1/1.43/2.09) -- i.e. raw MAE alone was measuring which horizon's
    labels are bigger/noisier, not which one the model forecasts better. Feeding in
    naive_mae_by_horizon that scales the same way must cancel that out and leave
    horizon_influence close to unchanged, unlike the old raw-MAE-only behavior in
    test_adapt_horizon_influence_favors_the_more_accurate_horizon above."""
    from trading_system.ml_engine import adapt_horizon_influence

    prev = {"5": 1 / 3, "10": 1 / 3, "20": 1 / 3}
    errors_by_horizon = {
        5: {"ridge": 0.0490, "lightgbm_reg": 0.0490},
        10: {"ridge": 0.0702, "lightgbm_reg": 0.0702},
        20: {"ridge": 0.1024, "lightgbm_reg": 0.1024},
    }
    family_weights = {"ridge": 0.5, "lightgbm_reg": 0.5}
    naive_mae_by_horizon = {5: 0.0490, 10: 0.0702, 20: 0.1024}  # model == naive at every horizon

    new_w, changed, composite, _target = adapt_horizon_influence(
        prev, errors_by_horizon, family_weights, naive_mae_by_horizon=naive_mae_by_horizon
    )
    # Every horizon scores exactly "as good as the naive baseline" (ratio 1.0) once
    # scaled, so none should be favored over another -- a sharp contrast with the
    # unscaled version of these exact numbers, which favors 5d by a wide margin.
    assert composite["5"] == pytest.approx(1.0)
    assert composite["10"] == pytest.approx(1.0)
    assert composite["20"] == pytest.approx(1.0)
    for k in ("5", "10", "20"):
        assert new_w[k] == pytest.approx(prev[k], abs=1e-9)
    assert changed is False


def test_adapt_horizon_influence_can_favor_higher_raw_mae_horizon_with_real_skill() -> None:
    """A horizon with the biggest raw MAE can still win once skill is measured
    relative to its own naive baseline -- exactly the "20d may be more valuable than
    5d despite a bigger MAE number" case."""
    from trading_system.ml_engine import adapt_horizon_influence

    prev = {"5": 1 / 3, "10": 1 / 3, "20": 1 / 3}
    errors_by_horizon = {
        5: {"ridge": 0.0490},   # 4.90% model MAE
        20: {"ridge": 0.1024},  # 10.24% model MAE -- looks "worse" by raw MAE alone
    }
    family_weights = {"ridge": 1.0}
    naive_mae_by_horizon = {
        5: 0.0490,   # model ties the naive baseline at 5d -> zero real skill
        20: 0.1150,  # model beats the naive baseline at 20d -> real skill there
    }
    new_w, changed, composite, _target = adapt_horizon_influence(
        prev, errors_by_horizon, family_weights, naive_mae_by_horizon=naive_mae_by_horizon
    )
    assert composite["20"] < composite["5"]  # 20d's scaled score is the better (lower) one
    assert new_w["20"] > new_w["5"]  # despite 20d having the larger raw MAE input
    assert changed is True


def test_adapt_horizon_influence_falls_back_to_raw_mae_without_naive_baseline() -> None:
    """No naive_mae_by_horizon supplied (older caller/tests) -> identical to before
    this parameter existed."""
    from trading_system.ml_engine import adapt_horizon_influence

    prev = {"5": 1 / 3, "10": 1 / 3, "20": 1 / 3}
    errors_by_horizon = {5: {"ridge": 0.01}, 20: {"ridge": 0.08}}
    family_weights = {"ridge": 1.0}
    with_none, _, composite_none, _t1 = adapt_horizon_influence(prev, errors_by_horizon, family_weights)
    with_empty, _, composite_empty, _t2 = adapt_horizon_influence(
        prev, errors_by_horizon, family_weights, naive_mae_by_horizon={}
    )
    assert composite_none == composite_empty == {"5": 0.01, "20": 0.08}
    assert with_none == with_empty


def test_adapt_horizon_influence_noop_without_matured_errors() -> None:
    from trading_system.ml_engine import adapt_horizon_influence

    prev = {"5": 1 / 3, "10": 1 / 3, "20": 1 / 3}
    new_w, changed, composite, target = adapt_horizon_influence(prev, {}, {"ridge": 1.0})
    assert new_w == prev
    assert changed is False
    assert composite == {}
    assert target == {}


def test_weighted_horizon_mean_applies_learned_horizon_trust() -> None:
    """The bug this fixes: allocate() persisted/loaded horizon_influence but blended
    5/10/20-day scores with a flat mean regardless of it. _weighted_horizon_mean must
    actually move the blended value toward whichever horizon carries more weight."""
    from trading_system.allocation import _weighted_horizon_mean

    hs = {5: 0.10, 10: 0.00, 20: -0.10}
    flat = _weighted_horizon_mean(hs, None)
    assert flat == pytest.approx(0.0)

    favor_5d = _weighted_horizon_mean(hs, {"5": 0.8, "10": 0.1, "20": 0.1})
    assert favor_5d > flat
    assert favor_5d == pytest.approx(0.10 * 0.8 + 0.00 * 0.1 + (-0.10) * 0.1)

    # A zero-sum or missing weight dict must fall back to the flat mean, not divide by zero.
    assert _weighted_horizon_mean(hs, {}) == pytest.approx(flat)
    assert _weighted_horizon_mean(hs, {"5": 0.0, "10": 0.0, "20": 0.0}) == pytest.approx(flat)
