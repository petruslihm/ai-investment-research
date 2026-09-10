"""Q01-Q10 audit follow-up: marked units, GPT gate, research freshness, ranking, UI."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from trading_system.allocation import allocate
from trading_system.btc_sleeve import BtcLiquidityState, BtcSleeveState
from trading_system.config import Settings
from trading_system.evidence_pack import cache_key, pack_is_fresh
from trading_system.ids import new_tick_id
from trading_system.judge_package import compact_judge_package, dumps_complete, verify_pack_sources
from trading_system.market.registry import stable_instrument_id
from trading_system.marked_units import marked_units_from_lots, stock_exposure_for_allocation
from trading_system.portfolio import Lot
from trading_system.portfolio_gate import apply_portfolio_gate
from trading_system.portfolio_service import add_lot, load_portfolio, mark_positions
from trading_system.recommendations import RecommendationAction, llm_final_record
from trading_system.research_agent import research_ticker
from trading_system.seed import seed_synthetic_history
from trading_system.storage import Store
from trading_system.ui.app import render_recommendations_html


def _settings(**kwargs: object) -> Settings:
    base = {
        "_env_file": None,
        "alpaca_api_key": None,
        "alpaca_secret_key": None,
        "openai_api_key": None,
        "gemini_api_key": None,
        "anthropic_api_key": None,
        "max_single_stock_weight": 0.25,
        "max_total_stock_weight": 0.80,
        "min_cash_weight": 0.10,
        "max_btc_weight": 0.30,
    }
    base.update(kwargs)
    return Settings(**base)  # type: ignore[arg-type]


def _rec(**over: object):
    data = {
        "tick_id": "tick_1",
        "decision_epoch_id": "epoch_1",
        "feature_snapshot_id": "fs_1",
        "instrument_id": "inst_aapl",
        "action": RecommendationAction.BUY,
        "current_units": 10.0,
        "recommended_units": 20.0,
        "delta_units": 10.0,
        "confidence": 0.5,
    }
    data.update(over)
    return llm_final_record(**data)  # type: ignore[arg-type]


def test_q01_per_lot_marked_units_not_weighted_average() -> None:
    class _Lot:
        def __init__(self, units: float, price: float) -> None:
            self.acquisition_units = units
            self.acquisition_price = price

    lots = [_Lot(100, 100), _Lot(100, 200)]
    got = marked_units_from_lots(lots, 200)
    wavg = (100 * 100 + 100 * 200) / 200
    naive = 200 * 200 / wavg
    assert got == pytest.approx(300.0)
    assert naive == pytest.approx(266.666, rel=1e-3)
    assert got != pytest.approx(naive)


def test_q01_mark_positions_different_purchase_prices(tmp_path: Path) -> None:
    store = Store(tmp_path / "q01.duckdb")
    store.open(acquire_writer=True)
    try:
        settings = _settings(smoke_universe=("AAPL", "MSFT", "NVDA", "BTC/USD"))
        seed_synthetic_history(store, settings, days=30, end=date(2024, 6, 14))
        aapl = stable_instrument_id("AAPL")
        add_lot(
            store.conn,
            Lot(
                instrument_id=aapl,  # type: ignore[arg-type]
                acquisition_units=100,
                acquisition_price=100,
                acquired_on=date(2024, 1, 2),
            ),
        )
        add_lot(
            store.conn,
            Lot(
                instrument_id=aapl,  # type: ignore[arg-type]
                acquisition_units=100,
                acquisition_price=200,
                acquired_on=date(2024, 1, 3),
            ),
        )
        marked = mark_positions(store.conn, load_portfolio(store.conn))
        pos = next(p for p in marked.positions if str(p.instrument_id) == str(aapl))
        close = float(
            store.conn.execute(
                "SELECT close FROM equity_daily_bars WHERE instrument_id = ? ORDER BY session_date DESC LIMIT 1",
                [aapl],
            ).fetchone()[0]
        )
        expected = 100 * close / 100 + 100 * close / 200
        assert pos.marked_units_final == pytest.approx(expected)
        wavg = (100 * 100 + 100 * 200) / 200
        assert pos.marked_units_final != pytest.approx(200 * close / wavg)
    finally:
        store.close()


def test_q02_allocation_uses_marked_not_acquisition() -> None:
    positions = [
        SimpleNamespace(
            instrument_id="inst_aapl",
            acquisition_units_total=200.0,
            marked_units_final=300.0,
            marked_units_intraday_preview=310.0,
        ),
        SimpleNamespace(
            instrument_id="inst_msft",
            acquisition_units_total=50.0,
            marked_units_final=None,
            marked_units_intraday_preview=None,
        ),
        SimpleNamespace(
            instrument_id="inst_btc_usd",
            acquisition_units_total=1.0,
            marked_units_final=1.2,
            marked_units_intraday_preview=None,
        ),
    ]
    marked, acq, preview, unpriced = stock_exposure_for_allocation(positions)
    assert marked["inst_aapl"] == 300.0
    assert acq["inst_aapl"] == 200.0
    assert preview["inst_aapl"] == 310.0
    assert "inst_msft" not in marked
    assert "inst_msft" in unpriced
    assert "inst_btc_usd" not in marked
    assert acq["inst_msft"] == 50.0


def test_q03_portfolio_gate_clamps_impossible_gpt_units() -> None:
    rec = _rec(recommended_units=5000.0, current_units=10.0)
    btc = BtcSleeveState(
        instrument_id=stable_instrument_id("BTC/USD"),
        liquidity=BtcLiquidityState.AVAILABLE,
        current_units=0,
    )
    out = apply_portfolio_gate(
        [rec],
        settings=_settings(),
        total_base_units=1000.0,
        stock_marked={"inst_aapl": 10.0},
        btc_state=btc,
    )
    gated = out[0]
    assert gated.requested_units == pytest.approx(5000.0)
    assert gated.constrained_units == pytest.approx(gated.recommended_units)
    assert gated.recommended_units == pytest.approx(250.0)
    assert "UNITS_CONSTRAINED" in (gated.override_reasons or [])


def test_q03_btc_transfer_block_keeps_current_units() -> None:
    rec = _rec(
        instrument_id=str(stable_instrument_id("BTC/USD")),
        recommended_units=80.0,
        current_units=12.0,
    )
    btc = BtcSleeveState(
        instrument_id=stable_instrument_id("BTC/USD"),
        liquidity=BtcLiquidityState.TRANSFER_PENDING,
        current_units=12.0,
    )
    out = apply_portfolio_gate(
        [rec],
        settings=_settings(),
        total_base_units=1000.0,
        stock_marked={},
        btc_state=btc,
    )
    assert out[0].recommended_units == pytest.approx(12.0)
    assert "BTC_TRANSFER_BLOCK" in (out[0].override_reasons or [])


def test_q04_judge_package_includes_portfolio_context() -> None:
    package = compact_judge_package(
        {
            "role": "final judge",
            "dq_level": "ok",
            "quant": {"action": "BUY", "recommended_units": 12, "horizons": []},
            "evidence_pack": {"ticker": "AAPL", "contrary_evidence": [{"claim": "risk", "source_url": ""}]},
            "portfolio": {
                "total_base_units": 1000,
                "cash_units": 200,
                "cash_floor_weight": 0.1,
                "holdings": [{"instrument_id": "inst_aapl", "marked_units_final": 80, "weight": 0.08}],
                "btc_sleeve": {"blocked": False, "current_units": 0},
                "constraints": {"max_single_stock_weight": 0.25},
            },
        }
    )
    assert package["portfolio"]["total_base_units"] == 1000
    assert package["portfolio"]["cash_units"] == 200
    assert package["portfolio"]["holdings"]
    assert package["portfolio"]["btc_sleeve"]["blocked"] is False
    raw = dumps_complete(package)
    json.loads(raw)
    assert "account" not in raw.lower() or "no account" in raw.lower()


def test_q04b_judge_package_includes_allocation_trace() -> None:
    package = compact_judge_package(
        {
            "dq_level": "ok",
            "quant": {"action": "BUY", "recommended_units": 12, "horizons": []},
            "evidence_pack": {"ticker": "AAPL"},
            "portfolio": {
                "total_base_units": 1000,
                "cash_units": 740,
                "allocation_trace": {
                    "universe": 409,
                    "passed_cutoff": 9,
                    "cutoff": 0.01,
                    "deployment_factor": 0.18,
                    "equity_budget": 0.16,
                    "cash_weight": 0.74,
                    "score_percentiles": {"p50": 0.0044, "p95": 0.009, "max": 0.0125},
                    "diagnosis": {"notes": ["calibration placeholder"], "sizing_compression": True},
                    "rows_passed": [
                        {
                            "ticker": "DY",
                            "pred_5d": 0.0078,
                            "pred_10d": 0.0287,
                            "pred_20d": 0.0557,
                            "opp_score": 0.0125,
                            "conviction": 0.19,
                            "final_weight": 0.02,
                        }
                    ],
                    "log_text": "too long to send " * 500,
                    "rows": [{"ticker": f"T{i}"} for i in range(400)],
                },
            },
        }
    )
    trace = package["allocation_trace"]
    assert trace["universe"] == 409
    assert trace["passed_cutoff"] == 9
    assert trace["top_passed"][0]["ticker"] == "DY"
    assert "rows" not in trace
    assert "log_text" not in trace
    assert "allocation_trace" not in package["portfolio"]
    dumped = dumps_complete(package)
    assert "too long to send" not in dumped
    assert len(dumped) < 8000


def test_q05_research_cache_expires_after_45_minutes_or_material_change() -> None:
    """Research is short-lived and can be invalidated even sooner by changed inputs."""
    key = cache_key(ticker="AAPL", accession="0001", quant_action="BUY", units_bucket="4")
    assert datetime.now(timezone.utc).date().isoformat() not in key
    assert "AAPL" in key
    now = datetime.now(timezone.utc)
    expired = {
        "researched_at": (now - timedelta(minutes=46)).isoformat(),
        "freshness": {"price": 100.0, "portfolio_sig": "p1", "regime": "neutral"},
    }
    fresh = {
        "researched_at": (now - timedelta(minutes=10)).isoformat(),
        "freshness": {"price": 100.0, "portfolio_sig": "p1", "regime": "neutral"},
    }
    assert not pack_is_fresh(expired, now=now, last_price=100.0, portfolio_sig="p1", regime="neutral")
    assert pack_is_fresh(fresh, now=now, last_price=101.0, portfolio_sig="p1", regime="neutral")
    assert not pack_is_fresh(fresh, now=now, last_price=110.0, portfolio_sig="p1", regime="neutral")
    assert not pack_is_fresh(fresh, now=now, last_price=100.0, portfolio_sig="p2", regime="neutral")
    assert not pack_is_fresh(fresh, now=now, last_price=100.0, portfolio_sig="p1", regime="risk_off")


def test_q06_adversarial_still_runs_when_first_pass_degraded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from trading_system.recommendations import RecommendationAction, quant_only_record

    searches: list[bool] = []

    def fake_generate(
        settings, *, system: str, user: str, use_search: bool = False, response_schema=None
    ):
        searches.append(use_search)
        if "adversarial" in system.lower() or "Attack" in system:
            return {
                "status": "AVAILABLE",
                "web_search_used": True,
                "main_objections": ["priced in"],
                "contrary_evidence": [{"claim": "slowing", "source_url": "https://sec.gov/x"}],
                "sources": [{"url": "https://sec.gov/x"}],
                "_exchange": {
                    "system": system,
                    "user": user,
                    "model": "gemini-3.6-flash",
                    "raw_response": "{}",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        return {
            "status": "DEGRADED",
            "ticker": "AAPL",
            "web_search_used": False,
            "supporting_evidence": [{"claim": "filings only", "source_url": "https://www.sec.gov/a"}],
            "contrary_evidence": [],
            "sources": [{"url": "https://www.sec.gov/a"}],
            "open_questions": ["search unavailable"],
            "data_quality": "no-search",
            "_exchange": {
                "system": system,
                "user": user,
                "model": "gemini-3.6-flash",
                "raw_response": "{}",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    monkeypatch.setattr("trading_system.research_agent.generate_json", fake_generate)
    store = Store(tmp_path / "q06.duckdb")
    store.open(acquire_writer=True)
    try:
        pack = research_ticker(
            _settings(gemini_api_key="test-gemini"),
            quant_only_record(
                tick_id="tick_1",
                decision_epoch_id="epoch_1",
                feature_snapshot_id="fs_1",
                instrument_id="inst_aapl",
                action=RecommendationAction.BUY,
                recommended_units=4,
                current_units=0,
            ),
            conn=store.conn,
            tick_id="tick_q06",
        )
        assert searches == [True, True]
        assert pack["adversarial_review"]["main_objections"]
        assert pack.get("researched_at")
        assert pack.get("freshness_state") == "fresh"
    finally:
        store.close()


def test_q07_compact_keeps_valid_json_and_priority_fields() -> None:
    pack = {
        "ticker": "AAPL",
        "news_events": [{"headline": f"n{i}", "url": f"https://example.com/{i}"} for i in range(40)],
        "filings": [{"accession": f"a{i}", "url": "https://www.sec.gov/x"} for i in range(20)],
        "supporting_evidence": [{"claim": f"s{i}", "source_url": "https://www.sec.gov/x"} for i in range(30)],
        "contrary_evidence": [{"claim": f"c{i}", "source_url": "https://evil.example/x"} for i in range(30)],
        "open_questions": [f"q{i}" for i in range(40)],
        "sources": [{"url": "https://www.sec.gov/x"}],
        "data_quality": "keep-me",
    }
    out = compact_judge_package(
        {
            "portfolio": {"total_base_units": 1000, "cash_units": 100, "holdings": [{"instrument_id": "inst_aapl"}]},
            "quant": {
                "action": "BUY",
                "horizons": [
                    {"horizon": 5, "expected_return": 0.01, "rank_score": 0.2},
                    {"horizon": 10, "expected_return": 0.02, "rank_score": 0.3},
                    {"horizon": 20, "expected_return": 0.03, "rank_score": 0.4},
                ],
            },
            "dq_level": "ok",
            "evidence_pack": pack,
        },
        max_chars=3500,
    )
    raw = dumps_complete(out)
    parsed = json.loads(raw)
    assert raw.startswith("{") and raw.endswith("}")
    assert parsed["portfolio"]["total_base_units"] == 1000
    assert parsed["quant"]["horizons"]
    assert parsed["evidence_pack"]["contrary_evidence"]
    assert parsed["evidence_pack"]["data_quality"] == "keep-me"
    assert len(raw) <= 3500 or len(parsed["evidence_pack"]["news_events"]) <= 10


def test_q08_unverified_urls_are_not_grounded() -> None:
    pack = verify_pack_sources(
        {
            "sources": [{"url": "https://www.reuters.com/markets/aapl"}],
            "supporting_evidence": [
                {"claim": "grounded", "source_url": "https://www.reuters.com/markets/aapl"},
                {"claim": "invented", "source_url": "https://not-a-real-source.invalid/aapl"},
            ],
            "contrary_evidence": [{"claim": "sec", "source_url": "https://www.sec.gov/Archives/edgar/data/1"}],
        }
    )
    by_claim = {row["claim"]: row["source_status"] for row in pack["supporting_evidence"]}
    assert by_claim["grounded"] == "CITED_URL"
    assert by_claim["invented"] == "UNVERIFIED"
    assert pack["contrary_evidence"][0]["source_status"] == "SEC_DOMAIN_ONLY"


def test_q09_lambdarank_orders_shortlist_without_entering_expected_return() -> None:
    settings = _settings(max_total_stock_weight=0.2, max_single_stock_weight=0.2, min_cash_weight=0.0)
    scores = {5: 0.04, 10: 0.04, 20: 0.04}
    out = allocate(
        stock_scores={
            "inst_low": scores,
            "inst_high": scores,
            "inst_mid": scores,
        },
        stock_vol={"inst_low": 0.02, "inst_high": 0.02, "inst_mid": 0.02},
        stock_rank_scores={
            "inst_low": {5: 0.1, 10: 0.1, 20: 0.1},
            "inst_high": {5: 0.9, 10: 0.9, 20: 0.9},
            "inst_mid": {5: 0.5, 10: 0.5, 20: 0.5},
        },
        btc_scores={5: -0.2, 10: -0.2, 20: -0.2},
        settings=settings,
        btc_state=BtcSleeveState(
            instrument_id=stable_instrument_id("BTC/USD"),
            liquidity=BtcLiquidityState.AVAILABLE,
            current_units=0,
        ),
        tick_id=str(new_tick_id()),
        epoch_id="epoch_q09",
        feature_snapshot_id="fs_q09",
        total_base_units=1000.0,
    )
    assert out["payload"]["shortlist_order"][0] == "inst_high"
    assert out["payload"]["rank_factor"] == "lambdarank_tiebreak"
    assert out["payload"]["lambdarank_mean"]["inst_high"] > out["payload"]["lambdarank_mean"]["inst_low"]
    high = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_high")
    assert "lambdarank_tiebreak=" in (high.thesis or "")
    for h in high.horizons:
        assert h.expected_return == pytest.approx(0.04)
        assert h.rank_score is not None


def test_q10_recommendation_cards_show_unit_fields() -> None:
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {"total_base_units": 1000},
            "quant": [
                {
                    "instrument_id": "inst_aapl",
                    "action": "BUY",
                    "actionable": True,
                    "acquisition_units": 100,
                    "marked_units_final": 150,
                    "marked_units_intraday_preview": 155,
                    "current_units": 150,
                    "recommended_units": 180,
                    "delta_units": 30,
                    "confidence": 0.8,
                    "horizons": [{"horizon": 5, "expected_return": 0.04}],
                }
            ],
            "llm_final": [
                {
                    "instrument_id": "inst_aapl",
                    "action": "ADD",
                    "override_reasons": [],
                    "thesis": "근거",
                    "recommended_units": 170,
                    "delta_units": 20,
                    "requested_units": 400,
                    "constrained_units": 170,
                }
            ],
        },
        settings=Settings(_env_file=None),
    )
    assert "현재 보유 100 u" in html
    assert "현재 평가 150 u" in html
    assert "AI 추천 170 u" in html
    assert "증감 20 u" in html
    assert "FINAL close" in html
    assert "장중 미리보기 (비공식)" in html
    assert "요청 400 u → 포트폴리오 제약 후 170 u" in html
