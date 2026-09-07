"""Gemini research desk + GPT-5.6 Sol shortlist. No live API calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from trading_system.config import Settings
from trading_system.evidence_pack import normalize_pack
from trading_system.filing_chunks import select_filing_passages
from trading_system.llm_budget import estimate_usd, utc_today, would_exceed_budget
from trading_system.openai_judge import resolve_judge_model
from trading_system.recommendations import RecommendationAction, quant_only_record
from trading_system.research_agent import research_ticker, select_final_judge_recs
from trading_system.storage import Store


def _settings(**kwargs: object) -> Settings:
    base = {
        "_env_file": None,
        "alpaca_api_key": None,
        "alpaca_secret_key": None,
        "openai_api_key": None,
        "gemini_api_key": None,
        "anthropic_api_key": None,
        "smoke_universe": ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "BTC/USD"),
        "llm_final_max_names": 3,
        "llm_daily_budget_usd": 2.0,
    }
    base.update(kwargs)
    return Settings(**base)  # type: ignore[arg-type]


def _rec(ticker: str, *, units: float = 0.0, rec_units: float = 0.0, action=RecommendationAction.HOLD):
    return quant_only_record(
        tick_id="tick_1",
        decision_epoch_id="epoch_1",
        feature_snapshot_id="fs_1",
        instrument_id=f"inst_{ticker.lower()}",
        action=action,
        current_units=units,
        recommended_units=rec_units,
        delta_units=rec_units - units,
        confidence=0.5,
    )


def test_ticker_lot_canonicalizes_to_inst_id() -> None:
    from datetime import date

    from trading_system.ids import canonicalize_instrument_id
    from trading_system.portfolio import Lot

    assert canonicalize_instrument_id("AMD") == "inst_amd"
    assert canonicalize_instrument_id("inst_amd") == "inst_amd"
    lot = Lot(
        instrument_id="GOOG",  # type: ignore[arg-type]
        acquisition_units=1,
        acquisition_price=1,
        acquired_on=date(2024, 1, 2),
    )
    assert lot.instrument_id == "inst_goog"
    recs = [
        _rec("aaa", rec_units=10, action=RecommendationAction.BUY),
        _rec("bbb", rec_units=9, action=RecommendationAction.BUY),
        _rec("ccc", rec_units=8, action=RecommendationAction.BUY),
        _rec("ddd", rec_units=7, action=RecommendationAction.BUY),
        _rec("hold", units=2.0, action=RecommendationAction.HOLD),
    ]
    chosen = select_final_judge_recs(recs, max_names=3)
    assert chosen == ["inst_hold", "inst_aaa", "inst_bbb", "inst_ccc"]


def test_select_final_judge_never_drops_holdings() -> None:
    recs = [
        _rec("hold_a", units=1.0, action=RecommendationAction.HOLD),
        _rec("hold_b", units=2.0, action=RecommendationAction.HOLD),
        _rec("aaa", rec_units=10, action=RecommendationAction.BUY),
    ]
    chosen = select_final_judge_recs(recs, max_names=1)
    assert chosen[:2] == ["inst_hold_a", "inst_hold_b"]
    assert "inst_aaa" in chosen


def test_btc_sleeve_does_not_consume_equity_judge_slots() -> None:
    recs = [
        _rec("aaa", rec_units=10, action=RecommendationAction.BUY),
        _rec("bbb", rec_units=9, action=RecommendationAction.BUY),
        _rec("ccc", rec_units=8, action=RecommendationAction.BUY),
        quant_only_record(
            tick_id="tick_1",
            decision_epoch_id="epoch_1",
            feature_snapshot_id="fs_1",
            instrument_id="inst_btc_usd",
            action=RecommendationAction.ENTER,
            current_units=0,
            recommended_units=999,
            delta_units=999,
            confidence=0.5,
        ),
    ]
    chosen = select_final_judge_recs(recs, max_names=3)
    assert chosen[:3] == ["inst_aaa", "inst_bbb", "inst_ccc"]
    assert "inst_btc_usd" in chosen


def test_llm_progress_shows_index_and_ticker() -> None:
    from trading_system.v1_cycle import llm_progress_message

    assert llm_progress_message("Gemini 리서치", 3, 12, "inst_pltr") == "Gemini 리서치 3/12 · PLTR"
    assert llm_progress_message("Sol 판단", 2, 7, "inst_shop") == "Sol 판단 2/7 · SHOP"
    # Sol UI index is processing order (1,2,3…), not shortlist rank.


def test_deadline_settings_flow_into_retry_policy() -> None:
    from trading_system.gemini_client import gemini_retry_policy
    from trading_system.openai_judge import sol_retry_policy

    s = _settings(gemini_deadline_seconds=300, sol_judge_deadline_seconds=240)
    assert gemini_retry_policy(s).total_deadline_seconds == 300
    assert sol_retry_policy(s).total_deadline_seconds == 240


def test_research_capability_stays_available_if_most_packs_succeed() -> None:
    from trading_system.v1_cycle import _research_capability

    assert (
        _research_capability(
            {
                "a": {"status": "AVAILABLE"},
                "b": {"status": "AVAILABLE"},
                "c": {"status": "RATE_LIMITED"},
            }
        )
        == "AVAILABLE"
    )
    assert _research_capability({"a": {"status": "DEGRADED"}, "b": {"status": "AVAILABLE"}}) == "AVAILABLE"
    assert (
        _research_capability(
            {
                "a": {"status": "AVAILABLE"},
                "b": {"status": "RATE_LIMITED"},
                "c": {"status": "RATE_LIMITED"},
                "d": {"status": "UNAVAILABLE"},
            }
        )
        == "DEGRADED"
    )


def test_filing_passages_prefer_item_not_prefix() -> None:
    junk = "COVER PAGE " * 400
    meat = (
        "Item 2.02 Results of Operations and Financial Condition "
        "The company lowered full-year guidance after an impairment and ongoing litigation. "
        "Revenue outlook was cut."
    )
    text = junk + meat
    passages = select_filing_passages(text, ticker="AAPL")
    blob = " ".join(p["text"] for p in passages)
    assert "lowered full-year guidance" in blob
    assert blob.lower().count("cover page") < 30


def test_resolve_judge_model_ignores_mini() -> None:
    assert resolve_judge_model(_settings(llm_model="gpt-4o-mini")) == "gpt-5.6-sol"
    assert resolve_judge_model(_settings(llm_judge_model="gpt-5.6-sol")) == "gpt-5.6-sol"


def test_sol_pricing_estimate() -> None:
    usd = estimate_usd("gpt-5.6-sol", 1_000_000, 1_000_000)
    assert abs(usd - 24.0) < 1e-9


def test_budget_blocks_when_spent(tmp_path: Path) -> None:
    store = Store(tmp_path / "budget.duckdb")
    store.open(acquire_writer=True)
    try:
        store.conn.execute(
            """
            INSERT INTO llm_cost_ledger (
                entry_id, utc_date, provider, model, kind, ticker,
                prompt_tokens, completion_tokens, estimated_usd, created_at
            )
            VALUES ('c1', ?, 'openai', 'gpt-5.6-sol', 'auto:judge', 'AAPL',
                    100, 50, 2.0, CURRENT_TIMESTAMP)
            """,
            [utc_today()],
        )
        assert would_exceed_budget(store.conn, _settings(llm_daily_budget_usd=2.0), model="gpt-5.6-sol")
        assert not would_exceed_budget(
            store.conn, _settings(llm_daily_budget_usd=50.0), model="gpt-5.6-sol"
        )
        store.conn.execute(
            """
            INSERT INTO llm_cost_ledger (
                entry_id, utc_date, provider, model, kind, ticker,
                prompt_tokens, completion_tokens, estimated_usd, created_at
            )
            VALUES ('c2', ?, 'openai', 'gpt-5.6-sol', 'judge', 'MSFT',
                    100, 50, 9.0, CURRENT_TIMESTAMP)
            """,
            [utc_today()],
        )
        assert would_exceed_budget(store.conn, _settings(llm_daily_budget_usd=2.0), model="gpt-5.6-sol")
        from trading_system.llm_budget import daily_auto_spend_usd, daily_spend_usd

        assert daily_auto_spend_usd(store.conn) == 2.0
        assert daily_spend_usd(store.conn) == 11.0
    finally:
        store.close()


def test_record_usage_manual_does_not_consume_auto_cap(tmp_path: Path) -> None:
    from trading_system.llm_budget import daily_auto_spend_usd, daily_spend_usd, record_usage

    store = Store(tmp_path / "budget_kinds.duckdb")
    store.open(acquire_writer=True)
    try:
        record_usage(
            store.conn,
            provider="openai",
            model="gpt-5.6-sol",
            kind="judge",
            ticker="AAPL",
            prompt_tokens=500_000,
            completion_tokens=0,
            toward_daily_cap=False,
        )
        assert daily_auto_spend_usd(store.conn) == 0.0
        assert daily_spend_usd(store.conn) > 0.0
        assert not would_exceed_budget(store.conn, _settings(llm_daily_budget_usd=3.0), model="gpt-5.6-sol")
        record_usage(
            store.conn,
            provider="openai",
            model="gpt-5.6-sol",
            kind="judge",
            ticker="MSFT",
            prompt_tokens=500_000,
            completion_tokens=0,
            toward_daily_cap=True,
        )
        kinds = {row[0] for row in store.conn.execute("SELECT kind FROM llm_cost_ledger").fetchall()}
        assert "judge" in kinds
        assert "auto:judge" in kinds
        assert would_exceed_budget(store.conn, _settings(llm_daily_budget_usd=1.0), model="gpt-5.6-sol")
    finally:
        store.close()


def test_record_usage_ledger_write_failure_is_logged_not_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """record_usage must never raise (never-block: cost tracking can't abort a scan), but
    a lost ledger row silently under-counts every later daily-cap check for the rest of
    the day -- that must not vanish with zero trace."""
    import logging

    import duckdb

    from trading_system.llm_budget import record_usage

    conn = duckdb.connect(":memory:")  # no schema applied -- llm_cost_ledger doesn't exist
    try:
        with caplog.at_level(logging.WARNING, logger="trading_system.llm_budget"):
            usd = record_usage(
                conn,
                provider="openai",
                model="gpt-5.6-sol",
                kind="judge",
                ticker="AAPL",
                prompt_tokens=1000,
                completion_tokens=100,
                toward_daily_cap=True,
            )
        assert usd > 0.0  # estimate is still returned despite the failed write
        assert any("failed to persist ledger row" in r.message for r in caplog.records)
    finally:
        conn.close()


def test_automatic_ledger_write_failure_latches_the_day_closed(tmp_path: Path) -> None:
    """A single lost automatic ledger row must not just under-count silently -- it
    must block every further automatic call for that UTC day, even ones whose own
    ledger read/write would otherwise succeed fine on a healthy connection."""
    import duckdb

    from trading_system.llm_budget import daily_auto_spend_usd, record_usage

    broken = duckdb.connect(":memory:")  # no schema -- INSERT fails
    try:
        record_usage(
            broken,
            provider="openai",
            model="gpt-5.6-sol",
            kind="judge",
            prompt_tokens=1000,
            completion_tokens=100,
            toward_daily_cap=True,
        )
    finally:
        broken.close()

    # A completely separate, healthy connection must still be blocked for today --
    # the latch is process-wide for the day, not tied to the connection that failed.
    store = Store(tmp_path / "healthy.duckdb")
    store.open(acquire_writer=True)
    try:
        assert daily_auto_spend_usd(store.conn) == float("inf")
        assert would_exceed_budget(store.conn, _settings(llm_daily_budget_usd=1_000_000.0), model="gpt-5.6-sol")
    finally:
        store.close()


def test_manual_ledger_write_failure_does_not_latch_the_auto_budget(tmp_path: Path) -> None:
    """Manual scans are uncapped by design -- a manual write failure must never
    consume or block the automatic budget."""
    import duckdb

    from trading_system.llm_budget import daily_auto_spend_usd, record_usage

    broken = duckdb.connect(":memory:")  # no schema -- INSERT fails
    try:
        record_usage(
            broken,
            provider="openai",
            model="gpt-5.6-sol",
            kind="manual_research",
            prompt_tokens=1000,
            completion_tokens=100,
            toward_daily_cap=False,
        )
    finally:
        broken.close()

    store = Store(tmp_path / "healthy2.duckdb")
    store.open(acquire_writer=True)
    try:
        assert daily_auto_spend_usd(store.conn) == 0.0  # not forced to infinity
        assert not would_exceed_budget(store.conn, _settings(llm_daily_budget_usd=1_000_000.0), model="gpt-5.6-sol")
    finally:
        store.close()


def test_auto_ledger_failure_latch_is_scoped_to_its_own_utc_day() -> None:
    """Day rollover must not need an explicit reset: a latch set for one day must
    never block a query for a different day."""
    from datetime import timedelta

    import duckdb

    from trading_system import llm_budget as llm_budget_module
    from trading_system.llm_budget import daily_auto_spend_usd

    # utc_today() inside record_usage always uses the real current day, so simulate
    # "yesterday's" failure by marking the latch directly for that day.
    yesterday = utc_today() - timedelta(days=1)
    llm_budget_module._mark_auto_ledger_write_failed(yesterday)

    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE llm_cost_ledger (entry_id VARCHAR, utc_date DATE, provider VARCHAR, "
            "model VARCHAR, kind VARCHAR, ticker VARCHAR, prompt_tokens INTEGER, "
            "completion_tokens INTEGER, estimated_usd DOUBLE, created_at TIMESTAMPTZ)"
        )
        # Yesterday is latched closed.
        assert daily_auto_spend_usd(conn, day=yesterday) == float("inf")
        # Today is a different day -- the stale latch must not apply.
        assert daily_auto_spend_usd(conn, day=utc_today()) == 0.0
    finally:
        conn.close()


def test_automatic_usage_without_ledger_connection_latches_day_closed(tmp_path: Path) -> None:
    from trading_system.llm_budget import daily_auto_spend_usd, record_usage

    record_usage(
        None,
        provider="openai",
        model="gpt-5.6-sol",
        kind="judge",
        toward_daily_cap=True,
        prompt_tokens=100,
        completion_tokens=10,
    )

    store = Store(tmp_path / "budget_after_missing_connection.duckdb")
    store.open(acquire_writer=True)
    try:
        assert daily_auto_spend_usd(store.conn) == float("inf")
    finally:
        store.close()


def test_automatic_budget_fails_closed_when_ledger_is_unavailable() -> None:
    from trading_system.llm_budget import budget_remaining, daily_auto_spend_usd

    class BrokenConnection:
        def execute(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("ledger unavailable")

    conn = BrokenConnection()
    settings = _settings(llm_daily_budget_usd=10.0)

    assert daily_auto_spend_usd(conn) == float("inf")  # type: ignore[arg-type]
    assert budget_remaining(conn, settings) == 0.0  # type: ignore[arg-type]
    assert would_exceed_budget(conn, settings, model="gpt-5.6-sol")  # type: ignore[arg-type]
    assert would_exceed_budget(None, settings, model="gpt-5.6-sol")


def test_manual_research_ignores_auto_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_generate(
        settings, *, system: str, user: str, use_search: bool = False, response_schema=None
    ):
        calls.append(system[:20])
        return {
            "status": "AVAILABLE",
            "ticker": "AAPL",
            "supporting_evidence": [{"claim": "ok", "source_url": "https://sec.gov"}],
            "web_search_used": True,
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
    store = Store(tmp_path / "budget_manual.duckdb")
    store.open(acquire_writer=True)
    try:
        store.conn.execute(
            """
            INSERT INTO llm_cost_ledger (
                entry_id, utc_date, provider, model, kind, ticker,
                prompt_tokens, completion_tokens, estimated_usd, created_at
            )
            VALUES ('c1', ?, 'gemini', 'gemini-3.6-flash', 'auto:research', 'AAPL',
                    100, 50, 2.0, CURRENT_TIMESTAMP)
            """,
            [utc_today()],
        )
        rec = _rec("aapl", rec_units=4, action=RecommendationAction.BUY)
        cfg = _settings(gemini_api_key="test-gemini", llm_daily_budget_usd=2.0)
        blocked = research_ticker(cfg, rec, conn=store.conn, tick_id="tick_auto", enforce_llm_budget=True)
        assert not calls
        assert any("BUDGET_EXCEEDED" in str(q) for q in blocked["open_questions"])
        ran = research_ticker(cfg, rec, conn=store.conn, tick_id="tick_manual", enforce_llm_budget=False)
        assert calls
        assert ran["supporting_evidence"]
    finally:
        store.close()


def test_research_without_gemini_keeps_filings(tmp_path: Path) -> None:
    store = Store(tmp_path / "research.duckdb")
    store.open(acquire_writer=True)
    try:
        pack = research_ticker(
            _settings(),
            _rec("aapl", rec_units=4, action=RecommendationAction.BUY),
            filings=[
                {
                    "ticker": "AAPL",
                    "accession": "0001",
                    "form": "8-K",
                    "passages": [{"section": "Item 2.02", "text": "guidance cut"}],
                }
            ],
            conn=store.conn,
            tick_id="tick_1",
        )
        assert pack["ticker"] == "AAPL"
        assert pack["status"] == "NOT_CONFIGURED"
        assert any("Gemini" in q for q in pack["open_questions"])
        assert pack["filings"]
        row = store.conn.execute("SELECT COUNT(*) FROM research_evidence_packs").fetchone()
        assert row and int(row[0]) >= 1
    finally:
        store.close()


def test_normalize_pack_rejects_sentiment_stub() -> None:
    pack = normalize_pack({"sentiment": "positive"}, ticker="AAPL")
    assert pack["ticker"] == "AAPL"
    assert pack["supporting_evidence"] == []


def test_legacy_b_prompts_have_q1_q5_and_fixed_output_contract() -> None:
    from trading_system.llm_client import JUDGE_SYSTEM
    from trading_system.research_agent import RESEARCH_SCHEMA, RESEARCH_SYSTEM

    assert all(f"[Q{i}]" in RESEARCH_SYSTEM for i in range(1, 6))
    assert "rerating" in RESEARCH_SYSTEM.lower()
    assert "attractiveness versus CASH" in RESEARCH_SYSTEM
    assert "choose the shape" not in RESEARCH_SYSTEM
    assert "[MANDATORY Q1-Q5]" in JUDGE_SYSTEM
    assert "ENTER (매수) or NO_ACTION (관망)" in JUDGE_SYSTEM
    assert "ADD (추가매수), HOLD (유지), or EXIT (매도)" in JUDGE_SYSTEM
    assert "choose the shape" not in JUDGE_SYSTEM
    assert "q1_consensus_revision" in RESEARCH_SCHEMA["required"]


def test_research_portfolio_compacts_full_allocation_trace() -> None:
    import json

    from trading_system.research_agent import _compact_research_portfolio

    portfolio = {
        "total_base_units": 1000,
        "cash_units": 200,
        "holdings": [
            {"instrument_id": f"inst_{i}", "marked_units_final": 10, "weight": 0.01}
            for i in range(12)
        ],
        "allocation_trace": {
            "universe": 503,
            "passed_cutoff": 9,
            "rows_passed": [
                {"ticker": f"T{i}", "allocation_score": 1 - i / 1000, "large_blob": "x" * 500}
                for i in range(400)
            ],
        },
    }
    compact = _compact_research_portfolio(portfolio)
    assert len(compact["holdings"]) == 12
    assert len(compact["allocation_trace"]["top_passed"]) == 8
    assert "large_blob" not in json.dumps(compact)
    assert len(json.dumps(compact)) < 10_000


def test_compact_omits_empty_template_keys() -> None:
    from trading_system.judge_package import compact_evidence_pack, compact_judge_package

    out = compact_evidence_pack({"ticker": "AAPL", "contrary_evidence": [{"claim": "risk"}]})
    assert out["ticker"] == "AAPL"
    assert out["contrary_evidence"]
    assert "industry" not in out
    assert "macro" not in out
    assert "open_questions" not in out
    assert "role" not in compact_judge_package({"quant": {"action": "BUY"}, "evidence_pack": out})


def test_gemini_research_two_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    searches: list[bool] = []

    def fake_generate(
        settings, *, system: str, user: str, use_search: bool = False, response_schema=None
    ):
        calls.append(system[:40])
        searches.append(use_search)
        if "adversarial" in system.lower() or "attack" in system.lower() or "critic" in system.lower():
            return {
                "status": "AVAILABLE",
                "main_objections": ["guidance already priced"],
                "contrary_evidence": [{"claim": "peers cheaper", "source_url": "https://example.com"}],
                "what_would_change_the_call": "another cut",
                "_exchange": {
                    "system": system,
                    "user": user,
                    "model": "gemini-3.6-flash",
                    "raw_response": "{}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        return {
            "status": "AVAILABLE",
            "ticker": "AAPL",
            "material_change": True,
            "supporting_evidence": [{"claim": "filing cut guidance", "source_url": "https://sec.gov"}],
            "contrary_evidence": [],
            "news_events": [{"headline": "guidance cut", "url": "https://example.com/n"}],
            "industry": {"sector": "tech", "competitors": ["MSFT"], "notes": "ok"},
            "macro": {"regime": "risk-on", "notes": ""},
            "sources": [{"title": "8-K", "url": "https://sec.gov"}],
            "open_questions": [],
            "data_quality": "filings+search",
            "web_search_used": use_search,
            "_exchange": {
                "system": system,
                "user": user,
                "model": "gemini-3.6-flash",
                "raw_response": "{}",
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
            },
        }

    monkeypatch.setattr("trading_system.research_agent.generate_json", fake_generate)
    store = Store(tmp_path / "gem.duckdb")
    store.open(acquire_writer=True)
    try:
        pack = research_ticker(
            _settings(gemini_api_key="test-gemini"),
            _rec("aapl", rec_units=4, action=RecommendationAction.BUY),
            filings=[{"ticker": "AAPL", "accession": "0001", "passages": [{"section": "Item 2.02", "text": "cut"}]}],
            conn=store.conn,
            tick_id="tick_g",
        )
        assert len(calls) == 2
        assert searches == [True, True]
        assert pack["supporting_evidence"]
        assert pack["adversarial_review"]["main_objections"]
        assert any("peers cheaper" in str(row) for row in pack["contrary_evidence"])
        statuses = {row.get("source_status") for row in pack["supporting_evidence"] if isinstance(row, dict)}
        assert "VERIFIED_SOURCE" in statuses
        contrary_status = {
            row.get("source_status") for row in pack["contrary_evidence"] if isinstance(row, dict)
        }
        assert "UNVERIFIED" in contrary_status
        n_live = store.conn.execute("SELECT COUNT(*) FROM llm_transcripts").fetchone()[0]
        assert int(n_live) == 2
        assert pack["status"] == "AVAILABLE"
        assert pack["web_search_used"] is True
    finally:
        store.close()


def test_evidence_pack_clamps_score_fields() -> None:
    from trading_system.evidence_pack import clamp_score_0_100

    assert clamp_score_0_100(None) is None
    assert clamp_score_0_100("not a number") is None
    assert clamp_score_0_100(float("nan")) is None
    assert clamp_score_0_100(float("inf")) is None
    assert clamp_score_0_100(-10) == pytest.approx(0.0)
    assert clamp_score_0_100(150) == pytest.approx(100.0)
    assert clamp_score_0_100("62.5") == pytest.approx(62.5)


def test_gemini_research_carries_bounded_scores_into_pack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """research_sizing_multiplier (allocation.py) can only move real position sizing
    if these bounded 0-100 fields actually survive normalize_pack + the adversarial
    merge -- this is the wiring proof, not just the schema declaration."""

    def fake_generate(
        settings, *, system: str, user: str, use_search: bool = False, response_schema=None
    ):
        if "adversarial" in system.lower() or "attack" in system.lower() or "critic" in system.lower():
            return {
                "status": "AVAILABLE",
                "main_objections": ["guidance already priced"],
                "bearish_severity_score": 35,
                "_exchange": {
                    "system": system, "user": user, "model": "gemini-3.6-flash",
                    "raw_response": "{}", "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                },
            }
        return {
            "status": "AVAILABLE",
            "ticker": "AAPL",
            "web_search_used": True,
            "rerating_score": 82,
            "valuation_support_score": 70,
            "cash_relative_score": 60,
            "data_quality_score": "91",  # string on purpose -- must still coerce
            "supporting_evidence": [{"claim": "filing cut guidance", "source_url": "https://sec.gov"}],
            "sources": [{"title": "8-K", "url": "https://sec.gov"}],
            "_exchange": {
                "system": system, "user": user, "model": "gemini-3.6-flash",
                "raw_response": "{}", "prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28,
            },
        }

    monkeypatch.setattr("trading_system.research_agent.generate_json", fake_generate)
    store = Store(tmp_path / "gem_scores.duckdb")
    store.open(acquire_writer=True)
    try:
        pack = research_ticker(
            _settings(gemini_api_key="test-gemini"),
            _rec("aapl", rec_units=4, action=RecommendationAction.BUY),
            conn=store.conn,
            tick_id="tick_scores",
        )
        assert pack["rerating_score"] == pytest.approx(82.0)
        assert pack["valuation_support_score"] == pytest.approx(70.0)
        assert pack["cash_relative_score"] == pytest.approx(60.0)
        assert pack["data_quality_score"] == pytest.approx(91.0)
        assert pack["bearish_severity_score"] == pytest.approx(35.0)

        from trading_system.allocation import research_sizing_multiplier

        mult = research_sizing_multiplier(pack)
        assert mult > 1.0  # net-bullish, decent quality, mild objection -> sizes up, not neutral
    finally:
        store.close()


def test_research_without_search_is_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_generate(
        settings, *, system: str, user: str, use_search: bool = False, response_schema=None
    ):
        return {
            "status": "AVAILABLE",
            "ticker": "AAPL",
            "web_search_used": False,
            "supporting_evidence": [{"claim": "8-K", "source_url": "https://sec.gov"}],
            "open_questions": [
                "Google Search grounding is not available on this Gemini key "
                "(paid Search often required). Research continued without live web search."
            ],
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
    store = Store(tmp_path / "gem_nosearch.duckdb")
    store.open(acquire_writer=True)
    try:
        pack = research_ticker(
            _settings(gemini_api_key="test-gemini"),
            _rec("aapl", rec_units=4, action=RecommendationAction.BUY),
            conn=store.conn,
            tick_id="tick_ns",
        )
        assert pack["status"] == "AVAILABLE"
        assert pack["web_search_used"] is False
        assert any("web search" in str(q).lower() or "Search grounding" in str(q) for q in pack["open_questions"])
    finally:
        store.close()


def test_held_names_join_scan_universe(tmp_path: Path) -> None:
    from datetime import date

    from trading_system.market.registry import bootstrap_smoke_universe
    from trading_system.portfolio import Lot
    from trading_system.portfolio_service import add_lot

    store = Store(tmp_path / "held.duckdb")
    store.open(acquire_writer=True)
    try:
        add_lot(
            store.conn,
            Lot(
                instrument_id="FRSH",
                acquisition_units=1,
                acquisition_price=13,
                acquired_on=date(2026, 1, 2),
            ),
        )
        mapping = bootstrap_smoke_universe(store.conn, _settings(), provider="alpaca")
        assert mapping["FRSH"] == "inst_frsh"
    finally:
        store.close()
