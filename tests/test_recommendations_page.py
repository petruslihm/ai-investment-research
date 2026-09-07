"""Recommendations board: ranked candidates, LLM rationale, today flag, ticker contains search."""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from trading_system.config import Settings
from trading_system.ui.app import (
    _fmt_signed_pct,
    _is_buy_candidate,
    _score,
    _today_priority,
    create_app,
    render_recommendations_html,
)


def _quant(**over: object) -> dict:
    row = {
        "instrument_id": "inst_aapl",
        "action": "BUY",
        "actionable": True,
        "current_units": 0,
        "acquisition_units": 2,
        "marked_units_final": 3,
        "recommended_units": 12,
        "delta_units": 9,
        "confidence": 0.8,
        "horizons": [{"horizon": 5, "expected_return": 0.04, "rank_score": 0.72}],
        "thesis": "quant thesis",
    }
    row.update(over)
    return row


def _llm(**over: object) -> dict:
    row = {
        "instrument_id": "inst_aapl",
        "action": "BUY",
        "override_reasons": [],
        "thesis": "공시와 모멘텀이 같은 방향입니다.",
    }
    row.update(over)
    return row


def test_score_uses_expected_return_not_rounded_rank() -> None:
    assert _fmt_signed_pct(-0.0) == "0.0%"
    assert _fmt_signed_pct(-0.0001) == "0.0%"
    assert _fmt_signed_pct(-0.1659) == "-16.6%"
    assert _score(_quant(horizons=[{"horizon": 5, "expected_return": 0.0052, "rank_score": -0.166}])) == "+0.5%"
    assert _score(_quant(horizons=[{"horizon": 5, "expected_return": -0.002, "rank_score": -0.2}])) == "-0.2%"


def test_buy_candidate_is_allocated_weight_only() -> None:
    cfg = Settings(_env_file=None)
    assert _is_buy_candidate(_quant(recommended_units=12), settings=cfg, total_base_units=1000)
    assert not _is_buy_candidate(_quant(recommended_units=12, action="HOLD"), settings=cfg, total_base_units=1000)
    assert not _is_buy_candidate(_quant(recommended_units=0, action="BUY"), settings=cfg, total_base_units=1000)
    assert not _is_buy_candidate(
        _quant(recommended_units=4.28, current_units=0, delta_units=4.28),
        settings=cfg,
        total_base_units=1200,
    )


def test_today_priority_requires_live_quant_and_llm_buy() -> None:
    q = _quant()
    cfg = Settings(_env_file=None)
    kw = {"settings": cfg, "total_base_units": 1000.0}
    assert _today_priority(q, _llm(), **kw)
    assert not _today_priority({**q, "actionable": False}, _llm(), **kw)
    assert not _today_priority(q, None, **kw)
    assert not _today_priority(q, _llm(override_reasons=["NOT_A_CANDIDATE"]), **kw)
    assert not _today_priority(q, _llm(action="HOLD"), **kw)
    assert not _today_priority({**q, "delta_units": 0}, _llm(), **kw)


def test_ranked_board_shows_rationale_today_and_search() -> None:
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {"total_base_units": 1000},
            "quant": [
                _quant(),
                _quant(
                    instrument_id="inst_msft",
                    action="HOLD",
                    recommended_units=0,
                    delta_units=0,
                    current_units=3,
                    horizons=[{"horizon": 5, "rank_score": 0.4}],
                ),
                _quant(
                    instrument_id="inst_aaon",
                    action="HOLD",
                    recommended_units=0,
                    delta_units=0,
                    current_units=0,
                    horizons=[{"horizon": 5, "rank_score": 0.11}],
                ),
            ],
            "llm_final": [
                _llm(),
                _llm(
                    instrument_id="inst_msft",
                    action="HOLD",
                    thesis="보유 포지션 유지가 낫습니다.",
                ),
            ],
        },
        settings=Settings(_env_file=None),
    )
    assert 'id="ticker-search"' in html
    assert "ticker.includes(q)" in html
    assert "data-ticker=\"aapl\"" in html
    assert "data-ticker=\"aaon\"" in html
    assert ">1</b><span>순위</span>" in html
    assert "최종 판단 근거" in html
    assert "현재 보유" in html
    assert "현재 평가" in html
    assert "AI 추천" in html
    assert "증감" in html
    assert "FINAL close" in html
    assert "공시와 모멘텀이 같은 방향입니다." in html
    assert "오늘 우선" in html
    assert 'data-today="1"' in html
    assert "보유 종목" in html
    assert "전체 종목 점수" in html
    assert "A → AA → AAPL" in html
    # Skip-style LLM dump of the whole universe must not appear.
    assert "이번 매수 후보가 아니라 LLM 판단을 건너뛰었습니다" not in html


def test_boring_names_collapse_into_a_compact_table_not_full_cards() -> None:
    """A ~500-name scan used to render every single name as a full _rec_card, which
    put ~500 large divs in the DOM at once. This reproduces the actual real-world
    shape of a boring row: allocation.py's _note_for_exclusion sets allocation_note
    for EVERY exclusion reason, so BELOW_MIN_OPPORTUNITY (most of a real scan) always
    carries a note too -- allocation_note alone must not be enough to force the full
    card, or the split never triggers. The note must still survive as a hover
    tooltip, not a visible paragraph."""
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {"total_base_units": 1000},
            "quant": [
                {
                    "instrument_id": "inst_bore",
                    "action": "NO_ACTION",
                    "actionable": False,
                    "current_units": 0,
                    "acquisition_units": 0,
                    "marked_units_final": 0,
                    "recommended_units": 0,
                    "delta_units": 0,
                    "confidence": 0.31,
                    "horizons": [{"horizon": 5, "expected_return": 0.001, "rank_score": 0.1}],
                    "exclusion_reason": "BELOW_MIN_OPPORTUNITY",
                    "allocation_note": "BORE은(는) 기회 점수가 최소 기준에 미달해 배분 대상이 아닙니다.",
                    # deliberately no thesis (thesis is always-set diagnostic noise, never a signal)
                }
            ],
            "llm_final": [],
        },
        settings=Settings(_env_file=None),
    )
    assert '<table class="trace-table rec-table">' in html
    assert 'class="rec-row"' in html
    assert 'data-ticker="bore"' in html
    assert "기회 점수가 최소 기준에 미달" in html  # kept, just as a title=... tooltip
    # The compact path must not pull in the full card's heavier markup for this name.
    assert 'class="rec-card"' not in html


def test_near_miss_names_still_get_the_full_card_in_all_scores() -> None:
    """BELOW_MIN_POSITION means Quant actually wanted to size this name and only the
    position floor zeroed it out -- a real near-miss, unlike the blanket
    BELOW_MIN_OPPORTUNITY rejection most of a ~500-name scan gets. The split must not
    hide that story behind the compact row."""
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {"total_base_units": 1000},
            "quant": [
                _quant(
                    instrument_id="inst_story",
                    action="NO_ACTION",
                    actionable=False,
                    current_units=0,
                    acquisition_units=0,
                    marked_units_final=0,
                    recommended_units=0,
                    delta_units=0,
                    exclusion_reason="BELOW_MIN_POSITION",
                    allocation_note="STORY는(는) 최소 비중에 못 미쳐 제외되었습니다.",
                )
            ],
            "llm_final": [],
        },
        settings=Settings(_env_file=None),
    )
    assert "최소 비중에 못 미쳐 제외되었습니다" in html
    assert 'data-ticker="story"' in html


def test_research_summary_renders_on_the_card_and_forces_detail_view() -> None:
    """The revived Gemini research pass (research_agent.research_ticker) must
    actually reach the recommendations page, not just move sizing silently -- and a
    name whose only story is a research pack (no allocation_note/thesis/LLM entry)
    must still get the full card, not the compact row, or the summary is invisible."""
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {"total_base_units": 1000},
            "quant": [
                {
                    "instrument_id": "inst_rsch",
                    "action": "NO_ACTION",
                    "actionable": False,
                    "current_units": 0,
                    "acquisition_units": 0,
                    "marked_units_final": 0,
                    "recommended_units": 0,
                    "delta_units": 0,
                    "confidence": 0.4,
                    "horizons": [{"horizon": 5, "expected_return": 0.01, "rank_score": 0.2}],
                    "research_summary_ko": "가이던스 상향, 리레이팅 초입 국면.",
                    "research_rerating_score": 78.0,
                    "research_valuation_support_score": 65.0,
                    "research_cash_relative_score": 60.0,
                    "research_data_quality_score": 85.0,
                    "research_sizing_multiplier": 1.12,
                }
            ],
            "llm_final": [],
        },
        settings=Settings(_env_file=None),
    )
    assert "Gemini 리서치" in html
    assert "가이던스 상향, 리레이팅 초입 국면." in html
    assert "리레이팅 78" in html
    assert "사이징 1.12x" in html
    assert 'class="rec-card"' in html  # forced into the detail view, not the compact table
    assert 'data-ticker="rsch"' in html


def test_allocation_note_explains_below_floor() -> None:
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {"total_base_units": 1200},
            "quant": [
                _quant(
                    instrument_id="inst_pltr",
                    action="NO_ACTION",
                    recommended_units=0,
                    delta_units=0,
                    current_units=0,
                    acquisition_units=0,
                    marked_units_final=0,
                    opportunity_score=0.012,
                    equity_budget=0.80,
                    initial_units=4.28,
                    pre_floor_units=4.28,
                    exclusion_reason="BELOW_MIN_POSITION",
                    allocation_note=(
                        "PLTR은(는) 계산된 목표가 0.36%로, 최소 의미 비중 1.0%에 미달해 추천하지 않습니다."
                    ),
                )
            ],
            "llm_final": [],
        },
        settings=Settings(_env_file=None),
    )
    assert "관망" in html
    assert "0.36%" in html
    assert "1.0%" in html
    assert "기회점수" in html
    assert "제외 BELOW_MIN_POSITION" in html
    assert "AI 추천 4.28 u" not in html


def test_deployment_banner_explains_partial_equity() -> None:
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {
                "total_base_units": 1200,
                "formula_version": "quant_alloc_v4",
                "deployment_note": (
                    "현재 후보들은 매수 기준은 통과했지만 신호 강도가 높지 않아 "
                    "최대 100% 중 24%만 주식에 배분합니다."
                ),
                "residual_cash_label": "기회 강도가 낮아 현금 유지",
                "deployment_factor": 0.3,
                "max_total_stock_weight": 1.0,
                "equity_budget": 0.24,
                "equity_budget_used": 0.24,
                "strong_opportunity": {
                    "strong_horizon_mean_return": 0.04,
                    "used_score": 0.023,
                    "source": "derived_from_strong_horizon_mean_return",
                    "why": "V1 stand-in, not OOS-calibrated",
                },
            },
            "quant": [],
            "llm_final": [],
        },
        settings=Settings(_env_file=None),
    )
    assert "최대 100% 중 24%만 주식에 배분합니다." in html
    assert "기회 강도가 낮아 현금 유지" in html
    assert "투입계수 0.30" in html
    assert "not OOS-calibrated" in html


def test_allocation_trace_table_on_recommendations_page() -> None:
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {
                "total_base_units": 1000,
                "formula_version": "quant_alloc_v4",
                "min_opportunity_score": 0.01,
                "equity_budget": 0.16,
                "deployment_factor": 0.16,
                "aggregate_conviction": 0.64,
                "full_deployment_equivalent_names": 4.0,
                "btc_weight": 0.10,
                "cash_weight": 0.74,
                "stock_weights": {"inst_nvda": 0.16},
                "shortlist_order": ["inst_nvda"],
                "max_equity_budget": 0.90,
                "residual_cash_reason": "LOW_CONVICTION",
                "strong_opportunity": {
                    "strong_horizon_mean_return": 0.04,
                    "used_score": 0.023,
                },
            },
            "quant": [
                _quant(
                    instrument_id="inst_nvda",
                    opportunity_score=0.0157,
                    conviction=0.44,
                    exclusion_reason=None,
                    recommended_units=160,
                    horizons=[
                        {"horizon": 5, "expected_return": 0.018},
                        {"horizon": 10, "expected_return": 0.027},
                        {"horizon": 20, "expected_return": 0.041},
                    ],
                ),
                _quant(
                    instrument_id="inst_msft",
                    action="HOLD",
                    recommended_units=0,
                    delta_units=0,
                    current_units=0,
                    opportunity_score=0.0097,
                    conviction=0.0,
                    exclusion_reason="BELOW_MIN_OPPORTUNITY",
                    horizons=[
                        {"horizon": 5, "expected_return": 0.004},
                        {"horizon": 10, "expected_return": 0.011},
                        {"horizon": 20, "expected_return": 0.016},
                    ],
                ),
            ],
            "llm_final": [],
        },
        settings=Settings(_env_file=None),
    )
    assert "배분 추적" in html
    assert "pass_0.01" in html
    assert "NVDA" in html
    assert "MSFT" in html
    assert "=== ALLOCATION TRACE ===" in html
    assert ">Y<" in html
    assert ">N<" in html
    assert "매수 컷오프 미달 보유" in html


def test_held_keep_note_renders_on_recommendations_page() -> None:
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {"total_base_units": 1200, "held_keep_weight": 0.184},
            "quant": [
                _quant(
                    instrument_id="inst_goog",
                    action="HOLD",
                    recommended_units=221,
                    delta_units=0,
                    current_units=221,
                    acquisition_units=221,
                    marked_units_final=221,
                    exclusion_reason="HOLD_NOT_NEW_BUY",
                    allocation_note="GOOG은(는) 새 매수 기준에는 못 미치지만, 전망이 나빠 청산할 정도는 아니라 보유를 유지합니다.",
                )
            ],
            "llm_final": [],
        },
        settings=Settings(_env_file=None),
    )
    assert "보유를 유지합니다" in html
    assert "Quant 추천 221 u" in html
    assert "새 매수 기준에는 못 미치지만" in html


def test_tiny_target_renders_as_watch() -> None:
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {"total_base_units": 1200},
            "quant": [
                _quant(
                    recommended_units=4.28,
                    delta_units=4.28,
                    current_units=0,
                    acquisition_units=0,
                    marked_units_final=0,
                )
            ],
            "llm_final": [],
        },
        settings=Settings(_env_file=None),
    )
    assert "관망" in html
    assert "추천 규모가 작아 실행하지 않습니다." in html
    assert "AI 추천 4.28 u" not in html
    assert "최종 목표 4.28 u" in html
    assert "기술 상세" in html
    assert ">1</b><span>순위</span>" not in html


def test_recommendations_route_has_search_box() -> None:
    html = TestClient(create_app()).get("/recommendations").text
    assert 'id="ticker-search"' in html
    assert "매수 후보" in html


def test_recommendations_show_legacy_b_committee_note_when_final_text_missing() -> None:
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {"total_base_units": 1000},
            "quant": [_quant()],
            "llm_final": [_llm(action="ENTER")],
            "portfolio_committee": {
                "status": "AVAILABLE",
            },
        },
        settings=Settings(_env_file=None),
    )
    assert "포트폴리오 최종 심사" in html
    assert "GPT 응답이 비어 있습니다." in html


def test_recommendations_show_raw_gpt_final_answer_without_json_parsing() -> None:
    raw = "최종 판단\n\nAAPL: 매수\nMSFT: 유지\n권장 현금: 20%"
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {"total_base_units": 1000},
            "quant": [_quant()],
            "llm_final": [],
            "portfolio_committee": {"status": "AVAILABLE", "final_text": raw},
        },
        settings=Settings(_env_file=None),
    )
    assert "GPT 최종 투자 판단" in html
    assert "AAPL: 매수" in html
    assert "MSFT: 유지" in html
    assert "권장 현금: 20%" in html


def test_settings_page_has_shutdown_button() -> None:
    html = TestClient(create_app()).get("/settings").text
    assert "action=\"/shutdown\"" in html
    assert "서버 종료" in html
    assert "confirm(" in html.split('action="/shutdown"')[1].split("</form>")[0]


def test_shutdown_route_sends_self_sigint_via_background_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The button must trigger the exact same graceful path a manual Ctrl+C in the
    server's own console takes (self os.kill(SIGINT), which uvicorn already handles)
    -- not an abrupt os._exit that could cut off an in-flight DB write. Must never
    actually kill anything in this test process, so os.kill is replaced first."""
    import signal

    calls: list[tuple[int, int]] = []
    monkeypatch.setattr("trading_system.ui.app.os.kill", lambda pid, sig: calls.append((pid, sig)))

    client = TestClient(create_app())
    resp = client.post("/shutdown")
    assert resp.status_code == 200
    assert "종료" in resp.text

    for _ in range(50):
        if calls:
            break
        time.sleep(0.05)
    assert calls == [(os.getpid(), signal.SIGINT)]


def test_shell_shows_server_start_time_on_every_page() -> None:
    """Lets the user tell a freshly restarted server apart from one that's been up
    for a while (e.g. after a stuck job forced a restart) -- rendered once per
    create_app() call, so it changes only on an actual process restart."""
    html = TestClient(create_app()).get("/settings").text
    assert "서버 시작" in html


def test_dashboard_has_separate_train_button() -> None:
    html = TestClient(create_app()).get("/").text
    assert "/train-once" in html
    assert "모델 학습" in html
    assert "지금 스캔 실행" in html
    assert "action='/run-once'" in html
    assert "source=auto" not in html
    assert "예산 기준값은 기본 $15" in html
    assert "soft limit" in html
    assert "실제 비용의 절대 상한은 아닙니다" in html


def test_unselected_sol_shows_quant_units_and_short_rationale() -> None:
    html = render_recommendations_html(
        {
            "quant_status": "AVAILABLE",
            "llm_judge_status": "AVAILABLE",
            "allocation": {"total_base_units": 1000},
            "quant": [
                _quant(
                    instrument_id="inst_nem",
                    action="ENTER",
                    recommended_units=18,
                    delta_units=18,
                    current_units=0,
                    acquisition_units=0,
                    marked_units_final=None,
                )
            ],
            "llm_final": [
                _llm(
                    instrument_id="inst_nem",
                    action="NO_ACTION",
                    override_reasons=["NOT_SELECTED_FOR_FINAL_JUDGE"],
                    thesis="Quant 매수 후보이지만 이번 Sol 최종 판단 한도에는 못 들어갔습니다. 표시된 단위는 Quant 숫자입니다.",
                    rationale_detail="보유 종목과 매수 상위만 GPT-5.6 Sol을 태웁니다.",
                )
            ],
        },
        settings=Settings(_env_file=None),
    )
    assert "Quant 추천 18 u" in html
    assert "AI 추천 18 u" not in html
    assert "상세보기" in html
    assert html.index("상세보기") < html.index("quant thesis")
    assert "Quant 매수 후보이지만" in html
