"""LLM transcript history: reconstruct past ticks and persist live calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_system.llm_client import JUDGE_SYSTEM, take_exchange, usage_from_openai
from trading_system.llm_log import (
    answer_summary_from_blob,
    insert_llm_transcript,
    list_llm_sessions,
    load_llm_session,
)
from trading_system.recommendations import RecommendationAction, llm_final_record, quant_only_record
from trading_system.sec_llm import skipped_llm_record
from trading_system.storage import Store
from trading_system.storage.ticks import TickCommitPayload, commit_tick
from trading_system.ui.app import NAV_ROUTES, create_app
from trading_system.ui.llm_view import render_llm_log_html


def _quant():
    return quant_only_record(
        tick_id="tick_hist",
        decision_epoch_id="epoch_1",
        feature_snapshot_id="fs_1",
        instrument_id="inst_aapl",
        action=RecommendationAction.ENTER,
        confidence=0.8,
        recommended_units=4,
        current_units=0,
        actionable=True,
        thesis="quant wants a starter sleeve",
    )


def _wrap(rec) -> dict:
    return {
        "recommendation_id": rec.recommendation_id,
        "source": str(getattr(rec.source, "value", rec.source)),
        "instrument_id": str(rec.instrument_id),
        "action": str(getattr(rec.action, "value", rec.action)),
        "payload": json.loads(rec.model_dump_json()),
    }


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "llm.duckdb")
    s.open(acquire_writer=True)
    yield s
    s.close()


def test_nav_includes_llm_log() -> None:
    assert any(href == "/llm" for href, *_ in NAV_ROUTES)
    paths = {route.path for route in create_app().routes if hasattr(route, "path")}
    assert "/llm" in paths
    assert "/llm/{tick_id}" in paths


def test_take_exchange_is_stripped() -> None:
    payload = {"status": "AVAILABLE", "action": "BUY", "_exchange": {"system": "s", "user": "u"}}
    ex = take_exchange(payload)
    assert ex == {"system": "s", "user": "u"}
    assert "_exchange" not in payload


def test_usage_from_openai_reads_chat_and_responses_shapes() -> None:
    assert usage_from_openai({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}) == (10, 5, 15)
    assert usage_from_openai({"input_tokens": 8, "output_tokens": 2}) == (8, 2, 10)


def test_reconstructs_finished_scan(store: Store) -> None:
    quant = _quant()
    judged = llm_final_record(
        tick_id=quant.tick_id,
        decision_epoch_id=quant.decision_epoch_id,
        feature_snapshot_id=quant.feature_snapshot_id,
        instrument_id=quant.instrument_id,
        action=RecommendationAction.BUY,
        thesis="공시와 모멘텀이 맞습니다.",
        override_reasons=[],
        confidence=0.7,
        override_of_recommendation_id=quant.recommendation_id,
    )
    skipped = skipped_llm_record(
        quant_only_record(
            tick_id=quant.tick_id,
            decision_epoch_id=quant.decision_epoch_id,
            feature_snapshot_id=quant.feature_snapshot_id,
            instrument_id="inst_msft",
            action=RecommendationAction.HOLD,
        ),
        quant_status="AVAILABLE",
    )
    commit_tick(
        store.conn,
        TickCommitPayload(
            tick_id=str(quant.tick_id),
            recommendations=[_wrap(quant), _wrap(judged), _wrap(skipped)],
        ),
    )
    sessions = list_llm_sessions(store.conn)
    assert sessions
    assert sessions[0]["tick_id"] == "tick_hist"
    assert sessions[0]["live_n"] == 0
    detail = load_llm_session(store.conn, "tick_hist")
    assert detail is not None
    assert detail["judged"] == 1
    assert detail["skipped"] == 1
    assert detail["reconstructed_n"] == 1
    turn = detail["turns"][0]
    assert turn["source"] == "reconstructed"
    assert turn["ticker"] == "AAPL"
    assert "공시와 모멘텀이 맞습니다." in (turn["thesis"] or "")
    assert JUDGE_SYSTEM in turn["system_prompt"]
    html = render_llm_log_html(sessions, detail, selected_id="tick_hist")
    assert "저장본에서 재구성" in html
    assert "AAPL" in html
    assert "질문" in html
    assert "답변" in html
    assert "판단 1건" in html
    assert "토큰 사용량을 남기지 않았습니다" in html


def test_raw_legacy_conversation_without_llm_final_appears_in_history(store: Store) -> None:
    commit_tick(store.conn, TickCommitPayload(tick_id="tick_raw", recommendations=[]))
    insert_llm_transcript(
        store.conn,
        tick_id="tick_raw",
        kind="legacy_b_analysis_part_1",
        status="AVAILABLE",
        ticker="PORTFOLIO",
        model="gpt-5.6-sol",
        prompt_version="legacy_b_raw_conversation_v1",
        system_prompt="같은 대화를 이어가라.",
        user_prompt="후보를 분석해라.",
        response_text="후보 분석 답변 원문",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )
    insert_llm_transcript(
        store.conn,
        tick_id="tick_raw",
        kind="legacy_b_final_execution",
        status="AVAILABLE",
        ticker="PORTFOLIO",
        model="gpt-5.6-sol",
        prompt_version="legacy_b_raw_conversation_v1",
        system_prompt="같은 대화를 이어가라.",
        user_prompt="최종안을 작성해라.",
        response_text="# 최종 판단\nNVDA 매수, 현금 20%",
        prompt_tokens=80,
        completion_tokens=30,
        total_tokens=110,
    )

    sessions = list_llm_sessions(store.conn)
    assert sessions[0]["tick_id"] == "tick_raw"
    assert sessions[0]["rec_n"] == 0
    assert sessions[0]["judged"] == 1
    detail = load_llm_session(store.conn, "tick_raw")
    assert detail is not None
    assert detail["judged"] == 1
    assert detail["live_n"] == 2
    assert detail["total_tokens"] == 230
    html = render_llm_log_html(sessions, detail, selected_id="tick_raw")
    assert "후보 분석 Part 1" in html
    assert "최종 실행안" in html
    assert "답변 원문" in html
    assert "Gemini 리서치와" not in html


def test_llm_page_renders() -> None:
    html = TestClient(create_app()).get("/llm").text
    assert "LLM 기록" in html
    assert "실행 시각" in html


def test_research_pack_answer_summary_without_thesis() -> None:
    assert answer_summary_from_blob({"summary_ko": "수요가 살아 있다."}) == "수요가 살아 있다."
    assert "claim text" in answer_summary_from_blob(
        {"supporting_evidence": [{"claim": "claim text", "source_url": "https://sec.gov"}]}
    )
    assert "hello" in answer_summary_from_blob({}, response_text="hello world")
    fenced = '```json {"ticker":"AMD","supporting_evidence":[{"claim":"가이던스를 올렸다"}]}```'
    assert "가이던스를 올렸다" in answer_summary_from_blob({}, response_text=fenced)
    numbers_only = '```json {"ticker":"AMD","material_change":{"quant_action":"EXIT"}}```'
    assert "```" not in answer_summary_from_blob({}, response_text=numbers_only)
    assert "AMD" in answer_summary_from_blob({}, response_text=numbers_only)


def test_research_turn_html_shows_claim_not_empty() -> None:
    html = render_llm_log_html(
        [
            {
                "tick_id": "tick_amd",
                "date": "2026-09-01",
                "time": "23:00:00",
                "judged": 0,
                "live_n": 1,
                "rec_n": 1,
            }
        ],
        {
            "tick_id": "tick_amd",
            "turns": [
                {
                    "kind": "research",
                    "ticker": "AMD",
                    "status": "AVAILABLE",
                    "source": "live",
                    "model": "gemini-3.6-flash",
                    "thesis": None,
                    "response_text": json.dumps(
                        {
                            "supporting_evidence": [
                                {"claim": "가이던스를 올렸다", "source_url": "https://sec.gov"}
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    "user_prompt": "{}",
                    "system_prompt": "s",
                    "prompt_tokens": 1433,
                    "completion_tokens": 2075,
                    "total_tokens": 6362,
                }
            ],
            "prompt_tokens": 1433,
            "completion_tokens": 2075,
            "total_tokens": 6362,
            "token_calls": 1,
        },
        selected_id="tick_amd",
    )
    assert "기록된 답변 없음" not in html
    assert "가이던스를 올렸다" in html
