"""legacy-b raw conversation request shape. No live OpenAI calls.

The structured per-name/portfolio judge (final_judge, final_portfolio_judge,
build_portfolio_judge_package) was removed 2026-09 along with its tests: it was
never wired into any production call path. legacy_b_raw_conversation is the only
GPT entry point the running system uses.
"""

from __future__ import annotations

import json

import httpx
import pytest

from trading_system.config import Settings
from trading_system.llm_client import STATUS_AVAILABLE, openai_error_snippet
from trading_system.openai_judge import (
    SEC_EXCERPT_CHARS_PER_CANDIDATE,
    _legacy_b_candidate_line,
    _reasoning_model,
    build_legacy_b_raw_turns,
    legacy_b_raw_conversation,
)


def _settings(**kwargs: object) -> Settings:
    base = {
        "_env_file": None,
        "alpaca_api_key": None,
        "alpaca_secret_key": None,
        "openai_api_key": "sk-test",
        "gemini_api_key": None,
        "anthropic_api_key": None,
        "sol_judge_deadline_seconds": 30.0,
        "smoke_universe": ("AAPL", "MSFT", "BTC/USD"),
    }
    base.update(kwargs)
    return Settings(**base)  # type: ignore[arg-type]


def _response(status: int, url: str, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", url))


def test_reasoning_model_detects_sol() -> None:
    assert _reasoning_model("gpt-5.6-sol")
    assert _reasoning_model("gpt-5.6")
    assert not _reasoning_model("gpt-4o")


def test_analysis_turns_keep_prose_but_final_requires_structured_decision() -> None:
    turns = build_legacy_b_raw_turns(
        {
            "portfolio": {"total_base_units": 1000, "cash_units": 300},
            "candidates": [
                {
                    "instrument_id": "inst_aapl",
                    "ticker": "AAPL",
                    "held": True,
                    "last_price": 200,
                    "quant": {"current_units": 100, "recommended_units": 120, "horizons": []},
                }
            ],
        }
    )
    assert [turn["stage"] for turn in turns][-5:] == [
        "competitive_table",
        "rescore",
        "cash_rank",
        "live_portfolio_check",
        "final_execution",
    ]
    assert not any("JSON 형식으로만" in str(turn["prompt"]) for turn in turns)
    assert "최종 추천을 JSON 스키마" in str(turns[-1]["prompt"])
    assert "WATCH/NO_ACTION/HOLD는 보유 상태를 바꾸지 않는다" in str(turns[-1]["prompt"])


def test_legacy_b_analysis_part_defers_ranking_to_final_stage() -> None:
    turns = build_legacy_b_raw_turns(
        {
            "portfolio": {},
            "candidates": [
                {"instrument_id": "inst_aapl", "ticker": "AAPL", "held": False, "quant": {}}
            ],
        }
    )
    part_prompt = str(turns[0]["prompt"])
    assert turns[0]["stage"] == "analysis_part_1"
    assert "순위는 매기지 마라" in part_prompt
    assert "- 순위 A:" not in part_prompt
    assert "- 순위 B:" not in part_prompt
    assert "- 순위 C:" not in part_prompt


def test_legacy_b_raw_conversation_keeps_response_chain_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict] = []
    final = {"input_id": "test_input", "input_as_of": "2026-09-04T14:00:00+00:00", "recommendations": [
        {"instrument_id": "inst_aapl", "action": "BUY", "recommended_units": 100, "rank": 1,
         "thesis": "검증할 근거", "contrary_evidence": "반대 근거", "change_conditions": "가이던스 변경"}
    ]}

    class FakeClient:
        def __init__(self, *a, **k):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            payloads.append(dict(json or {}))
            n = len(payloads)
            text = json_module.dumps(final, ensure_ascii=False) if n == 6 else f"중간 답변 {n}"
            return _response(
                200,
                url,
                {
                    "id": f"resp_{n}",
                    "output_text": text,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )

    json_module = json
    monkeypatch.setattr("trading_system.openai_judge.httpx.Client", FakeClient)
    out = legacy_b_raw_conversation(
        _settings(),
        {
            "input_id": final["input_id"], "input_as_of": final["input_as_of"],
            "portfolio": {"total_base_units": 1000, "cash_units": 300},
            "candidates": [
                {
                    "instrument_id": "inst_aapl",
                    "ticker": "AAPL",
                    "held": False,
                    "last_price": 200,
                    "quant": {"current_units": 0, "recommended_units": 100, "horizons": []},
                }
            ],
        },
    )
    assert out["status"] == STATUS_AVAILABLE
    assert out["final_text"] == ""
    assert out["structured_final"] == final
    assert len(payloads) == 6
    assert "previous_response_id" not in payloads[0]
    assert payloads[1]["previous_response_id"] == "resp_1"
    assert all(p["text"]["format"] == {"type": "text"} for p in payloads[:-1])
    assert payloads[-1]["text"]["format"]["type"] == "json_schema"
    assert payloads[-1]["text"]["format"]["strict"] is True


def test_candidate_line_quotes_real_sec_filing_when_available() -> None:
    """The GPT prompt used to carry only ticker/price/quant score -- SEC filings were
    fetched (ingest_sec_extracts) but never actually reached the prompt. A candidate
    with an AVAILABLE 8-K/10-Q excerpt must now have that real, dated text quoted."""
    row = {
        "instrument_id": "inst_aapl",
        "ticker": "AAPL",
        "held": False,
        "last_price": 200,
        "quant": {"current_units": 0, "recommended_units": 100, "horizons": []},
        "sec_filing": {
            "status": STATUS_AVAILABLE,
            "form": "8-K",
            "accepted_at": "2026-09-01T20:15:00+00:00",
            "passages": [{"section": "Item 2.02 Results of Operations", "text": "Revenue rose 12% YoY."}],
        },
    }
    line = _legacy_b_candidate_line(row)
    assert "실제 SEC 공시 원문 발췌" in line
    assert "8-K" in line
    assert "2026-09-01T20:15:00+00:00" in line
    assert "Revenue rose 12% YoY." in line


def test_candidate_line_excerpt_is_capped() -> None:
    row = {
        "instrument_id": "inst_aapl",
        "ticker": "AAPL",
        "held": False,
        "quant": {},
        "sec_filing": {
            "status": STATUS_AVAILABLE,
            "form": "10-Q",
            "accepted_at": "2026-09-01",
            "passages": [{"section": "Risk Factors", "text": "x" * 5000}],
        },
    }
    line = _legacy_b_candidate_line(row)
    quoted = line.split("\n")[-1].strip()
    assert len(quoted) <= SEC_EXCERPT_CHARS_PER_CANDIDATE


def test_candidate_line_omits_sec_block_when_not_available() -> None:
    no_filing = _legacy_b_candidate_line(
        {"instrument_id": "inst_aapl", "ticker": "AAPL", "held": False, "quant": {}}
    )
    assert "SEC 공시" not in no_filing

    unavailable = _legacy_b_candidate_line(
        {
            "instrument_id": "inst_aapl",
            "ticker": "AAPL",
            "held": False,
            "quant": {},
            "sec_filing": {"status": "unavailable", "form": "8-K", "passages": []},
        }
    )
    assert "SEC 공시" not in unavailable


def test_candidate_line_quotes_gemini_research_when_available() -> None:
    """The revived Gemini research pack (research_agent.research_ticker) must actually
    reach the GPT prompt, not just move sizing silently -- GPT should be told what
    Gemini already found (and can disagree with it)."""
    row = {
        "instrument_id": "inst_aapl",
        "ticker": "AAPL",
        "held": False,
        "quant": {},
        "research": {
            "status": STATUS_AVAILABLE,
            "summary_ko": "가이던스 상향, 리레이팅 초입 국면으로 판단.",
            "rerating_score": 78,
            "valuation_support_score": 65,
            "cash_relative_score": 60,
            "data_quality_score": 85,
            "adversarial_review": {"main_objections": ["동종업계 대비 이미 밸류에이션 부담"]},
        },
    }
    line = _legacy_b_candidate_line(row)
    assert "Gemini 리서치" in line
    assert "가이던스 상향" in line
    assert "리레이팅 78" in line
    assert "데이터품질 85" in line
    assert "동종업계 대비 이미 밸류에이션 부담" in line


def test_candidate_line_omits_research_block_when_not_available() -> None:
    no_research = _legacy_b_candidate_line(
        {"instrument_id": "inst_aapl", "ticker": "AAPL", "held": False, "quant": {}}
    )
    assert "Gemini 리서치" not in no_research

    unavailable = _legacy_b_candidate_line(
        {
            "instrument_id": "inst_aapl",
            "ticker": "AAPL",
            "held": False,
            "quant": {},
            "research": {"status": "unavailable", "summary_ko": "should not appear"},
        }
    )
    assert "Gemini 리서치" not in unavailable
    assert "should not appear" not in unavailable


def test_openai_error_snippet_reads_message() -> None:
    resp = _response(
        400,
        "https://api.openai.com/v1/responses",
        {"error": {"message": "Unsupported parameter: 'temperature'", "type": "invalid_request_error"}},
    )
    assert "temperature" in openai_error_snippet(resp)


def test_pending_package_is_scoped_to_its_trading_session(tmp_path) -> None:
    """The frozen candidate package exists to resume a conversation that was cut
    short (429, budget stop) within the same session. It must NOT survive into
    the next session, or a stalled judgement would drive tomorrow's
    recommendation off yesterday's research and yesterday's quant scores."""
    import json
    from datetime import datetime, timedelta, timezone

    from trading_system.openai_judge import (
        MARKET_TZ,
        _pending_package_path,
        clear_pending_package,
        load_pending_package,
        save_pending_package,
    )

    package = {"portfolio": {"cash_units": 10}, "candidates": [{"instrument_id": "inst_aapl"}]}
    save_pending_package(tmp_path, package)
    assert load_pending_package(tmp_path) == package

    path = _pending_package_path(tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    previous = (datetime.now(timezone.utc).astimezone(MARKET_TZ) - timedelta(days=1)).date()
    body["session"] = previous.isoformat()
    path.write_text(json.dumps(body), encoding="utf-8")
    assert load_pending_package(tmp_path) is None

    clear_pending_package(tmp_path)
    assert load_pending_package(tmp_path) is None


def test_pending_package_written_before_sessions_falls_back_to_its_timestamp(tmp_path) -> None:
    """Files saved before the session field existed still have to obey the same
    boundary -- derived from saved_at rather than trusted unconditionally."""
    import json
    from datetime import datetime, timedelta, timezone

    from trading_system.openai_judge import (
        _pending_package_path,
        load_pending_package,
        save_pending_package,
    )

    save_pending_package(tmp_path, {"portfolio": {}, "candidates": [{"instrument_id": "x"}]})
    path = _pending_package_path(tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    body.pop("session")

    body["saved_at"] = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    path.write_text(json.dumps(body), encoding="utf-8")
    assert load_pending_package(tmp_path) is None

    body["saved_at"] = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    path.write_text(json.dumps(body), encoding="utf-8")
    assert load_pending_package(tmp_path) is not None


def test_resumable_turns_are_limited_to_the_same_et_session(tmp_path) -> None:
    """An identical prompt from a previous ET date must not be reused."""
    from datetime import datetime, timezone

    from trading_system.openai_judge import _load_resumable_turns
    from trading_system.storage import Store

    now = datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc)  # 11:00 ET
    previous_session = datetime(2026, 9, 5, 3, 0, tzinfo=timezone.utc)  # 23:00 ET, previous date
    same_session = datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc)  # 10:00 ET
    turns = [{"stage": "sector", "prompt": "identical prompt"}]

    store = Store(tmp_path / "resume.duckdb")
    store.open(acquire_writer=True)
    try:
        def insert_turn(transcript_id: str, created_at: datetime) -> None:
            store.conn.execute(
                """
                INSERT INTO llm_transcripts (
                    transcript_id, kind, ticker, status, user_prompt, response_text, created_at
                ) VALUES (?, 'legacy_b_sector', 'PORTFOLIO', 'AVAILABLE', ?, 'saved response', ?)
                """,
                [transcript_id, "identical prompt", created_at],
            )

        insert_turn("previous", previous_session)
        assert _load_resumable_turns(store.conn, turns, now=now) == []

        insert_turn("same", same_session)
        reused = _load_resumable_turns(store.conn, turns, now=now)
        assert len(reused) == 1
        assert reused[0]["response"] == "saved response"
    finally:
        store.close()


def test_budget_check_counts_spend_already_made_inside_this_conversation(tmp_path) -> None:
    """A judgement is ~9 turns and measured $10.92 end to end (2026-09-04), while
    usage is only written to the ledger once the whole conversation finishes. So
    the per-turn check has to add what this run has already spent -- otherwise
    every turn sees the same stale daily total and the threshold is exceeded several
    times over inside one run."""
    from trading_system.openai_judge import _turn_would_exceed_budget
    from trading_system.storage import Store

    class _Settings:
        llm_daily_budget_usd = 3.0

    store = Store(tmp_path / "budget.duckdb")
    store.open(acquire_writer=True)
    try:
        # Nothing spent yet -> one more turn fits under the threshold.
        assert (
            _turn_would_exceed_budget(
                store.conn, _Settings(), model="gpt-5.6-sol", spent_prompt=0, spent_completion=0
            )
            is False
        )
        # Two turns already paid for in THIS conversation (not yet in the ledger)
        # -> the next turn would pass the threshold, so the run must stop here.
        assert (
            _turn_would_exceed_budget(
                store.conn,
                _Settings(),
                model="gpt-5.6-sol",
                spent_prompt=500_000,
                spent_completion=30_000,
            )
            is True
        )
    finally:
        store.close()


def test_budget_check_fails_closed_without_a_ledger_connection() -> None:
    """No connection means the day's spend cannot be verified. Refuse rather than
    assume zero -- daily_auto_spend_usd returns inf for exactly this reason."""
    from trading_system.openai_judge import _turn_would_exceed_budget

    class _Settings:
        llm_daily_budget_usd = 3.0

    assert (
        _turn_would_exceed_budget(
            None, _Settings(), model="gpt-5.6-sol", spent_prompt=0, spent_completion=0
        )
        is True
    )
