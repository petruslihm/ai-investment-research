"""legacy-b's raw multi-turn GPT conversation. Recommendation only — never places orders.

The per-name/portfolio structured JSON judge (final_judge, final_portfolio_judge) was
removed 2026-09: it was never wired into any production call path (v1_cycle.py only
calls legacy_b_raw_conversation below). JUDGE_SYSTEM/JUDGE_PROMPT_VERSION still live in
llm_client.py because llm_log.py reconstructs historical per-name judge transcripts from
older ticks that predate this change.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from trading_system.config import Settings
from trading_system.filing_chunks import passages_as_text
from trading_system.market.calendar import MARKET_TZ
from trading_system.llm_client import (
    LEGACY_B_RAW_PROMPT_VERSION,
    LEGACY_B_RAW_SYSTEM,
    STATUS_AVAILABLE,
    STATUS_NOT_CONFIGURED,
    STATUS_QUOTA_EXCEEDED,
    STATUS_RATE_LIMITED,
    STATUS_UNAVAILABLE,
    _exchange,
    _with_exchange,
    inspect_openai_failure,
    openai_configured,
    openai_error_snippet,
    usage_from_openai,
)
from trading_system.providers.never_block import ProviderHttpError, RetryKind, RetryPolicy, run_with_deadline

DEFAULT_JUDGE_MODEL = "gpt-5.6-sol"
CHEAP_JUDGE_ALIASES = frozenset({"gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"})
# Per-candidate cap on quoted SEC filing text. Bounded well below
# filing_chunks.PASSAGE_BUDGET_CHARS (12,000) because this is repeated for every
# candidate (up to llm_research_max_names, default 16) in every analysis_part turn --
# the point is a short, real, citable excerpt, not the whole filing.
SEC_EXCERPT_CHARS_PER_CANDIDATE = 600
# Same bounded-per-candidate rationale as SEC_EXCERPT_CHARS_PER_CANDIDATE, applied to
# the revived Gemini research pack's summary_ko.
RESEARCH_SUMMARY_CHARS_PER_CANDIDATE = 500


def _http_timeout(deadline_seconds: float) -> float:
    # No upper cap: the old min(..., 300.0) made the httpx client itself time out
    # before the outer run_with_deadline budget did on any web_search + reasoning
    # turn longer than ~5 minutes, which is exactly what caused live scans to fail
    # with "deadline expired" and an empty final_text.
    return max(20.0, float(deadline_seconds) - 15.0)


def sol_retry_policy(settings: Settings) -> RetryPolicy:
    deadline = max(30.0, float(settings.sol_judge_deadline_seconds))
    return RetryPolicy(
        max_attempts=2,
        base_delay_seconds=0.8,
        max_delay_seconds=10.0,
        jitter=0.2,
        total_deadline_seconds=deadline,
    )


def resolve_judge_model(settings: Settings) -> str:
    raw = (settings.llm_judge_model or settings.llm_model or DEFAULT_JUDGE_MODEL).strip()
    if not raw or raw.lower() in CHEAP_JUDGE_ALIASES:
        return DEFAULT_JUDGE_MODEL
    return raw


def _reasoning_model(model: str) -> bool:
    key = (model or "").strip().lower()
    return key.startswith(("gpt-5", "o1", "o3", "o4")) or "sol" in key


def _raise_openai_http(resp: httpx.Response) -> None:
    kind, status, retry_after = inspect_openai_failure(resp)
    detail = openai_error_snippet(resp)
    message = f"OpenAI {resp.status_code} {status}"
    if detail:
        message = f"{message}: {detail}"
    raise ProviderHttpError(
        message,
        status_code=resp.status_code,
        kind=kind,
        retry_after_seconds=retry_after,
    )


def _output_text(body: dict[str, Any]) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    chunks: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            text = part.get("text") or part.get("output_text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
    return "\n".join(chunks)


def _legacy_b_candidate_line(row: dict[str, Any]) -> str:
    quant = row.get("quant") if isinstance(row.get("quant"), dict) else {}
    horizons = quant.get("horizons") if isinstance(quant.get("horizons"), list) else []
    horizon_bits: list[str] = []
    for item in horizons:
        if not isinstance(item, dict):
            continue
        horizon = item.get("horizon")
        value = item.get("expected_return")
        try:
            horizon_bits.append(f"{int(horizon)}일 {float(value) * 100:+.2f}%")
        except (TypeError, ValueError):
            continue
    ticker = str(row.get("ticker") or row.get("instrument_id") or "?")
    held = bool(row.get("held"))
    position = "보유" if held else "신규 후보"
    current_units = quant.get("current_units")
    recommended_units = quant.get("recommended_units")
    confidence = quant.get("confidence")
    opportunity = quant.get("opportunity_score")
    rank_score = row.get("rank_score")
    last_price = row.get("last_price")
    line = (
        f"{ticker} | {position} | 현재가 {last_price if last_price is not None else '확인 필요'} | "
        f"Quant 전망 {' / '.join(horizon_bits) or '없음'} | 신뢰도 {confidence} | "
        f"기회점수 {opportunity} | 순위점수 {rank_score} | "
        f"현재단위 {current_units} | Quant 제안단위 {recommended_units}"
    )
    sec = row.get("sec_filing") if isinstance(row.get("sec_filing"), dict) else None
    if sec and sec.get("status") == STATUS_AVAILABLE:
        excerpt = passages_as_text(
            sec.get("passages") or [], limit=SEC_EXCERPT_CHARS_PER_CANDIDATE
        ).strip()
        if excerpt:
            form = sec.get("form") or "?"
            accepted = sec.get("accepted_at") or "?"
            line += (
                f"\n  [실제 SEC 공시 원문 발췌 — {form}, 접수 {accepted}, 지어내지 말고 이 발췌를 근거로 삼을 것]\n"
                f"  {excerpt}"
            )
    research = row.get("research") if isinstance(row.get("research"), dict) else None
    if research and str(research.get("status") or "") == STATUS_AVAILABLE:
        bits: list[str] = []
        summary = str(research.get("summary_ko") or "").strip()
        if summary:
            bits.append(f"요약: {summary[:RESEARCH_SUMMARY_CHARS_PER_CANDIDATE]}")
        score_bits: list[str] = []
        for label, key in (
            ("리레이팅", "rerating_score"),
            ("밸류에이션지지", "valuation_support_score"),
            ("현금대비매력", "cash_relative_score"),
            ("데이터품질", "data_quality_score"),
        ):
            val = research.get(key)
            if isinstance(val, (int, float)):
                score_bits.append(f"{label} {float(val):.0f}")
        if score_bits:
            bits.append("Gemini 자체 점수(0-100): " + " / ".join(score_bits))
        adv = research.get("adversarial_review") if isinstance(research.get("adversarial_review"), dict) else {}
        objections = [str(x) for x in (adv.get("main_objections") or []) if str(x).strip()]
        if objections:
            bits.append("역검토 반대논거: " + "; ".join(objections[:2])[:300])
        if bits:
            line += "\n  [Gemini 리서치 — 웹검색+역검토 완료, 이 요약을 참고자료로만 쓰고 직접 재검증할 것]\n  " + "\n  ".join(
                bits
            )
    return line


def build_legacy_b_raw_turns(
    package: dict[str, Any],
    *,
    per_part: int = 6,
) -> list[dict[str, Any]]:
    """Reproduce legacy-b's pasted prompts as one continuous GPT conversation.

    The old JSON-recording turn is deliberately omitted. The user-facing result is
    the model's untouched final prose response.
    """
    candidates = [row for row in (package.get("candidates") or []) if isinstance(row, dict)]
    part_size = max(1, min(10, int(per_part)))
    chunks = [candidates[i : i + part_size] for i in range(0, len(candidates), part_size)]
    total_parts = max(1, len(chunks))
    turns: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks or [[]], start=1):
        block = "\n".join(_legacy_b_candidate_line(row) for row in chunk) or "(후보 없음)"
        turns.append(
            {
                "stage": f"analysis_part_{index}",
                "use_search": True,
                "prompt": f"""아래는 내 Quant 스크리너가 선별한 미국 주식·BTC 리레이팅 후보야. Part {index}/{total_parts}.
보유 종목은 선별 기준 미달이어도 포함했다. Quant 수치는 방향을 확정하는 답이 아니라 후보 압축용 참고자료다.

분석 대상:
{block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[분석 목적]
"MU처럼 이미 많이 올랐지만, 실적 추정치·산업 병목·테마 때문에 계속 리레이팅될 종목"을 찾는다.
싸서 사는 게 아니라, 비싸도 계속 비싸질 수 있는지 확인하는 게 목적이다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

각 종목에 대해 웹에서 최신 자료를 직접 찾아 아래 5가지 질문에 답해라.

[Q1] 최근 실적 발표일과 EPS·매출 beat/miss, 실적 이후 컨센서스와 가이던스 변화. 진짜 리레이팅인가 단순 모멘텀인가?
[Q2] 다음 12개월 EPS·매출 전망이 주가 상승을 정당화하는가? Forward PER/EV·EBITDA의 역사·동종업계 위치와 추가 상향 여력은?
[Q3] TAM·업황·테마가 1분기짜리인가 1~2년짜리인가? 현재 사이클은 초입·중반·끝물 중 어디인가?
[Q4] 경쟁사와 섹터도 같이 움직이는가? 혼자 움직인다면 개별 이슈·작전성·진짜 내러티브 중 무엇인가?
[Q5] 최근 1~3개월 목표가·신규 커버리지·기관 재평가가 뒤늦게 따라오는가, 이미 선반영됐는가?

종목별로 Q1~Q5에 대한 답을 구체적으로 정리해라. 이 Part에는 아직 전체 후보와 현금이 다 모이지 않았으니
순위는 매기지 마라 — 순위 A/B/C는 전체 후보를 다 본 뒤 마지막 단계에서 한 번만 매긴다.
확인하지 못한 수치나 사건은 지어내지 말고 확인 불가라고 써라.""",
            }
        )

    portfolio = package.get("portfolio") if isinstance(package.get("portfolio"), dict) else {}
    held_lines = []
    for row in candidates:
        if bool(row.get("held")):
            held_lines.append(_legacy_b_candidate_line(row))
    portfolio_block = "\n".join(held_lines) or "(보유 종목 없음)"
    cash_units = portfolio.get("cash_units")
    total_units = portfolio.get("total_base_units")

    turns.extend(
        [
            {
                "stage": "competitive_table",
                "use_search": False,
                "prompt": (
                    "이제 앞에서 분석한 모든 종목을 한 표로 합쳐라. 각 종목 옆에 글로벌 경쟁력, "
                    "경쟁사 중 대략적인 순위, 마진의 질, Q1~Q5 핵심과 리레이팅/저평가/현금대비 판단을 함께 적어줘. "
                    "시간이 오래 걸려도 좋으니 앞선 조사 내용을 빠뜨리지 마라."
                ),
            },
            {
                "stage": "rescore",
                "use_search": False,
                "prompt": (
                    "회피해야 할 종목은 따로 명확히 표시하고, 나머지는 너무 러프하게 묶지 말고 100점 만점으로 정밀 재채점해라. "
                    "리레이팅 점수, 현재 저평가 점수, 현금 대비 매력도 점수를 분리하고 각 점수의 이유를 짧게 적어라."
                ),
            },
            {
                "stage": "cash_rank",
                "use_search": True,
                "prompt": (
                    "이제 모든 종목 사이에 <현금>을 순위에 넣는다. 주식시장에 영향을 줄 최신 이슈—금리, 연준, "
                    "최근 또는 예정된 주요 지표, 지정학, 테마와 수급 쏠림—를 먼저 웹에서 확인해 정리해라. "
                    "그 이슈를 반영해 현금의 순위와 권장 현금 비중을 정해라. 나는 평소 투자자금을 주식에 적극적으로 넣지만, "
                    "큰 이벤트와 기대수익 부족 시에는 현금을 보유한다."
                ),
            },
            {
                "stage": "live_portfolio_check",
                "use_search": True,
                "prompt": f"""현재 내 투자용 기준 자금은 총 {total_units}단위이고 현재 현금은 약 {cash_units}단위다.

<내 보유 종목>
{portfolio_block}

나머지 신규 후보도 앞에서 분석한 목록 그대로다. 섹터 겹침은 신경 쓰지 않는다.
지금 현재 시점의 각 보유 종목과 신규 후보에 대해 현재가, 최신 뉴스, 실적·컨센서스 변화와 필요한 수급 정보를 다시 확인해라.
보유 종목은 각각 <추가매수/유지/매도>, 신규 후보는 각각 <매수/관망> 중 하나로 판단해라.""",
            },
            {
                "stage": "final_execution",
                "use_search": False,
                "prompt": """이제 앞선 모든 조사와 판단을 종합해 지금 당장 실행할 최종안을 작성해라.

반드시 포함할 것:
1. 순위 A — MU식 리레이팅 가능성 전체 순위
2. 순위 B — 현재가 대비 저평가 전체 순위
3. 순위 C — 모든 종목과 CASH를 함께 놓은 현금 대비 매력도 전체 순위
4. 권장 현금 비중과 이유
5. 보유 종목별 최종 판단: 추가매수/유지/매도 중 하나
6. 신규 후보별 최종 판단: 매수/관망 중 하나
7. 그 실행 뒤의 최종 포트폴리오 비중
8. 판단을 뒤집을 핵심 위험과 확인 불가 항목

이 답변이 사용자에게 그대로 표시된다. JSON이나 코드블록으로 쓰지 말고, 보기 좋은 한국어 제목·표·목록으로 완성된 최종 보고서만 답해라. "
설명용 서문이나 API 관련 말은 붙이지 마라.""",
            },
        ]
    )
    return turns


def _current_session(moment: datetime) -> date:
    """The ET calendar date a moment belongs to -- the same day boundary the auto
    scan/train slots use (see daily_scan.py), so 'same session' means the same
    thing everywhere. Note an ET date spans roughly 13:00 KST to 13:00 KST next
    day, which is why an after-close retry is still the same session."""
    return moment.astimezone(MARKET_TZ).date()


def resume_session_bounds_utc(moment: datetime | None = None) -> tuple[datetime, datetime]:
    """UTC bounds for the current ET calendar session used by scan retries."""
    current = moment or datetime.now(timezone.utc)
    session = _current_session(current)
    start_local = datetime.combine(session, time.min, tzinfo=MARKET_TZ)
    end_local = datetime.combine(session + timedelta(days=1), time.min, tzinfo=MARKET_TZ)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

# Same literal v1_cycle.py and ui/app.py already use for the pre-flight block, so
# a mid-conversation stop shows the user the identical status/badge.
STATUS_BUDGET_EXCEEDED = "BUDGET_EXCEEDED"

# Cost of one more judgement turn, for the budget check below. Measured
# 2026-09-04 across a full 9-turn run: per-turn prompt grew 114K -> 321K as the
# chained context accumulated (2.13M total), completion 8.6K -> 18.4K. The
# average turn is used rather than the peak so the guard does not refuse to
# start a conversation it could actually afford.
TURN_PROMPT_TOKENS_ESTIMATE = 236_000
TURN_COMPLETION_TOKENS_ESTIMATE = 13_500


def _turn_would_exceed_budget(
    conn: Any,
    settings: Settings,
    *,
    model: str,
    spent_prompt: int,
    spent_completion: int,
) -> bool:
    """Would one more turn pass the automatic UTC-day soft budget threshold?

    Counts what this conversation has already spent (not yet in the ledger --
    usage is recorded once, after the whole conversation) plus one more turn.
    """
    try:
        from trading_system.llm_budget import estimate_usd, would_exceed_budget
    except Exception:  # noqa: BLE001 -- budget guard must never break the judgement
        return False
    try:
        in_flight = estimate_usd(model, int(spent_prompt), int(spent_completion))
        return would_exceed_budget(
            conn,
            settings,
            model=model,
            prompt_tokens=TURN_PROMPT_TOKENS_ESTIMATE,
            completion_tokens=TURN_COMPLETION_TOKENS_ESTIMATE,
            extra_usd=in_flight,
        )
    except Exception:  # noqa: BLE001
        return False


PENDING_JUDGE_FILENAME = "gpt_judge_pending.json"


def _pending_package_path(artifacts_dir: Any) -> Path:
    return Path(artifacts_dir) / PENDING_JUDGE_FILENAME


def save_pending_package(artifacts_dir: Any, package: dict[str, Any]) -> None:
    """Freeze the exact candidate/portfolio package a GPT judge conversation is
    about to use, to a local JSON file (not the DB -- this must survive even if
    the writer lease/DB gets fought over by the next attempt).

    Why this exists: the online SGD model updates a little on every cycle (see
    apply_online_updates in v1_cycle.py), which nudges borderline names across
    the BUY threshold between attempts -- so a plain retry can end up with a
    different candidate set turn to turn, which breaks _load_resumable_turns'
    exact-prompt-text match beyond whichever turn already succeeded. Freezing
    the package sidesteps that: every turn's prompt is a pure function of this
    package (see build_legacy_b_raw_turns), so replaying the same package always
    regenerates byte-identical prompts, no matter how much quant scores drift
    upstream in the meantime.

    Scoped to the trading session it was built in (see load_pending_package):
    resuming an interrupted conversation is the point, carrying yesterday's
    research into today's recommendation is not.
    """
    path = _pending_package_path(artifacts_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        body = {
            "saved_at": now.isoformat(),
            "session": _current_session(now).isoformat(),
            "package": package,
        }
        path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # best-effort cache -- a failed write just means no resume next time


def load_pending_package(artifacts_dir: Any) -> dict[str, Any] | None:
    """Load a package saved by save_pending_package, if it belongs to the current
    trading session.

    Not a clock: the bound is the ET session the package was frozen in, the same
    day boundary the auto scan/train slots use. Resuming an interrupted
    conversation later the same session (including the after-close retry window,
    which is still the same ET date) is exactly what this is for; letting a
    budget- or 429-stopped conversation carry yesterday's research into today's
    recommendation is not. Gemini research has its own short TTL and independent
    price/portfolio/regime invalidation checks.
    """
    path = _pending_package_path(artifacts_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        body = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    package = body.get("package")
    if not isinstance(package, dict):
        return None

    now = datetime.now(timezone.utc)
    saved_session = body.get("session")
    if saved_session is None:
        # Written before sessions were recorded -- fall back to the session the
        # save timestamp falls in rather than trusting it unconditionally.
        try:
            stamp = datetime.fromisoformat(str(body.get("saved_at")))
        except (TypeError, ValueError):
            return None
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        saved_session = _current_session(stamp).isoformat()
    if str(saved_session) != _current_session(now).isoformat():
        return None
    return package


def clear_pending_package(artifacts_dir: Any) -> None:
    path = _pending_package_path(artifacts_dir)
    try:
        path.unlink()
    except OSError:
        pass


def _load_resumable_turns(
    conn: Any,
    turns: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Find a prior attempt's already-succeeded turns so a retried conversation
    (e.g. after an OpenAI 429 mid-conversation) does not re-pay for turns that
    already completed. Returns the PREFIX of `turns` whose exact prompt text
    matches a saved, successful transcript for that stage -- stops at the first
    stage that has no match, so a single changed candidate list (different
    prompt text) safely falls back to a normal from-scratch run instead of
    silently mixing unrelated turns.
    """
    if conn is None:
        return []
    session_start, session_end = resume_session_bounds_utc(now)
    reused: list[dict[str, Any]] = []
    for turn in turns:
        stage = str(turn["stage"])
        try:
            row = conn.execute(
                """
                SELECT user_prompt, response_text, model, prompt_tokens, completion_tokens
                FROM llm_transcripts
                WHERE kind = ? AND ticker = 'PORTFOLIO' AND status = 'AVAILABLE'
                  AND created_at >= ? AND created_at < ?
                ORDER BY created_at DESC LIMIT 1
                """,
                [f"legacy_b_{stage}", session_start, session_end],
            ).fetchone()
        except Exception:  # noqa: BLE001 -- resume is an optimization, never a hard dependency
            return reused
        if not row or row[0] != turn["prompt"] or not str(row[1] or "").strip():
            break
        reused.append(
            {
                "stage": stage,
                "prompt": str(turn["prompt"]),
                "response": str(row[1]),
                "model": row[2],
                "prompt_tokens": row[3],
                "completion_tokens": row[4],
                "total_tokens": int(row[3] or 0) + int(row[4] or 0),
                "reused": True,
            }
        )
    return reused


def legacy_b_raw_conversation(
    settings: Settings,
    package: dict[str, Any],
    *,
    conn: Any = None,
    enforce_llm_budget: bool = False,
) -> dict[str, Any]:
    """Run legacy-b's multi-turn GPT workflow and return the final prose untouched.

    conn (optional): when given, a retried conversation first checks for a recent
    attempt's already-succeeded turns (see _load_resumable_turns) and resumes
    after them instead of re-running the whole conversation from turn 1 -- the
    early analysis turns are by far the most expensive (full candidate research
    dumped into the prompt), so this matters most after a 429 partway through.

    enforce_llm_budget: check the daily soft threshold before EVERY turn, not just
    once before the conversation. A judgement is 9-ish turns and measured 2.13M
    prompt tokens end to end (2026-09-04), so a single pre-flight estimate can
    wave through a run that then costs many times the threshold. Stopping mid-
    conversation is cheap here precisely because the turns already paid for are
    saved and a later attempt resumes from them.
    """
    model = resolve_judge_model(settings)
    turns = build_legacy_b_raw_turns(package)
    joined_prompts = "\n\n".join(str(turn["prompt"]) for turn in turns)
    if not openai_configured(settings):
        return _with_exchange(
            {
                "status": STATUS_NOT_CONFIGURED,
                "prompt_version": LEGACY_B_RAW_PROMPT_VERSION,
                "final_text": "",
                "turns": [],
                "note": "OpenAI key not configured.",
            },
            _exchange(system=LEGACY_B_RAW_SYSTEM, user=joined_prompts, model=model, error=STATUS_NOT_CONFIGURED),
        )

    api_key = str(settings.openai_api_key)
    reused_turns = _load_resumable_turns(conn, turns)
    resume_from = len(reused_turns)
    previous_response_id: str | None = None
    completed: list[dict[str, Any]] = list(reused_turns)
    total_prompt = sum(int(t.get("prompt_tokens") or 0) for t in reused_turns)
    total_completion = sum(int(t.get("completion_tokens") or 0) for t in reused_turns)
    final_text = str(reused_turns[-1]["response"]) if reused_turns else ""

    for i, turn in enumerate(turns[resume_from:], start=resume_from):
        prompt = str(turn["prompt"])
        if enforce_llm_budget and _turn_would_exceed_budget(
            conn, settings, model=model, spent_prompt=total_prompt, spent_completion=total_completion
        ):
            # Stop here rather than at the end: the turns already paid for are
            # returned (and saved) below, so the next attempt resumes from them
            # instead of paying for them twice.
            return _with_exchange(
                {
                    "status": STATUS_BUDGET_EXCEEDED,
                    "degraded": True,
                    "prompt_version": LEGACY_B_RAW_PROMPT_VERSION,
                    "final_text": final_text,
                    "turns": completed,
                    "failed_stage": str(turn["stage"]),
                    "error": "자동 실행의 UTC 일일 LLM 예산 기준값에 도달했습니다(soft limit).",
                },
                _exchange(
                    system=LEGACY_B_RAW_SYSTEM,
                    user=joined_prompts,
                    model=model,
                    raw_response=final_text or None,
                    prompt_tokens=total_prompt or None,
                    completion_tokens=total_completion or None,
                    total_tokens=(total_prompt + total_completion) or None,
                    error=STATUS_BUDGET_EXCEEDED,
                ),
            )
        # The first live call after a resumed prefix has no previous_response_id to
        # chain from (OpenAI's response id for the reused turns was never persisted --
        # see _load_resumable_turns), so it inlines the reused turns as explicit prior
        # messages instead. Every call after that chains normally via
        # previous_response_id, same as a from-scratch run.
        resume_boundary = i == resume_from and bool(reused_turns)

        def _one_turn() -> dict[str, Any]:
            if resume_boundary:
                input_payload: Any = []
                for t in reused_turns:
                    input_payload.append({"role": "user", "content": str(t["prompt"])})
                    input_payload.append({"role": "assistant", "content": str(t["response"])})
                input_payload.append({"role": "user", "content": prompt})
            else:
                input_payload = prompt
            payload: dict[str, Any] = {
                "model": model,
                "instructions": LEGACY_B_RAW_SYSTEM,
                "input": input_payload,
                "store": True,
                "text": {"format": {"type": "text"}, "verbosity": "high"},
            }
            if previous_response_id and not resume_boundary:
                payload["previous_response_id"] = previous_response_id
            if bool(turn.get("use_search")):
                payload["tools"] = [{"type": "web_search"}]
            if _reasoning_model(model):
                payload["reasoning"] = {"effort": "medium"}
            with httpx.Client(timeout=_http_timeout(float(settings.sol_judge_deadline_seconds))) as client:
                response = client.post(
                    "https://api.openai.com/v1/responses",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
            if response.status_code >= 400:
                _raise_openai_http(response)
            return response.json()

        result = run_with_deadline(
            _one_turn,
            policy=sol_retry_policy(settings),
            label=f"legacy_b_{turn['stage']}",
        )
        if not result.ok or not isinstance(result.value, dict):
            status = STATUS_UNAVAILABLE
            if result.kind == RetryKind.BILLING:
                status = STATUS_QUOTA_EXCEEDED
            elif result.kind == RetryKind.RATE_LIMITED:
                status = STATUS_RATE_LIMITED
            return _with_exchange(
                {
                    "status": status,
                    "degraded": True,
                    "prompt_version": LEGACY_B_RAW_PROMPT_VERSION,
                    "final_text": final_text,
                    "turns": completed,
                    "failed_stage": turn["stage"],
                    "error": str(result.error) if result.error else status,
                },
                _exchange(
                    system=LEGACY_B_RAW_SYSTEM,
                    user=joined_prompts,
                    model=model,
                    raw_response=final_text or None,
                    prompt_tokens=total_prompt or None,
                    completion_tokens=total_completion or None,
                    total_tokens=(total_prompt + total_completion) or None,
                    error=str(result.error) if result.error else status,
                ),
            )
        body = result.value
        text = _output_text(body).strip()
        if not text:
            return _with_exchange(
                {
                    "status": STATUS_UNAVAILABLE,
                    "degraded": True,
                    "prompt_version": LEGACY_B_RAW_PROMPT_VERSION,
                    "final_text": final_text,
                    "turns": completed,
                    "failed_stage": turn["stage"],
                    "error": "OpenAI returned empty text",
                },
                _exchange(
                    system=LEGACY_B_RAW_SYSTEM,
                    user=joined_prompts,
                    model=model,
                    raw_response=final_text or None,
                    prompt_tokens=total_prompt or None,
                    completion_tokens=total_completion or None,
                    total_tokens=(total_prompt + total_completion) or None,
                    error="OpenAI returned empty text",
                ),
            )
        previous_response_id = str(body.get("id") or "") or previous_response_id
        prompt_n, completion_n, _total_n = usage_from_openai(body.get("usage"))
        total_prompt += int(prompt_n or 0)
        total_completion += int(completion_n or 0)
        final_text = text
        completed.append(
            {
                "stage": str(turn["stage"]),
                "prompt": prompt,
                "response": text,
                "response_id": previous_response_id,
                "web_search_requested": bool(turn.get("use_search")),
                "model": model,
                "prompt_tokens": prompt_n,
                "completion_tokens": completion_n,
                "total_tokens": (int(prompt_n or 0) + int(completion_n or 0)),
            }
        )

    return _with_exchange(
        {
            "status": STATUS_AVAILABLE,
            "prompt_version": LEGACY_B_RAW_PROMPT_VERSION,
            "final_text": final_text,
            "turns": completed,
            "response_id": previous_response_id,
        },
        _exchange(
            system=LEGACY_B_RAW_SYSTEM,
            user=joined_prompts,
            model=model,
            raw_response=final_text,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        ),
    )
