"""Optional LLM extract + judge. Bounded timeout. Never executes trades.

Missing keys → NOT_CONFIGURED (no invented extract/judge).
Rate limits / quota → RATE_LIMITED / QUOTA_EXCEEDED. Other failures → UNAVAILABLE.
Never synthesizes an LLM-final recommendation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from trading_system.config import Settings
from trading_system.providers.http_client import parse_retry_after
from trading_system.providers.never_block import ProviderHttpError, RetryKind, RetryPolicy, run_with_deadline

DEFAULT_LLM_MODEL = "gpt-5.6-sol"
EXTRACT_PROMPT_VERSION = "extract_v2"
JUDGE_PROMPT_VERSION = "judge_legacy_b_v6"
LEGACY_B_RAW_PROMPT_VERSION = "legacy_b_structured_final_v5"
EXTRACT_USER_CHARS = 8000
EXTRACT_SYSTEM = (
    "Read this filing and return JSON with whatever actually changed. "
    "Do not invent facts. Skip empty fields. There is no required key list."
)
JUDGE_SYSTEM = (
    "You are the final investment judge for a personal US-stock + BTC portfolio.\n\n"
    "[ANALYSIS PURPOSE]\n"
    "Follow the legacy-b decision format. Find names that can continue to rerate even after a rise: "
    "the question is not merely whether a stock is cheap, but whether earnings revisions, a durable cycle, "
    "and institutional repricing can make an expensive stock become more expensive. Quant is one input, "
    "not the verdict. allocation_trace explains the cross-section and sizing; min_opportunity_score is a "
    "gate for a new buy, never an automatic sell rule for a holding.\n\n"
    "[MANDATORY Q1-Q5]\n"
    "Q1 Consensus revision: after the latest earnings, determine beat/miss, EPS and revenue estimate direction, "
    "guidance changes, and whether this is genuine rerating or mere price momentum.\n"
    "Q2 Earnings support and valuation: decide whether next-12-month EPS/revenue growth supports the price move; "
    "compare forward valuation with history and peers; distinguish priced-in expectations from revision headroom.\n"
    "Q3 TAM/cycle duration: decide whether the catalyst is a one-quarter event or a one-to-two-year cycle and "
    "classify the cycle as early, middle, late, or unknown.\n"
    "Q4 Peer confirmation: determine whether peers and the sector confirm the move; flag an isolated move, "
    "temporary theme, manipulation/liquidity risk, or a defensible company-specific narrative.\n"
    "Q5 Institutional repricing: assess recent target-price/coverage changes and institutional reassessment; "
    "distinguish early catch-up from already-completed repricing.\n\n"
    "[KEEP THREE JUDGMENTS SEPARATE]\n"
    "Score rerating potential, current undervaluation, and attractiveness versus CASH separately. Never call a "
    "stock attractive merely because it is cheap, and never call it undervalued merely because momentum is strong. "
    "For CASH comparison, incorporate the latest supplied macro/regime, rates, liquidity, geopolitical risk, "
    "theme rotation, target-price gap, valuation crowding, catalyst stage, and evidence quality.\n\n"
    "Also score global competitive position, margin quality, and an overall investment score precisely on a "
    "0-to-100 scale; do not use rough buckets as a substitute for the numeric scores.\n\n"
    "[FINAL ACTION — EXACT ENUMS]\n"
    "If current_units is zero, action must be ENTER (매수) or NO_ACTION (관망). "
    "If current_units is positive, action must be ADD (추가매수), HOLD (유지), or EXIT (매도). "
    "For HOLD, recommended_units must equal current_units. For EXIT or NO_ACTION, recommended_units must be 0. "
    "For ENTER, recommended_units must be positive. For ADD, it must be greater than current_units. "
    "HOLD_NOT_NEW_BUY means retain the current holding, not zero it. EXIT_NEGATIVE_OUTLOOK is a Quant exit clue, "
    "not an order. Give the decision for now, while distinguishing a valid thesis from a good entry point.\n\n"
    "Never place or route orders. Do not invent prices, filings, URLs, news, estimates, or flows. State unknowns "
    "plainly. Return only the required structured JSON. summary_ko and all explanatory fields must be Korean."
)

LEGACY_B_RAW_SYSTEM = (
    "너는 미국 주식과 BTC를 검토하는 투자 분석가다. 주문을 실행하지 않고 사용자가 읽을 판단만 제공한다. "
    "이 대화는 legacy-b에서 사용자가 여러 프롬프트를 같은 GPT 대화에 차례로 붙여 넣던 과정을 그대로 재현한다. "
    "앞선 답변과 조사 내용을 다음 질문에서도 계속 기억하고, 제공된 Quant 수치는 후보 선별용 참고자료로만 사용하라. "
    "최신 사실이 필요한 단계에서는 웹 검색으로 뉴스, 실적, 컨센서스, 밸류에이션, 업황, 기관 재평가와 시장 이슈를 확인하라. "
    "확인하지 못한 사실은 추측하지 말고 확인 불가라고 밝혀라. 분석 답변은 한국어로 작성하되 최종 판단 단계는 지정된 JSON 스키마만 반환하라."
)

STATUS_NOT_CONFIGURED = "NOT_CONFIGURED"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_AVAILABLE = "AVAILABLE"
STATUS_DEGRADED = "DEGRADED"
STATUS_RATE_LIMITED = "RATE_LIMITED"
STATUS_QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

_JUDGE_FAILURE_STATUSES = {
    STATUS_UNAVAILABLE,
    STATUS_DEGRADED,
    STATUS_RATE_LIMITED,
    STATUS_QUOTA_EXCEEDED,
}

_LLM_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay_seconds=0.5,
    max_delay_seconds=8.0,
    jitter=0.2,
    total_deadline_seconds=30.0,
)

_log = logging.getLogger("trading_system.llm_client")
_RATE_HEADERS = (
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
)


def openai_configured(settings: Settings) -> bool:
    return bool(settings.openai_api_key)


def is_judge_failure(status: str) -> bool:
    return status in _JUDGE_FAILURE_STATUSES or status == STATUS_NOT_CONFIGURED


def _not_configured(prompt_version: str) -> dict[str, Any]:
    return {
        "status": STATUS_NOT_CONFIGURED,
        "degraded": True,
        "action": None,
        "prompt_version": prompt_version,
        "note": "OpenAI key not configured. Enter it in Settings. Quant path is independent.",
    }


def _openai_error_fields(payload: object) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return "", ""
    err = payload.get("error")
    if not isinstance(err, dict):
        return "", ""
    return str(err.get("code") or ""), str(err.get("type") or "")


def _openai_error_message(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    err = payload.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or "").strip()
    if isinstance(err, str):
        return err.strip()
    return ""


def openai_error_snippet(resp: httpx.Response, *, limit: int = 180) -> str:
    """Short provider message for logs/UI. Never includes request bodies or keys."""
    try:
        payload = resp.json()
    except (ValueError, json.JSONDecodeError):
        return ""
    msg = " ".join(_openai_error_message(payload).split())
    return msg[: max(0, int(limit))]


def _is_quota_error(status_code: int, code: str, err_type: str) -> bool:
    blob = f"{code} {err_type}".lower()
    if "insufficient_quota" in blob or ("quota" in blob and "rate" not in blob):
        return True
    if status_code == 402:
        return True
    return err_type.lower() in {"insufficient_quota", "billing_not_active"}


def inspect_openai_failure(resp: httpx.Response) -> tuple[RetryKind, str, float | None]:
    """Classify OpenAI errors. Does not return secrets or full bodies."""
    retry_after = parse_retry_after(resp.headers)
    payload: object = None
    try:
        payload = resp.json()
    except (ValueError, json.JSONDecodeError):
        payload = None
    code, err_type = _openai_error_fields(payload)
    rate_hdrs = " ".join(f"{name}={resp.headers.get(name) or '-'}" for name in _RATE_HEADERS)
    _log.info(
        "openai status=%s code=%s type=%s message=%s retry_after=%s %s",
        resp.status_code,
        code or "-",
        err_type or "-",
        (_openai_error_message(payload) or "-")[:180],
        retry_after,
        rate_hdrs,
    )
    if _is_quota_error(resp.status_code, code, err_type):
        return RetryKind.BILLING, STATUS_QUOTA_EXCEEDED, retry_after
    if resp.status_code == 429 or code == "rate_limit_exceeded" or "rate_limit" in err_type:
        return RetryKind.RATE_LIMITED, STATUS_RATE_LIMITED, retry_after
    if resp.status_code in {401, 403}:
        return RetryKind.NON_RETRYABLE, STATUS_UNAVAILABLE, retry_after
    if 500 <= resp.status_code <= 599:
        return RetryKind.RETRYABLE, STATUS_UNAVAILABLE, retry_after
    return RetryKind.NON_RETRYABLE, STATUS_UNAVAILABLE, retry_after


def take_exchange(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Remove the private transcript blob so it never lands in recommendation JSON."""
    if not isinstance(payload, dict):
        return None
    raw = payload.pop("_exchange", None)
    return raw if isinstance(raw, dict) else None


def _exchange(
    *,
    system: str,
    user: str,
    model: str,
    raw_response: str | None = None,
    error: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "system": system,
        "user": user,
        "model": model,
        "raw_response": raw_response,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def usage_from_openai(usage: object) -> tuple[int | None, int | None, int | None]:
    """Chat Completions uses prompt/completion; Responses uses input/output."""
    if not isinstance(usage, dict):
        return None, None, None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    try:
        prompt_n = int(prompt) if prompt is not None else None
    except (TypeError, ValueError):
        prompt_n = None
    try:
        completion_n = int(completion) if completion is not None else None
    except (TypeError, ValueError):
        completion_n = None
    try:
        total_n = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_n = None
    if total_n is None and prompt_n is not None and completion_n is not None:
        total_n = prompt_n + completion_n
    return prompt_n, completion_n, total_n


def _with_exchange(payload: dict[str, Any], exchange: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["_exchange"] = exchange
    return out


def _post_openai(api_key: str, system: str, user: str, model: str) -> dict[str, Any]:
    captured: dict[str, Any] = {"content": None, "usage": None}

    def _call() -> dict[str, Any]:
        with httpx.Client(timeout=25.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
        if resp.status_code >= 400:
            kind, status, retry_after = inspect_openai_failure(resp)
            raise ProviderHttpError(
                f"OpenAI {resp.status_code} {status}",
                status_code=resp.status_code,
                kind=kind,
                retry_after_seconds=retry_after,
            )
        body = resp.json()
        captured["usage"] = body.get("usage")
        content = body["choices"][0]["message"]["content"]
        captured["content"] = content
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            parsed.setdefault("status", STATUS_AVAILABLE)
        return parsed

    result = run_with_deadline(_call, policy=_LLM_POLICY, label="llm_openai")
    prompt_n, completion_n, total_n = usage_from_openai(captured.get("usage"))
    if result.ok and isinstance(result.value, dict):
        return _with_exchange(
            result.value,
            _exchange(
                system=system,
                user=user,
                model=model,
                raw_response=captured.get("content"),
                prompt_tokens=prompt_n,
                completion_tokens=completion_n,
                total_tokens=total_n,
            ),
        )
    status = STATUS_UNAVAILABLE
    if result.kind == RetryKind.BILLING:
        status = STATUS_QUOTA_EXCEEDED
    elif result.kind == RetryKind.RATE_LIMITED:
        status = STATUS_RATE_LIMITED
    return _with_exchange(
        {
            "status": status,
            "degraded": True,
            "action": None,
            "error": result.error,
            "note": f"OpenAI {status}. Quant path is independent.",
        },
        _exchange(
            system=system,
            user=user,
            model=model,
            error=str(result.error) if result.error else status,
            prompt_tokens=prompt_n,
            completion_tokens=completion_n,
            total_tokens=total_n,
        ),
    )


def extract_filing(settings: Settings, text: str) -> dict[str, Any]:
    if not openai_configured(settings):
        return _not_configured(EXTRACT_PROMPT_VERSION)
    model = settings.llm_model or DEFAULT_LLM_MODEL
    return _post_openai(
        settings.openai_api_key,  # type: ignore[arg-type]
        EXTRACT_SYSTEM,
        text[:EXTRACT_USER_CHARS],
        model,
    )

