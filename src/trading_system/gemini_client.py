"""Gemini research calls. JSON evidence packs, optional Google Search grounding.

Search grounding often needs a paid Gemini plan. Free keys should still return
a pack from filings/quant, with the limitation recorded in open_questions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from trading_system.config import Settings
from trading_system.llm_client import (
    STATUS_AVAILABLE,
    STATUS_NOT_CONFIGURED,
    STATUS_QUOTA_EXCEEDED,
    STATUS_RATE_LIMITED,
    STATUS_UNAVAILABLE,
    _exchange,
    _with_exchange,
)
from trading_system.providers.http_client import parse_retry_after
from trading_system.providers.never_block import ProviderHttpError, RetryKind, RetryPolicy, run_with_deadline

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_log = logging.getLogger("trading_system.gemini_client")

def _http_timeout(deadline_seconds: float) -> float:
    return max(20.0, min(float(deadline_seconds) - 15.0, 300.0))


def gemini_retry_policy(settings: Settings) -> RetryPolicy:
    deadline = max(30.0, float(settings.gemini_deadline_seconds))
    return RetryPolicy(
        max_attempts=3,
        base_delay_seconds=1.5,
        max_delay_seconds=20.0,
        jitter=0.2,
        total_deadline_seconds=deadline,
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.I)


def gemini_configured(settings: Settings) -> bool:
    return bool((settings.gemini_api_key or "").strip())


def research_model(settings: Settings) -> str:
    return (settings.gemini_research_model or DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def _not_configured(system: str, user: str, model: str) -> dict[str, Any]:
    return _with_exchange(
        {
            "status": STATUS_NOT_CONFIGURED,
            "degraded": True,
            "note": "Gemini key not configured.",
        },
        _exchange(system=system, user=user, model=model, error="NOT_CONFIGURED"),
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parts_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = (content or {}).get("parts") if isinstance(content, dict) else None
    chunks: list[str] = []
    for part in parts or []:
        if isinstance(part, dict) and part.get("text"):
            chunks.append(str(part["text"]))
    return "\n".join(chunks)


def _grounding_urls(payload: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    candidates = payload.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return out
    meta = candidates[0].get("groundingMetadata") or candidates[0].get("grounding_metadata") or {}
    chunks = meta.get("groundingChunks") or meta.get("grounding_chunks") or []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        web = chunk.get("web") if isinstance(chunk.get("web"), dict) else {}
        url = str(web.get("uri") or web.get("url") or "").strip()
        title = str(web.get("title") or "").strip()
        if not url:
            retrieved = chunk.get("retrievedContext") or chunk.get("retrieved_context") or {}
            if isinstance(retrieved, dict):
                url = str(retrieved.get("uri") or retrieved.get("url") or "").strip()
                title = title or str(retrieved.get("title") or "").strip()
        if url:
            out.append({"title": title, "url": url})
    return out


def _urls_from_text(text: str) -> list[dict[str, str]]:
    """Gemini 3 thinking often omits groundingMetadata but still cites redirect URLs."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in _URL_RE.findall(text or ""):
        url = raw.rstrip(").,]")
        if "vertexaisearch.cloud.google.com" not in url and "grounding-api-redirect" not in url:
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append({"title": "", "url": url})
    return out


def _usage(payload: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    meta = payload.get("usageMetadata") or payload.get("usage_metadata") or {}
    if not isinstance(meta, dict):
        return None, None, None
    prompt = meta.get("promptTokenCount") or meta.get("prompt_token_count")
    completion = meta.get("candidatesTokenCount") or meta.get("candidates_token_count")
    total = meta.get("totalTokenCount") or meta.get("total_token_count")
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


def _classify_gemini(resp: httpx.Response) -> tuple[RetryKind, str]:
    if resp.status_code == 429:
        return RetryKind.RATE_LIMITED, STATUS_RATE_LIMITED
    if resp.status_code in {401, 403}:
        return RetryKind.NON_RETRYABLE, STATUS_UNAVAILABLE
    if resp.status_code == 402:
        return RetryKind.BILLING, STATUS_QUOTA_EXCEEDED
    if 500 <= resp.status_code <= 599:
        return RetryKind.RETRYABLE, STATUS_UNAVAILABLE
    return RetryKind.NON_RETRYABLE, STATUS_UNAVAILABLE


def generate_json(
    settings: Settings,
    *,
    system: str,
    user: str,
    use_search: bool = False,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = research_model(settings)
    if not gemini_configured(settings):
        return _not_configured(system, user, model)
    api_key = str(settings.gemini_api_key)
    captured: dict[str, Any] = {"text": None, "usage": (None, None, None), "urls": []}

    def _post(*, search: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        if isinstance(response_schema, dict) and response_schema:
            body["generationConfig"]["responseJsonSchema"] = response_schema
        if search:
            body["tools"] = [{"google_search": {}}]
        with httpx.Client(timeout=_http_timeout(float(settings.gemini_deadline_seconds))) as client:
            resp = client.post(
                GEMINI_URL.format(model=model),
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=body,
            )
        if resp.status_code >= 400:
            kind, status = _classify_gemini(resp)
            retry_after = parse_retry_after(resp.headers)
            snippet = (resp.text or "")[:180].replace(api_key, "")
            _log.info("gemini status=%s search=%s body=%s", resp.status_code, search, snippet)
            raise ProviderHttpError(
                f"Gemini {resp.status_code} {status}",
                status_code=resp.status_code,
                kind=kind,
                retry_after_seconds=retry_after,
            )
        payload = resp.json()
        text = _parts_text(payload)
        captured["text"] = text
        captured["usage"] = _usage(payload)
        captured["urls"] = _grounding_urls(payload)
        if not captured["urls"]:
            captured["urls"] = _urls_from_text(text)
        parsed = _extract_json(text)
        if parsed is None:
            raise ProviderHttpError(
                "Gemini returned no JSON object",
                status_code=422,
                kind=RetryKind.NON_RETRYABLE,
            )
        required = response_schema.get("required") if isinstance(response_schema, dict) else None
        if isinstance(required, list) and any(str(key) not in parsed for key in required):
            raise ProviderHttpError(
                "Gemini JSON did not match the required research schema",
                status_code=422,
                kind=RetryKind.NON_RETRYABLE,
            )
        parsed.setdefault("status", STATUS_AVAILABLE)
        parsed["grounding_urls"] = list(captured["urls"])
        if search:
            parsed["web_search_used"] = True
            sources = list(parsed.get("sources") or [])
            existing = {str(s.get("url") if isinstance(s, dict) else s) for s in sources}
            for row in captured["urls"]:
                if row["url"] not in existing:
                    sources.append(row)
            parsed["sources"] = sources
        else:
            parsed["web_search_used"] = False
        return parsed

    search_wanted = bool(use_search and settings.gemini_web_search)
    search_note = ""

    def _call() -> dict[str, Any]:
        nonlocal search_note
        if search_wanted:
            try:
                return _post(search=True)
            except ProviderHttpError as exc:
                code = exc.status_code or 0
                if code in {400, 403, 404, 422}:
                    search_note = (
                        "Gemini rejected the Search tool on this request "
                        f"(HTTP {code}). Research continued without live web search."
                    )
                    _log.info("gemini search unavailable; retrying without search")
                    return _post(search=False)
                raise
        return _post(search=False)

    result = run_with_deadline(_call, policy=gemini_retry_policy(settings), label="llm_gemini")
    prompt_n, completion_n, total_n = captured.get("usage") or (None, None, None)
    if result.ok and isinstance(result.value, dict):
        payload = dict(result.value)
        if search_note:
            questions = list(payload.get("open_questions") or [])
            questions.append(search_note)
            payload["open_questions"] = questions
            payload["web_search_used"] = False
        return _with_exchange(
            payload,
            _exchange(
                system=system,
                user=user,
                model=model,
                raw_response=captured.get("text"),
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
            "error": result.error,
            "note": f"Gemini {status}.",
            "open_questions": [search_note] if search_note else [],
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
