"""Cheapest-possible reachability probes for every configured provider.

Each probe is one request. Where a free metadata endpoint proves the credential we use
it; only OpenAI needs a real (1-token) completion because auth alone does not reveal a
billing/quota block. Probes run concurrently so a single button covers all providers.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from trading_system.llm_client import inspect_openai_failure
from trading_system.providers.never_block import RetryKind
from trading_system.sec_edgar import _headers as sec_headers

STATUS_OK = "AVAILABLE"
STATUS_NOT_CONFIGURED = "NOT_CONFIGURED"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_AUTH = "AUTH_REQUIRED"
STATUS_RATE_LIMITED = "RATE_LIMITED"
STATUS_QUOTA = "QUOTA_EXCEEDED"

# Cheapest widely available OpenAI chat model; only used for a 1-token liveness ping.
PING_MODEL = "gpt-4o-mini"
# Cheapest Anthropic model; used only if GET /v1/models is missing for this key.
ANTHROPIC_PING_MODEL = "claude-haiku-4-5"
_TIMEOUT = 12.0


def _clean_secret(value: str | None) -> str | None:
    text = (value or "").strip().strip('"').strip("'").strip()
    return text or None


def _provider_error_detail(resp: httpx.Response) -> str:
    """Human-readable status without leaking keys or full response bodies."""
    code = resp.status_code
    err_type = ""
    try:
        payload = resp.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            err_type = str(err.get("type") or err.get("code") or "").strip()
        elif isinstance(payload.get("type"), str):
            err_type = payload["type"].strip()
    if err_type:
        return f"HTTP {code} ({err_type})"
    return f"HTTP {code}"


@dataclass
class ProbeResult:
    key: str
    label: str
    status: str
    detail: str
    usage: str | None = None
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def configured(self) -> bool:
        return self.status != STATUS_NOT_CONFIGURED

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "usage": self.usage,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class ConnectionReport:
    checked_at: datetime
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def failures(self) -> list[ProbeResult]:
        return [r for r in self.results if r.configured and not r.ok]

    def as_dict(self) -> dict[str, object]:
        return {
            "checked_at": self.checked_at.isoformat(),
            "results": [r.as_dict() for r in self.results],
            "ok": not self.failures,
        }


def _status_from_code(code: int) -> str:
    if code in (401, 403):
        return STATUS_AUTH
    if code == 402:
        return STATUS_QUOTA
    if code == 429:
        return STATUS_RATE_LIMITED
    return STATUS_UNAVAILABLE


def _not_configured(key: str, label: str, hint: str) -> ProbeResult:
    return ProbeResult(key=key, label=label, status=STATUS_NOT_CONFIGURED, detail=hint)


def _rate_limit_note(headers: httpx.Headers) -> str | None:
    remaining = headers.get("x-ratelimit-remaining-tokens")
    limit = headers.get("x-ratelimit-limit-tokens")
    if remaining is None:
        return None
    try:
        left = f"{int(remaining):,}"
        total = f"{int(limit):,}" if limit is not None else None
    except (TypeError, ValueError):
        return None
    return f"분당 토큰 한도 잔여 {left}" + (f" / {total}" if total else "")


def probe_alpaca(api_key: str | None, secret_key: str | None) -> ProbeResult:
    label = "Alpaca 시장 데이터"
    api_key = _clean_secret(api_key)
    secret_key = _clean_secret(secret_key)
    if not (api_key and secret_key):
        return _not_configured("alpaca", label, "API Key와 Secret Key를 모두 입력하세요.")
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    started = time.monotonic()
    last = "응답이 없습니다."
    status = STATUS_UNAVAILABLE
    # Paper and live accounts answer on different hosts; the clock endpoint costs no data quota.
    for host in ("https://paper-api.alpaca.markets", "https://api.alpaca.markets"):
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(f"{host}/v2/clock", headers=headers)
        except httpx.HTTPError as exc:
            last = f"네트워크 오류 ({type(exc).__name__})"
            continue
        if resp.status_code == 200:
            return ProbeResult(
                key="alpaca",
                label=label,
                status=STATUS_OK,
                detail="키가 유효합니다.",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
        status = _status_from_code(resp.status_code)
        last = f"HTTP {resp.status_code}"
    return ProbeResult(
        key="alpaca",
        label=label,
        status=status,
        detail=last,
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )


def probe_openai(api_key: str | None) -> ProbeResult:
    label = "OpenAI"
    api_key = _clean_secret(api_key)
    if not api_key:
        return _not_configured("openai", label, "OpenAI API Key를 입력하세요.")
    started = time.monotonic()
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": PING_MODEL,
                    "max_tokens": 1,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
    except httpx.HTTPError as exc:
        return ProbeResult(
            key="openai",
            label=label,
            status=STATUS_UNAVAILABLE,
            detail=f"네트워크 오류 ({type(exc).__name__})",
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    elapsed = int((time.monotonic() - started) * 1000)
    if resp.status_code == 200:
        used = None
        try:
            usage = resp.json().get("usage") or {}
            total = usage.get("total_tokens")
            if total is not None:
                used = f"이번 확인에 {int(total)} 토큰 사용"
        except (ValueError, TypeError, AttributeError):
            used = None
        note = _rate_limit_note(resp.headers)
        return ProbeResult(
            key="openai",
            label=label,
            status=STATUS_OK,
            detail=f"{PING_MODEL} 응답을 받았습니다.",
            usage=" · ".join(x for x in (used, note) if x),
            elapsed_ms=elapsed,
        )
    kind, reason, _retry_after = inspect_openai_failure(resp)
    if resp.status_code == 429:
        status = STATUS_QUOTA if kind is RetryKind.BILLING else STATUS_RATE_LIMITED
    else:
        status = _status_from_code(resp.status_code)
    return ProbeResult(
        key="openai",
        label=label,
        status=status,
        detail=f"HTTP {resp.status_code} ({reason})",
        usage=_rate_limit_note(resp.headers),
        elapsed_ms=elapsed,
    )


def _simple_probe(
    key: str,
    label: str,
    hint: str,
    api_key: str | None,
    url: str,
    headers: dict[str, str],
) -> ProbeResult:
    api_key = _clean_secret(api_key)
    if not api_key:
        return _not_configured(key, label, hint)
    started = time.monotonic()
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return ProbeResult(
            key=key,
            label=label,
            status=STATUS_UNAVAILABLE,
            detail=f"네트워크 오류 ({type(exc).__name__})",
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    elapsed = int((time.monotonic() - started) * 1000)
    if resp.status_code == 200:
        return ProbeResult(key=key, label=label, status=STATUS_OK, detail="키가 유효합니다.", elapsed_ms=elapsed)
    return ProbeResult(
        key=key,
        label=label,
        status=_status_from_code(resp.status_code),
        detail=f"HTTP {resp.status_code}",
        elapsed_ms=elapsed,
    )


def probe_gemini(api_key: str | None) -> ProbeResult:
    return _simple_probe(
        "gemini",
        "Gemini",
        "Gemini API Key를 입력하세요.",
        api_key,
        f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key or ''}&pageSize=1",
        {},
    )


def probe_anthropic(api_key: str | None) -> ProbeResult:
    label = "Anthropic"
    key = _clean_secret(api_key)
    if not key:
        return _not_configured("anthropic", label, "Anthropic API Key를 입력하세요.")
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    started = time.monotonic()
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get("https://api.anthropic.com/v1/models", headers=headers, params={"limit": "1"})
            # Older or restricted keys sometimes cannot list models; a 1-token Haiku ping still proves the key.
            if resp.status_code == 404:
                resp = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json={
                        "model": ANTHROPIC_PING_MODEL,
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                )
    except httpx.HTTPError as exc:
        return ProbeResult(
            key="anthropic",
            label=label,
            status=STATUS_UNAVAILABLE,
            detail=f"네트워크 오류 ({type(exc).__name__})",
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    elapsed = int((time.monotonic() - started) * 1000)
    if resp.status_code == 200:
        usage = None
        if resp.request.method == "POST":
            usage = f"모델 목록이 없어 {ANTHROPIC_PING_MODEL} 1토큰 핑으로 확인했습니다."
        return ProbeResult(
            key="anthropic",
            label=label,
            status=STATUS_OK,
            detail="키가 유효합니다.",
            usage=usage,
            elapsed_ms=elapsed,
        )
    status = _status_from_code(resp.status_code)
    detail = _provider_error_detail(resp)
    if resp.status_code == 401:
        detail = "키가 거부됐습니다. console.anthropic.com의 API Keys에서 sk-ant-로 시작하는 키를 다시 복사하세요."
    elif resp.status_code == 403:
        detail = "이 키에 Messages API 권한이 없습니다. Anthropic 콘솔에서 키 권한을 확인하세요."
    elif resp.status_code == 402:
        detail = "크레딧/결제 한도입니다. Anthropic 콘솔의 Billing을 확인하세요."
    return ProbeResult(key="anthropic", label=label, status=status, detail=detail, elapsed_ms=elapsed)


def probe_sec() -> ProbeResult:
    """SEC needs no key, but 403 blocks are the usual cause of degraded filings."""
    label = "SEC EDGAR"
    started = time.monotonic()
    headers = sec_headers()
    try:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = client.head("https://data.sec.gov/submissions/CIK0000320193.json", headers=headers)
            if resp.status_code == 405:
                resp = client.get(
                    "https://data.sec.gov/submissions/CIK0000320193.json",
                    headers={**headers, "Range": "bytes=0-511"},
                )
    except httpx.HTTPError as exc:
        return ProbeResult(
            key="sec",
            label=label,
            status=STATUS_UNAVAILABLE,
            detail=f"네트워크 오류 ({type(exc).__name__})",
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    elapsed = int((time.monotonic() - started) * 1000)
    if resp.status_code in (200, 206):
        return ProbeResult(key="sec", label=label, status=STATUS_OK, detail="EDGAR에 접근할 수 있습니다.", elapsed_ms=elapsed)
    if resp.status_code == 403:
        return ProbeResult(
            key="sec",
            label=label,
            status=STATUS_UNAVAILABLE,
            detail=(
                "미국 SEC 공시 서버가 이 네트워크를 차단했습니다. API 키가 있는 서비스가 아닙니다. "
                "VPN(미국)을 쓰거나, 차단이 풀릴 때까지 앱은 직전 공시 기록으로 계속합니다."
            ),
            elapsed_ms=elapsed,
        )
    return ProbeResult(
        key="sec",
        label=label,
        status=_status_from_code(resp.status_code),
        detail=f"HTTP {resp.status_code}",
        elapsed_ms=elapsed,
    )


def probe_kakao(access_token: str | None) -> ProbeResult:
    label = "카카오 알림"
    if not access_token:
        return _not_configured("kakao", label, "카카오 연결을 먼저 진행하세요.")
    started = time.monotonic()
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(
                "https://kapi.kakao.com/v1/user/access_token_info",
                headers={"Authorization": f"Bearer {access_token}"},
            )
    except httpx.HTTPError as exc:
        return ProbeResult(
            key="kakao",
            label=label,
            status=STATUS_UNAVAILABLE,
            detail=f"네트워크 오류 ({type(exc).__name__})",
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    elapsed = int((time.monotonic() - started) * 1000)
    if resp.status_code == 200:
        return ProbeResult(key="kakao", label=label, status=STATUS_OK, detail="토큰이 유효합니다.", elapsed_ms=elapsed)
    detail = "토큰이 만료됐습니다. 다시 연결하세요." if resp.status_code == 401 else f"HTTP {resp.status_code}"
    return ProbeResult(
        key="kakao",
        label=label,
        status=_status_from_code(resp.status_code),
        detail=detail,
        elapsed_ms=elapsed,
    )


PROBE_ORDER: tuple[str, ...] = ("alpaca", "openai", "gemini", "anthropic", "sec", "kakao")


def check_all(env: dict[str, str]) -> ConnectionReport:
    """Run every probe in parallel. One click, one round-trip per provider."""
    jobs = {
        "alpaca": lambda: probe_alpaca(env.get("ALPACA_API_KEY"), env.get("ALPACA_SECRET_KEY")),
        "openai": lambda: probe_openai(env.get("OPENAI_API_KEY")),
        "gemini": lambda: probe_gemini(env.get("GEMINI_API_KEY")),
        "anthropic": lambda: probe_anthropic(env.get("ANTHROPIC_API_KEY")),
        "sec": probe_sec,
        "kakao": lambda: probe_kakao(env.get("KAKAO_ACCESS_TOKEN")),
    }
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {key: pool.submit(fn) for key, fn in jobs.items()}
        done = {key: fut.result() for key, fut in futures.items()}
    return ConnectionReport(
        checked_at=datetime.now(timezone.utc),
        results=[done[key] for key in PROBE_ORDER if key in done],
    )
