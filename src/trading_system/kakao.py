"""KakaoTalk personal 'Send to Me' notifier. Failures never block the app."""

from __future__ import annotations

import json
import time
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from trading_system.alerts import AlertRecord
from trading_system.credentials import env_path, read_env_map, upsert_env
from trading_system.providers.never_block import NeverBlockResult, RetryKind, RetryPolicy, run_with_deadline

KAKAO_AUTH_URL = "https://kauth.kakao.com/oauth/authorize"
KAKAO_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
KAKAO_MEMO_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
KAKAO_ME_URL = "https://kapi.kakao.com/v2/user/me"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8743/auth/kakao/callback"
KAKAO_SCOPES = "talk_message,profile_nickname"
KAKAO_TEXT_MAX = 200
TOKEN_SKEW_SECONDS = 60
HTTP_TIMEOUT = 8.0

KAKAO_TOKEN_KEYS = (
    "KAKAO_ACCESS_TOKEN",
    "KAKAO_REFRESH_TOKEN",
    "KAKAO_ACCESS_EXPIRES_AT",
    "KAKAO_NICKNAME",
)


class KakaoStatus(StrEnum):
    CONNECTED = "CONNECTED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


def default_redirect_uri(env: dict[str, str] | None = None) -> str:
    env = env or {}
    return (env.get("KAKAO_REDIRECT_URI") or DEFAULT_REDIRECT_URI).strip() or DEFAULT_REDIRECT_URI


def oauth_state_path(root: Path) -> Path:
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d / ".kakao_oauth_state"


def redact(text: object, env: dict[str, str] | None = None) -> str:
    s = "" if text is None else str(text)
    secrets = []
    for src in (env or {},):
        for k, v in src.items():
            if v and k.startswith("KAKAO_") and "URI" not in k and "NICKNAME" not in k:
                secrets.append(v)
    for v in secrets:
        if len(v) >= 8:
            s = s.replace(v, "[redacted]")
    return s


def connection_status(env: dict[str, str]) -> KakaoStatus:
    if not (env.get("KAKAO_REST_API_KEY") or "").strip():
        return KakaoStatus.NOT_CONFIGURED
    refresh = (env.get("KAKAO_REFRESH_TOKEN") or "").strip()
    access = (env.get("KAKAO_ACCESS_TOKEN") or "").strip()
    if not refresh and not access:
        return KakaoStatus.AUTH_REQUIRED
    if not refresh and access and _access_expired(env):
        return KakaoStatus.TOKEN_EXPIRED
    return KakaoStatus.CONNECTED


def _access_expired(env: dict[str, str]) -> bool:
    raw = (env.get("KAKAO_ACCESS_EXPIRES_AT") or "").strip()
    if not raw:
        return True
    try:
        return time.time() >= float(raw)
    except ValueError:
        return True


def authorize_url(env: dict[str, str], state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": env["KAKAO_REST_API_KEY"],
        "redirect_uri": default_redirect_uri(env),
        "scope": KAKAO_SCOPES,
        "state": state,
    }
    return f"{KAKAO_AUTH_URL}?{urlencode(params)}"


def kakao_request(
    method: str,
    url: str,
    *,
    timeout: float = HTTP_TIMEOUT,
    client: httpx.Client | None = None,
    **kwargs: Any,
) -> httpx.Response:
    own = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        return http.request(method, url, timeout=timeout, **kwargs)
    finally:
        if own:
            http.close()


def _token_form(env: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    data = {"client_id": env["KAKAO_REST_API_KEY"], **extra}
    secret = (env.get("KAKAO_CLIENT_SECRET") or "").strip()
    if secret:
        data["client_secret"] = secret
    return data


def persist_tokens(path: Path, payload: dict[str, Any], *, nickname: str | None = None) -> None:
    updates: dict[str, str] = {}
    if payload.get("access_token"):
        updates["KAKAO_ACCESS_TOKEN"] = str(payload["access_token"])
    if payload.get("refresh_token"):
        updates["KAKAO_REFRESH_TOKEN"] = str(payload["refresh_token"])
    expires_in = payload.get("expires_in")
    if expires_in is not None:
        updates["KAKAO_ACCESS_EXPIRES_AT"] = str(int(time.time() + float(expires_in) - TOKEN_SKEW_SECONDS))
    if nickname:
        updates["KAKAO_NICKNAME"] = nickname
    if updates:
        upsert_env(path, updates)


def exchange_code(
    env: dict[str, str],
    code: str,
    *,
    env_file: Path,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    resp = kakao_request(
        "POST",
        KAKAO_TOKEN_URL,
        client=client,
        data=_token_form(
            env,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": default_redirect_uri(env),
            },
        ),
    )
    if resp.status_code >= 400:
        raise httpx.HTTPStatusError(
            redact(resp.text, env),
            request=resp.request,
            response=resp,
        )
    payload = resp.json()
    nick = fetch_nickname(str(payload.get("access_token", "")), env=env, client=client)
    persist_tokens(env_file, payload, nickname=nick)
    return {"ok": True, "nickname": nick}


def refresh_access_token(
    env: dict[str, str],
    *,
    env_file: Path,
    client: httpx.Client | None = None,
) -> dict[str, str]:
    refresh = (env.get("KAKAO_REFRESH_TOKEN") or "").strip()
    if not refresh:
        raise RuntimeError("AUTH_REQUIRED")
    resp = kakao_request(
        "POST",
        KAKAO_TOKEN_URL,
        client=client,
        data=_token_form(env, {"grant_type": "refresh_token", "refresh_token": refresh}),
    )
    if resp.status_code in {400, 401}:
        raise RuntimeError("TOKEN_EXPIRED")
    if resp.status_code >= 400:
        raise httpx.HTTPStatusError(
            redact(resp.text, env),
            request=resp.request,
            response=resp,
        )
    payload = resp.json()
    persist_tokens(env_file, payload)
    merged = dict(env)
    merged.update(read_env_map(env_file))
    return merged


def fetch_nickname(
    access_token: str,
    *,
    env: dict[str, str],
    client: httpx.Client | None = None,
) -> str | None:
    if not access_token:
        return None
    try:
        resp = kakao_request(
            "GET",
            KAKAO_ME_URL,
            client=client,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if resp.status_code >= 400:
            return None
        body = resp.json()
        props = body.get("properties") or {}
        acct = (body.get("kakao_account") or {}).get("profile") or {}
        nick = props.get("nickname") or acct.get("nickname")
        return str(nick) if nick else None
    except Exception:  # noqa: BLE001 — profile is optional
        return None


def ensure_access_token(
    env: dict[str, str],
    *,
    env_file: Path,
    client: httpx.Client | None = None,
) -> dict[str, str]:
    env = dict(env)
    if (env.get("KAKAO_ACCESS_TOKEN") or "").strip() and not _access_expired(env):
        return env
    return refresh_access_token(env, env_file=env_file, client=client)


def format_kakao_text(alert: AlertRecord) -> str:
    text = (alert.message or "").strip() or f"[{alert.kind}]"
    header = "[Stock AI]\n"
    body = text if text.startswith("[Stock AI]") else header + text
    if len(body) > KAKAO_TEXT_MAX:
        body = body[: KAKAO_TEXT_MAX - 1] + "…"
    return body


def _send_memo(access_token: str, text: str, *, client: httpx.Client | None = None) -> httpx.Response:
    template = {
        "object_type": "text",
        "text": text,
        "link": {
            "web_url": "http://127.0.0.1:8743/",
            "mobile_web_url": "http://127.0.0.1:8743/",
        },
    }
    return kakao_request(
        "POST",
        KAKAO_MEMO_URL,
        client=client,
        headers={"Authorization": f"Bearer {access_token}"},
        data={"template_object": json.dumps(template, ensure_ascii=False)},
    )


def send_to_me(
    env: dict[str, str],
    alert: AlertRecord,
    *,
    env_file: Path,
    client: httpx.Client | None = None,
) -> KakaoStatus:
    status = connection_status(env)
    if status is KakaoStatus.NOT_CONFIGURED:
        return status
    if status in {KakaoStatus.AUTH_REQUIRED, KakaoStatus.TOKEN_EXPIRED}:
        return status
    env = ensure_access_token(env, env_file=env_file, client=client)
    text = format_kakao_text(alert)
    token = env["KAKAO_ACCESS_TOKEN"]
    resp = _send_memo(token, text, client=client)
    if resp.status_code == 401:
        env = refresh_access_token(env, env_file=env_file, client=client)
        resp = _send_memo(env["KAKAO_ACCESS_TOKEN"], text, client=client)
    if resp.status_code >= 400:
        raise httpx.HTTPStatusError(
            redact(resp.text, env),
            request=resp.request,
            response=resp,
        )
    return KakaoStatus.CONNECTED


def notify_kakao(
    alert: AlertRecord,
    *,
    env_file: Path | None = None,
    client: httpx.Client | None = None,
) -> NeverBlockResult:
    path = env_file or env_path()
    env = read_env_map(path)

    def _run() -> KakaoStatus:
        return send_to_me(env, alert, env_file=path, client=client)

    result = run_with_deadline(
        _run,
        policy=RetryPolicy(max_attempts=1, total_deadline_seconds=HTTP_TIMEOUT),
        classify=lambda exc: RetryKind.NON_RETRYABLE,
        label="kakao",
    )
    if result.error:
        result.error = redact(result.error, env)
    return result
