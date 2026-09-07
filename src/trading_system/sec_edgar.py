"""SEC EDGAR 8-K / 10-Q ingest with public accepted timestamp (UTC). No API key required.

Official hosts only: data.sec.gov and www.sec.gov. Never fabricate filings.
403/429 → DEGRADED when last-known metadata exists, else UNAVAILABLE.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from trading_system.config import discover_project_root
from trading_system.credentials import read_env_map
from trading_system.providers.http_client import parse_retry_after
from trading_system.providers.never_block import (
    ProviderHttpError,
    RetryKind,
    RetryPolicy,
    classify_http_status,
    run_with_deadline,
)

APP_VERSION = "0.1.0"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_TXT = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{accession}.txt"
ARCHIVE_DOC = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{document}"
_OFFICIAL_HOSTS = frozenset({"www.sec.gov", "data.sec.gov"})

# Public CIKs for the default smoke universe — ticker map fallback only, not filings.
SMOKE_CIKS: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "AMZN": "0001018723",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "SPY": "0000884394",
}

# SEC fair-access cap is 10 req/s. Stay well under that.
_MIN_INTERVAL_SECONDS = 0.2
_SEC_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay_seconds=0.5,
    max_delay_seconds=4.0,
    jitter=0.2,
    total_deadline_seconds=20.0,
)

_log = logging.getLogger("trading_system.sec_edgar")
_rate_lock = threading.Lock()
_last_request_mono = 0.0
_CIK_MAP: dict[str, str] | None = None
_LAST_ERROR: str | None = None
_USED_LKG = False


def last_sec_error() -> str | None:
    return _LAST_ERROR


def used_sec_lkg() -> bool:
    return _USED_LKG


def _set_error(message: str) -> None:
    global _LAST_ERROR
    text = (message or "").strip()
    _LAST_ERROR = text or None


def _assert_official_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    allowed_path = (
        path == "/files/company_tickers.json"
        or path.startswith("/submissions/CIK")
        or path.startswith("/Archives/edgar/data/")
    )
    if parsed.scheme != "https" or host not in _OFFICIAL_HOSTS or not allowed_path:
        raise ProviderHttpError(
            f"refusing non-official SEC endpoint {host}{path}",
            status_code=None,
            kind=RetryKind.NON_RETRYABLE,
        )


def _tickers_cache() -> Path:
    return discover_project_root() / "data" / "sec_company_tickers.json"


def _lkg_dir() -> Path:
    return discover_project_root() / "data" / "sec_lkg"


def _lkg_path(cik: str) -> Path:
    return _lkg_dir() / f"CIK{cik}.json"


def _headers() -> dict[str, str]:
    contact = os.getenv("SEC_CONTACT_EMAIL", "").strip()
    if not contact:
        try:
            contact = read_env_map(discover_project_root() / ".env").get("SEC_CONTACT_EMAIL", "").strip()
        except OSError:
            contact = ""
    identity = contact or "contact-not-configured"
    headers = {
        "User-Agent": f"InvestAssist/{APP_VERSION} (personal investment assistant; {identity})",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json, text/plain, */*",
    }
    if contact:
        headers["From"] = contact
    return headers


def _client() -> httpx.Client:
    return httpx.Client(timeout=20.0, headers=_headers(), follow_redirects=True)


def _classify_sec(exc: BaseException) -> RetryKind:
    if isinstance(exc, ProviderHttpError):
        return exc.kind
    if isinstance(exc, httpx.HTTPStatusError):
        return classify_http_status(exc.response.status_code)
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, TimeoutError)):
        return RetryKind.RETRYABLE
    return RetryKind.NON_RETRYABLE


def _throttle() -> None:
    global _last_request_mono
    with _rate_lock:
        wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request_mono)
        if wait > 0:
            time.sleep(wait)
        _last_request_mono = time.monotonic()


def _log_status(url: str, resp: httpx.Response) -> None:
    parsed = urlparse(url)
    safe = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    retry_after = parse_retry_after(resp.headers)
    _log.info("SEC status=%s retry_after=%s url=%s", resp.status_code, retry_after, safe)


def _get(url: str) -> httpx.Response:
    _assert_official_url(url)
    _throttle()
    with _client() as client:
        resp = client.get(url)
    _log_status(url, resp)
    retry_after = parse_retry_after(resp.headers)
    if resp.status_code in {403, 429}:
        raise ProviderHttpError(
            f"SEC {resp.status_code} {urlparse(url).path}",
            status_code=resp.status_code,
            kind=RetryKind.RATE_LIMITED,
            retry_after_seconds=retry_after,
        )
    resp.raise_for_status()
    return resp


def _load_disk_cache() -> dict[str, str]:
    path = _tickers_cache()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data:
            return {str(k).upper(): str(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def _save_disk_cache(mapping: dict[str, str]) -> None:
    path = _tickers_cache()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping), encoding="utf-8")
    except OSError:
        pass


def load_cik_map() -> dict[str, str]:
    """Ticker → 10-digit CIK. Disk-cached so we do not hammer EDGAR."""
    global _CIK_MAP
    if _CIK_MAP:
        return _CIK_MAP

    cached = _load_disk_cache()
    if cached:
        _CIK_MAP = cached
        return _CIK_MAP

    def _fetch() -> dict[str, str]:
        resp = _get(TICKERS_URL)
        data = resp.json()
        out: dict[str, str] = {}
        rows = data.values() if isinstance(data, dict) else []
        for row in rows:
            ticker = str(row.get("ticker", "")).upper()
            if not ticker:
                continue
            out[ticker] = str(int(row["cik_str"])).zfill(10)
        if not out:
            raise ValueError("SEC company_tickers.json parsed empty")
        return out

    result = run_with_deadline(_fetch, policy=_SEC_POLICY, classify=_classify_sec, label="sec_tickers")
    if result.ok and isinstance(result.value, dict) and result.value:
        _CIK_MAP = result.value
        _save_disk_cache(_CIK_MAP)
        _set_error("")
        return _CIK_MAP

    _set_error(result.error or "SEC ticker map UNAVAILABLE")
    _CIK_MAP = dict(SMOKE_CIKS)
    return _CIK_MAP


def lookup_cik(ticker: str) -> str | None:
    ticker = ticker.upper().replace(".", "-")
    found = load_cik_map().get(ticker)
    if found:
        return found
    return SMOKE_CIKS.get(ticker)


def _parse_submissions(payload: dict[str, Any], cik: str, forms: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    recent = (payload.get("filings") or {}).get("recent") or {}
    forms_l = recent.get("form") or []
    acc = recent.get("accessionNumber") or []
    accepted = recent.get("acceptanceDateTime") or []
    docs = recent.get("primaryDocument") or []
    out: list[dict[str, Any]] = []
    for i, (form, accession, acc_at) in enumerate(zip(forms_l, acc, accepted, strict=False)):
        if form not in forms:
            continue
        try:
            accepted_at = datetime.fromisoformat(str(acc_at).replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            continue
        primary = docs[i] if i < len(docs) else None
        out.append(
            {
                "form": form,
                "accession": accession,
                "accepted_at": accepted_at,
                "cik": cik,
                "primary_document": primary,
                "source": "live",
            }
        )
        if len(out) >= limit:
            break
    return out


def _save_lkg(cik: str, payload: dict[str, Any]) -> None:
    path = _lkg_path(cik)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def _load_lkg(cik: str, forms: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    path = _lkg_path(cik)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    rows = _parse_submissions(payload, cik, forms, limit)
    for row in rows:
        row["source"] = "lkg"
    return rows


def recent_filings(cik: str, *, forms: tuple[str, ...] = ("8-K", "10-Q"), limit: int = 8) -> list[dict[str, Any]]:
    global _USED_LKG
    url = SUBMISSIONS_URL.format(cik=cik)

    def _fetch() -> list[dict[str, Any]]:
        resp = _get(url)
        payload = resp.json()
        if not isinstance(payload, dict):
            raise ValueError("SEC submissions payload malformed")
        rows = _parse_submissions(payload, cik, forms, limit)
        _save_lkg(cik, payload)
        return rows

    result = run_with_deadline(_fetch, policy=_SEC_POLICY, classify=_classify_sec, label="sec_submissions")
    if result.ok and isinstance(result.value, list):
        _USED_LKG = False
        _set_error("")
        return list(result.value)

    lkg = _load_lkg(cik, forms, limit)
    if lkg:
        _USED_LKG = True
        _set_error(result.error or "SEC submissions using LKG metadata")
        _log.info("SEC LKG metadata used for CIK%s n=%s", cik, len(lkg))
        return lkg
    _set_error(result.error or "SEC submissions UNAVAILABLE")
    return []


def _plain_text(raw: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_filing_text(cik: str, accession: str, *, primary_document: str | None = None, max_chars: int = 8000) -> str | None:
    cik_n = str(int(cik))
    acc_nodash = accession.replace("-", "")
    txt_url = ARCHIVE_TXT.format(cik=cik_n, acc_nodash=acc_nodash, accession=accession)
    doc_url = (
        ARCHIVE_DOC.format(cik=cik_n, acc_nodash=acc_nodash, document=primary_document)
        if primary_document
        else None
    )

    def _fetch() -> str:
        resp = _get(txt_url)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text
        if doc_url:
            doc = _get(doc_url)
            if doc.status_code == 200 and doc.text.strip():
                return doc.text
        return ""

    result = run_with_deadline(_fetch, policy=_SEC_POLICY, classify=_classify_sec, label="sec_filing_body")
    if not result.ok or not result.value:
        return None
    plain = _plain_text(str(result.value))
    return plain[:max_chars] if plain else None
