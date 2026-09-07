"""SEC EDGAR 403/LKG and OpenAI 429/quota handling. Never fabricate filings or LLM judges."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from trading_system.config import Settings
from trading_system.llm_client import (
    STATUS_QUOTA_EXCEEDED,
    STATUS_RATE_LIMITED,
    inspect_openai_failure,
)
from trading_system.providers.http_client import parse_retry_after
from trading_system.providers.never_block import ProviderHttpError, RetryKind, RetryPolicy
from trading_system.recommendations import RecommendationAction, quant_only_record
from trading_system.sec_edgar import (
    _get,
    _headers,
    _parse_submissions,
    recent_filings,
)
from trading_system.sec_llm import ingest_sec_extracts
from trading_system.storage import Store
from trading_system.v1_cycle import _aggregate_llm_status


def _settings(**kwargs: object) -> Settings:
    base = {
        "_env_file": None,
        "alpaca_api_key": None,
        "alpaca_secret_key": None,
        "openai_api_key": None,
        "gemini_api_key": None,
        "anthropic_api_key": None,
        "smoke_universe": ("AAPL", "MSFT", "BTC/USD"),
    }
    base.update(kwargs)
    return Settings(**base)  # type: ignore[arg-type]


def _quant(*, action: RecommendationAction = RecommendationAction.ENTER):
    return quant_only_record(
        tick_id="tick_1",
        decision_epoch_id="epoch_1",
        feature_snapshot_id="fs_1",
        instrument_id="inst_aapl",
        action=action,
        confidence=0.8,
    )


def _openai_response(status: int, body: dict | None = None, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx.Response(status, json=body, headers=headers or {}, request=request)


def _lkg_payload() -> dict:
    return {
        "filings": {
            "recent": {
                "form": ["8-K"],
                "accessionNumber": ["0000320193-26-000123"],
                "acceptanceDateTime": ["2026-08-01T20:00:00-04:00"],
                "primaryDocument": ["aapl-8k.htm"],
            }
        }
    }


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "sec.duckdb")
    s.open(acquire_writer=True)
    yield s
    s.close()


def test_sec_user_agent_loads_contact_without_hardcoding_personal_email(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import trading_system.sec_edgar as edgar

    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(edgar, "discover_project_root", lambda: tmp_path)
    assert "From" not in _headers()
    assert "contact-not-configured" in _headers()["User-Agent"]

    monkeypatch.setenv("SEC_CONTACT_EMAIL", "owner@example.com")
    headers = _headers()
    assert headers["From"] == "owner@example.com"
    assert "owner@example.com" in headers["User-Agent"]


def test_sec_rejects_non_official_and_cgi_browse() -> None:
    with pytest.raises(ProviderHttpError) as blocked:
        _get("https://example.com/filings")
    assert blocked.value.kind == RetryKind.NON_RETRYABLE
    with pytest.raises(ProviderHttpError) as cgi:
        _get("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany")
    assert cgi.value.kind == RetryKind.NON_RETRYABLE


def test_parse_retry_after_numeric_and_http_date() -> None:
    assert parse_retry_after({"Retry-After": "7"}) == 7.0
    seconds = parse_retry_after({"retry-after": "Thu, 01 Jan 2099 00:00:00 GMT"})
    assert seconds is not None and seconds > 0


def test_sec_403_without_lkg_is_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import trading_system.sec_edgar as edgar

    monkeypatch.setattr(edgar, "discover_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        edgar,
        "_SEC_POLICY",
        RetryPolicy(max_attempts=3, base_delay_seconds=0.0, max_delay_seconds=1.0, jitter=0.0, total_deadline_seconds=5.0),
    )

    def forbidden(_url: str) -> httpx.Response:
        raise ProviderHttpError(
            "SEC 403 /submissions/CIK0000320193.json",
            status_code=403,
            kind=RetryKind.RATE_LIMITED,
        )

    monkeypatch.setattr(edgar, "_get", forbidden)
    edgar._USED_LKG = False
    edgar._LAST_ERROR = None

    rows = recent_filings("0000320193", limit=1)
    assert rows == []
    assert edgar.used_sec_lkg() is False
    assert edgar.last_sec_error()


def test_sec_403_preserves_lkg_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import trading_system.sec_edgar as edgar

    monkeypatch.setattr(edgar, "discover_project_root", lambda: tmp_path)
    lkg = tmp_path / "data" / "sec_lkg"
    lkg.mkdir(parents=True)
    (lkg / "CIK0000320193.json").write_text(
        __import__("json").dumps(_lkg_payload()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        edgar,
        "_SEC_POLICY",
        RetryPolicy(max_attempts=2, base_delay_seconds=0.0, max_delay_seconds=1.0, jitter=0.0, total_deadline_seconds=5.0),
    )
    monkeypatch.setattr(
        edgar,
        "_get",
        lambda _url: (_ for _ in ()).throw(
            ProviderHttpError("SEC 403 /submissions/CIK0000320193.json", status_code=403, kind=RetryKind.RATE_LIMITED)
        ),
    )

    rows = recent_filings("0000320193", limit=1)
    assert len(rows) == 1
    assert rows[0]["accession"] == "0000320193-26-000123"
    assert rows[0]["source"] == "lkg"
    assert edgar.used_sec_lkg() is True


def test_ingest_live_fail_with_lkg_is_degraded(monkeypatch: pytest.MonkeyPatch, store: Store) -> None:
    accepted = datetime(2026, 8, 1, tzinfo=timezone.utc)
    monkeypatch.setattr("trading_system.sec_llm.lookup_cik", lambda _t: "0000320193")
    monkeypatch.setattr(
        "trading_system.sec_llm.recent_filings",
        lambda _cik, limit=1: [
            {
                "form": "8-K",
                "accession": "0000320193-26-000123",
                "accepted_at": accepted,
                "cik": "0000320193",
                "primary_document": "aapl-8k.htm",
                "source": "lkg",
            }
        ],
    )
    monkeypatch.setattr("trading_system.sec_llm.last_sec_error", lambda: "SEC 403 /submissions/CIK0000320193.json")
    pack = ingest_sec_extracts(store.conn, _settings(), tickers=["AAPL", "MSFT"])
    assert pack["used_lkg"] is True
    assert pack["sec_status"] == "DEGRADED"
    assert pack["filings"]
    assert all(f.get("source") == "lkg" for f in pack["filings"])
    notes = " ".join(str(f.get("note") or "") for f in pack["filings"])
    assert "invent" in notes.lower() or "LKG" in notes


def test_ingest_403_without_filings_does_not_fabricate(monkeypatch: pytest.MonkeyPatch, store: Store) -> None:
    monkeypatch.setattr("trading_system.sec_llm.lookup_cik", lambda _t: "0000320193")
    monkeypatch.setattr("trading_system.sec_llm.recent_filings", lambda *_a, **_k: [])
    monkeypatch.setattr("trading_system.sec_llm.last_sec_error", lambda: "SEC 403 /submissions/CIK0000320193.json")
    pack = ingest_sec_extracts(store.conn, _settings(), tickers=["AAPL"])
    assert pack["filings"] == []
    assert pack["sec_status"] == "UNAVAILABLE"
    assert pack["used_lkg"] is False


def test_inspect_openai_rate_limit_vs_quota() -> None:
    rate = inspect_openai_failure(
        _openai_response(
            429,
            {"error": {"code": "rate_limit_exceeded", "type": "tokens"}},
            {"Retry-After": "2", "x-ratelimit-remaining-requests": "0"},
        )
    )
    assert rate[0] == RetryKind.RATE_LIMITED
    assert rate[1] == STATUS_RATE_LIMITED
    assert rate[2] == 2.0

    quota = inspect_openai_failure(
        _openai_response(
            429,
            {"error": {"code": "insufficient_quota", "type": "insufficient_quota"}},
            {"Retry-After": "1"},
        )
    )
    assert quota[0] == RetryKind.BILLING
    assert quota[1] == STATUS_QUOTA_EXCEEDED


def test_aggregate_keeps_quant_independent() -> None:
    rec = _quant()
    rec.override_reasons = [STATUS_RATE_LIMITED]
    rec.action = RecommendationAction.NO_ACTION
    rec.actionable = False
    assert _aggregate_llm_status(configured=True, recs=[rec]) == STATUS_RATE_LIMITED
    quota = rec.model_copy(update={"override_reasons": [STATUS_QUOTA_EXCEEDED]})
    assert _aggregate_llm_status(configured=True, recs=[quota]) == STATUS_QUOTA_EXCEEDED
    assert _aggregate_llm_status(configured=False, recs=[rec]) == "NOT_CONFIGURED"


def test_aggregate_ignores_skipped_non_candidates() -> None:
    from trading_system.recommendations import llm_final_record
    from trading_system.sec_llm import skipped_llm_record

    skip = skipped_llm_record(_quant(), quant_status="AVAILABLE")
    assert _aggregate_llm_status(configured=True, recs=[skip]) == "UNAVAILABLE"
    ok = llm_final_record(
        tick_id="tick_1",
        decision_epoch_id="epoch_1",
        feature_snapshot_id="fs_1",
        instrument_id="inst_aapl",
        action=RecommendationAction.BUY,
        thesis="ok",
    )
    assert _aggregate_llm_status(configured=True, recs=[skip, ok]) == "AVAILABLE"


def test_evidence_does_not_fall_back_to_another_ticker() -> None:
    from trading_system.sec_llm import evidence_for_instrument

    filings = [{"ticker": "MSFT", "accession": "msft-1"}]
    assert evidence_for_instrument(filings, "inst_aapl") is None
    assert evidence_for_instrument([{"ticker": "AAPL", "accession": "aapl-1"}], "inst_aapl")["accession"] == "aapl-1"


def test_parse_submissions_does_not_invent_rows() -> None:
    assert _parse_submissions({}, "0000320193", ("8-K",), 8) == []
    assert _parse_submissions({"filings": {"recent": {}}}, "0000320193", ("8-K",), 8) == []
