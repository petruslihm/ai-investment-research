"""Persist and reconstruct LLM prompt/response logs for the UI.

Live OpenAI transcripts were not stored before this table existed. Those older
scans are reconstructed from tick_recommendations (answer is stored; the exact
HTTP body is not).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import duckdb

from trading_system.judge_package import compact_judge_package, dumps_complete
from trading_system.llm_client import (
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM,
    take_exchange,
)

_SKIP_REASONS = {"NOT_A_CANDIDATE", "NOT_SELECTED_FOR_FINAL_JUDGE"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def format_ts(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    text = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return text.split(".")[0]


def table_exists(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [name],
    ).fetchone()
    return row is not None


def ticker_from_instrument(instrument_id: object) -> str:
    s = str(instrument_id or "")
    if s.startswith("inst_"):
        s = s[5:]
    return s.upper()


def _as_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def ensure_transcript_columns(conn: duckdb.DuckDBPyConnection) -> bool:
    if not table_exists(conn, "llm_transcripts"):
        return False
    try:
        conn.execute("ALTER TABLE llm_transcripts ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER")
        conn.execute("ALTER TABLE llm_transcripts ADD COLUMN IF NOT EXISTS completion_tokens INTEGER")
        conn.execute("ALTER TABLE llm_transcripts ADD COLUMN IF NOT EXISTS total_tokens INTEGER")
    except Exception:  # noqa: BLE001
        pass
    cols = {
        str(r[0]).lower()
        for r in conn.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'main' AND table_name = 'llm_transcripts'
            """
        ).fetchall()
    }
    return "prompt_tokens" in cols


def insert_llm_transcript(
    conn: duckdb.DuckDBPyConnection,
    *,
    tick_id: str | None,
    kind: str,
    status: str | None = None,
    instrument_id: str | None = None,
    ticker: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    response_text: str | None = None,
    error: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> str | None:
    if not table_exists(conn, "llm_transcripts"):
        return None
    has_tokens = ensure_transcript_columns(conn)
    transcript_id = f"llm_{uuid4().hex}"
    if has_tokens:
        conn.execute(
            """
            INSERT INTO llm_transcripts (
                transcript_id, tick_id, kind, instrument_id, ticker, model, prompt_version,
                status, system_prompt, user_prompt, response_text, error,
                prompt_tokens, completion_tokens, total_tokens, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                transcript_id,
                tick_id,
                kind,
                instrument_id,
                ticker or (ticker_from_instrument(instrument_id) if instrument_id else None),
                model,
                prompt_version,
                status,
                system_prompt,
                user_prompt,
                response_text,
                error,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                _utcnow(),
            ],
        )
    else:
        conn.execute(
            """
            INSERT INTO llm_transcripts (
                transcript_id, tick_id, kind, instrument_id, ticker, model, prompt_version,
                status, system_prompt, user_prompt, response_text, error, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                transcript_id,
                tick_id,
                kind,
                instrument_id,
                ticker or (ticker_from_instrument(instrument_id) if instrument_id else None),
                model,
                prompt_version,
                status,
                system_prompt,
                user_prompt,
                response_text,
                error,
                _utcnow(),
            ],
        )
    return transcript_id


def persist_exchange(
    conn: duckdb.DuckDBPyConnection | None,
    payload: dict[str, Any],
    *,
    tick_id: str | None,
    kind: str,
    instrument_id: str | None = None,
    ticker: str | None = None,
    prompt_version: str | None = None,
    default_system: str | None = None,
    default_user: str | None = None,
) -> None:
    exchange = take_exchange(payload) or {}
    if conn is None:
        return
    dumped = {k: v for k, v in payload.items() if k != "_exchange"}
    response = exchange.get("raw_response")
    raw = str(response or "").strip()
    if not raw or raw in {"{}", "null"}:
        response = json.dumps(dumped, ensure_ascii=False) if dumped else response
    insert_llm_transcript(
        conn,
        tick_id=tick_id,
        kind=kind,
        status=str(payload.get("status") or "") or None,
        instrument_id=instrument_id,
        ticker=ticker,
        model=exchange.get("model"),
        prompt_version=prompt_version,
        system_prompt=exchange.get("system") or default_system,
        user_prompt=exchange.get("user") or default_user,
        response_text=str(response) if response is not None else None,
        error=exchange.get("error") or payload.get("error") or payload.get("note"),
        prompt_tokens=_as_int(exchange.get("prompt_tokens")),
        completion_tokens=_as_int(exchange.get("completion_tokens")),
        total_tokens=_as_int(exchange.get("total_tokens")),
    )


def _parse_rec(payload_json: object) -> dict[str, Any] | None:
    try:
        wrapper = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(wrapper, dict):
        return None
    body = wrapper.get("payload")
    rec = body if isinstance(body, dict) else wrapper
    return rec if isinstance(rec, dict) else None


def _pretty(text: object) -> str:
    raw = "" if text is None else str(text)
    if not raw.strip():
        return ""
    parsed = _json_object(raw)
    if parsed:
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    return raw


def _strip_json_fence(raw: str) -> str:
    text = (raw or "").strip()
    if "```" in text:
        text = text.replace("```json", "```").replace("```JSON", "```")
        if text.startswith("```"):
            text = text[3:]
        fence = text.rfind("```")
        if fence >= 0:
            text = text[:fence]
        text = text.strip()
    if text[:4].lower() == "json":
        maybe = text[4:].lstrip(" \r\n\t:")
        if maybe.startswith("{"):
            text = maybe
    return text.strip()


def _json_object(text: object) -> dict[str, Any]:
    raw = _strip_json_fence("" if text is None else str(text))
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(raw[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}


def _clip(text: str, n: int = 280) -> str:
    raw = " ".join(text.split())
    if len(raw) <= n:
        return raw
    return raw[: n - 1] + "…"


def answer_summary_from_blob(blob: object, *, response_text: str = "") -> str:
    """Human-readable answer for the LLM log. Research packs rarely have `thesis`."""
    data = blob if isinstance(blob, dict) else {}
    if not data:
        data = _json_object(response_text)
    for key in (
        "summary_ko",
        "executive_thesis",
        "thesis",
        "rationale",
        "summary",
        "executive_summary",
        "notes",
    ):
        text = str(data.get(key) or "").strip()
        if text:
            return _clip(text)
    industry = data.get("industry") if isinstance(data.get("industry"), dict) else {}
    notes = str(industry.get("notes") or "").strip()
    if notes:
        return _clip(notes)
    material = data.get("material_change")
    if isinstance(material, str) and material.strip():
        return _clip(material)
    if isinstance(material, dict):
        for key in ("summary", "description", "notes", "event"):
            text = str(material.get(key) or "").strip()
            if text:
                return _clip(text)
    for item in data.get("supporting_evidence") or []:
        if isinstance(item, dict) and str(item.get("claim") or "").strip():
            return _clip(str(item["claim"]))
        if isinstance(item, str) and item.strip():
            return _clip(item)
    for item in data.get("news_events") or []:
        if isinstance(item, dict):
            headline = str(item.get("headline") or item.get("title") or item.get("claim") or "").strip()
            if headline:
                return _clip(headline)
        if isinstance(item, str) and item.strip():
            return _clip(item)
    adv = data.get("adversarial_review") if isinstance(data.get("adversarial_review"), dict) else {}
    for item in adv.get("main_objections") or data.get("main_objections") or []:
        if str(item).strip():
            return _clip(str(item))
    raw = (response_text or "").strip()
    ticker = str(data.get("ticker") or "").strip()
    if data:
        if ticker:
            return f"{ticker} 근거 팩이 저장되어 있습니다. 아래 답변 JSON에서 전문을 볼 수 있습니다."
        return "근거 팩이 저장되어 있습니다. 아래 답변 JSON에서 전문을 볼 수 있습니다."
    if raw.startswith("```") or raw.lstrip().startswith("{"):
        return "저장된 답변 JSON이 있습니다. 아래 답변 JSON에서 전문을 볼 수 있습니다."
    return _clip(raw) if raw else ""


def _is_skip(rec: dict[str, Any]) -> bool:
    reasons = {str(x) for x in (rec.get("override_reasons") or [])}
    return bool(reasons & _SKIP_REASONS)


def list_llm_sessions(conn: duckdb.DuckDBPyConnection, *, limit: int = 80) -> list[dict[str, Any]]:
    if not table_exists(conn, "ticks"):
        return []
    has_transcripts = table_exists(conn, "llm_transcripts")
    transcript_filter = (
        "OR EXISTS (SELECT 1 FROM llm_transcripts lt WHERE lt.tick_id = t.tick_id)"
        if has_transcripts
        else ""
    )
    rows = conn.execute(
        f"""
        SELECT t.tick_id, t.committed_at, COUNT(r.recommendation_id) AS rec_n
        FROM ticks t
        LEFT JOIN tick_recommendations r ON r.tick_id = t.tick_id AND r.source = 'llm_final'
        WHERE t.status = 'committed'
          AND (r.recommendation_id IS NOT NULL {transcript_filter})
        GROUP BY t.tick_id, t.committed_at
        ORDER BY t.committed_at DESC
        LIMIT ?
        """,
        [limit],
    ).fetchall()
    live_counts: dict[str, int] = {}
    raw_judged: dict[str, int] = {}
    if has_transcripts:
        for tick_id, n in conn.execute(
            "SELECT tick_id, COUNT(*) FROM llm_transcripts WHERE tick_id IS NOT NULL GROUP BY tick_id"
        ).fetchall():
            live_counts[str(tick_id)] = int(n)
        for tick_id, n in conn.execute(
            """
            SELECT tick_id, COUNT(*)
            FROM llm_transcripts
            WHERE tick_id IS NOT NULL
              AND kind IN ('portfolio_judge', 'legacy_b_final_execution')
            GROUP BY tick_id
            """
        ).fetchall():
            raw_judged[str(tick_id)] = 1 if int(n) else 0
    judged_counts: dict[str, int] = {}
    skipped_counts: dict[str, int] = {}
    if rows:
        tick_ids = [str(r[0]) for r in rows]
        placeholders = ", ".join(["?"] * len(tick_ids))
        rec_rows = conn.execute(
            f"""
            SELECT tick_id, payload_json FROM tick_recommendations
            WHERE source = 'llm_final' AND tick_id IN ({placeholders})
            """,
            tick_ids,
        ).fetchall()
        for tick_id, raw in rec_rows:
            rec = _parse_rec(raw)
            key = str(tick_id)
            if rec and _is_skip(rec):
                skipped_counts[key] = skipped_counts.get(key, 0) + 1
            elif rec:
                judged_counts[key] = judged_counts.get(key, 0) + 1
    sessions: list[dict[str, Any]] = []
    for tick_id, committed_at, rec_n in rows:
        stamp = format_ts(committed_at)
        date_part, _, time_part = stamp.partition(" ")
        key = str(tick_id)
        sessions.append(
            {
                "tick_id": key,
                "committed_at": committed_at,
                "stamp": stamp,
                "date": date_part,
                "time": time_part,
                "rec_n": int(rec_n),
                "judged": judged_counts.get(key, 0) or raw_judged.get(key, 0),
                "skipped": skipped_counts.get(key, 0),
                "live_n": live_counts.get(key, 0),
            }
        )
    return sessions


def _live_turns(conn: duckdb.DuckDBPyConnection, tick_id: str) -> list[dict[str, Any]]:
    if not table_exists(conn, "llm_transcripts"):
        return []
    has_tokens = ensure_transcript_columns(conn)
    token_cols = ", prompt_tokens, completion_tokens, total_tokens" if has_tokens else ""
    rows = conn.execute(
        f"""
        SELECT transcript_id, kind, instrument_id, ticker, model, prompt_version, status,
               system_prompt, user_prompt, response_text, error, created_at{token_cols}
        FROM llm_transcripts
        WHERE tick_id = ?
        ORDER BY created_at ASC
        """,
        [tick_id],
    ).fetchall()
    turns: list[dict[str, Any]] = []
    for row in rows:
        response = _pretty(row[9])
        parsed_action = None
        parsed_thesis = None
        reasons: list[str] = []
        blob = _json_object(row[9])
        if blob:
            parsed_action = blob.get("action")
            parsed_thesis = answer_summary_from_blob(blob, response_text=str(row[9] or ""))
            reasons = [str(x) for x in (blob.get("override_reasons") or [])]
        else:
            parsed_thesis = answer_summary_from_blob({}, response_text=str(row[9] or ""))
        prompt_tokens = _as_int(row[12]) if has_tokens and len(row) > 12 else None
        completion_tokens = _as_int(row[13]) if has_tokens and len(row) > 13 else None
        total_tokens = _as_int(row[14]) if has_tokens and len(row) > 14 else None
        turns.append(
            {
                "id": row[0],
                "kind": row[1],
                "instrument_id": row[2],
                "ticker": row[3] or ticker_from_instrument(row[2]),
                "model": row[4],
                "prompt_version": row[5],
                "status": row[6],
                "system_prompt": row[7] or "",
                "user_prompt": _pretty(row[8]) or (row[8] or ""),
                "response_text": response,
                "error": row[10],
                "created_at": format_ts(row[11]) if row[11] else "",
                "source": "live",
                "action": parsed_action,
                "thesis": parsed_thesis,
                "override_reasons": reasons,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )
    # The aggregate portfolio_judge row exists to restore the latest dashboard.
    # When the individual legacy-b turns are also present, hiding that aggregate
    # avoids showing the final answer twice in the conversation history.
    if any(str(turn.get("kind") or "").startswith("legacy_b_") for turn in turns):
        turns = [turn for turn in turns if turn.get("kind") != "portfolio_judge"]
    return turns


def _quant_map(conn: duckdb.DuckDBPyConnection, tick_id: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT payload_json FROM tick_recommendations
        WHERE tick_id = ? AND source = 'quant_only'
        """,
        [tick_id],
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for (raw,) in rows:
        rec = _parse_rec(raw)
        if rec and rec.get("instrument_id"):
            out[str(rec["instrument_id"])] = rec
    return out


def _evidence_for_ticker(conn: duckdb.DuckDBPyConnection, ticker: str) -> dict[str, Any]:
    if not ticker or not table_exists(conn, "sec_extracts"):
        return {}
    rows = conn.execute(
        "SELECT payload_json FROM sec_extracts ORDER BY accepted_at DESC"
    ).fetchall()
    needle = ticker.upper().replace("/", "").replace(".", "")
    for (raw,) in rows:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        got = str(payload.get("ticker", "")).upper().replace("/", "").replace(".", "")
        if got and (got == needle or needle.endswith(got) or got in needle):
            return payload
    return {}


def _reconstruct_turns(conn: duckdb.DuckDBPyConnection, tick_id: str, *, skip_keys: set[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT payload_json FROM tick_recommendations
        WHERE tick_id = ? AND source = 'llm_final'
        """,
        [tick_id],
    ).fetchall()
    quants = _quant_map(conn, tick_id)
    turns: list[dict[str, Any]] = []
    for (raw,) in rows:
        rec = _parse_rec(raw)
        if not rec or _is_skip(rec):
            continue
        instrument_id = str(rec.get("instrument_id") or "")
        ticker = ticker_from_instrument(instrument_id)
        if ("judge", instrument_id) in skip_keys or ("judge", ticker) in skip_keys:
            continue
        quant = quants.get(instrument_id) or {}
        dq_level = "ok" if quant.get("actionable") else "not_configured"
        package = {
            "quant": quant or rec,
            "dq_level": dq_level,
            "evidence": _evidence_for_ticker(conn, ticker),
        }
        user = dumps_complete(compact_judge_package(package))
        answer = {
            "action": rec.get("action"),
            "thesis": rec.get("thesis"),
            "override_reasons": rec.get("override_reasons") or [],
            "confidence": rec.get("confidence"),
            "status": (rec.get("override_reasons") or ["AVAILABLE"])[0]
            if rec.get("override_reasons")
            else "AVAILABLE",
        }
        turns.append(
            {
                "id": f"recon_{instrument_id}",
                "kind": "judge",
                "instrument_id": instrument_id,
                "ticker": ticker,
                "model": None,
                "prompt_version": JUDGE_PROMPT_VERSION,
                "status": str(answer["status"]),
                "system_prompt": JUDGE_SYSTEM,
                "user_prompt": _pretty(user),
                "response_text": _pretty(json.dumps(answer, ensure_ascii=False)),
                "error": None,
                "created_at": "",
                "source": "reconstructed",
                "action": rec.get("action"),
                "thesis": rec.get("thesis"),
                "override_reasons": [str(x) for x in (rec.get("override_reasons") or [])],
            }
        )
    turns.sort(key=lambda t: str(t.get("ticker") or ""))
    return turns


def load_llm_session(conn: duckdb.DuckDBPyConnection, tick_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT tick_id, committed_at, status FROM ticks WHERE tick_id = ?",
        [tick_id],
    ).fetchone()
    if not row:
        return None
    recs = conn.execute(
        """
        SELECT payload_json FROM tick_recommendations
        WHERE tick_id = ? AND source = 'llm_final'
        """,
        [tick_id],
    ).fetchall()
    judged = 0
    skipped = 0
    for (raw,) in recs:
        rec = _parse_rec(raw)
        if not rec:
            continue
        if _is_skip(rec):
            skipped += 1
        else:
            judged += 1
    live = _live_turns(conn, tick_id)
    if judged == 0 and any(
        str(turn.get("kind") or "") in {"portfolio_judge", "legacy_b_final_execution"}
        for turn in live
    ):
        judged = 1
    skip_keys: set[tuple[str, str]] = set()
    for turn in live:
        if turn.get("kind") == "judge":
            if turn.get("instrument_id"):
                skip_keys.add(("judge", str(turn["instrument_id"])))
            if turn.get("ticker"):
                skip_keys.add(("judge", str(turn["ticker"])))
    reconstructed = _reconstruct_turns(conn, tick_id, skip_keys=skip_keys)
    turns = live + reconstructed
    stamp = format_ts(row[1])
    date_part, _, time_part = stamp.partition(" ")
    prompt_vals = [_as_int(t.get("prompt_tokens")) for t in live]
    completion_vals = [_as_int(t.get("completion_tokens")) for t in live]
    total_vals = [_as_int(t.get("total_tokens")) for t in live]
    prompt_ok = [n for n in prompt_vals if n is not None]
    completion_ok = [n for n in completion_vals if n is not None]
    total_ok = [n for n in total_vals if n is not None]
    return {
        "tick_id": str(row[0]),
        "committed_at": row[1],
        "stamp": stamp,
        "date": date_part,
        "time": time_part,
        "status": row[2],
        "judged": judged,
        "skipped": skipped,
        "live_n": len(live),
        "reconstructed_n": len(reconstructed),
        "prompt_tokens": sum(prompt_ok) if prompt_ok else None,
        "completion_tokens": sum(completion_ok) if completion_ok else None,
        "total_tokens": sum(total_ok) if total_ok else None,
        "token_calls": len(total_ok),
        "models": sorted({str(t.get("model")) for t in live if t.get("model")}),
        "turns": turns,
    }


__all__ = [
    "EXTRACT_SYSTEM",
    "JUDGE_SYSTEM",
    "insert_llm_transcript",
    "list_llm_sessions",
    "load_llm_session",
    "persist_exchange",
    "table_exists",
    "ticker_from_instrument",
]
