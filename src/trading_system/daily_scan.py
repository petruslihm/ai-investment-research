"""Once-per-NYSE-session scan window: 1 hour before regular open, weekdays only.

Does not spawn a second DuckDB writer. The UI process posts /run-once to itself.
Windows Task Scheduler may start the UI if the PC is on; it must not launch a
competing writer while the UI already holds the lease.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from trading_system.market.calendar import (
    MARKET_TZ,
    SESSION_CLOSE_ET,
    is_trading_day,
    prev_trading_day,
    session_open_datetime,
)

LAST_SESSION_NAME = "daily_scan_last.txt"
LAST_WEEKLY_TRAIN_NAME = "weekly_train_last.txt"
LAST_DAILY_TRAIN_NAME = "daily_train_last.txt"

# How many times one day's auto job may be started in total. The slot is claimed
# before the run (so a crash mid-LLM-call cannot silently loop), but a run that
# demonstrably did no work -- or died -- refunds it via release_auto_slot so the
# day is not lost to one transient failure. The cap is what keeps that refund
# from becoming an unbounded retry loop when every attempt fails the same way;
# the daily LLM budget (enforce_llm_budget on auto runs) is the money backstop
# underneath it.
#
# 5, not 3: on 2026-09-04 it took five attempts (one hang, three OpenAI 429s,
# then a success) to get one judgement out. Each attempt now resumes from the
# turns that already succeeded rather than restarting the conversation, so a
# 9-turn judgement that only advances a turn or two per attempt still needs
# roughly this many. The per-turn budget check in legacy_b_raw_conversation is
# what bounds the money, not this count.
MAX_AUTO_ATTEMPTS = 5

# The marker file holds "<session ISO date> used=<n> held=<0|1>":
#   used -- attempts started today, capped by MAX_AUTO_ATTEMPTS
#   held -- 1 while an attempt owns the slot (running, or finished and done with
#           it), 0 once an attempt refunded it and a retry may claim it again
# Readers only ever take the leading 10 chars for the date, so marker files
# written before these fields existed still parse (and read as held, used=1,
# i.e. exactly the old "one shot per day" behaviour).
_USED_PREFIX = "used="
_HELD_PREFIX = "held="


def _read_marker(path: Path) -> tuple[date | None, int, bool]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None, 0, False
    try:
        session = date.fromisoformat(raw[:10])
    except ValueError:
        return None, 0, False
    used, held = 1, True
    for token in raw.split():
        if token.startswith(_USED_PREFIX):
            try:
                used = int(token[len(_USED_PREFIX) :])
            except ValueError:
                used = 1
        elif token.startswith(_HELD_PREFIX):
            held = token[len(_HELD_PREFIX) :].strip() not in {"0", "false", "False"}
    return session, used, held


def _write_marker(path: Path, session: date, used: int, held: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{session.isoformat()} {_USED_PREFIX}{max(1, int(used))} {_HELD_PREFIX}{1 if held else 0}\n",
        encoding="utf-8",
    )


def _slot_is_spent(path: Path, session: date) -> bool:
    """True when this session may not start another auto run right now."""
    marked, used, held = _read_marker(path)
    if marked != session:
        return False
    return held or used >= MAX_AUTO_ATTEMPTS


def _claim_slot(path: Path, session: date) -> bool:
    """Take the slot for one attempt. False if held or the day's cap is spent."""
    marked, used, held = _read_marker(path)
    if marked != session:
        _write_marker(path, session, 1, True)
        return True
    if held or used >= MAX_AUTO_ATTEMPTS:
        return False
    _write_marker(path, session, used + 1, True)
    return True


def _release_slot(path: Path, session: date) -> bool:
    """Hand the slot back so today can retry (up to MAX_AUTO_ATTEMPTS).

    Called when a claimed attempt did no work -- another run already held the
    in-process scan lock, or the cycle died before finishing. Without this one
    transient failure silently costs the whole day, which is what happened on
    2026-09-04. `used` is deliberately NOT decremented: the refund reopens the
    slot but still counts against the day's cap, so a run that always fails the
    same way stops after MAX_AUTO_ATTEMPTS instead of looping forever.
    """
    marked, used, held = _read_marker(path)
    if marked != session or not held:
        return False
    _write_marker(path, session, used, False)
    return True


def last_session_path(duckdb_path: Path) -> Path:
    return Path(duckdb_path).with_name(LAST_SESSION_NAME)


def read_last_session(duckdb_path: Path) -> date | None:
    """The session that must not be auto-scanned again right now.

    A refunded slot with attempts left reads as None (i.e. "not done today"), so
    due_daily_scan keeps offering the session and _claim_slot can hand out the
    retry -- see _release_slot.
    """
    session, _used, _held = _read_marker(last_session_path(duckdb_path))
    if session is None or not _slot_is_spent(last_session_path(duckdb_path), session):
        return None
    return session


def mark_session_run(duckdb_path: Path, session: date) -> None:
    _write_marker(last_session_path(duckdb_path), session, MAX_AUTO_ATTEMPTS, True)


def claim_session_run(duckdb_path: Path, session: date) -> bool:
    """Consume one auto-scan attempt for the session (see MAX_AUTO_ATTEMPTS)."""
    return _claim_slot(last_session_path(duckdb_path), session)


def release_session_run(duckdb_path: Path, session: date) -> bool:
    """Refund an auto-scan attempt that did no work (see _release_slot)."""
    return _release_slot(last_session_path(duckdb_path), session)


def preopen_datetime(session_date: date, minutes_before: int = 60) -> datetime:
    return session_open_datetime(session_date) - timedelta(minutes=max(0, int(minutes_before)))


def due_daily_scan(
    now: datetime,
    *,
    last_session: date | None,
    minutes_before: int = 60,
) -> date | None:
    """Return today's NYSE session date when a daily scan should start, else None."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local = now.astimezone(MARKET_TZ)
    session = local.date()
    if not is_trading_day(session):
        return None
    if last_session == session:
        return None
    start = preopen_datetime(session, minutes_before)
    close = datetime.combine(session, SESSION_CLOSE_ET, tzinfo=MARKET_TZ)
    if local < start or local >= close:
        return None
    return session


def week_friday(d: date) -> date:
    """Friday of the Monday–Sunday week that contains d (America/New_York calendar date)."""
    return d - timedelta(days=d.weekday()) + timedelta(days=4)


def week_train_session(friday: date) -> date | None:
    """Friday if NYSE is open; otherwise the last session earlier that week."""
    if is_trading_day(friday):
        return friday
    prev = prev_trading_day(friday)
    monday = friday - timedelta(days=4)
    if prev >= monday:
        return prev
    return None


def last_weekly_train_path(duckdb_path: Path) -> Path:
    return Path(duckdb_path).with_name(LAST_WEEKLY_TRAIN_NAME)


def read_last_weekly_train(duckdb_path: Path) -> date | None:
    path = last_weekly_train_path(duckdb_path)
    try:
        raw = path.read_text(encoding="utf-8").strip()[:10]
        return date.fromisoformat(raw)
    except (OSError, ValueError):
        return None


def mark_weekly_train(duckdb_path: Path, session: date) -> None:
    path = last_weekly_train_path(duckdb_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(session.isoformat() + "\n", encoding="utf-8")


def claim_weekly_train(duckdb_path: Path, session: date) -> bool:
    """Consume this week's auto-train slot before the job starts."""
    if read_last_weekly_train(duckdb_path) == session:
        return False
    mark_weekly_train(duckdb_path, session)
    return True


def last_daily_train_path(duckdb_path: Path) -> Path:
    return Path(duckdb_path).with_name(LAST_DAILY_TRAIN_NAME)


def read_last_daily_train(duckdb_path: Path) -> date | None:
    """Session that must not be auto-trained again right now (see read_last_session)."""
    session, _used, _held = _read_marker(last_daily_train_path(duckdb_path))
    if session is None or not _slot_is_spent(last_daily_train_path(duckdb_path), session):
        return None
    return session


def mark_daily_train(duckdb_path: Path, session: date) -> None:
    _write_marker(last_daily_train_path(duckdb_path), session, MAX_AUTO_ATTEMPTS, True)


def claim_daily_train(duckdb_path: Path, session: date) -> bool:
    """Consume one auto-train attempt for the session (see MAX_AUTO_ATTEMPTS)."""
    return _claim_slot(last_daily_train_path(duckdb_path), session)


def release_daily_train(duckdb_path: Path, session: date) -> bool:
    """Refund an auto-train attempt that did no work (see _release_slot)."""
    return _release_slot(last_daily_train_path(duckdb_path), session)


def due_daily_train(
    now: datetime,
    *,
    last_session: date | None,
    minutes_after_close: int = 60,
) -> date | None:
    """Every NYSE session, N minutes after close, once per day.

    train_only mode (see run_v1_cycle) makes no SEC/Gemini/GPT calls, so unlike
    due_weekly_train there is no cost reason to hold this to once a week -- see
    apply_online_updates' pending[-16:] cap in ml_engine.py for why infrequent
    retraining actually loses matured labels the online models never catch up on.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local = now.astimezone(MARKET_TZ)
    session = local.date()
    if not is_trading_day(session):
        return None
    if last_session == session:
        return None
    start = datetime.combine(session, SESSION_CLOSE_ET, tzinfo=MARKET_TZ) + timedelta(
        minutes=max(0, int(minutes_after_close))
    )
    deadline = datetime.combine(session + timedelta(days=1), time(0, 0), tzinfo=MARKET_TZ)
    if local < start or local >= deadline:
        return None
    return session


def claim_due_auto_job(
    duckdb_path: Path,
    now: datetime,
    *,
    scan_enabled: bool = True,
    train_enabled: bool = True,
    minutes_before: int = 60,
    minutes_after_close: int = 60,
) -> str | None:
    """Claim at most one auto job. Returns 'scan' or 'train', or None.

    Claiming happens before the HTTP start, so an interrupted cycle is not retried.
    Manual dashboard buttons do not go through this.
    """
    if scan_enabled:
        session = due_daily_scan(
            now,
            last_session=read_last_session(duckdb_path),
            minutes_before=minutes_before,
        )
        if session is not None and claim_session_run(duckdb_path, session):
            return "scan"
    if train_enabled:
        session = due_daily_train(
            now,
            last_session=read_last_daily_train(duckdb_path),
            minutes_after_close=minutes_after_close,
        )
        if session is not None and claim_daily_train(duckdb_path, session):
            return "train"
    return None


def due_weekly_train(
    now: datetime,
    *,
    last_session: date | None,
    minutes_after_close: int = 60,
) -> date | None:
    """Friday (or last session of a holiday week) 1h after NYSE close, once per week."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local = now.astimezone(MARKET_TZ)
    monday = local.date() - timedelta(days=local.weekday())
    friday = monday + timedelta(days=4)
    session = week_train_session(friday)
    if session is None:
        return None
    if last_session == session:
        return None
    start = datetime.combine(session, SESSION_CLOSE_ET, tzinfo=MARKET_TZ) + timedelta(
        minutes=max(0, int(minutes_after_close))
    )
    next_monday = monday + timedelta(days=7)
    deadline = datetime.combine(next_monday, time(0, 0), tzinfo=MARKET_TZ)
    if local < start or local >= deadline:
        return None
    return session
