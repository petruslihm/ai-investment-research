"""NYSE pre-open daily scan window (weekends and holidays skipped)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_system.daily_scan import (
    MAX_AUTO_ATTEMPTS,
    claim_daily_train,
    claim_due_auto_job,
    claim_session_run,
    due_daily_scan,
    due_daily_train,
    due_weekly_train,
    last_session_path,
    mark_session_run,
    read_last_daily_train,
    read_last_session,
    release_daily_train,
    release_session_run,
)
from trading_system.market.calendar import is_trading_day

ET = ZoneInfo("America/New_York")


def _et(y: int, m: int, d: int, hh: int, mm: int) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_due_skips_weekend_and_before_window() -> None:
    saturday = _et(2026, 9, 5, 8, 30)
    assert is_trading_day(saturday.date()) is False
    assert due_daily_scan(saturday, last_session=None) is None
    monday_early = _et(2026, 8, 31, 8, 29)
    assert is_trading_day(monday_early.date()) is True
    assert due_daily_scan(monday_early, last_session=None) is None


def test_due_at_one_hour_before_open() -> None:
    monday_preopen = _et(2026, 8, 31, 8, 30)
    assert due_daily_scan(monday_preopen, last_session=None) == date(2026, 8, 31)
    assert due_daily_scan(monday_preopen, last_session=date(2026, 8, 31)) is None
    after_close = _et(2026, 8, 31, 16, 0)
    assert due_daily_scan(after_close, last_session=None) is None


def test_due_skips_nyse_holiday() -> None:
    labor_day = _et(2026, 9, 7, 8, 30)
    assert is_trading_day(labor_day.date()) is False
    assert due_daily_scan(labor_day, last_session=None) is None


def test_last_session_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "investment_assistant.duckdb"
    assert read_last_session(db) is None
    mark_session_run(db, date(2026, 9, 1))
    assert read_last_session(db) == date(2026, 9, 1)
    assert db.with_name("daily_scan_last.txt").is_file()


def test_weekly_train_friday_one_hour_after_close() -> None:
    friday_before = _et(2026, 9, 4, 16, 59)
    assert due_weekly_train(friday_before, last_session=None) is None
    friday_go = _et(2026, 9, 4, 17, 0)
    assert is_trading_day(friday_go.date()) is True
    assert due_weekly_train(friday_go, last_session=None) == date(2026, 9, 4)
    assert due_weekly_train(friday_go, last_session=date(2026, 9, 4)) is None
    saturday = _et(2026, 9, 5, 10, 0)
    assert due_weekly_train(saturday, last_session=None) == date(2026, 9, 4)
    monday = _et(2026, 9, 7, 0, 0)
    assert due_weekly_train(monday, last_session=None) is None


def test_weekly_train_uses_thursday_when_friday_is_holiday() -> None:
    # Good Friday 2026-04-03. Last session that week is Thursday.
    assert is_trading_day(date(2026, 4, 3)) is False
    thursday_go = _et(2026, 4, 2, 17, 0)
    assert due_weekly_train(thursday_go, last_session=None) == date(2026, 4, 2)
    friday_holiday = _et(2026, 4, 3, 17, 0)
    assert due_weekly_train(friday_holiday, last_session=None) == date(2026, 4, 2)


def test_auto_job_claims_once_even_if_work_never_finishes(tmp_path: Path) -> None:
    db = tmp_path / "investment_assistant.duckdb"
    now = _et(2026, 8, 31, 8, 30)
    assert claim_due_auto_job(db, now) == "scan"
    assert claim_due_auto_job(db, now) is None
    assert claim_due_auto_job(db, now + timedelta(minutes=5)) is None
    assert due_daily_scan(now, last_session=read_last_session(db)) is None
    assert claim_session_run(db, date(2026, 8, 31)) is False


def test_due_daily_train_fires_any_weekday_not_just_friday() -> None:
    """train_only mode (see run_v1_cycle) makes no paid API calls, so the auto
    scheduler retrains every session instead of being held to once a week."""
    monday = _et(2026, 8, 31, 17, 0)
    assert is_trading_day(monday.date()) is True
    assert due_daily_train(monday, last_session=None) == date(2026, 8, 31)
    tuesday = _et(2026, 9, 1, 17, 0)
    assert due_daily_train(tuesday, last_session=date(2026, 8, 31)) == date(2026, 9, 1)
    # Already trained today -> not due again.
    assert due_daily_train(tuesday, last_session=date(2026, 9, 1)) is None
    # Before the after-close window.
    assert due_daily_train(_et(2026, 9, 1, 16, 59), last_session=None) is None
    # Weekend -> never due.
    saturday = _et(2026, 9, 5, 17, 0)
    assert due_daily_train(saturday, last_session=None) is None


def test_daily_train_watermark_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "investment_assistant.duckdb"
    assert read_last_daily_train(db) is None
    assert claim_daily_train(db, date(2026, 9, 1)) is True
    assert read_last_daily_train(db) == date(2026, 9, 1)
    assert claim_daily_train(db, date(2026, 9, 1)) is False
    assert claim_daily_train(db, date(2026, 9, 2)) is True


def test_claim_due_auto_job_trains_daily_not_weekly(tmp_path: Path) -> None:
    db = tmp_path / "investment_assistant.duckdb"
    friday = _et(2026, 9, 4, 17, 0)
    assert claim_due_auto_job(db, friday, scan_enabled=False) == "train"
    assert claim_due_auto_job(db, friday, scan_enabled=False) is None
    saturday = _et(2026, 9, 5, 10, 0)
    assert claim_due_auto_job(db, saturday, scan_enabled=False) is None
    # The old weekly scheduler would refuse to train again until next Friday --
    # the daily one must fire again on the very next trading day. 2026-09-07 is
    # Labor Day (a holiday), so that next session is Tuesday 2026-09-08.
    assert is_trading_day(date(2026, 9, 7)) is False
    next_session = _et(2026, 9, 8, 17, 0)
    assert claim_due_auto_job(db, next_session, scan_enabled=False) == "train"


def test_failed_auto_run_refunds_the_slot_so_the_day_is_not_lost(tmp_path: Path) -> None:
    """2026-09-04: the auto scan claimed the day's slot, then hung -- and because
    the claim happens before the run, nothing retried it and the day's automatic
    scan was simply lost. A claimed attempt that did no work must hand the slot
    back so a later tick can pick it up."""
    db = tmp_path / "investment_assistant.duckdb"
    session = date(2026, 9, 4)

    assert claim_session_run(db, session) is True
    # While an attempt holds the slot nothing else may start it.
    assert claim_session_run(db, session) is False
    assert read_last_session(db) == session

    assert release_session_run(db, session) is True
    # Refunded -> the session reads as not-yet-run, so due_daily_scan offers it again.
    assert read_last_session(db) is None
    assert claim_session_run(db, session) is True


def test_auto_retry_is_capped_so_a_broken_job_cannot_loop(tmp_path: Path) -> None:
    """The refund above must not become an unbounded retry loop: a job that fails
    the same way every time stops after MAX_AUTO_ATTEMPTS for that session."""
    db = tmp_path / "investment_assistant.duckdb"
    session = date(2026, 9, 4)

    for _ in range(MAX_AUTO_ATTEMPTS):
        assert claim_session_run(db, session) is True
        assert release_session_run(db, session) is True
    assert claim_session_run(db, session) is False
    assert read_last_session(db) == session
    # A different session is unaffected by the previous day's exhausted budget.
    assert claim_session_run(db, date(2026, 9, 8)) is True


def test_successful_auto_run_keeps_the_slot_spent_for_the_day(tmp_path: Path) -> None:
    """No refund on success -- otherwise the 30s scheduler tick would re-fire the
    same session immediately and pay for the LLM calls all over again."""
    db = tmp_path / "investment_assistant.duckdb"
    session = date(2026, 9, 4)
    assert claim_session_run(db, session) is True
    assert read_last_session(db) == session
    assert claim_session_run(db, session) is False
    assert due_daily_scan(_et(2026, 9, 4, 9, 0), last_session=read_last_session(db)) is None


def test_daily_train_slot_refunds_and_caps_the_same_way(tmp_path: Path) -> None:
    db = tmp_path / "investment_assistant.duckdb"
    session = date(2026, 9, 4)
    assert claim_daily_train(db, session) is True
    assert claim_daily_train(db, session) is False
    assert release_daily_train(db, session) is True
    assert read_last_daily_train(db) is None
    for _ in range(MAX_AUTO_ATTEMPTS - 1):
        assert claim_daily_train(db, session) is True
        assert release_daily_train(db, session) is True
    assert claim_daily_train(db, session) is False


def test_pre_attempts_marker_file_still_reads_as_spent(tmp_path: Path) -> None:
    """Marker files written before the used/held fields existed are a bare date.
    They must keep meaning "this session is done", never "free to run again"."""
    db = tmp_path / "investment_assistant.duckdb"
    path = last_session_path(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("2026-09-04\n", encoding="utf-8")
    assert read_last_session(db) == date(2026, 9, 4)
    assert claim_session_run(db, date(2026, 9, 4)) is False


def test_refund_without_a_claim_is_a_no_op(tmp_path: Path) -> None:
    db = tmp_path / "investment_assistant.duckdb"
    assert release_session_run(db, date(2026, 9, 4)) is False
    assert claim_session_run(db, date(2026, 9, 4)) is True
    # A second refund for an already-refunded slot must not grant extra attempts.
    assert release_session_run(db, date(2026, 9, 4)) is True
    assert release_session_run(db, date(2026, 9, 4)) is False
