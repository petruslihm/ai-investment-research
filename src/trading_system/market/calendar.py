"""US equity session calendar (NYSE regular holidays; adapted from legacy-a)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

from trading_system.providers.interfaces import BarFinality

MARKET_TZ = ZoneInfo("America/New_York")
SESSION_OPEN_ET = time(9, 30)
SESSION_CLOSE_ET = time(16, 0)

# One-off NYSE closures not covered by the regular holiday rules.
SPECIAL_CLOSURES: frozenset[date] = frozenset(
    {
        date(2025, 1, 9),  # National Day of Mourning (Jimmy Carter)
    }
)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    return d + timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lam) // 451
    month = (h + lam - 7 * m + 114) // 31
    day = ((h + lam - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=64)
def holidays(year: int) -> frozenset[date]:
    out = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    if year >= 2021:
        out.add(_observed(date(year, 6, 19)))
    return frozenset(out)


def is_trading_day(d: date) -> bool:
    if d in SPECIAL_CLOSURES:
        return False
    if d.weekday() >= 5:
        return False
    return d not in holidays(d.year)


def next_trading_day(d: date, *, inclusive: bool = False) -> date:
    cur = d if inclusive else d + timedelta(days=1)
    for _ in range(30):
        if is_trading_day(cur):
            return cur
        cur += timedelta(days=1)
    return cur


def prev_trading_day(d: date, *, inclusive: bool = False) -> date:
    cur = d if inclusive else d - timedelta(days=1)
    for _ in range(30):
        if is_trading_day(cur):
            return cur
        cur -= timedelta(days=1)
    return cur


@lru_cache(maxsize=8192)
def add_sessions(d: date, n: int) -> date:
    cur = d
    for _ in range(max(0, int(n))):
        cur = next_trading_day(cur)
    return cur


def session_open_datetime(session_date: date) -> datetime:
    """NYSE regular open (09:30 America/New_York) for a session date."""
    return datetime.combine(session_date, SESSION_OPEN_ET, tzinfo=MARKET_TZ)


def equity_session_date(as_of_utc: datetime) -> date:
    """Map UTC instant to US equity session date in America/New_York."""
    local = as_of_utc.astimezone(MARKET_TZ)
    return local.date()


def equity_session_closed(session_date: date, as_of_utc: datetime) -> bool:
    """True when the NY equity regular session for session_date has ended."""
    local = as_of_utc.astimezone(MARKET_TZ)
    if local.date() > session_date:
        return True
    if local.date() < session_date:
        return False
    return local.time() >= SESSION_CLOSE_ET


def equity_bar_finality(session_date: date, as_of_utc: datetime) -> BarFinality:
    """Assign PRELIMINARY until the equity session has closed."""
    if not is_trading_day(session_date):
        raise ValueError(f"not an equity trading session: {session_date}")
    if equity_session_closed(session_date, as_of_utc):
        return BarFinality.FINAL
    return BarFinality.PRELIMINARY


def btc_day_finality(session_date: date, as_of_utc: datetime) -> BarFinality:
    """BTC daily bars finalize at UTC day boundary."""
    today_utc = as_of_utc.astimezone(timezone.utc).date()
    if session_date < today_utc:
        return BarFinality.FINAL
    return BarFinality.PRELIMINARY
