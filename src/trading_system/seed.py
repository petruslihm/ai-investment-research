"""Deterministic smoke-universe history for first run without API keys."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from trading_system.config import Settings
from trading_system.ids import new_request_set_id
from trading_system.market.calendar import is_trading_day
from trading_system.market.registry import bootstrap_smoke_universe, stable_instrument_id
from trading_system.market.repository import upsert_btc_daily_bars, upsert_equity_daily_bars
from trading_system.providers.interfaces import BarFinality, DailyBar
from trading_system.storage import Store


def trading_days_ending(end: date, n: int) -> list[date]:
    out: list[date] = []
    cur = end
    while len(out) < n:
        if is_trading_day(cur):
            out.append(cur)
        cur -= timedelta(days=1)
        if cur.year < 2018:
            break
    return list(reversed(out))


def calendar_days_ending(end: date, n: int) -> list[date]:
    return [end - timedelta(days=n - 1 - i) for i in range(n)]


def seed_synthetic_history(store: Store, settings: Settings, *, days: int = 90, end: date | None = None) -> None:
    end = end or date.today()
    sessions = trading_days_ending(end, days)
    cal = calendar_days_ending(end, days)
    now = datetime.now(timezone.utc)
    mapping = bootstrap_smoke_universe(store.conn, settings, provider="fixture")
    rs = new_request_set_id()
    store.conn.execute(
        "INSERT OR IGNORE INTO request_sets VALUES (?, ?, ?)",
        [rs, now, "synthetic_seed"],
    )
    equity_syms = [s for s in settings.smoke_universe if s != settings.btc_symbol]
    for si, sym in enumerate(equity_syms):
        inst = mapping[sym]
        base = 80.0 + si * 15.0
        bars = []
        for i, d in enumerate(sessions):
            close = base + i * 0.15 + (si * 0.03)
            bars.append(
                DailyBar(
                    instrument_id=inst,
                    session_date=d,
                    open=close - 0.4,
                    high=close + 0.8,
                    low=close - 0.9,
                    close=close,
                    volume=1_000_000 + i * 1000,
                    finality=BarFinality.FINAL,
                    adjustment_revision="fixture_rev_1",
                    provider="fixture",
                    receive_ts=now,
                )
            )
        upsert_equity_daily_bars(store.conn, bars)
    btc_bars = []
    btc_inst = stable_instrument_id(settings.btc_symbol)
    for i, d in enumerate(cal):
        close = 40000.0 + i * 12.0
        btc_bars.append(
            DailyBar(
                instrument_id=btc_inst,
                session_date=d,
                open=close - 50,
                high=close + 80,
                low=close - 90,
                close=close,
                volume=100.0,
                finality=BarFinality.FINAL,
                adjustment_revision="fixture_rev_1",
                provider="fixture",
                receive_ts=now,
            )
        )
    upsert_btc_daily_bars(store.conn, btc_bars)
