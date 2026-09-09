"""legacy-b style technical/chart-shape factors, ported from etf-radar/screener/factors.py.

The ML Quant path (features.py + ml_engine.py) reduces every instrument to 7 purely
statistical features (ret_1/5/10, vol_10, hl_range_5, volume_z, mom_20) and has no
independent notion of "does this chart currently look broken" -- a name only needs to
clear a thin opportunity_score cutoff (default 0.01) to become a BUY candidate, no
matter how far it has fallen or how violently. This module restores legacy-b's
explicit, human-readable technical read of the chart: moving averages, 52-week high
proximity, breakout/box detection, candle-pattern flags (big bearish day, long upper
wick), volume-surge/distribution detection, and max-drawdown, combined into the same
composite scores legacy-b used (leader/momentum/volume/buyable/breakout/top_risk/
quality_of_trend/catalyst), plus its two hard exclusion gates:

  1. momentum_score < 30                                    -> track "Avoid / Weak"
  2. top_risk_score >= 90 and ret_5d < 0 and
     (big bearish candle or long upper wick in the last 5 sessions) -> hard exclude

Simplifications vs legacy-b (no equivalent data source exists in this system):
  - No sector ETF mapping -> sector relative strength is omitted (weight folded into
    the market/SPY relative-strength component instead of being a separate term).
  - No market-cap table -> the turnover-vs-market-cap component of volume_score is
    omitted (its weight is redistributed across the remaining volume_score terms).
  - No cross-sectional universe pass here -> RS-percentile-rank-across-universe is
    approximated with a smooth function of the absolute SPY-relative return instead
    of a true percentile rank. Callers that already have the full scored batch may
    pass `market_rs_percentile` (0-100) to use a real cross-sectional percentile.

All raw formulas are ported 1:1 in spirit from legacy-b (US market variant only --
legacy-b's KR-only factors like foreign/institution supply flow have no source data
here and are not ported).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

MIN_SESSIONS_FOR_FACTORS = 20
MIN_SESSIONS_FOR_BREAKOUT = 60

AVOID_MOMENTUM_THRESHOLD = 30.0
HARD_EXCLUDE_RISK_THRESHOLD = 90.0

TRACK_AVOID = "Avoid / Weak"


# ============================================================================
# Raw indicator helpers (all operate on a close-ascending-by-date pandas Series/
# DataFrame with columns open/high/low/close/volume; "today" is the last row).
# ============================================================================


def _ret(close: pd.Series, d: int) -> float:
    if len(close) <= d:
        return 0.0
    prior = close.iloc[-1 - d]
    current = close.iloc[-1]
    if not np.isfinite(prior) or prior == 0 or not np.isfinite(current):
        return 0.0
    return float((current / prior - 1.0) * 100.0)


def _ma(close: pd.Series, w: int) -> float | None:
    if len(close) < w:
        return None
    val = close.tail(w).mean()
    return float(val) if np.isfinite(val) else None


def _above_ma(close: pd.Series, w: int) -> bool:
    m = _ma(close, w)
    if m is None or m == 0:
        return False
    return bool(close.iloc[-1] > m)


def _dist_from_ma(close: pd.Series, w: int) -> float:
    """% distance of the current close from its w-day moving average."""
    m = _ma(close, w)
    if m is None or m == 0:
        return 0.0
    return float((close.iloc[-1] / m - 1.0) * 100.0)


def _ma_slope(close: pd.Series, w: int, lookback: int = 5) -> float:
    """% change of the w-day MA itself over the last `lookback` sessions (trend direction)."""
    if len(close) < w + lookback:
        return 0.0
    now = close.tail(w).mean()
    then = close.iloc[: len(close) - lookback].tail(w).mean()
    if not np.isfinite(then) or then == 0:
        return 0.0
    return float((now / then - 1.0) * 100.0)


def _high_prox(close: pd.Series, days: int) -> float:
    """% of the current close relative to the highest close in the last `days` sessions."""
    window = close.tail(min(days, len(close)))
    if window.empty:
        return 0.0
    hi = window.max()
    if not np.isfinite(hi) or hi == 0:
        return 0.0
    return float(close.iloc[-1] / hi * 100.0)


def week52_prox(close: pd.Series) -> float:
    return _high_prox(close, 252)


def week52_low_prox(close: pd.Series) -> float:
    """% the current close has recovered above the 52-week low (0 = at the low)."""
    window = close.tail(min(252, len(close)))
    if window.empty:
        return 0.0
    lo = window.min()
    if not np.isfinite(lo) or lo == 0:
        return 0.0
    return float((close.iloc[-1] / lo - 1.0) * 100.0)


def _atr(df: pd.DataFrame, w: int = 14) -> float:
    if len(df) < w + 1:
        return 0.0
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    val = tr.tail(w).mean()
    return float(val) if np.isfinite(val) else 0.0


def _ann_vol(close: pd.Series, w: int = 20) -> float:
    if len(close) < w + 1:
        return 0.0
    rets = close.pct_change().tail(w)
    std = rets.std(ddof=1)
    if not np.isfinite(std):
        return 0.0
    return float(std * np.sqrt(252) * 100.0)


def _amount(df: pd.DataFrame) -> pd.Series:
    return df["close"] * df["volume"]


def _dv_surge(df: pd.DataFrame) -> float:
    """5-day avg $volume vs the prior 20-day (6..25 sessions ago) avg $volume."""
    amt = _amount(df)
    if len(amt) < 25:
        return 1.0
    recent = amt.tail(5).mean()
    prior = amt.iloc[-25:-5].mean()
    if not np.isfinite(prior) or prior <= 0:
        return 1.0
    return float(recent / prior)


def _dv_surge_long(df: pd.DataFrame) -> float:
    """20-day avg $volume vs the prior 60-day (21..80 sessions ago) avg $volume."""
    amt = _amount(df)
    if len(amt) < 80:
        return 1.0
    recent = amt.tail(20).mean()
    prior = amt.iloc[-80:-20].mean()
    if not np.isfinite(prior) or prior <= 0:
        return 1.0
    return float(recent / prior)


def _up_dv_ratio(df: pd.DataFrame, w: int = 15) -> float:
    """Avg $volume on up-days / avg $volume on down-days over the last w sessions."""
    tail = df.tail(w)
    if len(tail) < 3:
        return 1.0
    ret = tail["close"].pct_change()
    amt = tail["close"] * tail["volume"]
    up = amt[ret > 0].mean()
    down = amt[ret < 0].mean()
    if not np.isfinite(down) or down <= 0:
        return 2.0 if np.isfinite(up) and up > 0 else 1.0
    if not np.isfinite(up):
        return 0.0
    return float(up / down)


def has_big_bearish(df: pd.DataFrame, thr: float = -0.05, w: int = 5) -> bool:
    """Any single-day close-to-close drop of `thr` (-5% default) or worse in the last w sessions."""
    tail = df["close"].tail(w + 1).pct_change().dropna()
    if tail.empty:
        return False
    return bool((tail <= thr).any())


def has_long_upper_wick(df: pd.DataFrame, w: int = 5) -> bool:
    """Any candle in the last w sessions whose upper wick exceeds 2x its body."""
    tail = df.tail(w)
    if tail.empty:
        return False
    body = (tail["close"] - tail["open"]).abs()
    upper_wick = tail["high"] - tail[["open", "close"]].max(axis=1)
    safe_body = body.replace(0, np.nan)
    ratio = upper_wick / safe_body
    return bool((ratio > 2.0).fillna(upper_wick > 0).any())


def vol_up_ret_weak(df: pd.DataFrame, thr: float = 3.0) -> bool:
    """Today's volume >= thr x its 21-day average, but today's return is negative.

    "Volume exploded but price fell" -- distribution / selling pressure.
    """
    if len(df) < 22:
        return False
    avg21 = df["volume"].iloc[-22:-1].mean()
    if not np.isfinite(avg21) or avg21 <= 0:
        return False
    today_vol = df["volume"].iloc[-1]
    today_ret = df["close"].iloc[-1] / df["close"].iloc[-2] - 1.0 if len(df) >= 2 else 0.0
    return bool(today_vol >= thr * avg21 and today_ret < 0)


def consecutive_up(close: pd.Series, w: int = 7) -> int:
    rets = close.tail(w + 1).pct_change().dropna()
    count = 0
    for r in reversed(rets.tolist()):
        if r > 0:
            count += 1
        else:
            break
    return count


def recent_pullback(close: pd.Series, df: pd.DataFrame, days: int = 7) -> bool:
    """Healthy pullback: price drifting down over `days` while $volume also fades."""
    if len(close) < days * 2:
        return False
    recent_price = close.tail(days).mean()
    prior_price = close.iloc[-days * 2 : -days].mean()
    amt = _amount(df)
    recent_amt = amt.tail(days).mean()
    prior_amt = amt.iloc[-days * 2 : -days].mean()
    if not (np.isfinite(prior_price) and prior_price > 0 and np.isfinite(prior_amt) and prior_amt > 0):
        return False
    return bool(recent_price < prior_price and recent_amt < prior_amt)


def max_drawdown(close: pd.Series, w: int = 60) -> float:
    """Max peak-to-trough drawdown (%) over the last w sessions. Negative or zero."""
    window = close.tail(min(w, len(close)))
    if len(window) < 2:
        return 0.0
    running_max = window.cummax()
    dd = (window / running_max - 1.0) * 100.0
    val = dd.min()
    return float(val) if np.isfinite(val) else 0.0


def is_recent_box_breakout(close: pd.Series, w: int = 20, within_days: int = 3) -> tuple[bool, bool]:
    """(broke_out, marginal) -- did the close break above the prior w-day box high
    within the last `within_days` sessions? `marginal` = broke by less than 1%."""
    if len(close) < w + within_days + 1:
        return False, False
    for back in range(within_days):
        idx = len(close) - 1 - back
        if idx < w:
            continue
        box_high = close.iloc[idx - w : idx].max()
        if not np.isfinite(box_high) or box_high <= 0:
            continue
        today = close.iloc[idx]
        if today > box_high:
            return True, bool(today < box_high * 1.01)
    return False, False


def volatility_contraction_ratio(close: pd.Series) -> float:
    """Recent-5-day realized vol vs the prior 10-30-session vol -- >1 means expansion
    after a contraction (classic pre-breakout volatility-squeeze-then-release shape)."""
    if len(close) < 35:
        return 1.0
    rets = close.pct_change()
    recent = rets.tail(5).std(ddof=1)
    prior = rets.iloc[-35:-5].std(ddof=1)
    if not np.isfinite(prior) or prior <= 0 or not np.isfinite(recent):
        return 1.0
    return float(recent / prior)


# ============================================================================
# Composite scores (0-100 unless noted). US-market port of legacy-b's factor set.
# ============================================================================


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, v)))


def _rs_component(rs: float, *, market_rs_percentile: float | None, budget: float) -> float:
    """Scale a relative-strength value (stock return - SPY return, in %) into a
    [0, budget] contribution. Uses a true cross-sectional percentile if the caller
    supplies one (see module docstring); otherwise a smooth absolute-RS approximation."""
    if market_rs_percentile is not None:
        return _clamp(market_rs_percentile / 100.0 * budget, 0.0, budget)
    # Smooth logistic-ish squash: rs=0 -> half budget, +-15% RS saturates the budget.
    scaled = 0.5 + max(-0.5, min(0.5, rs / 30.0))
    return _clamp(scaled * budget, 0.0, budget)


def leader_score(df: pd.DataFrame, spy_close: pd.Series | None, *, market_rs_percentile: float | None = None) -> float:
    """Capital-inflow / accumulation-strength score -- "how much money is actually
    flowing into this name right now", not "how strong is the price trend" (that's
    momentum_score). The two used to share ~all of their inputs (2026-09-04 real-scan
    audit: correlation 0.97 -- effectively the same signal shown twice). Redesigned to
    lean almost entirely on $volume behavior instead:
      - _up_dv_ratio (buying $ vs selling $ over 15 sessions): 35 -- the most direct
        accumulation-vs-distribution read.
      - _dv_surge (5d/20d $volume surge): 25 -- is fresh money showing up now.
      - _dv_surge_long (20d/60d $volume surge): 25 -- is that inflow sustained, not a
        one-day spike.
      - ret20 direction: 15 -- a light confirmation so heavy volume on a name that is
        actually falling (distribution, not accumulation) doesn't score as a "leader".
    `spy_close`/`market_rs_percentile` are accepted for call-signature compatibility
    with momentum_score but no longer used -- this score is intentionally price/RS-free.
    """
    close = df["close"]
    ret20 = _ret(close, 20)
    updv = _up_dv_ratio(df)
    dv = _dv_surge(df)
    dv_long = _dv_surge_long(df)
    score = 0.0
    score += _clamp(50 + (updv - 1.0) * 30.0, 0, 100) * 0.35
    score += _clamp(50 + (dv - 1.0) * 30.0, 0, 100) * 0.25
    score += _clamp(50 + (dv_long - 1.0) * 30.0, 0, 100) * 0.25
    score += _clamp(50 + ret20 * 1.0, 0, 100) * 0.15
    return _clamp(score)


def momentum_score(df: pd.DataFrame, spy_close: pd.Series | None, *, market_rs_percentile: float | None = None) -> float:
    """RS-driven momentum score. This is the score the hard "Avoid / Weak" gate reads."""
    close = df["close"]
    ret20 = _ret(close, 20)
    ret60 = _ret(close, 60)
    rs20 = ret20 - (_ret(spy_close, 20) if spy_close is not None and len(spy_close) else 0.0)
    score = 0.0
    score += _clamp(50 + ret20 * 1.1, 0, 100) * 0.22
    score += _clamp(50 + ret60 * 0.7, 0, 100) * 0.18
    score += _rs_component(rs20, market_rs_percentile=market_rs_percentile, budget=30.0)
    score += _clamp(week52_prox(close), 0, 100) * 0.15
    ma_bonus = sum(2.0 for w in (20, 50, 200) if _above_ma(close, w))
    score += min(5.0, ma_bonus / 6.0 * 5.0)
    return _clamp(score)


def volume_score(df: pd.DataFrame) -> float:
    """Turnover-vs-market-cap term omitted (no market-cap data); remaining weights
    rescaled from legacy's 35/20/30 (+10 turnover) to 41/24/35 so the max is still 100."""
    dv5 = _dv_surge(df)
    dv20 = _dv_surge_long(df)
    updv = _up_dv_ratio(df)
    score = 0.0
    score += _clamp((dv5 - 1.0) * 20.5 + 20.5, 0, 41)
    score += _clamp((dv20 - 1.0) * 12.0 + 12.0, 0, 24)
    score += _clamp((updv - 1.0) * 17.5 + 17.5, 0, 35)
    if vol_up_ret_weak(df, thr=3.0):
        score -= 15.0
    return _clamp(score)


def buyable_score(df: pd.DataFrame) -> float:
    """"Is this a reasonable entry right now" score."""
    close = df["close"]
    score = 50.0
    prox = week52_prox(close)
    if prox >= 95:
        score += 15
    elif prox >= 90:
        score += 10
    elif prox >= 80:
        score += 5
    ma20_dist = _dist_from_ma(close, 20)
    if 0 <= ma20_dist <= 10:
        score += 15
    if recent_pullback(close, df):
        score += 10
    if len(df) >= 2 and df["close"].iloc[-1] > df["open"].iloc[-1] and df["close"].iloc[-2] > df["open"].iloc[-2]:
        score += 5
    cons_up = consecutive_up(close)
    if cons_up <= 2:
        score += 5
    ma50_dist = _dist_from_ma(close, 50)
    ret5 = _ret(close, 5)
    if ma20_dist >= 20:
        score -= 25
    elif ma20_dist >= 15:
        score -= 15
    if ma50_dist >= 30:
        score -= 15
    elif ma50_dist >= 20:
        score -= 8
    if ret5 >= 15:
        score -= 20
    elif ret5 >= 10:
        score -= 10
    if has_big_bearish(df):
        score -= 15
    if has_long_upper_wick(df):
        score -= 10
    if cons_up >= 4:
        score -= 15
    elif cons_up == 3:
        score -= 8
    return _clamp(score)


def breakout_score(df: pd.DataFrame) -> float:
    if len(df) < MIN_SESSIONS_FOR_BREAKOUT:
        return 0.0
    close = df["close"]
    score = 0.0
    prox = week52_prox(close)
    if prox >= 95:
        score += 30
    elif prox >= 90:
        score += 15
    broke, marginal = is_recent_box_breakout(close)
    if broke:
        score += 10 if marginal else 20
        dv = _dv_surge(df)
        if dv >= 3.0:
            score += 25
        elif dv >= 2.0:
            score += 15
        elif dv >= 1.5:
            score += 8
    vcr = volatility_contraction_ratio(close)
    if vcr >= 2.5:
        score += 15
    elif vcr >= 1.5:
        score += 8
    if abs(_ret(close, 20)) < 5 and _dv_surge_long(df) >= 1.3:
        score += 10
    return _clamp(score)


def top_risk_score(df: pd.DataFrame) -> float:
    """Higher = more overextended/risky right now. Display + hard-gate input, not a
    standalone exclusion signal by itself (see passes_hard_gates)."""
    close = df["close"]
    score = 0.0
    ma20_dist = _dist_from_ma(close, 20)
    if ma20_dist >= 25:
        score += 30
    elif ma20_dist >= 20:
        score += 20
    elif ma20_dist >= 15:
        score += 10
    ma50_dist = _dist_from_ma(close, 50)
    if ma50_dist >= 35:
        score += 20
    elif ma50_dist >= 25:
        score += 12
    elif ma50_dist >= 18:
        score += 6
    cons_up = consecutive_up(close)
    if cons_up >= 5:
        score += 20
    elif cons_up >= 4:
        score += 12
    elif cons_up == 3:
        score += 6
    ret5 = _ret(close, 5)
    if ret5 >= 15:
        score += 25
    elif ret5 >= 10:
        score += 15
    elif ret5 >= 7:
        score += 8
    if vol_up_ret_weak(df, thr=3.0):
        score += 20
    elif vol_up_ret_weak(df, thr=2.0):
        score += 12
    if has_big_bearish(df):
        score += 15
    if has_long_upper_wick(df):
        score += 10
    if ret5 >= 5 and _dv_surge(df) < 0.8:
        score += 10
    return _clamp(score)


def quality_of_trend_score(df: pd.DataFrame) -> float:
    """Computed and surfaced for display, but -- matching legacy-b's own documented
    backtest finding -- given zero weight in the hard-gate/track decisions below."""
    close = df["close"]
    score = 50.0
    vol20 = _ann_vol(close, 20) / 100.0
    if vol20 > 0:
        score += _clamp(_ret(close, 20) / (vol20 * 100 + 1e-9), -20, 20) * 0.5
    vol60 = _ann_vol(close, min(60, len(close)))
    if vol60 > 0:
        score += _clamp(_ret(close, 60) / (vol60 + 1e-9) * 10, -15, 15) * 0.5
    mdd = max_drawdown(close, 60)
    if mdd <= -20:
        score -= 20
    elif mdd > -5:
        score += 8
    slope20 = _ma_slope(close, 20)
    score += 8 if slope20 > 0 else -5
    slope50 = _ma_slope(close, 50)
    score += 7 if slope50 > 0 else -5
    ma20 = close.rolling(20).mean()
    tail = close.tail(20)
    ma_tail = ma20.tail(20)
    above_ratio = float((tail > ma_tail).mean()) if len(tail) else 0.0
    score += (above_ratio - 0.5) * 20
    return _clamp(score)


def catalyst_proxy_score(df: pd.DataFrame) -> float:
    """Price/volume-pattern proxy for "something happened" -- no news source needed."""
    close = df["close"]
    score = 50.0
    if len(df) >= 2:
        gap = (df["open"].iloc[-1] / df["close"].iloc[-2] - 1.0) * 100.0
        bullish_close = df["close"].iloc[-1] > df["open"].iloc[-1]
        if gap >= 3 and bullish_close:
            score += 20
        elif gap >= 5:
            score += 15
    prox = week52_prox(close)
    if prox >= 99:
        score += 20 if _dv_surge(df) >= 1.5 else 12
    ret3 = _ret(close, 3)
    if ret3 >= 8 and _dv_surge_long(df) >= 1.2:
        score += 15
    low_prox = week52_low_prox(close)
    if 15 <= low_prox <= 40:
        score += 10
    elif low_prox < 15 and len(close) >= 252:
        score -= 10
    ret5 = _ret(close, 5)
    down_days = int((close.tail(6).pct_change().dropna() <= -0.02).sum())
    if ret5 >= 5 and down_days == 0:
        score += 10
    elif down_days >= 3:
        score -= 10
    return _clamp(score)


# ============================================================================
# Track classification + hard exclusion gates
# ============================================================================


@dataclass
class TechnicalFactors:
    leader_score: float
    momentum_score: float
    volume_score: float
    buyable_score: float
    breakout_score: float
    top_risk_score: float
    quality_of_trend_score: float
    catalyst_score: float
    track: str
    ret_5d: float
    week52_prox: float
    max_drawdown_60d: float
    has_big_bearish: bool
    has_long_upper_wick: bool
    hard_excluded: bool = False
    exclusion_reason: str | None = None
    notes: list[str] = field(default_factory=list)


CATALYST_SIGNAL_THRESHOLD = 60.0


def classify_track(f: TechnicalFactors) -> str:
    """Descriptive label for display. Only the momentum<30 branch feeds a hard
    exclusion (see passes_hard_gates) -- that is the one rule explicitly requested
    and is the only branch calibrated with confidence against legacy-b's documented
    threshold. The other labels are informational only and never gate anything, since
    this port's leader/buyable/breakout scoring is a best-effort reconstruction from
    legacy-b's point-budget descriptions, not a byte-exact copy of its formulas --
    using them to hard-exclude would risk silently dropping perfectly healthy names.

    2026-09-04 fixes (real-scan audit found both):
      - "Hot Leader / Buyable" used to fire whenever leader_score>=70 and
        top_risk_score<65, without checking buyable_score at all. 6 of 12 names
        labeled that way that day (CRM included: leader=80, buyable=20) were
        actually bad entries. Now split into an explicit /Buyable vs /Wait leg.
      - catalyst_score fed nothing here, so a real price/volume event on a
        non-leader name (66 names that day, catalyst_score>=60 with
        leader_score<=45 -- AMZN, BA, CVS among them) was indistinguishable from
        an ordinary day and got buried in "Watch Only". Added as its own track,
        checked last so it never outranks a stronger leader/breakout signal.
    """
    if f.momentum_score < AVOID_MOMENTUM_THRESHOLD:
        return TRACK_AVOID
    if f.leader_score >= 70 and f.top_risk_score >= 65:
        return "Hot Leader / Extended"
    if f.leader_score >= 70 and f.buyable_score >= 55:
        return "Hot Leader / Buyable"
    if f.leader_score >= 70:
        return "Hot Leader / Wait"
    if f.breakout_score >= 50:
        return "Breakout Signal"
    if f.leader_score >= 55 and f.buyable_score >= 55 and f.top_risk_score < 55:
        return "Leader / Buyable"
    if f.leader_score >= 55 and f.top_risk_score < 60:
        return "Leader / Pullback Wait"
    if f.catalyst_score >= CATALYST_SIGNAL_THRESHOLD:
        return "Catalyst Signal"
    return "Watch Only"


def passes_hard_gates(f: TechnicalFactors) -> tuple[bool, str | None]:
    """Returns (passes, reason). Mirrors legacy-b's two hard exclusions:
      1. screener/factors.py classify_track(): momentum_score < 30 -> "Avoid / Weak",
         and screener/engine.py _split_tracks(): Avoid-track rows are dropped entirely.
      2. app.py build_candidates(): risk >= 90 and ret_5d < 0 and
         (big bearish candle or long upper wick) -> hard removed from candidates.
    """
    if f.track == TRACK_AVOID:
        return False, "TECH_AVOID_MOMENTUM_BELOW_30"
    if (
        f.top_risk_score >= HARD_EXCLUDE_RISK_THRESHOLD
        and f.ret_5d < 0
        and (f.has_big_bearish or f.has_long_upper_wick)
    ):
        return False, "TECH_HIGH_RISK_BREAKDOWN"
    return True, None


def fetch_bars_frames(
    conn: duckdb.DuckDBPyConnection, instrument_ids: list[str], *, max_sessions: int = 300
) -> dict[str, pd.DataFrame]:
    """One batched query for every requested instrument's OHLCV, newest-trimmed to
    max_sessions per name, ascending by date. Missing/empty history simply yields no
    entry for that instrument_id (callers should treat that as "unknown", not "safe")."""
    if not instrument_ids:
        return {}
    placeholders = ",".join("?" for _ in instrument_ids)
    rows = conn.execute(
        f"""
        SELECT instrument_id, session_date, open, high, low, close, volume
        FROM equity_daily_bars
        WHERE instrument_id IN ({placeholders}) AND finality = 'final'
        ORDER BY instrument_id, session_date
        """,
        list(instrument_ids),
    ).fetchall()
    by_inst: dict[str, list[tuple]] = {}
    for inst, d, o, h, low_, c, v in rows:
        by_inst.setdefault(inst, []).append((d, o, h, low_, c, v))
    out: dict[str, pd.DataFrame] = {}
    for inst, series in by_inst.items():
        frame = pd.DataFrame(series, columns=["session_date", "open", "high", "low", "close", "volume"])
        frame = frame.tail(max_sessions).reset_index(drop=True)
        out[inst] = frame
    return out


def fetch_spy_close(conn: duckdb.DuckDBPyConnection, *, spy_instrument_id: str, max_sessions: int = 300) -> pd.Series | None:
    frames = fetch_bars_frames(conn, [spy_instrument_id], max_sessions=max_sessions)
    frame = frames.get(spy_instrument_id)
    if frame is None or frame.empty:
        return None
    return frame["close"]


def compute_rs_percentiles(
    frames: dict[str, pd.DataFrame], spy_close: pd.Series | None, *, lookback: int = 20
) -> dict[str, float]:
    """Cross-sectional percentile rank (0-100) of each instrument's `lookback`-session
    return-vs-SPY across the given batch -- a real percentile rank (legacy-b's RS
    percentile-rank term), not the single-name approximation in `_rs_component`.
    Pass the result's per-instrument value as `market_rs_percentile` below."""
    spy_ret = _ret(spy_close, lookback) if spy_close is not None and len(spy_close) else 0.0
    rs_values: dict[str, float] = {}
    for inst, frame in frames.items():
        if len(frame) < lookback + 1:
            continue
        rs_values[inst] = _ret(frame["close"], lookback) - spy_ret
    if not rs_values:
        return {}
    series = pd.Series(rs_values)
    pct = series.rank(pct=True) * 100.0
    return {str(k): float(v) for k, v in pct.to_dict().items()}


def compute_technical_factors(
    df: pd.DataFrame,
    spy_close: pd.Series | None = None,
    *,
    market_rs_percentile: float | None = None,
) -> TechnicalFactors | None:
    """df must have columns open/high/low/close/volume, ascending by date.

    Returns None if there isn't enough history (< MIN_SESSIONS_FOR_FACTORS sessions)
    to compute a meaningful read -- callers should treat that as "unknown", not "safe".
    """
    if df is None or len(df) < MIN_SESSIONS_FOR_FACTORS:
        return None
    close = df["close"]
    ld = leader_score(df, spy_close, market_rs_percentile=market_rs_percentile)
    mo = momentum_score(df, spy_close, market_rs_percentile=market_rs_percentile)
    vo = volume_score(df)
    by = buyable_score(df)
    br = breakout_score(df)
    rk = top_risk_score(df)
    qt = quality_of_trend_score(df)
    ca = catalyst_proxy_score(df)
    bearish = has_big_bearish(df)
    wick = has_long_upper_wick(df)
    notes: list[str] = []
    if bearish:
        notes.append("장대음봉(5일 내 -5% 이상 급락)")
    if wick:
        notes.append("긴 윗꼬리(고점 매도압력)")
    if vol_up_ret_weak(df):
        notes.append("거래량 폭증 + 하락 (분산/매도세)")
    f = TechnicalFactors(
        leader_score=ld,
        momentum_score=mo,
        volume_score=vo,
        buyable_score=by,
        breakout_score=br,
        top_risk_score=rk,
        quality_of_trend_score=qt,
        catalyst_score=ca,
        track="",
        ret_5d=_ret(close, 5),
        week52_prox=week52_prox(close),
        max_drawdown_60d=max_drawdown(close, 60),
        has_big_bearish=bearish,
        has_long_upper_wick=wick,
        notes=notes,
    )
    f.track = classify_track(f)
    ok, reason = passes_hard_gates(f)
    f.hard_excluded = not ok
    f.exclusion_reason = reason
    return f
