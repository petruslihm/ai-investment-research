"""Vectorized point-in-time history for the legacy-style technical factors.

Every value at position ``i`` is computed from rows ``0..i`` only.  The formulas
mirror :mod:`trading_system.technical_factors`, whose scalar implementation is the
reference definition for a single as-of slice.
"""

from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd

from trading_system.technical_factors import fetch_bars_frames


TECHNICAL_FEATURE_NAMES: tuple[str, ...] = (
    "leader_score",
    "momentum_score",
    "volume_score",
    "buyable_score",
    "breakout_score",
    "top_risk_score",
    "quality_of_trend_score",
    "catalyst_score",
    "ret_5d",
    "week52_prox",
    "max_drawdown_60d",
    "has_big_bearish",
    "has_long_upper_wick",
)


def _clamp(values: np.ndarray, lo: float = 0.0, hi: float = 100.0) -> np.ndarray:
    """Vector form of technical_factors._clamp, including its NaN behavior."""
    values = np.asarray(values, dtype=float)
    return np.fmax(lo, np.fmin(hi, values))


def _returns(close: np.ndarray, lookback: int) -> np.ndarray:
    out = np.zeros(len(close), dtype=float)
    if lookback <= 0 or len(close) <= lookback:
        return out
    prior = close[:-lookback]
    current = close[lookback:]
    valid = np.isfinite(prior) & (prior != 0) & np.isfinite(current)
    values = np.zeros(len(current), dtype=float)
    np.divide(current, prior, out=values, where=valid)
    values[valid] = (values[valid] - 1.0) * 100.0
    out[lookback:] = values
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy(
        dtype=float, copy=True
    )


def _moving_average(close: np.ndarray, window: int) -> np.ndarray:
    out = _rolling_mean(close, window)
    out[np.arange(len(close)) < window - 1] = np.nan
    return out


def _above_ma(close: np.ndarray, moving_average: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(moving_average)
        & (moving_average != 0)
        & (close > moving_average)
    )


def _distance_from_ma(close: np.ndarray, moving_average: np.ndarray) -> np.ndarray:
    out = np.zeros(len(close), dtype=float)
    valid = np.isfinite(moving_average) & (moving_average != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = (close[valid] / moving_average[valid] - 1.0) * 100.0
    return out


def _ma_slope(close: np.ndarray, window: int, lookback: int = 5) -> np.ndarray:
    ma = _rolling_mean(close, window)
    then = pd.Series(ma).shift(lookback).to_numpy(dtype=float)
    out = np.zeros(len(close), dtype=float)
    eligible = np.arange(len(close)) >= window + lookback - 1
    valid = eligible & np.isfinite(then) & (then != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = (ma[valid] / then[valid] - 1.0) * 100.0
    return out


def _high_proximity(close: np.ndarray, window: int = 252) -> np.ndarray:
    high = pd.Series(close).rolling(window, min_periods=1).max().to_numpy(dtype=float)
    out = np.zeros(len(close), dtype=float)
    valid = np.isfinite(high) & (high != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = close[valid] / high[valid] * 100.0
    return out


def _low_proximity(close: np.ndarray, window: int = 252) -> np.ndarray:
    low = pd.Series(close).rolling(window, min_periods=1).min().to_numpy(dtype=float)
    out = np.zeros(len(close), dtype=float)
    valid = np.isfinite(low) & (low != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = (close[valid] / low[valid] - 1.0) * 100.0
    return out


def _annualized_volatility(
    daily_return: pd.Series, *, window: int, minimum_length: int
) -> np.ndarray:
    values = (
        daily_return.rolling(window, min_periods=2).std(ddof=1).to_numpy(dtype=float)
        * np.sqrt(252.0)
        * 100.0
    )
    values[np.arange(len(values)) < minimum_length - 1] = 0.0
    return np.nan_to_num(values, nan=0.0)


def _dollar_volume_surge(amount: np.ndarray) -> np.ndarray:
    recent = _rolling_mean(amount, 5)
    prior = pd.Series(amount).rolling(20, min_periods=1).mean().shift(5).to_numpy(dtype=float)
    out = np.ones(len(amount), dtype=float)
    eligible = np.arange(len(amount)) >= 24
    valid = eligible & np.isfinite(prior) & (prior > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = recent[valid] / prior[valid]
    return out


def _dollar_volume_surge_long(amount: np.ndarray) -> np.ndarray:
    recent = _rolling_mean(amount, 20)
    prior = pd.Series(amount).rolling(60, min_periods=1).mean().shift(20).to_numpy(dtype=float)
    out = np.ones(len(amount), dtype=float)
    eligible = np.arange(len(amount)) >= 79
    valid = eligible & np.isfinite(prior) & (prior > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = recent[valid] / prior[valid]
    return out


def _up_dollar_volume_ratio(
    amount: np.ndarray, daily_return: pd.Series, window: int = 15
) -> np.ndarray:
    # A scalar tail(window).pct_change() excludes the return into the first row of
    # each window, so the rolling return window is window - 1 sessions wide.
    up_amount = pd.Series(amount).where(daily_return > 0)
    down_amount = pd.Series(amount).where(daily_return < 0)
    up = up_amount.rolling(window - 1, min_periods=1).mean().to_numpy(dtype=float)
    down = down_amount.rolling(window - 1, min_periods=1).mean().to_numpy(dtype=float)

    out = np.ones(len(amount), dtype=float)
    enough_rows = np.arange(len(amount)) >= 2
    down_valid = np.isfinite(down) & (down > 0)
    ratio_rows = enough_rows & down_valid
    up_valid = np.isfinite(up) & (up > 0)
    out[enough_rows & ~down_valid & up_valid] = 2.0
    out[enough_rows & down_valid & ~np.isfinite(up)] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ratio_rows & np.isfinite(up)] = (
            up[ratio_rows & np.isfinite(up)] / down[ratio_rows & np.isfinite(up)]
        )
    return out


def _big_bearish(daily_return: pd.Series, window: int = 5) -> np.ndarray:
    return (
        daily_return.le(-0.05)
        .rolling(window, min_periods=1)
        .max()
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _long_upper_wick(
    open_: np.ndarray, high: np.ndarray, close: np.ndarray, window: int = 5
) -> np.ndarray:
    body = np.abs(close - open_)
    upper_wick = high - np.maximum(open_, close)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            upper_wick,
            body,
            out=np.full(len(close), np.nan, dtype=float),
            where=body != 0,
        )
    # The scalar implementation compares before fillna, so zero-body NaNs compare
    # False. Preserve that behavior exactly.
    flag = pd.Series(ratio > 2.0)
    return flag.rolling(window, min_periods=1).max().to_numpy(dtype=float)


def _volume_up_return_weak(
    volume: np.ndarray, daily_return: pd.Series, threshold: float
) -> np.ndarray:
    prior_average = (
        pd.Series(volume).rolling(21, min_periods=1).mean().shift(1).to_numpy(dtype=float)
    )
    ret = daily_return.to_numpy(dtype=float)
    eligible = np.arange(len(volume)) >= 21
    return (
        eligible
        & np.isfinite(prior_average)
        & (prior_average > 0)
        & (volume >= threshold * prior_average)
        & (ret < 0)
    )


def _consecutive_up(daily_return: pd.Series, window: int = 7) -> np.ndarray:
    positive = daily_return.gt(0).to_numpy(dtype=bool)
    groups = np.cumsum(~positive)
    streak = pd.Series(positive, dtype=int).groupby(groups).cumsum().to_numpy(dtype=float)
    return np.minimum(streak, float(window))


def _recent_pullback(close: np.ndarray, amount: np.ndarray, days: int = 7) -> np.ndarray:
    recent_price = _rolling_mean(close, days)
    prior_price = pd.Series(close).rolling(days, min_periods=1).mean().shift(days).to_numpy(dtype=float)
    recent_amount = _rolling_mean(amount, days)
    prior_amount = pd.Series(amount).rolling(days, min_periods=1).mean().shift(days).to_numpy(dtype=float)
    eligible = np.arange(len(close)) >= days * 2 - 1
    return (
        eligible
        & np.isfinite(prior_price)
        & (prior_price > 0)
        & np.isfinite(prior_amount)
        & (prior_amount > 0)
        & (recent_price < prior_price)
        & (recent_amount < prior_amount)
    )


def _rolling_max_drawdown(close: np.ndarray, window: int = 60) -> np.ndarray:
    """Exact trailing-window drawdown with the running peak reset per window."""
    n = len(close)
    out = np.zeros(n, dtype=float)
    if n < 2:
        return out

    prefix = pd.Series(close)
    prefix_dd = (prefix / prefix.cummax() - 1.0) * 100.0
    prefix_min = prefix_dd.expanding(min_periods=1).min().to_numpy(dtype=float)
    prefix_count = min(n, window - 1)
    out[:prefix_count] = np.nan_to_num(prefix_min[:prefix_count], nan=0.0)

    if n >= window:
        windows = np.lib.stride_tricks.sliding_window_view(close, window)
        running_high = np.maximum.accumulate(windows, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdowns = (windows / running_high - 1.0) * 100.0
        values = np.min(drawdowns, axis=1)
        out[window - 1 :] = np.where(np.isfinite(values), values, 0.0)
    return out


def _recent_box_breakout(
    close: np.ndarray, window: int = 20, within_days: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    box_high = (
        pd.Series(close).shift(1).rolling(window, min_periods=1).max().to_numpy(dtype=float)
    )
    positions = np.arange(len(close))
    event = (
        (positions >= window)
        & np.isfinite(box_high)
        & (box_high > 0)
        & (close > box_high)
    )
    event_marginal = event & (close < box_high * 1.01)

    event_1 = np.r_[False, event[:-1]] if len(event) else event.copy()
    event_2 = np.r_[[False, False], event[:-2]] if len(event) >= 2 else np.zeros(len(event), dtype=bool)
    marginal_1 = np.r_[False, event_marginal[:-1]] if len(event) else event_marginal.copy()
    marginal_2 = (
        np.r_[[False, False], event_marginal[:-2]]
        if len(event_marginal) >= 2
        else np.zeros(len(event_marginal), dtype=bool)
    )

    broke = event | event_1 | event_2
    marginal = np.where(event, event_marginal, np.where(event_1, marginal_1, marginal_2))
    enough_history = positions >= window + within_days
    return broke & enough_history, marginal & broke & enough_history


def _volatility_contraction_ratio(daily_return: pd.Series) -> np.ndarray:
    recent = daily_return.rolling(5, min_periods=2).std(ddof=1).to_numpy(dtype=float)
    prior = (
        daily_return.rolling(30, min_periods=2).std(ddof=1).shift(5).to_numpy(dtype=float)
    )
    out = np.ones(len(daily_return), dtype=float)
    eligible = np.arange(len(out)) >= 34
    valid = eligible & np.isfinite(prior) & (prior > 0) & np.isfinite(recent)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = recent[valid] / prior[valid]
    return out


def _rs_component(relative_strength: np.ndarray, budget: float) -> np.ndarray:
    scaled = 0.5 + np.fmax(-0.5, np.fmin(0.5, relative_strength / 30.0))
    return _clamp(scaled * budget, 0.0, budget)


def historical_technical_columns(
    df: pd.DataFrame,
    spy_close: pd.Series | None,
) -> dict[str, np.ndarray]:
    """Return all 13 technical feature histories without using future rows.

    ``df`` and a non-None ``spy_close`` must be positionally aligned and ascending
    by date. No centered windows or negative shifts are used.
    """
    required = ("open", "high", "low", "close", "volume")
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"missing required OHLCV columns: {', '.join(missing)}")

    n = len(df)
    open_ = df["open"].to_numpy(dtype=float, copy=False)
    high = df["high"].to_numpy(dtype=float, copy=False)
    close = df["close"].to_numpy(dtype=float, copy=False)
    volume = df["volume"].to_numpy(dtype=float, copy=False)
    close_series = pd.Series(close)
    daily_return = close_series.pct_change()
    amount = close * volume

    if spy_close is not None and len(spy_close) != n:
        raise ValueError("spy_close must be positionally aligned with df and have equal length")
    if spy_close is None or len(spy_close) == 0:
        spy_ret20 = np.zeros(n, dtype=float)
    else:
        spy_ret20 = _returns(spy_close.to_numpy(dtype=float, copy=False), 20)

    ret3 = _returns(close, 3)
    ret5 = _returns(close, 5)
    ret20 = _returns(close, 20)
    ret60 = _returns(close, 60)
    ma20 = _moving_average(close, 20)
    ma50 = _moving_average(close, 50)
    ma200 = _moving_average(close, 200)
    dist20 = _distance_from_ma(close, ma20)
    dist50 = _distance_from_ma(close, ma50)
    slope20 = _ma_slope(close, 20)
    slope50 = _ma_slope(close, 50)
    high_prox = _high_proximity(close)
    low_prox = _low_proximity(close)
    dv5 = _dollar_volume_surge(amount)
    dv20 = _dollar_volume_surge_long(amount)
    up_dv = _up_dollar_volume_ratio(amount, daily_return)
    bearish = _big_bearish(daily_return)
    wick = _long_upper_wick(open_, high, close)
    weak_volume_3 = _volume_up_return_weak(volume, daily_return, 3.0)
    weak_volume_2 = _volume_up_return_weak(volume, daily_return, 2.0)
    up_streak = _consecutive_up(daily_return)
    pullback = _recent_pullback(close, amount)
    drawdown = _rolling_max_drawdown(close)
    broke_out, marginal_breakout = _recent_box_breakout(close)
    contraction = _volatility_contraction_ratio(daily_return)

    leader = (
        _clamp(50.0 + (up_dv - 1.0) * 30.0) * 0.35
        + _clamp(50.0 + (dv5 - 1.0) * 30.0) * 0.25
        + _clamp(50.0 + (dv20 - 1.0) * 30.0) * 0.25
        + _clamp(50.0 + ret20) * 0.15
    )
    leader = _clamp(leader)

    ma_bonus = (
        _above_ma(close, ma20).astype(float)
        + _above_ma(close, ma50).astype(float)
        + _above_ma(close, ma200).astype(float)
    )
    momentum = (
        _clamp(50.0 + ret20 * 1.1) * 0.22
        + _clamp(50.0 + ret60 * 0.7) * 0.18
        + _rs_component(ret20 - spy_ret20, 30.0)
        + _clamp(high_prox) * 0.15
        + np.minimum(5.0, ma_bonus / 3.0 * 5.0)
    )
    momentum = _clamp(momentum)

    volume_score = (
        _clamp((dv5 - 1.0) * 20.5 + 20.5, 0.0, 41.0)
        + _clamp((dv20 - 1.0) * 12.0 + 12.0, 0.0, 24.0)
        + _clamp((up_dv - 1.0) * 17.5 + 17.5, 0.0, 35.0)
        - weak_volume_3.astype(float) * 15.0
    )
    volume_score = _clamp(volume_score)

    buyable = np.full(n, 50.0, dtype=float)
    buyable += np.select(
        [high_prox >= 95.0, high_prox >= 90.0, high_prox >= 80.0],
        [15.0, 10.0, 5.0],
        default=0.0,
    )
    buyable += ((dist20 >= 0.0) & (dist20 <= 10.0)) * 15.0
    buyable += pullback * 10.0
    two_bullish = np.zeros(n, dtype=bool)
    if n >= 2:
        bullish = close > open_
        two_bullish[1:] = bullish[1:] & bullish[:-1]
    buyable += two_bullish * 5.0
    buyable += (up_streak <= 2.0) * 5.0
    buyable -= np.select(
        [dist20 >= 20.0, dist20 >= 15.0], [25.0, 15.0], default=0.0
    )
    buyable -= np.select(
        [dist50 >= 30.0, dist50 >= 20.0], [15.0, 8.0], default=0.0
    )
    buyable -= np.select([ret5 >= 15.0, ret5 >= 10.0], [20.0, 10.0], default=0.0)
    buyable -= bearish * 15.0
    buyable -= wick * 10.0
    buyable -= np.select([up_streak >= 4.0, up_streak == 3.0], [15.0, 8.0], default=0.0)
    buyable = _clamp(buyable)

    breakout = np.zeros(n, dtype=float)
    breakout_candidate = np.select(
        [high_prox >= 95.0, high_prox >= 90.0], [30.0, 15.0], default=0.0
    )
    breakout_candidate += np.where(broke_out, np.where(marginal_breakout, 10.0, 20.0), 0.0)
    breakout_candidate += np.where(
        broke_out,
        np.select([dv5 >= 3.0, dv5 >= 2.0, dv5 >= 1.5], [25.0, 15.0, 8.0], default=0.0),
        0.0,
    )
    breakout_candidate += np.select(
        [contraction >= 2.5, contraction >= 1.5], [15.0, 8.0], default=0.0
    )
    breakout_candidate += ((np.abs(ret20) < 5.0) & (dv20 >= 1.3)) * 10.0
    enough_breakout_history = np.arange(n) >= 59
    breakout[enough_breakout_history] = _clamp(breakout_candidate)[enough_breakout_history]

    risk = np.select(
        [dist20 >= 25.0, dist20 >= 20.0, dist20 >= 15.0],
        [30.0, 20.0, 10.0],
        default=0.0,
    )
    risk += np.select(
        [dist50 >= 35.0, dist50 >= 25.0, dist50 >= 18.0],
        [20.0, 12.0, 6.0],
        default=0.0,
    )
    risk += np.select(
        [up_streak >= 5.0, up_streak >= 4.0, up_streak == 3.0],
        [20.0, 12.0, 6.0],
        default=0.0,
    )
    risk += np.select([ret5 >= 15.0, ret5 >= 10.0, ret5 >= 7.0], [25.0, 15.0, 8.0], default=0.0)
    risk += np.select([weak_volume_3, weak_volume_2], [20.0, 12.0], default=0.0)
    risk += bearish * 15.0
    risk += wick * 10.0
    risk += ((ret5 >= 5.0) & (dv5 < 0.8)) * 10.0
    risk = _clamp(risk)

    annual_vol20 = _annualized_volatility(daily_return, window=20, minimum_length=21)
    annual_vol60 = _annualized_volatility(daily_return, window=60, minimum_length=61)
    quality = np.full(n, 50.0, dtype=float)
    valid_vol20 = annual_vol20 > 0
    quality[valid_vol20] += (
        _clamp(ret20[valid_vol20] / (annual_vol20[valid_vol20] + 1e-9), -20.0, 20.0)
        * 0.5
    )
    valid_vol60 = annual_vol60 > 0
    quality[valid_vol60] += (
        _clamp(ret60[valid_vol60] / (annual_vol60[valid_vol60] + 1e-9) * 10.0, -15.0, 15.0)
        * 0.5
    )
    quality += np.select([drawdown <= -20.0, drawdown > -5.0], [-20.0, 8.0], default=0.0)
    quality += np.where(slope20 > 0, 8.0, -5.0)
    quality += np.where(slope50 > 0, 7.0, -5.0)
    above_ma20_raw = close_series > close_series.rolling(20).mean()
    above_ratio = above_ma20_raw.rolling(20, min_periods=1).mean().to_numpy(dtype=float)
    quality += (above_ratio - 0.5) * 20.0
    quality = _clamp(quality)

    catalyst = np.full(n, 50.0, dtype=float)
    gap = np.full(n, np.nan, dtype=float)
    bullish = close > open_
    if n >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            gap[1:] = (open_[1:] / close[:-1] - 1.0) * 100.0
    catalyst += np.select(
        [(gap >= 3.0) & bullish, gap >= 5.0], [20.0, 15.0], default=0.0
    )
    catalyst += np.where(high_prox >= 99.0, np.where(dv5 >= 1.5, 20.0, 12.0), 0.0)
    catalyst += ((ret3 >= 8.0) & (dv20 >= 1.2)) * 15.0
    catalyst += ((low_prox >= 15.0) & (low_prox <= 40.0)) * 10.0
    catalyst -= ((low_prox < 15.0) & (np.arange(n) >= 251)) * 10.0
    down_days = (
        daily_return.le(-0.02).rolling(5, min_periods=1).sum().to_numpy(dtype=float)
    )
    catalyst += ((ret5 >= 5.0) & (down_days == 0.0)) * 10.0
    catalyst -= (down_days >= 3.0) * 10.0
    catalyst = _clamp(catalyst)

    return {
        "leader_score": leader,
        "momentum_score": momentum,
        "volume_score": volume_score,
        "buyable_score": buyable,
        "breakout_score": breakout,
        "top_risk_score": risk,
        "quality_of_trend_score": quality,
        "catalyst_score": catalyst,
        "ret_5d": ret5,
        "week52_prox": high_prox,
        "max_drawdown_60d": drawdown,
        "has_big_bearish": bearish.astype(float),
        "has_long_upper_wick": wick.astype(float),
    }


def build_historical_technical_features(
    conn: duckdb.DuckDBPyConnection,
    *,
    instrument_ids: list[str],
    spy_instrument_id: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Batch-load full OHLCV histories and build aligned PIT technical columns."""
    requested = list(dict.fromkeys(str(instrument_id) for instrument_id in instrument_ids))
    fetch_ids = list(requested)
    if spy_instrument_id not in fetch_ids:
        fetch_ids.append(spy_instrument_id)
    frames = fetch_bars_frames(conn, fetch_ids, max_sessions=sys.maxsize)

    spy_frame = frames.get(spy_instrument_id)
    spy_by_date: pd.Series | None = None
    if spy_frame is not None and not spy_frame.empty:
        spy_by_date = (
            spy_frame.drop_duplicates("session_date", keep="last")
            .set_index("session_date")["close"]
        )

    result: dict[str, dict[str, np.ndarray]] = {}
    for instrument_id in requested:
        frame = frames.get(instrument_id)
        if frame is None or frame.empty:
            continue
        aligned_spy = (
            None
            if spy_by_date is None
            else frame["session_date"].map(spy_by_date).reset_index(drop=True)
        )
        result[instrument_id] = historical_technical_columns(frame, aligned_spy)
    return result
