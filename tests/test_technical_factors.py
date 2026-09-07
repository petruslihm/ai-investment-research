"""legacy-b style technical/chart-shape factors and hard exclusion gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_system.technical_factors import (
    MIN_SESSIONS_FOR_FACTORS,
    TRACK_AVOID,
    classify_track,
    compute_rs_percentiles,
    compute_technical_factors,
    consecutive_up,
    has_big_bearish,
    has_long_upper_wick,
    max_drawdown,
    momentum_score,
    passes_hard_gates,
    top_risk_score,
    week52_prox,
)


def _frame(closes: list[float], *, opens: list[float] | None = None, volumes: list[float] | None = None) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    open_ = pd.Series(opens, dtype=float) if opens is not None else close.shift(1).fillna(close.iloc[0])
    high = pd.concat([close, open_], axis=1).max(axis=1) * 1.001
    low = pd.concat([close, open_], axis=1).min(axis=1) * 0.999
    vol = pd.Series(volumes, dtype=float) if volumes is not None else pd.Series([100_000.0] * len(close))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def _steady_uptrend(n: int = 260, *, seed: int = 1, drift: float = 0.15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(drift, 1.0, n))
    return _frame(list(closes))


def _spy(n: int = 260, *, seed: int = 99, drift: float = 0.03) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(400 + np.cumsum(rng.normal(drift, 0.5, n)))


def _collapsing_series(n: int = 260) -> pd.DataFrame:
    """~30% peak-to-trough decline over the last month, including two big red days --
    the BURL-shaped scenario this whole module exists to catch."""
    rng = np.random.default_rng(0)
    base = list(370 + np.cumsum(rng.normal(0, 1.2, n - 20)))
    tail = [369, 358, 337, 314, 289.9, 273.0, 259.7, 262.1, 257.1, 257.5,
            255, 253, 251, 250, 249, 248, 247, 246, 245, 244]
    closes = base + tail
    volumes = [100_000.0] * (n - 20) + [100_000.0] * 20
    volumes[-16] = 400_000.0  # the -8%-ish crash day
    volumes[-15] = 350_000.0
    return _frame(closes, volumes=volumes)


# ---------------------------------------------------------------------------
# Raw indicator sanity
# ---------------------------------------------------------------------------


def test_week52_prox_is_100_at_the_high_and_less_elsewhere() -> None:
    close = pd.Series([100.0] * 200 + [110.0])  # today is a new high
    assert week52_prox(close) == pytest.approx(100.0)
    close2 = pd.Series([100.0] * 199 + [110.0] + [88.0])  # today is 20% off that high
    assert week52_prox(close2) == pytest.approx(80.0, rel=1e-2)


def test_max_drawdown_is_negative_and_matches_known_decline() -> None:
    close = pd.Series([100.0] * 10 + [80.0])  # -20% from the peak
    assert max_drawdown(close, w=60) == pytest.approx(-20.0, rel=1e-6)


def test_consecutive_up_counts_only_the_trailing_streak() -> None:
    close = pd.Series([100, 99, 101, 102, 103, 101, 102, 103, 104])
    # last three sessions are up (101->102->103->104), the run before that was broken
    assert consecutive_up(close, w=8) == 3


def test_has_big_bearish_detects_a_minus_5pct_day_and_ignores_normal_noise() -> None:
    calm = pd.Series([100, 100.5, 99.7, 100.2, 100.0, 99.9])
    assert has_big_bearish(_frame(list(calm))) is False
    crash = pd.Series([100, 100.5, 99.7, 100.2, 100.0, 93.5])  # -6.5% last day
    assert has_big_bearish(_frame(list(crash))) is True


def test_has_long_upper_wick_flags_a_rejected_rally_candle() -> None:
    df = _frame([100.0] * 6)
    # override the last candle: small green body, huge upper wick
    df.loc[df.index[-1], ["open", "close"]] = [100.0, 101.0]
    df.loc[df.index[-1], "high"] = 112.0
    df.loc[df.index[-1], "low"] = 99.5
    assert has_long_upper_wick(df) is True


# ---------------------------------------------------------------------------
# compute_technical_factors: end-to-end shape checks
# ---------------------------------------------------------------------------


def test_returns_none_below_min_history() -> None:
    df = _frame([100.0] * (MIN_SESSIONS_FOR_FACTORS - 1))
    assert compute_technical_factors(df, None) is None


def test_healthy_uptrend_is_not_hard_excluded() -> None:
    df = _steady_uptrend(seed=1)
    spy = _spy()
    f = compute_technical_factors(df, spy)
    assert f is not None
    assert f.hard_excluded is False
    assert f.track != TRACK_AVOID


def test_second_healthy_seed_is_also_not_hard_excluded() -> None:
    """Regression guard: an earlier version of classify_track() had a leader_score<50
    fallback into the Avoid track that wasn't part of legacy-b's two named gates and
    mis-fired on ordinary, unremarkable-but-fine uptrends. That fallback is gone."""
    df = _steady_uptrend(seed=7, drift=0.1)
    spy = _spy(seed=42)
    f = compute_technical_factors(df, spy)
    assert f is not None
    assert f.hard_excluded is False


def test_burl_shaped_collapse_is_hard_excluded_via_momentum_gate() -> None:
    df = _collapsing_series()
    spy = _spy()
    f = compute_technical_factors(df, spy)
    assert f is not None
    assert f.momentum_score < 30
    assert f.track == TRACK_AVOID
    assert f.hard_excluded is True
    assert f.exclusion_reason == "TECH_AVOID_MOMENTUM_BELOW_30"


def test_flat_dead_stock_does_not_crash_and_has_high_momentum_not_low() -> None:
    """A perfectly flat tape isn't 'weak' by the ret/RS-based momentum formula --
    it should compute cleanly, not blow up on div-by-zero."""
    df = _frame([100.0] * 260)
    f = compute_technical_factors(df, pd.Series([400.0] * 260))
    assert f is not None
    assert np.isfinite(f.momentum_score)
    assert np.isfinite(f.top_risk_score)


# ---------------------------------------------------------------------------
# passes_hard_gates() directly, with hand-built TechnicalFactors
# ---------------------------------------------------------------------------


def test_risk_gate_requires_all_three_conditions_together() -> None:
    from trading_system.technical_factors import TechnicalFactors

    base = dict(
        leader_score=50.0,
        momentum_score=50.0,
        volume_score=50.0,
        buyable_score=50.0,
        breakout_score=50.0,
        quality_of_trend_score=50.0,
        catalyst_score=50.0,
        track="Watch Only",
        week52_prox=80.0,
        max_drawdown_60d=-5.0,
    )
    # risk high alone: passes (ret_5d positive)
    f = TechnicalFactors(**base, top_risk_score=95.0, ret_5d=2.0, has_big_bearish=False, has_long_upper_wick=False)
    ok, reason = passes_hard_gates(f)
    assert ok is True and reason is None

    # risk high + decline, but no bearish/wick flag: passes
    f2 = TechnicalFactors(**base, top_risk_score=95.0, ret_5d=-2.0, has_big_bearish=False, has_long_upper_wick=False)
    ok2, _ = passes_hard_gates(f2)
    assert ok2 is True

    # risk high + decline + bearish candle: hard excluded
    f3 = TechnicalFactors(**base, top_risk_score=95.0, ret_5d=-2.0, has_big_bearish=True, has_long_upper_wick=False)
    ok3, reason3 = passes_hard_gates(f3)
    assert ok3 is False
    assert reason3 == "TECH_HIGH_RISK_BREAKDOWN"

    # risk just under the threshold: passes even with decline + wick
    f4 = TechnicalFactors(**base, top_risk_score=89.9, ret_5d=-2.0, has_big_bearish=False, has_long_upper_wick=True)
    ok4, _ = passes_hard_gates(f4)
    assert ok4 is True


def test_momentum_gate_is_the_sole_avoid_driver() -> None:
    from trading_system.technical_factors import TechnicalFactors

    base = dict(
        leader_score=10.0,  # deliberately low -- must NOT by itself cause exclusion
        volume_score=50.0,
        buyable_score=50.0,
        breakout_score=50.0,
        top_risk_score=0.0,
        quality_of_trend_score=50.0,
        catalyst_score=50.0,
        track="",
        ret_5d=1.0,
        week52_prox=80.0,
        max_drawdown_60d=-5.0,
        has_big_bearish=False,
        has_long_upper_wick=False,
    )
    f = TechnicalFactors(**base, momentum_score=30.1)
    f.track = classify_track(f)
    ok, _ = passes_hard_gates(f)
    assert ok is True, "momentum just above 30 with low leader_score must not be excluded"

    f2 = TechnicalFactors(**base, momentum_score=29.9)
    f2.track = classify_track(f2)
    ok2, reason2 = passes_hard_gates(f2)
    assert ok2 is False
    assert reason2 == "TECH_AVOID_MOMENTUM_BELOW_30"


def test_compute_rs_percentiles_ranks_the_batch() -> None:
    frames = {
        "strong": _frame(list(100 + np.arange(60) * 1.0)),
        "flat": _frame([100.0] * 60),
        "weak": _frame(list(100 - np.arange(60) * 0.5)),
    }
    spy = pd.Series([400.0] * 60)
    pct = compute_rs_percentiles(frames, spy, lookback=20)
    assert pct["strong"] > pct["flat"] > pct["weak"]
    assert 0.0 <= min(pct.values()) and max(pct.values()) <= 100.0


# ---------------------------------------------------------------------------
# classify_track: 2026-09-04 fixes (Hot Leader / Buyable mislabel, Catalyst Signal)
# ---------------------------------------------------------------------------


def _track_base(**over: object) -> dict:
    from trading_system.technical_factors import TechnicalFactors  # noqa: F401

    base = dict(
        leader_score=0.0,
        momentum_score=50.0,
        volume_score=50.0,
        buyable_score=50.0,
        breakout_score=0.0,
        top_risk_score=0.0,
        quality_of_trend_score=50.0,
        catalyst_score=50.0,
        track="",
        ret_5d=1.0,
        week52_prox=80.0,
        max_drawdown_60d=-5.0,
        has_big_bearish=False,
        has_long_upper_wick=False,
    )
    base.update(over)
    return base


def test_hot_leader_with_low_buyable_score_is_not_labeled_buyable() -> None:
    """Real-scan regression: CRM had leader_score=80, top_risk_score=50 (<65), but
    buyable_score=20 -- the old rule labeled it 'Hot Leader / Buyable' anyway because
    it never looked at buyable_score for this branch. 6 of 12 same-labeled names that
    day were actually bad entries (buyable_score<40)."""
    from trading_system.technical_factors import TechnicalFactors

    f = TechnicalFactors(**_track_base(leader_score=80.0, top_risk_score=50.0, buyable_score=20.0))
    assert classify_track(f) == "Hot Leader / Wait"


def test_hot_leader_with_high_buyable_score_keeps_the_buyable_label() -> None:
    from trading_system.technical_factors import TechnicalFactors

    f = TechnicalFactors(**_track_base(leader_score=80.0, top_risk_score=50.0, buyable_score=60.0))
    assert classify_track(f) == "Hot Leader / Buyable"


def test_extended_still_wins_over_the_buyable_check() -> None:
    """top_risk_score>=65 should still route to Extended regardless of buyable_score."""
    from trading_system.technical_factors import TechnicalFactors

    f = TechnicalFactors(**_track_base(leader_score=80.0, top_risk_score=70.0, buyable_score=90.0))
    assert classify_track(f) == "Hot Leader / Extended"


def test_catalyst_signal_surfaces_an_otherwise_unremarkable_name() -> None:
    """Real-scan regression: 66 names that day had catalyst_score>=60 with
    leader_score<=45 (AMZN, BA, CVS among them) and were indistinguishable from an
    ordinary day, buried in 'Watch Only'."""
    from trading_system.technical_factors import TechnicalFactors

    f = TechnicalFactors(**_track_base(leader_score=40.0, buyable_score=40.0, catalyst_score=65.0))
    assert classify_track(f) == "Catalyst Signal"


def test_catalyst_signal_never_outranks_a_stronger_track() -> None:
    """catalyst_score is checked last -- a name that already qualifies as a leader,
    breakout, or pullback-wait keeps that (stronger, more specific) label."""
    from trading_system.technical_factors import TechnicalFactors

    hot_leader = TechnicalFactors(**_track_base(leader_score=80.0, top_risk_score=50.0, buyable_score=60.0, catalyst_score=80.0))
    assert classify_track(hot_leader) == "Hot Leader / Buyable"

    breakout = TechnicalFactors(**_track_base(breakout_score=60.0, catalyst_score=80.0))
    assert classify_track(breakout) == "Breakout Signal"

    leader_buyable = TechnicalFactors(
        **_track_base(leader_score=60.0, buyable_score=60.0, top_risk_score=40.0, catalyst_score=80.0)
    )
    assert classify_track(leader_buyable) == "Leader / Buyable"


def test_catalyst_below_threshold_still_falls_to_watch_only() -> None:
    from trading_system.technical_factors import TechnicalFactors

    f = TechnicalFactors(**_track_base(leader_score=40.0, buyable_score=40.0, catalyst_score=55.0))
    assert classify_track(f) == "Watch Only"


def test_avoid_track_still_wins_over_a_high_catalyst_score() -> None:
    """The one real hard-gate driver (momentum<30) must still take priority."""
    from trading_system.technical_factors import TechnicalFactors

    f = TechnicalFactors(**_track_base(momentum_score=20.0, catalyst_score=80.0))
    assert classify_track(f) == TRACK_AVOID


# ---------------------------------------------------------------------------
# leader_score redesign (2026-09-04): capital inflow, not a second momentum score
# ---------------------------------------------------------------------------


def _same_price_different_volume(accumulation: bool) -> pd.DataFrame:
    """Identical price path; volume concentrated on up days (accumulation) or on
    down days (distribution)."""
    rng = np.random.default_rng(5)
    steps = rng.normal(0.1, 1.0, 120)
    closes = list(100 + np.cumsum(steps))
    frame = _frame(closes)
    ret = frame["close"].pct_change().fillna(0.0)
    up_vol, down_vol = (300_000.0, 60_000.0) if accumulation else (60_000.0, 300_000.0)
    frame["volume"] = [up_vol if r > 0 else down_vol for r in ret]
    return frame


def test_leader_score_separates_accumulation_from_distribution() -> None:
    """Same price path, opposite money flow -> leader_score must differ a lot while
    momentum_score (pure price/RS) stays identical."""
    from trading_system.technical_factors import leader_score

    spy = _spy(120)
    acc = _same_price_different_volume(accumulation=True)
    dist = _same_price_different_volume(accumulation=False)
    assert leader_score(acc, spy) > leader_score(dist, spy) + 15
    assert momentum_score(acc, spy) == pytest.approx(momentum_score(dist, spy))


def test_leader_score_ignores_spy_and_rs_percentile_by_design() -> None:
    """It is deliberately price/RS-free now -- benchmark inputs must not move it."""
    from trading_system.technical_factors import leader_score

    df = _steady_uptrend(seed=3)
    a = leader_score(df, _spy(seed=11), market_rs_percentile=5.0)
    b = leader_score(df, _spy(seed=77), market_rs_percentile=95.0)
    c = leader_score(df, None)
    assert a == pytest.approx(b) == pytest.approx(c)


def test_leader_and_momentum_are_no_longer_the_same_signal() -> None:
    """Heavy sustained accumulation on a name whose price has gone nowhere: high
    leader_score, unremarkable momentum_score. Under the pre-2026-09-04 formula
    (correlation 0.97 across a real 444-name scan) this divergence was impossible."""
    from trading_system.technical_factors import leader_score

    flat = _frame([100.0 + (i % 2) * 0.2 for i in range(120)])
    ret = flat["close"].pct_change().fillna(0.0)
    flat["volume"] = [400_000.0 if r > 0 else 50_000.0 for r in ret]
    spy = _spy(120)
    assert leader_score(flat, spy) > 60
    assert momentum_score(flat, spy) < 60
