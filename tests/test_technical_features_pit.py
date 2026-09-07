"""PIT parity checks for vectorized historical technical features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_system.technical_factors import compute_technical_factors
from trading_system.technical_features_pit import (
    TECHNICAL_FEATURE_NAMES,
    build_historical_technical_features,
    historical_technical_columns,
)


COMPOSITE_FIELDS = (
    "leader_score",
    "momentum_score",
    "volume_score",
    "buyable_score",
    "breakout_score",
    "top_risk_score",
    "quality_of_trend_score",
    "catalyst_score",
)
NUMERIC_RAW_FIELDS = ("ret_5d", "week52_prox", "max_drawdown_60d")
FLAG_FIELDS = ("has_big_bearish", "has_long_upper_wick")


def _synthetic_frame(n: int, *, seed: int, shape: str) -> pd.DataFrame:
    """Match test_technical_factors.py's deterministic, positive OHLCV convention."""
    rng = np.random.default_rng(seed)
    if shape == "trend":
        returns = rng.normal(0.0010, 0.009, n)
    elif shape == "oscillating":
        returns = 0.009 * np.sin(np.arange(n) / 8.0) + rng.normal(0.0, 0.006, n)
    elif shape == "shocky":
        returns = rng.normal(0.0002, 0.013, n)
        returns[np.arange(37, n, 61)] = -0.065
        returns[np.arange(68, n, 73)] = 0.055
    else:  # pragma: no cover - parametrization owns the valid shapes
        raise ValueError(shape)

    close = 100.0 * np.exp(np.cumsum(returns))
    previous = np.r_[close[0], close[:-1]]
    open_ = previous * np.exp(rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.018, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.018, n))
    volume = rng.lognormal(mean=np.log(150_000.0), sigma=0.45, size=n)

    if shape == "shocky":
        volume[np.arange(37, n, 61)] *= 4.0
        wick_rows = np.arange(45, n, 67)
        high[wick_rows] = np.maximum(open_[wick_rows], close[wick_rows]) * 1.14

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _synthetic_spy(n: int, *, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.006, n)
    return pd.Series(400.0 * np.exp(np.cumsum(returns)), dtype=float)


@pytest.mark.parametrize(
    ("seed", "shape"),
    [(11, "trend"), (29, "oscillating"), (47, "shocky")],
)
def test_historical_columns_match_every_prefix_without_lookahead(
    seed: int, shape: str
) -> None:
    n = 300
    df = _synthetic_frame(n, seed=seed, shape=shape)
    spy_close = _synthetic_spy(n, seed=seed + 1_000)

    columns = historical_technical_columns(df, spy_close)

    assert tuple(columns) == TECHNICAL_FEATURE_NAMES
    assert all(values.shape == (n,) for values in columns.values())
    sample_indices = sorted(
        set(range(25, n, 18)) | {19, 20, 23, 24, 34, 59, 60, 79, 199, 251, n - 1}
    )
    assert len(sample_indices) >= 15
    assert sample_indices[-1] == n - 1

    for i in sample_indices:
        expected = compute_technical_factors(df.iloc[: i + 1], spy_close.iloc[: i + 1])
        assert expected is not None

        for field in COMPOSITE_FIELDS + NUMERIC_RAW_FIELDS:
            assert columns[field][i] == pytest.approx(float(getattr(expected, field)))
        for field in FLAG_FIELDS:
            assert columns[field][i] == float(getattr(expected, field))


def test_batch_builder_fetches_full_histories_once_and_aligns_spy(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _synthetic_frame(300, seed=71, shape="trend")
    frame.insert(0, "session_date", pd.date_range("2025-01-02", periods=len(frame), freq="B").date)
    spy_frame = frame.copy()
    spy_frame["close"] = _synthetic_spy(len(frame), seed=72).to_numpy()
    calls: list[tuple[object, list[str], int]] = []
    connection = object()

    def fake_fetch(
        conn: object, instrument_ids: list[str], *, max_sessions: int
    ) -> dict[str, pd.DataFrame]:
        calls.append((conn, instrument_ids, max_sessions))
        return {"stock-1": frame, "spy-1": spy_frame}

    monkeypatch.setattr(
        "trading_system.technical_features_pit.fetch_bars_frames", fake_fetch
    )
    result = build_historical_technical_features(
        connection,  # type: ignore[arg-type]
        instrument_ids=["stock-1"],
        spy_instrument_id="spy-1",
    )

    assert len(calls) == 1
    assert calls[0][0] is connection
    assert calls[0][1] == ["stock-1", "spy-1"]
    assert calls[0][2] > len(frame)
    expected = historical_technical_columns(frame, spy_frame["close"])
    assert result.keys() == {"stock-1"}
    for field in TECHNICAL_FEATURE_NAMES:
        np.testing.assert_allclose(result["stock-1"][field], expected[field])
