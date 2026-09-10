"""Correctness tests for the experiment-only extended PIT feature pipeline."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from trading_system import features
from trading_system.features_extended import (
    EXTENDED_FEATURE_NAMES,
    build_and_persist_extended_features,
    extended_feature_columns,
)
from trading_system.market.calendar import add_sessions, is_trading_day
from trading_system.market.registry import stable_instrument_id
from trading_system.storage import Store
from trading_system.technical_features_pit import (
    TECHNICAL_FEATURE_NAMES,
    historical_technical_columns,
)


def _synthetic_frame(n: int, *, seed: int, shape: str) -> pd.DataFrame:
    """Match test_technical_features_pit.py's positive deterministic convention."""
    rng = np.random.default_rng(seed)
    if shape == "trend":
        returns = rng.normal(0.0010, 0.009, n)
    elif shape == "oscillating":
        returns = 0.009 * np.sin(np.arange(n) / 8.0) + rng.normal(0.0, 0.006, n)
    elif shape == "shocky":
        returns = rng.normal(0.0002, 0.013, n)
        returns[np.arange(37, n, 61)] = -0.065
        returns[np.arange(68, n, 73)] = 0.055
    else:  # pragma: no cover - parametrization owns valid shapes
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


def _session_dates(n: int, *, start: date = date(2024, 1, 2)) -> list[date]:
    sessions: list[date] = []
    current = start
    while len(sessions) < n:
        if is_trading_day(current):
            sessions.append(current)
        current += timedelta(days=1)
    return sessions


def _dated_frame(n: int, *, seed: int, shape: str, dates: list[date]) -> pd.DataFrame:
    frame = _synthetic_frame(n, seed=seed, shape=shape)
    frame.insert(0, "session_date", dates)
    return frame


def _series(frame: pd.DataFrame) -> list[tuple[date, float, float]]:
    return [
        (row.session_date, float(row.close), float(row.volume))
        for row in frame.itertuples(index=False)
    ]


@pytest.mark.parametrize(("seed", "shape"), [(11, "trend"), (47, "shocky")])
def test_extended_columns_are_exact_unperturbed_merge(seed: int, shape: str) -> None:
    dates = _session_dates(300)
    frame = _dated_frame(300, seed=seed, shape=shape, dates=dates)
    spy = _dated_frame(300, seed=seed + 1_000, shape="oscillating", dates=dates)
    series = _series(frame)

    actual = extended_feature_columns(series, frame, spy["close"])
    expected_base = features._feature_columns(series)
    expected_technical = historical_technical_columns(frame, spy["close"])

    assert tuple(actual) == EXTENDED_FEATURE_NAMES
    assert EXTENDED_FEATURE_NAMES == features.FEATURE_NAMES + TECHNICAL_FEATURE_NAMES
    assert all(values.shape == (len(frame),) for values in actual.values())
    for name in features.FEATURE_NAMES:
        np.testing.assert_array_equal(actual[name], expected_base[name])
    for name in TECHNICAL_FEATURE_NAMES:
        np.testing.assert_array_equal(actual[name], expected_technical[name])


def test_extended_columns_reject_date_misalignment() -> None:
    dates = _session_dates(30)
    frame = _dated_frame(30, seed=101, shape="trend", dates=dates)
    misaligned = frame.copy()
    misaligned.loc[10, "session_date"] = add_sessions(dates[10], 1)

    with pytest.raises(ValueError, match="positionally aligned dates"):
        extended_feature_columns(_series(frame), misaligned, frame["close"])


def _insert_frame(
    store: Store,
    instrument_id: str,
    frame: pd.DataFrame,
) -> None:
    received = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        (
            instrument_id,
            row.session_date,
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
            "final",
            "test-revision",
            "fixture",
            received,
        )
        for row in frame.itertuples(index=False)
    ]
    store.conn.executemany(
        """
        INSERT INTO equity_daily_bars (
            instrument_id, session_date, open, high, low, close, volume,
            finality, adjustment_revision, provider, receive_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_sentinel_rows(store: Store) -> None:
    sentinel_date = date(2023, 1, 3)
    for instrument_id, asset_class in (("inst_stale", "us_equity"), ("inst_btc_usd", "btc")):
        store.conn.execute(
            "INSERT INTO feature_rows VALUES (?, ?, 5, ?, ?, NULL)",
            [instrument_id, sentinel_date, asset_class, json.dumps({"sentinel": 1.0})],
        )
        store.conn.execute(
            """
            INSERT INTO label_rows VALUES (
                ?, ?, 5, ?, NULL, ?, ?, ?, ?, FALSE
            )
            """,
            [
                instrument_id,
                sentinel_date,
                asset_class,
                add_sessions(sentinel_date, 5),
                sentinel_date,
                add_sessions(sentinel_date, 5),
                features.TARGET_VERSION,
            ],
        )


def test_extended_builder_round_trip_is_lossless_and_pit_correct(tmp_path: Path) -> None:
    db_dir = tmp_path / "ml_experiment"
    store = Store(db_dir / "roundtrip.duckdb")
    store.open(acquire_writer=True)
    try:
        n = 55
        dates = _session_dates(n)
        spy_id = str(stable_instrument_id("SPY"))
        frames = {
            spy_id: _dated_frame(n, seed=200, shape="oscillating", dates=dates),
            "inst_alpha": _dated_frame(n, seed=201, shape="trend", dates=dates),
            "inst_beta": _dated_frame(n, seed=202, shape="shocky", dates=dates),
        }
        for instrument_id, frame in frames.items():
            _insert_frame(store, instrument_id, frame)
        _insert_sentinel_rows(store)

        written = build_and_persist_extended_features(
            store.conn,
            last_available=dates[-1],
        )

        expected_rows = 2 * (n - 20) * 3
        assert written == expected_rows
        assert store.conn.execute(
            "SELECT count(*) FROM feature_rows WHERE asset_class = 'us_equity'"
        ).fetchone()[0] == expected_rows
        assert store.conn.execute(
            "SELECT count(*) FROM feature_rows WHERE instrument_id = 'inst_stale'"
        ).fetchone()[0] == 0
        assert store.conn.execute(
            "SELECT count(*) FROM feature_rows WHERE asset_class = 'btc'"
        ).fetchone()[0] == 1
        assert store.conn.execute(
            "SELECT count(*) FROM label_rows WHERE asset_class = 'btc'"
        ).fetchone()[0] == 1

        spy_frame = frames[spy_id]
        for instrument_id, index, horizon in (
            ("inst_alpha", 20, 5),
            ("inst_alpha", 33, 10),
            ("inst_beta", 27, 20),
            ("inst_beta", n - 1, 5),
        ):
            as_of = dates[index]
            row = store.conn.execute(
                """
                SELECT f.features_json, l.label, l.label_available_date,
                       l.target_start, l.target_end, l.matured
                FROM feature_rows f
                JOIN label_rows l USING (instrument_id, as_of_date, horizon, asset_class)
                WHERE f.instrument_id = ? AND f.as_of_date = ?
                  AND f.horizon = ? AND f.asset_class = 'us_equity'
                """,
                [instrument_id, as_of, horizon],
            ).fetchone()
            assert row is not None
            payload = json.loads(row[0])

            # Recompute from only the prefix available at as_of; future rows are not
            # present in this independent PIT calculation.
            prefix = frames[instrument_id].iloc[: index + 1].reset_index(drop=True)
            spy_prefix = spy_frame.iloc[: index + 1].reset_index(drop=True)
            expected = extended_feature_columns(_series(prefix), prefix, spy_prefix["close"])
            assert tuple(payload) == EXTENDED_FEATURE_NAMES
            for name in EXTENDED_FEATURE_NAMES:
                assert payload[name] == expected[name][-1]

            end = add_sessions(as_of, horizon)
            assert row[2] == end
            assert row[3] == as_of
            assert row[4] == end
            if end in dates:
                end_index = dates.index(end)
                stock_return = frames[instrument_id]["close"].iloc[end_index] / frames[
                    instrument_id
                ]["close"].iloc[index] - 1.0
                spy_return = spy_frame["close"].iloc[end_index] / spy_frame["close"].iloc[
                    index
                ] - 1.0
                assert row[1] == pytest.approx(stock_return - spy_return)
                assert row[5] is True
            else:
                assert row[1] is None
                assert row[5] is False
    finally:
        store.close()


def test_extended_builder_rejects_non_experiment_database_before_writes() -> None:
    # Keep this path outside pytest's basetemp because the required Windows
    # workaround deliberately places basetemp under data/ml_experiment.
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="phase_b_guard_", dir=data_dir) as temp_dir:
        store = Store(Path(temp_dir) / "guard.duckdb")
        store.open(acquire_writer=True)
        try:
            _insert_sentinel_rows(store)
            before_features = store.conn.execute(
                "SELECT instrument_id, asset_class, features_json FROM feature_rows ORDER BY instrument_id"
            ).fetchall()
            before_labels = store.conn.execute(
                "SELECT instrument_id, asset_class, matured FROM label_rows ORDER BY instrument_id"
            ).fetchall()

            with pytest.raises(RuntimeError, match="restricted.*ml_experiment"):
                build_and_persist_extended_features(
                    store.conn,
                    last_available=date(2024, 1, 31),
                )

            assert store.conn.execute(
                "SELECT instrument_id, asset_class, features_json FROM feature_rows ORDER BY instrument_id"
            ).fetchall() == before_features
            assert store.conn.execute(
                "SELECT instrument_id, asset_class, matured FROM label_rows ORDER BY instrument_id"
            ).fetchall() == before_labels
        finally:
            store.close()
