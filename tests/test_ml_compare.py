"""Tests for the isolated 7-vs-20 feature comparison experiment."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import pytest

from trading_system import features
from trading_system.features_extended import (
    EXTENDED_FEATURE_NAMES,
    extended_load_matrix,
    extended_load_sequences,
)
from trading_system.ml_compare import compare_feature_sets
from trading_system.storage import Store


def _seed_comparison_rows(store: Store, *, n_dates: int = 130) -> None:
    feature_rows: list[tuple[object, ...]] = []
    label_rows: list[tuple[object, ...]] = []
    start = date(2024, 1, 2)
    for day_index in range(n_dates):
        as_of = start + timedelta(days=day_index)
        for instrument_index, instrument_id in enumerate(("inst_alpha", "inst_beta")):
            values = {
                name: float(
                    np.sin((day_index + 1) / (feature_index + 2.0))
                    + instrument_index * 0.05
                    + day_index * 0.001
                )
                for feature_index, name in enumerate(EXTENDED_FEATURE_NAMES)
            }
            label = float(
                0.025 * values[features.FEATURE_NAMES[1]]
                - 0.012 * values[features.FEATURE_NAMES[3]]
                + 0.008 * values[EXTENDED_FEATURE_NAMES[-1]]
            )
            target_end = as_of + timedelta(days=5)
            feature_rows.append(
                (
                    instrument_id,
                    as_of,
                    5,
                    "us_equity",
                    json.dumps(values),
                    None,
                )
            )
            label_rows.append(
                (
                    instrument_id,
                    as_of,
                    5,
                    "us_equity",
                    label,
                    target_end,
                    as_of,
                    target_end,
                    features.TARGET_VERSION,
                    True,
                )
            )
    store.conn.executemany(
        "INSERT INTO feature_rows VALUES (?, ?, ?, ?, ?, ?)", feature_rows
    )
    store.conn.executemany(
        "INSERT INTO label_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", label_rows
    )

    # The synthetic database intentionally contains both persisted JSON shapes.
    # The 7-key row is BTC and therefore outside this US-equity comparison.
    btc_as_of = date(2024, 1, 2)
    baseline_values = {name: float(i) for i, name in enumerate(features.FEATURE_NAMES)}
    store.conn.execute(
        "INSERT INTO feature_rows VALUES (?, ?, ?, ?, ?, NULL)",
        ["inst_btc_usd", btc_as_of, 5, "btc", json.dumps(baseline_values)],
    )
    store.conn.execute(
        "INSERT INTO label_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE)",
        [
            "inst_btc_usd",
            btc_as_of,
            5,
            "btc",
            0.01,
            btc_as_of + timedelta(days=5),
            btc_as_of,
            btc_as_of + timedelta(days=5),
            features.TARGET_VERSION,
        ],
    )


def test_compare_feature_sets_runs_end_to_end_for_both_widths(tmp_path: Path) -> None:
    db_dir = tmp_path / "ml_experiment"
    store = Store(db_dir / "compare.duckdb")
    store.open(acquire_writer=True)
    try:
        _seed_comparison_rows(store)

        baseline_x, baseline_y, baseline_keys = features.load_matrix(
            store.conn,
            asset_class="us_equity",
            horizon=5,
            matured_only=True,
        )
        extended_x, extended_y, extended_keys = extended_load_matrix(
            store.conn,
            asset_class="us_equity",
            horizon=5,
            matured_only=True,
        )
        assert baseline_x.shape == (260, 7)
        assert extended_x.shape == (260, 20)
        assert baseline_keys == extended_keys
        np.testing.assert_array_equal(baseline_y, extended_y)
        np.testing.assert_array_equal(baseline_x, extended_x[:, :7])

        sequence, sequence_y, sequence_keys = extended_load_sequences(
            store.conn,
            asset_class="us_equity",
            horizon=5,
            matured_only=True,
        )
        assert sequence.shape == (242, features.LSTM_LOOKBACK, 20)
        assert len(sequence_y) == len(sequence_keys) == 242

        before_journal = store.conn.execute(
            "SELECT count(*) FROM model_change_journal"
        ).fetchone()[0]
        before_predictions = store.conn.execute(
            "SELECT count(*) FROM prediction_rows"
        ).fetchone()[0]

        comparison = compare_feature_sets(store.conn, horizons=(5,))

        expected_metric_names = {
            "status",
            "mae",
            "rmse",
            "directional_accuracy",
            "pearson_correlation",
            "n_test",
        }
        for feature_set in ("baseline_7", "extended_20"):
            for family in (
                "ridge",
                "lightgbm_reg",
                "torch_sequence",
                "equal_weight_blend",
            ):
                metrics = comparison["metrics"][(5, feature_set, family)]
                assert metrics["status"] == "ok"
                assert expected_metric_names <= metrics.keys()
                assert metrics["n_test"] >= 50

        pairing = comparison["metadata"]["paired_samples"][5]
        assert pairing["extended_20"]["matches_baseline_samples"] is True
        assert pairing["extended_20"]["matches_baseline_first_seven"] is True
        assert comparison["metadata"]["skipped"] == []
        assert store.conn.execute(
            "SELECT count(*) FROM model_change_journal"
        ).fetchone()[0] == before_journal
        assert store.conn.execute(
            "SELECT count(*) FROM prediction_rows"
        ).fetchone()[0] == before_predictions
    finally:
        store.close()


def test_compare_refuses_non_experiment_database_before_table_reads() -> None:
    with TemporaryDirectory(prefix="phase_c_guard_", dir=Path.cwd() / "data") as temp_dir:
        conn = duckdb.connect(str(Path(temp_dir) / "guard.duckdb"))
        try:
            # This deliberately empty database has no feature_rows/label_rows. A
            # table read would fail differently, so the guard demonstrably runs first.
            with pytest.raises(RuntimeError, match="restricted.*ml_experiment"):
                compare_feature_sets(conn, horizons=(5,))
        finally:
            conn.close()
