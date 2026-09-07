"""Isolated baseline-vs-extended feature comparison experiment.

This module is intentionally not imported by any production code path.  It reads
the experiment database, fits in memory, and never persists models, predictions,
journal entries, or artifacts.
"""

from __future__ import annotations

import hashlib
from datetime import date
from time import perf_counter
from typing import Callable

import duckdb
import numpy as np
from sklearn.preprocessing import StandardScaler

from trading_system import features, features_extended
from trading_system.ids import AssetClass
from trading_system.ml_engine import (
    FittedBundle,
    _fit_lgbm,
    _fit_lstm,
    _fit_ridge,
    _predict,
    _training_window,
    chronological_split,
)


MIN_TRAIN_ROWS = 20
MIN_TEST_ROWS = 50
MAX_LSTM_MATRIX_ROWS = 50_000
FAMILIES = ("ridge", "lightgbm_reg", "torch_sequence", "equal_weight_blend")

MatrixLoader = Callable[..., tuple[np.ndarray, np.ndarray, list[tuple[str, date]]]]
SequenceLoader = Callable[..., tuple[np.ndarray, np.ndarray, list[tuple[str, date]]]]


def _assert_experiment_database(conn: duckdb.DuckDBPyConnection) -> None:
    """Refuse non-experiment paths before reading feature or label rows."""
    path = features_extended._database_path(conn)
    if path is None or "ml_experiment" not in path.casefold():
        shown = path if path is not None else "<in-memory or unknown>"
        raise RuntimeError(
            "ML feature comparison is restricted to a database path containing "
            f"'ml_experiment'; connected path is {shown}"
        )


def _sample_fingerprint(keys: list[tuple[str, date]], y: np.ndarray) -> str:
    digest = hashlib.sha256()
    for (instrument_id, as_of), value in zip(keys, y, strict=True):
        digest.update(str(instrument_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(as_of.isoformat().encode("ascii"))
        digest.update(b"\0")
        digest.update(np.float64(value).tobytes())
    return digest.hexdigest()


def _feature_fingerprint(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def _metrics(prediction: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    if len(prediction) != len(actual):
        raise ValueError("prediction and actual arrays must have the same length")
    finite = np.isfinite(prediction) & np.isfinite(actual)
    prediction = prediction[finite]
    actual = actual[finite]
    if len(actual) == 0:
        return {
            "status": "skipped",
            "reason": "no finite prediction/label pairs",
            "n_test": 0,
        }
    error = prediction - actual
    if len(actual) < 2 or np.std(prediction) == 0 or np.std(actual) == 0:
        correlation: float | None = None
    else:
        correlation = float(np.corrcoef(prediction, actual)[0, 1])
    return {
        "status": "ok",
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "directional_accuracy": float(np.mean(np.sign(prediction) == np.sign(actual))),
        "pearson_correlation": correlation,
        "n_test": int(len(actual)),
    }


def _prediction_maps(
    keys: list[tuple[str, date]],
    prediction: np.ndarray,
    actual: np.ndarray,
) -> tuple[dict[tuple[str, date], float], dict[tuple[str, date], float]]:
    return (
        {key: float(value) for key, value in zip(keys, prediction, strict=True)},
        {key: float(value) for key, value in zip(keys, actual, strict=True)},
    )


def _skip(
    results: dict[tuple[int, str, str], dict[str, object]],
    skipped: list[dict[str, object]],
    *,
    horizon: int,
    feature_set: str,
    family: str,
    reason: str,
    n_test: int = 0,
) -> None:
    results[(horizon, feature_set, family)] = {
        "status": "skipped",
        "reason": reason,
        "n_test": int(n_test),
    }
    skipped.append(
        {
            "horizon": horizon,
            "feature_set": feature_set,
            "family": family,
            "reason": reason,
        }
    )


def compare_feature_sets(
    conn: duckdb.DuckDBPyConnection,
    *,
    horizons=(5, 10, 20),
) -> dict:
    """Compare 7 and 20 PIT features with production's chronological split.

    All fitting and prediction is in-memory.  The connection is guarded to an
    ``ml_experiment`` path before the first feature/label-table query.
    """
    started = perf_counter()
    _assert_experiment_database(conn)
    asset_class = AssetClass.US_EQUITY.value
    loaders: tuple[tuple[str, MatrixLoader, SequenceLoader, int], ...] = (
        (
            "baseline_7",
            features.load_matrix,
            features.load_sequences,
            len(features.FEATURE_NAMES),
        ),
        (
            "extended_20",
            features_extended.extended_load_matrix,
            features_extended.extended_load_sequences,
            len(features_extended.EXTENDED_FEATURE_NAMES),
        ),
    )
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    results: dict[tuple[int, str, str], dict[str, object]] = {}
    skipped: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "asset_class": asset_class,
        "horizons": requested_horizons,
        "feature_sets": {
            "baseline_7": list(features.FEATURE_NAMES),
            "extended_20": list(features_extended.EXTENDED_FEATURE_NAMES),
        },
        "thresholds": {
            "min_train_rows": MIN_TRAIN_ROWS,
            "min_test_rows": MIN_TEST_ROWS,
            "max_lstm_matrix_rows": MAX_LSTM_MATRIX_ROWS,
            "rationale": (
                "20 train rows avoids LightGBM's built-in <20-row Ridge fallback; "
                "50 test rows avoids reporting extremely unstable metrics; the LSTM "
                "cap bounds the production helper's 25-epoch full-batch memory/CPU cost."
            ),
        },
        "split_method": (
            "chronological_split(len(y), purge=h, keys=keys), separately repeated "
            "for matured LSTM sequences"
        ),
        "blend_method": (
            "simple arithmetic mean over fitted-family predictions on their common "
            "test keys; not the live adaptive EWMA ensemble"
        ),
        "splits": {},
        "paired_samples": {},
        "skipped": skipped,
    }
    splits: dict[int, dict[str, object]] = metadata["splits"]  # type: ignore[assignment]
    paired: dict[int, dict[str, object]] = metadata["paired_samples"]  # type: ignore[assignment]

    for horizon in requested_horizons:
        splits[horizon] = {}
        paired[horizon] = {}
        reference_sample_fingerprint: str | None = None
        reference_base_feature_fingerprint: str | None = None
        for feature_set, matrix_loader, sequence_loader, n_features in loaders:
            combo_started = perf_counter()
            try:
                x, y, keys = matrix_loader(
                    conn,
                    asset_class=asset_class,
                    horizon=horizon,
                    matured_only=True,
                )
            except Exception as exc:  # noqa: BLE001 - one bad combination must not abort all
                reason = f"matured matrix load failed: {type(exc).__name__}: {exc}"
                for family in FAMILIES:
                    _skip(
                        results,
                        skipped,
                        horizon=horizon,
                        feature_set=feature_set,
                        family=family,
                        reason=reason,
                    )
                splits[horizon][feature_set] = {"status": "skipped", "reason": reason}
                continue

            sample_fingerprint = _sample_fingerprint(keys, y)
            first_seven_fingerprint = _feature_fingerprint(x[:, : len(features.FEATURE_NAMES)])
            if reference_sample_fingerprint is None:
                reference_sample_fingerprint = sample_fingerprint
                reference_base_feature_fingerprint = first_seven_fingerprint
            samples_match = sample_fingerprint == reference_sample_fingerprint
            first_seven_match = first_seven_fingerprint == reference_base_feature_fingerprint
            paired[horizon][feature_set] = {
                "sample_fingerprint": sample_fingerprint,
                "matches_baseline_samples": samples_match,
                "first_seven_feature_fingerprint": first_seven_fingerprint,
                "matches_baseline_first_seven": first_seven_match,
            }

            if not samples_match:
                reason = "feature sets did not load identical keys and labels"
                for family in FAMILIES:
                    _skip(
                        results,
                        skipped,
                        horizon=horizon,
                        feature_set=feature_set,
                        family=family,
                        reason=reason,
                    )
                splits[horizon][feature_set] = {
                    "status": "skipped",
                    "reason": reason,
                    "n_rows": int(len(y)),
                }
                continue
            if x.ndim != 2 or x.shape != (len(y), n_features):
                reason = f"unexpected matrix shape {x.shape}; expected ({len(y)}, {n_features})"
                for family in FAMILIES:
                    _skip(
                        results,
                        skipped,
                        horizon=horizon,
                        feature_set=feature_set,
                        family=family,
                        reason=reason,
                    )
                splits[horizon][feature_set] = {
                    "status": "skipped",
                    "reason": reason,
                    "n_rows": int(len(y)),
                }
                continue
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                reason = "matrix or labels contain non-finite values"
                for family in FAMILIES:
                    _skip(
                        results,
                        skipped,
                        horizon=horizon,
                        feature_set=feature_set,
                        family=family,
                        reason=reason,
                    )
                splits[horizon][feature_set] = {
                    "status": "skipped",
                    "reason": reason,
                    "n_rows": int(len(y)),
                }
                continue

            train_idx, test_idx = chronological_split(len(y), purge=horizon, keys=keys)
            split_info: dict[str, object] = {
                "status": "ok",
                "n_features": n_features,
                "n_rows": int(len(y)),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_period": _training_window(keys, train_idx),
                "test_period": _training_window(keys, test_idx),
            }
            splits[horizon][feature_set] = split_info
            if len(train_idx) < MIN_TRAIN_ROWS or len(test_idx) < MIN_TEST_ROWS:
                reason = (
                    f"insufficient split rows: train={len(train_idx)} "
                    f"(minimum {MIN_TRAIN_ROWS}), test={len(test_idx)} "
                    f"(minimum {MIN_TEST_ROWS})"
                )
                split_info["status"] = "skipped"
                split_info["reason"] = reason
                for family in FAMILIES:
                    _skip(
                        results,
                        skipped,
                        horizon=horizon,
                        feature_set=feature_set,
                        family=family,
                        reason=reason,
                        n_test=len(test_idx),
                    )
                continue

            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]
            flat_test_keys = [keys[int(i)] for i in test_idx]
            predictions_by_family: dict[str, dict[tuple[str, date], float]] = {}
            actuals_by_family: dict[str, dict[tuple[str, date], float]] = {}

            ridge_scaler: StandardScaler | None = None
            try:
                ridge_scaler, ridge = _fit_ridge(x_train, y_train)
                bundle = FittedBundle(
                    "ridge", horizon, asset_class, ridge_scaler, ridge, "compare_v1"
                )
                prediction = _predict(bundle, x_test)
                results[(horizon, feature_set, "ridge")] = _metrics(prediction, y_test)
                predictions_by_family["ridge"], actuals_by_family["ridge"] = (
                    _prediction_maps(flat_test_keys, prediction, y_test)
                )
            except Exception as exc:  # noqa: BLE001 - preserve other family comparisons
                _skip(
                    results,
                    skipped,
                    horizon=horizon,
                    feature_set=feature_set,
                    family="ridge",
                    reason=f"fit/predict failed: {type(exc).__name__}: {exc}",
                    n_test=len(test_idx),
                )

            try:
                lightgbm = _fit_lgbm(x_train, y_train, rank=False, groups=None)
                if ridge_scaler is None:
                    ridge_scaler = StandardScaler().fit(x_train)
                bundle = FittedBundle(
                    "lightgbm_reg",
                    horizon,
                    asset_class,
                    ridge_scaler,
                    lightgbm,
                    "compare_v1",
                )
                prediction = _predict(bundle, x_test)
                results[(horizon, feature_set, "lightgbm_reg")] = _metrics(
                    prediction, y_test
                )
                (
                    predictions_by_family["lightgbm_reg"],
                    actuals_by_family["lightgbm_reg"],
                ) = _prediction_maps(flat_test_keys, prediction, y_test)
            except Exception as exc:  # noqa: BLE001 - preserve other family comparisons
                _skip(
                    results,
                    skipped,
                    horizon=horizon,
                    feature_set=feature_set,
                    family="lightgbm_reg",
                    reason=f"fit/predict failed: {type(exc).__name__}: {exc}",
                    n_test=len(test_idx),
                )

            if len(y) > MAX_LSTM_MATRIX_ROWS:
                reason = (
                    f"matured matrix has {len(y)} rows, above the "
                    f"{MAX_LSTM_MATRIX_ROWS}-row LSTM full-batch safety cap"
                )
                _skip(
                    results,
                    skipped,
                    horizon=horizon,
                    feature_set=feature_set,
                    family="torch_sequence",
                    reason=reason,
                )
                split_info["sequence_split"] = {"status": "skipped", "reason": reason}
            else:
                try:
                    sequence, sequence_y, sequence_keys = sequence_loader(
                        conn,
                        asset_class=asset_class,
                        horizon=horizon,
                        lookback=features.LSTM_LOOKBACK,
                        matured_only=True,
                    )
                    sequence_train, sequence_test = chronological_split(
                        len(sequence_y), purge=horizon, keys=sequence_keys
                    )
                    sequence_info: dict[str, object] = {
                        "status": "ok",
                        "lookback": features.LSTM_LOOKBACK,
                        "n_rows": int(len(sequence_y)),
                        "n_train": int(len(sequence_train)),
                        "n_test": int(len(sequence_test)),
                        "train_period": _training_window(sequence_keys, sequence_train),
                        "test_period": _training_window(sequence_keys, sequence_test),
                    }
                    split_info["sequence_split"] = sequence_info
                    if (
                        len(sequence_train) < MIN_TRAIN_ROWS
                        or len(sequence_test) < MIN_TEST_ROWS
                    ):
                        reason = (
                            f"insufficient sequence split rows: train={len(sequence_train)} "
                            f"(minimum {MIN_TRAIN_ROWS}), test={len(sequence_test)} "
                            f"(minimum {MIN_TEST_ROWS})"
                        )
                        sequence_info["status"] = "skipped"
                        sequence_info["reason"] = reason
                        _skip(
                            results,
                            skipped,
                            horizon=horizon,
                            feature_set=feature_set,
                            family="torch_sequence",
                            reason=reason,
                            n_test=len(sequence_test),
                        )
                    else:
                        lstm = _fit_lstm(
                            sequence[sequence_train], sequence_y[sequence_train]
                        )
                        bundle = FittedBundle(
                            "torch_sequence",
                            horizon,
                            asset_class,
                            StandardScaler(),
                            lstm,
                            "compare_v1",
                            lookback=features.LSTM_LOOKBACK,
                        )
                        sequence_prediction = _predict(bundle, sequence[sequence_test])
                        sequence_actual = sequence_y[sequence_test]
                        sequence_test_keys = [
                            sequence_keys[int(i)] for i in sequence_test
                        ]
                        results[(horizon, feature_set, "torch_sequence")] = _metrics(
                            sequence_prediction, sequence_actual
                        )
                        (
                            predictions_by_family["torch_sequence"],
                            actuals_by_family["torch_sequence"],
                        ) = _prediction_maps(
                            sequence_test_keys, sequence_prediction, sequence_actual
                        )
                except Exception as exc:  # noqa: BLE001 - tabular results remain valid
                    reason = f"sequence fit/predict failed: {type(exc).__name__}: {exc}"
                    _skip(
                        results,
                        skipped,
                        horizon=horizon,
                        feature_set=feature_set,
                        family="torch_sequence",
                        reason=reason,
                    )
                    split_info["sequence_split"] = {
                        "status": "skipped",
                        "reason": reason,
                    }

            fitted_families = list(predictions_by_family)
            if not fitted_families:
                _skip(
                    results,
                    skipped,
                    horizon=horizon,
                    feature_set=feature_set,
                    family="equal_weight_blend",
                    reason="no fitted families available for blending",
                )
            else:
                common_keys = set(predictions_by_family[fitted_families[0]])
                for family in fitted_families[1:]:
                    common_keys.intersection_update(predictions_by_family[family])
                ordered_common_keys = [
                    key
                    for key in predictions_by_family[fitted_families[0]]
                    if key in common_keys
                ]
                if len(ordered_common_keys) < MIN_TEST_ROWS:
                    reason = (
                        f"only {len(ordered_common_keys)} common fitted-family test rows; "
                        f"minimum is {MIN_TEST_ROWS}"
                    )
                    _skip(
                        results,
                        skipped,
                        horizon=horizon,
                        feature_set=feature_set,
                        family="equal_weight_blend",
                        reason=reason,
                        n_test=len(ordered_common_keys),
                    )
                else:
                    component_predictions = np.asarray(
                        [
                            [predictions_by_family[family][key] for key in ordered_common_keys]
                            for family in fitted_families
                        ],
                        dtype=float,
                    )
                    blend_prediction = np.mean(component_predictions, axis=0)
                    blend_actual = np.asarray(
                        [
                            actuals_by_family[fitted_families[0]][key]
                            for key in ordered_common_keys
                        ],
                        dtype=float,
                    )
                    for family in fitted_families[1:]:
                        family_actual = np.asarray(
                            [actuals_by_family[family][key] for key in ordered_common_keys],
                            dtype=float,
                        )
                        if not np.allclose(blend_actual, family_actual, rtol=0.0, atol=1e-12):
                            raise ValueError(
                                "fitted families have inconsistent labels on common test keys"
                            )
                    blend_metrics = _metrics(blend_prediction, blend_actual)
                    blend_metrics["component_families"] = fitted_families
                    results[(horizon, feature_set, "equal_weight_blend")] = blend_metrics
                    split_info["blend"] = {
                        "families": fitted_families,
                        "n_test": len(ordered_common_keys),
                        "test_period": (
                            f"{min(key[1] for key in ordered_common_keys).isoformat()} ~ "
                            f"{max(key[1] for key in ordered_common_keys).isoformat()}"
                        ),
                    }
            split_info["fit_predict_seconds"] = perf_counter() - combo_started

    metadata["wall_clock_seconds"] = perf_counter() - started
    return {"metrics": results, "metadata": metadata}
