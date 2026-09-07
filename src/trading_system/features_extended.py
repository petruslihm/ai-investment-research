"""Experiment-only 20-column point-in-time equity feature builder.

The production feature pipeline in :mod:`trading_system.features` remains the
seven-feature source of truth.  This sibling module only builds the parallel
experiment dataset, preserving the production feature and label calculations.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from trading_system import features, technical_features_pit
from trading_system.ids import AssetClass
from trading_system.market.calendar import add_sessions, is_trading_day
from trading_system.market.registry import stable_instrument_id
from trading_system.technical_factors import fetch_bars_frames


EXTENDED_FEATURE_NAMES: tuple[str, ...] = (
    features.FEATURE_NAMES + technical_features_pit.TECHNICAL_FEATURE_NAMES
)


def load_equity_ohlcv_frames(
    conn: duckdb.DuckDBPyConnection,
    instrument_ids: list[str],
) -> dict[str, pd.DataFrame]:
    """Load complete ascending OHLCV histories for all requested instruments once."""
    requested = list(dict.fromkeys(str(instrument_id) for instrument_id in instrument_ids))
    return fetch_bars_frames(conn, requested, max_sessions=sys.maxsize)


def _series_from_ohlcv(frame: pd.DataFrame) -> list[tuple[date, float, float]]:
    return [
        (features._as_date(row.session_date), float(row.close), float(row.volume))
        for row in frame.itertuples(index=False)
    ]


def extended_feature_columns(
    series: list[tuple[date, float, float]],
    ohlcv_df: pd.DataFrame,
    spy_close: pd.Series | None,
) -> dict[str, np.ndarray]:
    """Return the unchanged seven production columns plus 13 PIT technical columns."""
    series_dates = [features._as_date(row[0]) for row in series]
    frame_dates = [features._as_date(value) for value in ohlcv_df["session_date"]]
    if len(series) != len(ohlcv_df) or series_dates != frame_dates:
        raise ValueError(
            "series and ohlcv_df must have identical lengths and positionally aligned dates"
        )
    if frame_dates != sorted(frame_dates) or len(frame_dates) != len(set(frame_dates)):
        raise ValueError("series and ohlcv_df dates must be unique and ascending")

    base = features._feature_columns(series)
    technical = technical_features_pit.historical_technical_columns(ohlcv_df, spy_close)
    merged = {**base, **technical}
    if tuple(merged) != EXTENDED_FEATURE_NAMES:
        raise AssertionError("extended feature columns do not match EXTENDED_FEATURE_NAMES")
    return merged


def _database_path(conn: duckdb.DuckDBPyConnection) -> str | None:
    current_row = conn.execute("SELECT current_database()").fetchone()
    current = str(current_row[0]) if current_row and current_row[0] is not None else None
    for row in conn.execute("PRAGMA database_list").fetchall():
        if len(row) >= 3 and str(row[1]) == current and row[2]:
            return str(row[2])
    return None


def _assert_experiment_database(conn: duckdb.DuckDBPyConnection) -> None:
    path = _database_path(conn)
    if path is None or "ml_experiment" not in path.casefold():
        shown = path if path is not None else "<in-memory or unknown>"
        raise RuntimeError(
            "extended feature rebuild is restricted to a database path containing "
            f"'ml_experiment'; connected path is {shown}"
        )


def _feature_json_at(columns: dict[str, np.ndarray], index: int) -> str:
    return json.dumps(
        {name: float(columns[name][index]) for name in EXTENDED_FEATURE_NAMES}
    )


def _append_extended_equity_rows(
    *,
    feat_rows: list[tuple[object, ...]],
    label_rows: list[tuple[object, ...]],
    instrument_id: str,
    series: list[tuple[date, float, float]],
    ohlcv_df: pd.DataFrame,
    aligned_spy_close: pd.Series,
    spy_map: dict[date, int],
    spy_closes: list[float],
    last_available: date,
    horizons: tuple[int, ...],
) -> int:
    if len(series) < 25:
        return 0

    columns = extended_feature_columns(series, ohlcv_df, aligned_spy_close)
    index = features._idx(series)
    n_added = 0
    for i, (as_of, close_i, _volume) in enumerate(series):
        if i < 20 or not is_trading_day(as_of) or as_of not in spy_map:
            continue
        payload = _feature_json_at(columns, i)
        for horizon in horizons:
            # Keep this label/maturity math byte-for-byte equivalent in behavior to
            # features._append_equity_rows; only the feature JSON differs here.
            end = add_sessions(as_of, horizon)
            matured = end <= last_available and end in index and end in spy_map
            label = None
            if matured:
                s1 = series[index[end]][1]
                p0, p1 = spy_closes[spy_map[as_of]], spy_closes[spy_map[end]]
                if close_i and p0:
                    label = ((s1 / close_i) - 1.0) - ((p1 / p0) - 1.0)
            feat_rows.append(
                (
                    instrument_id,
                    as_of,
                    horizon,
                    AssetClass.US_EQUITY.value,
                    payload,
                    None,
                )
            )
            label_rows.append(
                (
                    instrument_id,
                    as_of,
                    horizon,
                    AssetClass.US_EQUITY.value,
                    label,
                    end,
                    as_of,
                    end,
                    features.TARGET_VERSION,
                    label is not None
                    and not (isinstance(label, float) and math.isnan(label)),
                )
            )
            n_added += 1
    return n_added


def build_and_persist_extended_features(
    conn: duckdb.DuckDBPyConnection,
    *,
    last_available: date,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> int:
    """Cleanly rebuild experiment-only US-equity rows with all 20 PIT features."""
    _assert_experiment_database(conn)

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            "DELETE FROM feature_rows WHERE asset_class = ?",
            [AssetClass.US_EQUITY.value],
        )
        conn.execute(
            "DELETE FROM label_rows WHERE asset_class = ?",
            [AssetClass.US_EQUITY.value],
        )

        spy_id = str(stable_instrument_id("SPY"))
        rows = conn.execute(
            """
            SELECT DISTINCT instrument_id
            FROM equity_daily_bars
            WHERE instrument_id <> ?
            ORDER BY instrument_id
            """,
            [spy_id],
        ).fetchall()
        equity_ids = [str(row[0]) for row in rows]
        frames = load_equity_ohlcv_frames(conn, [*equity_ids, spy_id])

        spy_frame = frames.get(spy_id)
        if spy_frame is None or len(spy_frame) < 25:
            conn.execute("COMMIT")
            return 0
        spy_series = _series_from_ohlcv(spy_frame)
        spy_map = features._idx(spy_series)
        spy_closes = [close for _session, close, _volume in spy_series]
        spy_by_date = (
            spy_frame.drop_duplicates("session_date", keep="last")
            .set_index("session_date")["close"]
        )

        n_added = 0
        feat_rows: list[tuple[object, ...]] = []
        label_rows: list[tuple[object, ...]] = []

        def flush() -> None:
            nonlocal feat_rows, label_rows
            features._bulk_upsert(conn, feat_rows, label_rows)
            feat_rows = []
            label_rows = []

        for instrument_id in equity_ids:
            frame = frames.get(instrument_id)
            if frame is None:
                continue
            series = _series_from_ohlcv(frame)
            aligned_spy = frame["session_date"].map(spy_by_date).reset_index(drop=True)
            n_added += _append_extended_equity_rows(
                feat_rows=feat_rows,
                label_rows=label_rows,
                instrument_id=instrument_id,
                series=series,
                ohlcv_df=frame,
                aligned_spy_close=aligned_spy,
                spy_map=spy_map,
                spy_closes=spy_closes,
                last_available=last_available,
                horizons=horizons,
            )
            if len(feat_rows) >= 40_000:
                flush()

        flush()
        conn.execute("COMMIT")
        return n_added
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def extended_load_matrix(
    conn: duckdb.DuckDBPyConnection,
    *,
    asset_class: str,
    horizon: int,
    matured_only: bool,
    as_of: date | None = None,
    min_as_of: date | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, date]]]:
    """Load the experiment matrix in the declared 20-feature order."""
    clauses = ["f.asset_class = ?", "f.horizon = ?"]
    params: list[object] = [asset_class, horizon]
    if as_of is not None:
        clauses.append("f.as_of_date = ?")
        params.append(as_of)
    elif min_as_of is not None:
        clauses.append("f.as_of_date >= ?")
        params.append(min_as_of)
    rows = conn.execute(
        f"""
        SELECT f.instrument_id, f.as_of_date, f.features_json, l.label, l.matured
        FROM feature_rows f
        JOIN label_rows l
          ON f.instrument_id = l.instrument_id
         AND f.as_of_date = l.as_of_date
         AND f.horizon = l.horizon
         AND f.asset_class = l.asset_class
        WHERE {" AND ".join(clauses)}
        ORDER BY f.as_of_date, f.instrument_id
        """,
        params,
    ).fetchall()
    xs: list[list[float]] = []
    ys: list[float] = []
    keys: list[tuple[str, date]] = []
    for inst, as_of, feat_json, label, matured in rows:
        if matured_only and (not matured or label is None):
            continue
        feat = json.loads(feat_json)
        xs.append([float(feat[n]) for n in EXTENDED_FEATURE_NAMES])
        ys.append(float(label) if label is not None else 0.0)
        keys.append((inst, as_of))
    if not xs:
        return np.zeros((0, len(EXTENDED_FEATURE_NAMES))), np.zeros((0,)), []
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), keys


def extended_load_sequences(
    conn: duckdb.DuckDBPyConnection,
    *,
    asset_class: str,
    horizon: int,
    lookback: int = features.LSTM_LOOKBACK,
    matured_only: bool = True,
    as_of: date | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, date]]]:
    """LSTM sequences of 20 PIT features ending at as_of (no future bars)."""
    min_as_of: date | None = None
    if as_of is not None:
        date_rows = conn.execute(
            """
            SELECT DISTINCT as_of_date FROM feature_rows
            WHERE asset_class = ? AND horizon = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC
            LIMIT ?
            """,
            [asset_class, horizon, as_of, lookback],
        ).fetchall()
        if not date_rows:
            return (
                np.zeros((0, lookback, len(EXTENDED_FEATURE_NAMES))),
                np.zeros((0,)),
                [],
            )
        min_as_of = min(r[0] for r in date_rows)
    x, y, keys = extended_load_matrix(
        conn,
        asset_class=asset_class,
        horizon=horizon,
        matured_only=False,
        min_as_of=min_as_of,
    )
    if len(keys) == 0:
        return (
            np.zeros((0, lookback, len(EXTENDED_FEATURE_NAMES))),
            np.zeros((0,)),
            [],
        )
    by_inst: dict[str, list[int]] = {}
    for i, (inst, _d) in enumerate(keys):
        by_inst.setdefault(inst, []).append(i)
    seqs: list[np.ndarray] = []
    ys: list[float] = []
    out_keys: list[tuple[str, date]] = []
    for inst, idxs in by_inst.items():
        idxs = sorted(idxs, key=lambda i: keys[i][1])
        for pos in range(lookback - 1, len(idxs)):
            window = idxs[pos - lookback + 1 : pos + 1]
            i = idxs[pos]
            if as_of is not None and keys[i][1] != as_of:
                continue
            if matured_only:
                row = conn.execute(
                    """
                    SELECT matured, label FROM label_rows
                    WHERE instrument_id=? AND as_of_date=? AND horizon=? AND asset_class=?
                    """,
                    [inst, keys[i][1], horizon, asset_class],
                ).fetchone()
                if not row or not row[0] or row[1] is None:
                    continue
                label = float(row[1])
            else:
                label = float(y[i])
            seqs.append(x[window])
            ys.append(label)
            out_keys.append(keys[i])
    if not seqs:
        return (
            np.zeros((0, lookback, len(EXTENDED_FEATURE_NAMES))),
            np.zeros((0,)),
            [],
        )
    # Global chronological order across instruments (not instrument-blocked).
    order = sorted(range(len(out_keys)), key=lambda i: (out_keys[i][1], out_keys[i][0]))
    seqs = [seqs[i] for i in order]
    ys = [ys[i] for i in order]
    out_keys = [out_keys[i] for i in order]
    return np.stack(seqs), np.asarray(ys, dtype=float), out_keys
