"""Point-in-time features and multi-horizon targets (equity sessions vs BTC calendar)."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta

import duckdb
import numpy as np
import pandas as pd

from trading_system.ids import AssetClass
from trading_system.market.calendar import add_sessions, is_trading_day
from trading_system.market.registry import stable_instrument_id
from trading_system.market.decision_data import daily_input_fingerprints

FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_10",
    "vol_10",
    "hl_range_5",
    "volume_z",
    "mom_20",
)
TARGET_VERSION = "v1_excess_or_usd"
FEATURE_VERSION = "v1_pit_bars"
LSTM_LOOKBACK = 10


def _closes(conn: duckdb.DuckDBPyConnection, table: str, instrument_id: str | None) -> list[tuple[date, float, float]]:
    if table == "equity_daily_bars":
        rows = conn.execute(
            """
            SELECT session_date, close, volume
            FROM equity_daily_bars
            WHERE instrument_id = ? AND finality = 'final'
            ORDER BY session_date
            """,
            [instrument_id],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT session_date, close, volume FROM btc_daily_bars WHERE finality = 'final' ORDER BY session_date"
        ).fetchall()
    return [(r[0], float(r[1]), float(r[2])) for r in rows]


def _idx(series: list[tuple[date, float, float]]) -> dict[date, int]:
    return {d: i for i, (d, _c, _v) in enumerate(series)}


def _ret(closes: list[float], i: int, lookback: int) -> float:
    if i - lookback < 0:
        return 0.0
    a, b = closes[i - lookback], closes[i]
    if a == 0:
        return 0.0
    return (b / a) - 1.0


def _vol(closes: list[float], i: int, window: int) -> float:
    start = max(1, i - window + 1)
    rets = []
    for j in range(start, i + 1):
        if closes[j - 1] == 0:
            continue
        rets.append((closes[j] / closes[j - 1]) - 1.0)
    if len(rets) < 2:
        return 0.0
    return float(np.std(rets, ddof=1))


def _features_at(series: list[tuple[date, float, float]], i: int) -> dict[str, float]:
    closes = [c for _d, c, _v in series]
    vols = [v for _d, _c, v in series]
    hi = max(closes[max(0, i - 4) : i + 1])
    lo = min(closes[max(0, i - 4) : i + 1])
    mean_v = float(np.mean(vols[max(0, i - 9) : i + 1])) if vols else 0.0
    vz = 0.0 if mean_v == 0 else (vols[i] - mean_v) / (mean_v + 1e-9)
    return {
        "ret_1": _ret(closes, i, 1),
        "ret_5": _ret(closes, i, 5),
        "ret_10": _ret(closes, i, 10),
        "vol_10": _vol(closes, i, 10),
        "hl_range_5": 0.0 if lo == 0 else (hi - lo) / lo,
        "volume_z": vz,
        "mom_20": _ret(closes, i, 20),
    }


def _next_calendar(d: date, n: int) -> date:
    return d + timedelta(days=n)


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _feature_columns(series: list[tuple[date, float, float]]) -> dict[str, np.ndarray]:
    """Vectorized equivalent of ``_features_at`` for every index."""
    closes = np.asarray([c for _d, c, _v in series], dtype=float)
    vols = np.asarray([v for _d, _c, v in series], dtype=float)
    n = len(closes)
    daily = np.full(n, np.nan)
    if n > 1:
        prev = closes[:-1]
        daily[1:] = np.where(prev == 0, np.nan, closes[1:] / prev - 1.0)
    vol_10 = pd.Series(daily).rolling(10, min_periods=2).std(ddof=1).to_numpy()
    vol_10 = np.nan_to_num(vol_10, nan=0.0)
    hi = pd.Series(closes).rolling(5, min_periods=1).max().to_numpy()
    lo = pd.Series(closes).rolling(5, min_periods=1).min().to_numpy()
    hl = np.where(lo == 0, 0.0, (hi - lo) / lo)
    mean_v = pd.Series(vols).rolling(10, min_periods=1).mean().to_numpy()
    vz = np.where(mean_v == 0, 0.0, (vols - mean_v) / (mean_v + 1e-9))

    def shifted_ret(lookback: int) -> np.ndarray:
        out = np.zeros(n)
        if lookback <= 0 or n <= lookback:
            return out
        base = closes[:-lookback]
        out[lookback:] = np.where(base == 0, 0.0, closes[lookback:] / base - 1.0)
        return out

    return {
        "ret_1": shifted_ret(1),
        "ret_5": shifted_ret(5),
        "ret_10": shifted_ret(10),
        "vol_10": vol_10,
        "hl_range_5": hl,
        "volume_z": vz,
        "mom_20": shifted_ret(20),
    }


def _feat_json_at(cols: dict[str, np.ndarray], i: int) -> str:
    return json.dumps({name: float(cols[name][i]) for name in FEATURE_NAMES})


def _covered_instruments(
    conn: duckdb.DuckDBPyConnection,
    *,
    asset_class: str,
    last_available: date,
    n_horizons: int,
) -> set[str]:
    rows = conn.execute(
        """
        SELECT instrument_id, max(as_of_date), count(DISTINCT horizon)
        FROM feature_rows
        WHERE asset_class = ?
        GROUP BY instrument_id
        """,
        [asset_class],
    ).fetchall()
    out: set[str] = set()
    for inst, mx, nh in rows:
        if mx is None:
            continue
        if _as_date(mx) >= last_available and int(nh or 0) >= n_horizons:
            out.add(str(inst))
    return out


def _bulk_upsert(
    conn: duckdb.DuckDBPyConnection,
    feat_rows: list[tuple[object, ...]],
    label_rows: list[tuple[object, ...]],
) -> None:
    if not feat_rows:
        return
    feat_df = pd.DataFrame(
        feat_rows,
        columns=[
            "instrument_id",
            "as_of_date",
            "horizon",
            "asset_class",
            "features_json",
            "live_overlay_json",
        ],
    )
    lab_df = pd.DataFrame(
        label_rows,
        columns=[
            "instrument_id",
            "as_of_date",
            "horizon",
            "asset_class",
            "label",
            "label_available_date",
            "target_start",
            "target_end",
            "target_version",
            "matured",
        ],
    )
    conn.register("_feat_in", feat_df)
    conn.register("_lab_in", lab_df)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO feature_rows
            SELECT instrument_id, as_of_date, horizon, asset_class, features_json, live_overlay_json
            FROM _feat_in
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO label_rows
            SELECT instrument_id, as_of_date, horizon, asset_class, label, label_available_date,
                   target_start, target_end, target_version, matured
            FROM _lab_in
            """
        )
    finally:
        conn.unregister("_feat_in")
        conn.unregister("_lab_in")


def _append_equity_rows(
    *,
    feat_rows: list[tuple[object, ...]],
    label_rows: list[tuple[object, ...]],
    inst_id: str,
    series: list[tuple[date, float, float]],
    spy_map: dict[date, int],
    spy_closes: list[float],
    last_available: date,
    horizons: tuple[int, ...],
) -> int:
    if len(series) < 25:
        return 0
    cols = _feature_columns(series)
    index = _idx(series)
    n_added = 0
    for i, (as_of, close_i, _v) in enumerate(series):
        if i < 20 or as_of > last_available or not is_trading_day(as_of) or as_of not in spy_map:
            continue
        payload = _feat_json_at(cols, i)
        for h in horizons:
            end = add_sessions(as_of, h)
            matured = end <= last_available and end in index and end in spy_map
            label = None
            if matured:
                s1 = series[index[end]][1]
                p0, p1 = spy_closes[spy_map[as_of]], spy_closes[spy_map[end]]
                if close_i and p0:
                    label = ((s1 / close_i) - 1.0) - ((p1 / p0) - 1.0)
            feat_rows.append((inst_id, as_of, h, AssetClass.US_EQUITY.value, payload, None))
            label_rows.append(
                (
                    inst_id,
                    as_of,
                    h,
                    AssetClass.US_EQUITY.value,
                    label,
                    end,
                    as_of,
                    end,
                    TARGET_VERSION,
                    label is not None and not (isinstance(label, float) and math.isnan(label)),
                )
            )
            n_added += 1
    return n_added


def _append_btc_rows(
    *,
    feat_rows: list[tuple[object, ...]],
    label_rows: list[tuple[object, ...]],
    series: list[tuple[date, float, float]],
    btc_last: date,
    horizons: tuple[int, ...],
) -> int:
    if len(series) < 21:
        return 0
    inst_id = str(stable_instrument_id("BTC/USD"))
    cols = _feature_columns(series)
    ordinals = np.asarray([d.toordinal() for d, _c, _v in series], dtype=np.int64)
    n_added = 0
    for i, (as_of, close_i, _v) in enumerate(series):
        if i < 20 or as_of > btc_last:
            continue
        payload = _feat_json_at(cols, i)
        for h in horizons:
            end = _next_calendar(as_of, h)
            matured = end <= btc_last
            label = None
            if matured and close_i:
                end_i = int(np.searchsorted(ordinals, end.toordinal(), side="right") - 1)
                if end_i > i:
                    label = (series[end_i][1] / close_i) - 1.0
            feat_rows.append((inst_id, as_of, h, AssetClass.BTC.value, payload, None))
            label_rows.append(
                (
                    inst_id,
                    as_of,
                    h,
                    AssetClass.BTC.value,
                    label,
                    end,
                    as_of,
                    end,
                    TARGET_VERSION,
                    label is not None and not (isinstance(label, float) and math.isnan(label)),
                )
            )
            n_added += 1
    return n_added


def _load_equity_series(
    conn: duckdb.DuckDBPyConnection,
    instrument_ids: list[str],
) -> dict[str, list[tuple[date, float, float]]]:
    if not instrument_ids:
        return {}
    placeholders = ", ".join(["?"] * len(instrument_ids))
    rows = conn.execute(
        f"""
        SELECT instrument_id, session_date, close, volume
        FROM equity_daily_bars
        WHERE instrument_id IN ({placeholders}) AND finality = 'final'
        ORDER BY instrument_id, session_date
        """,
        instrument_ids,
    ).fetchall()
    out: dict[str, list[tuple[date, float, float]]] = {}
    for inst, session, close, volume in rows:
        out.setdefault(str(inst), []).append((_as_date(session), float(close), float(volume)))
    return out


def features_cover_latest(
    conn: duckdb.DuckDBPyConnection,
    *,
    last_available: date,
    last_available_btc: date | None,
) -> bool:
    """Coverage AND content identity; same-session corrections must rebuild."""
    fingerprints = daily_input_fingerprints(
        conn, equity_end=last_available, btc_end=last_available_btc or last_available
    )
    for asset, digest in fingerprints.items():
        stored = conn.execute("SELECT value FROM schema_meta WHERE key=?", [f"daily_feature_input:{asset}"]).fetchone()
        if not stored or stored[0] != digest:
            return False
    eq_latest, eq_names = conn.execute(
        """
        SELECT max(as_of_date), count(DISTINCT instrument_id)
        FROM feature_rows WHERE asset_class = 'us_equity'
        """
    ).fetchone() or (None, 0)
    spy_id = str(stable_instrument_id("SPY"))
    bar_names = conn.execute(
        "SELECT count(DISTINCT instrument_id) FROM equity_daily_bars WHERE instrument_id <> ? AND finality='final'",
        [spy_id],
    ).fetchone()
    n_bars = int(bar_names[0] or 0) if bar_names else 0
    n_feat = int(eq_names or 0)
    if eq_latest is None or n_feat == 0:
        return False
    if n_bars and n_feat < max(1, int(n_bars * 0.8)):
        return False
    if eq_latest < last_available:
        return False
    if last_available_btc is None:
        return True
    btc = conn.execute(
        "SELECT max(as_of_date) FROM feature_rows WHERE asset_class = 'btc'"
    ).fetchone()
    return bool(btc and btc[0] is not None and btc[0] >= last_available_btc)


def build_and_persist_features(
    conn: duckdb.DuckDBPyConnection,
    *,
    last_available: date,
    last_available_btc: date | None = None,
    horizons: tuple[int, ...] = (5, 10, 20),
    progress: object | None = None,
) -> int:
    """Write PIT feature/label rows. Labels after last_available are not matured.

    Equity maturity uses ``last_available`` (equity sessions). BTC maturity uses
    ``last_available_btc`` (BTC's own last bar), never the equity watermark.
    """
    btc_last = last_available_btc
    if btc_last is None:
        brow = conn.execute("SELECT max(session_date) FROM btc_daily_bars WHERE finality='final'").fetchone()
        btc_last = brow[0] if brow and brow[0] else last_available
    spy_id = stable_instrument_id("SPY")
    fingerprints = daily_input_fingerprints(conn, equity_end=last_available, btc_end=btc_last)
    changed = set()
    for asset, digest in fingerprints.items():
        stored = conn.execute("SELECT value FROM schema_meta WHERE key=?", [f"daily_feature_input:{asset}"]).fetchone()
        if not stored or stored[0] != digest:
            changed.add(asset)
    # Drop only mutable derived rows past the completed-data boundary. Historical
    # decision inputs are frozen separately in tick_observations, never rewritten.
    for asset, end in [("us_equity", last_available), ("btc", btc_last)]:
        if asset in changed:
            conn.execute("DELETE FROM feature_rows WHERE asset_class=? AND as_of_date>?", [asset, end])
            conn.execute("DELETE FROM label_rows WHERE asset_class=? AND as_of_date>?", [asset, end])
    spy = _closes(conn, "equity_daily_bars", spy_id)
    if len(spy) < 25:
        return 0
    spy_map = _idx(spy)
    spy_closes = [c for _d, c, _v in spy]

    equities = conn.execute(
        """
        SELECT DISTINCT instrument_id FROM equity_daily_bars
        WHERE instrument_id <> ?
        """,
        [spy_id],
    ).fetchall()

    equity_ids = [str(row[0]) for row in equities]
    covered_eq = _covered_instruments(
        conn,
        asset_class=AssetClass.US_EQUITY.value,
        last_available=last_available,
        n_horizons=len(horizons),
    )
    if "us_equity" in changed:
        covered_eq = set()
    todo = [inst_id for inst_id in equity_ids if inst_id not in covered_eq]
    btc_id = str(stable_instrument_id("BTC/USD"))
    covered_btc = btc_id in _covered_instruments(
        conn,
        asset_class=AssetClass.BTC.value,
        last_available=btc_last,
        n_horizons=len(horizons),
    )
    if "btc" in changed:
        covered_btc = False

    n = 0
    total = len(equity_ids)
    feat_rows: list[tuple[object, ...]] = []
    label_rows: list[tuple[object, ...]] = []
    series_by_id = _load_equity_series(conn, todo)

    def _flush() -> None:
        nonlocal feat_rows, label_rows
        _bulk_upsert(conn, feat_rows, label_rows)
        feat_rows = []
        label_rows = []

    for i_eq, inst_id in enumerate(equity_ids, start=1):
        if callable(progress) and (i_eq == 1 or i_eq == total or i_eq % 10 == 0):
            progress(i_eq, total)
        if inst_id in covered_eq:
            continue
        n += _append_equity_rows(
            feat_rows=feat_rows,
            label_rows=label_rows,
            inst_id=inst_id,
            series=series_by_id.get(inst_id, []),
            spy_map=spy_map,
            spy_closes=spy_closes,
            last_available=last_available,
            horizons=horizons,
        )
        if len(feat_rows) >= 40_000:
            _flush()

    if not covered_btc:
        n += _append_btc_rows(
            feat_rows=feat_rows,
            label_rows=label_rows,
            series=_closes(conn, "btc_daily_bars", None),
            btc_last=btc_last,
            horizons=horizons,
        )
    _flush()
    for asset, digest in fingerprints.items():
        conn.execute("INSERT OR REPLACE INTO schema_meta (key,value) VALUES (?,?)", [f"daily_feature_input:{asset}", digest])
    if callable(progress) and total:
        progress(total, total)
    return n


def load_matrix(
    conn: duckdb.DuckDBPyConnection,
    *,
    asset_class: str,
    horizon: int,
    matured_only: bool,
    as_of: date | None = None,
    min_as_of: date | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, date]]]:
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
        xs.append([float(feat[n]) for n in FEATURE_NAMES])
        ys.append(float(label) if label is not None else 0.0)
        keys.append((inst, as_of))
    if not xs:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0,)), []
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), keys


def load_sequences(
    conn: duckdb.DuckDBPyConnection,
    *,
    asset_class: str,
    horizon: int,
    lookback: int = LSTM_LOOKBACK,
    matured_only: bool = True,
    as_of: date | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, date]]]:
    """LSTM sequences of PIT features ending at as_of (no future bars)."""
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
            return np.zeros((0, lookback, len(FEATURE_NAMES))), np.zeros((0,)), []
        min_as_of = min(r[0] for r in date_rows)
    x, y, keys = load_matrix(
        conn,
        asset_class=asset_class,
        horizon=horizon,
        matured_only=False,
        min_as_of=min_as_of,
    )
    if len(keys) == 0:
        return np.zeros((0, lookback, len(FEATURE_NAMES))), np.zeros((0,)), []
    by_inst: dict[str, list[int]] = {}
    for i, (inst, _d) in enumerate(keys):
        by_inst.setdefault(inst, []).append(i)
    # One batch query for the whole (asset_class, horizon) slice instead of a
    # per-(instrument, date) query inside the loop below. At current data volumes
    # (~500 equity names x ~750 matured sessions x 3 horizons) the old per-row
    # conn.execute() issued 1M+ individual round trips and made a single train_only
    # run (see v1_cycle.py) take 30-60+ minutes instead of ~30 seconds -- growing
    # worse every day as more sessions mature, which defeats the point of a cheap
    # daily retrain (see due_daily_train in daily_scan.py).
    matured_labels: dict[tuple[str, date], float] | None = None
    if matured_only:
        matured_labels = {
            (str(inst), d): float(label)
            for inst, d, label in conn.execute(
                """
                SELECT instrument_id, as_of_date, label FROM label_rows
                WHERE asset_class = ? AND horizon = ? AND matured AND label IS NOT NULL
                """,
                [asset_class, horizon],
            ).fetchall()
        }
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
                assert matured_labels is not None
                label_val = matured_labels.get((inst, keys[i][1]))
                if label_val is None:
                    continue
                label = label_val
            else:
                label = float(y[i])
            seqs.append(x[window])
            ys.append(label)
            out_keys.append(keys[i])
    if not seqs:
        return np.zeros((0, lookback, len(FEATURE_NAMES))), np.zeros((0,)), []
    # Global chronological order across instruments (not instrument-blocked).
    order = sorted(range(len(out_keys)), key=lambda i: (out_keys[i][1], out_keys[i][0]))
    seqs = [seqs[i] for i in order]
    ys = [ys[i] for i in order]
    out_keys = [out_keys[i] for i in order]
    return np.stack(seqs), np.asarray(ys, dtype=float), out_keys


def assert_no_future_in_features(conn: duckdb.DuckDBPyConnection) -> None:
    """Features at as_of must not depend on later bars (structural: lookbacks only)."""
    bad = conn.execute(
        """
        SELECT COUNT(*) FROM label_rows
        WHERE matured AND label_available_date < as_of_date
        """
    ).fetchone()[0]
    if bad:
        raise AssertionError("label_available_date precedes as_of_date")


def live_overlay(ret_intraday: float | None, hist: dict[str, float]) -> dict[str, float]:
    """Live overlay is separate from historical training features."""
    out = dict(hist)
    out["live_ret_intraday"] = 0.0 if ret_intraday is None else float(ret_intraday)
    return out
