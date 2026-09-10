"""Horizon-aware batch models, online SGD, and adaptive ensemble."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import duckdb
import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler

from trading_system.features import FEATURE_NAMES, LSTM_LOOKBACK, load_matrix, load_sequences
from trading_system.ids import AssetClass
from trading_system.models import (
    DecisionEpochManifest,
    ModelArtifactRef,
    ModelChangeJournalEvent,
    ModelChangeKind,
    ModelFamily,
)

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


HORIZONS = (5, 10, 20)
# 2026-09-05: adapt_ensemble_from_matured used to evaluate family/horizon skill over
# the *entire* chronological_split test window, which at current data volumes spans
# roughly 10 months (2025-10-20 ~ 2026-08-27 on a real scan) -- meaning a family that
# started under/over-performing recently barely moved the ensemble at all, since one
# bad quarter was averaged against three good ones. This restricts weight-adaptation
# scoring to the most recent N unique as_of dates within that test window. It does
# NOT touch how train_batch_models fits the actual Ridge/LightGBM/LSTM models --
# those still train on the full history_years window; only the "which family/horizon
# do we currently trust more" scoring looks at recent data. ~63 trading days = ~1
# quarter, chosen over a shorter window because the 20d horizon's forward-return
# labels overlap 19 days at a time -- a much shorter window would not even contain
# enough non-overlapping 20d observations to be a stable estimate.
RECENT_EVAL_WINDOW_DAYS = 63
RETURN_FAMILIES = ("ridge", "lightgbm_reg", "torch_sequence", "online")


def family_enum(name: str) -> ModelFamily:
    return {
        "ridge": ModelFamily.RIDGE,
        "lightgbm_reg": ModelFamily.LIGHTGBM_REG,
        "lambdarank": ModelFamily.LAMBDARANK,
        "torch_sequence": ModelFamily.TORCH_SEQUENCE,
        "online": ModelFamily.ONLINE_SGD,
    }.get(name, ModelFamily.RIDGE)


@dataclass
class FittedBundle:
    family: str
    horizon: int
    asset_class: str
    scaler: StandardScaler
    model: object
    version: str
    lookback: int | None = None


class TinyLSTM(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, n_in: int = 7, hidden: int = 16) -> None:
        if nn is None:
            return
        super().__init__()
        self.rnn = nn.LSTM(n_in, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # noqa: ANN001
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _hash_arr(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def bundle_hash(bundle: FittedBundle) -> str:
    h = hashlib.sha256()
    h.update(f"{bundle.family}:{bundle.horizon}:{bundle.asset_class}:{bundle.version}".encode())
    model = bundle.model
    try:
        if isinstance(model, Ridge):
            h.update(np.asarray(model.coef_).tobytes())
            h.update(np.asarray(model.intercept_).tobytes())
        elif isinstance(model, tuple) and model[0] == "ridge_fallback":
            h.update(b"ridge_fallback")
            h.update(np.asarray(model[2].coef_).tobytes())
        elif torch is not None and isinstance(model, TinyLSTM):
            for p in model.parameters():
                h.update(p.detach().cpu().numpy().tobytes())
        elif lgb is not None and isinstance(model, lgb.Booster):
            h.update(model.model_to_string().encode())
        elif hasattr(model, "booster_"):
            h.update(str(model.booster_.num_trees()).encode())
        else:
            h.update(type(model).__name__.encode())
    except Exception:
        h.update(b"unhashable")
    return h.hexdigest()[:16]


def _fit_ridge(x: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    model = Ridge(alpha=1.0)
    model.fit(xs, y)
    return scaler, model


def _rank_labels(y: np.ndarray, keys: list[tuple[str, date]], idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Integer relevance within each prediction date (higher excess return = higher rank)."""
    rows = [(keys[int(i)][1], keys[int(i)][0], int(i), float(y[int(i)])) for i in idx]
    rows.sort(key=lambda r: (r[0], r[1]))
    xs_idx: list[int] = []
    rel: list[int] = []
    groups: list[int] = []
    i = 0
    while i < len(rows):
        d = rows[i][0]
        chunk = []
        while i < len(rows) and rows[i][0] == d:
            chunk.append(rows[i])
            i += 1
        order = sorted(range(len(chunk)), key=lambda j: chunk[j][3])
        ranks = [0] * len(chunk)
        for rank, j in enumerate(order):
            ranks[j] = rank
        for j, row in enumerate(chunk):
            xs_idx.append(row[2])
            rel.append(ranks[j])
        groups.append(len(chunk))
    return np.asarray(xs_idx, dtype=int), np.asarray(rel, dtype=int), groups


def _fit_lgbm(x: np.ndarray, y: np.ndarray, *, rank: bool, groups: list[int] | None) -> object:
    if lgb is None or len(y) < 20:
        return ("ridge_fallback", *_fit_ridge(x, y))
    try:
        if rank and groups and min(groups) >= 1 and len(groups) >= 2 and np.issubdtype(y.dtype, np.integer):
            dtrain = lgb.Dataset(x, label=y, group=groups)
            return lgb.train(
                {
                    "objective": "lambdarank",
                    "metric": "ndcg",
                    "verbosity": -1,
                    "min_data_in_leaf": 1,
                    "label_gain": list(range(int(y.max()) + 1)),
                },
                dtrain,
                num_boost_round=40,
            )
        model = lgb.LGBMRegressor(n_estimators=80, max_depth=4, verbosity=-1)
        model.fit(x, y)
        return model
    except Exception:
        return ("ridge_fallback", *_fit_ridge(x, y))


def _fit_lstm(seq: np.ndarray, y: np.ndarray) -> object:
    if torch is None or len(y) < 16 or seq.ndim != 3:
        flat = seq.reshape(len(seq), -1) if len(seq) else np.zeros((0, 1))
        return ("ridge_fallback", *_fit_ridge(flat if flat.size else np.zeros((0, 7)), y))
    torch.manual_seed(42)
    net = TinyLSTM(seq.shape[-1], hidden=16)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    xt = torch.tensor(seq, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    net.train()
    for _ in range(25):
        opt.zero_grad()
        pred = net(xt)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()
    net.eval()
    return net


def _predict(bundle: FittedBundle, x: np.ndarray) -> np.ndarray:
    model = bundle.model
    if len(x) == 0:
        return np.zeros((0,))
    if isinstance(model, tuple) and model[0] == "ridge_fallback":
        _, scaler, ridge = model
        flat = x.reshape(len(x), -1) if x.ndim == 3 else x
        return ridge.predict(scaler.transform(flat))
    if isinstance(model, Ridge):
        flat = x.reshape(len(x), -1) if x.ndim == 3 else x
        return model.predict(bundle.scaler.transform(flat))
    if lgb is not None and (isinstance(model, lgb.Booster) or hasattr(model, "predict")):
        flat = x.reshape(len(x), -1) if x.ndim == 3 else x
        return np.asarray(model.predict(flat), dtype=float)
    if torch is not None and isinstance(model, TinyLSTM):
        # D10: refuse 1-step row inference; LSTM needs the trained lookback window.
        if x.ndim != 3:
            return np.zeros(len(x))
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32)
            return model(xt).detach().cpu().numpy()
    return np.zeros(len(x))


def chronological_split(
    n: int,
    *,
    purge: int,
    keys: list[tuple[str, date]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """No random split. Purge is unique session/date count, not flattened row count."""
    if keys is not None and len(keys) == n:
        dates = sorted({d for _inst, d in keys})
        if len(dates) < 10:
            idx = np.arange(n)
            return idx, idx
        cut = max(1, int(len(dates) * 0.7))
        train_end = dates[cut - 1]
        test_dates = set(dates[cut + max(0, purge) :])
        train = np.array([i for i, k in enumerate(keys) if k[1] <= train_end], dtype=int)
        test = np.array([i for i, k in enumerate(keys) if k[1] in test_dates], dtype=int)
        train_set = set(train.tolist())
        test = np.array([i for i in test if i not in train_set], dtype=int)
        if len(train) == 0:
            train = np.arange(min(4, n))
        if len(test) == 0:
            last = dates[-1]
            test = np.array([i for i, k in enumerate(keys) if k[1] == last and i not in set(train.tolist())], dtype=int)
            if len(test) == 0:
                test = train[-min(3, len(train)) :]
        return train, test
    if n < 10:
        idx = np.arange(n)
        return idx, idx
    cut = max(4, int(n * 0.7))
    train = np.arange(0, cut)
    test_start = min(n, cut + purge)
    test = np.arange(test_start, n)
    if len(test) == 0:
        test = np.arange(max(0, n - 3), n)
        train = np.array([i for i in train if i not in set(test.tolist())], dtype=int)
        if len(train) == 0:
            train = np.arange(0, min(cut, n))
    return train, test


def _training_window(keys: list[tuple[str, date]], idx: np.ndarray) -> str | None:
    """Actual first/last label date used for training. No invented ranges."""
    dates = [keys[i][1] for i in idx.tolist() if 0 <= i < len(keys)]
    if not dates:
        return None
    return f"{min(dates).isoformat()} ~ {max(dates).isoformat()}"


def last_promoted(
    conn: duckdb.DuckDBPyConnection,
    *,
    asset: str,
    horizon: int,
    family: str,
) -> tuple[str | None, float | None]:
    """(version, metric) of the previous recorded promotion, or (None, None) if first ever."""
    try:
        row = conn.execute(
            """
            SELECT json_extract_string(payload_json, '$.new_version'),
                   json_extract(payload_json, '$.metric_after')
            FROM model_change_journal
            WHERE kind = 'promoted'
              AND json_extract_string(payload_json, '$.asset_class') = ?
              AND json_extract_string(payload_json, '$.model_family') = ?
              AND CAST(json_extract(payload_json, '$.horizon') AS INTEGER) = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [asset, family_enum(family).value, horizon],
        ).fetchone()
    except Exception:  # noqa: BLE001 — journal history is diagnostic only
        return None, None
    if not row:
        return None, None
    metric: float | None
    try:
        metric = None if row[1] is None else float(row[1])
    except (TypeError, ValueError):
        metric = None
    return (row[0] or None), metric


def _holdout_mae(
    bundle: FittedBundle,
    x: np.ndarray,
    y: np.ndarray,
    keys: list[tuple[str, date]],
    train: np.ndarray,
    test: np.ndarray,
    *,
    purge: int,
) -> tuple[float | None, str, int]:
    """Report only this estimator's purged chronological holdout error.

    Training keeps its historical small-sample fallbacks. Those fallbacks must
    not become out-of-sample evidence merely because they return test indices.
    LambdaRank outputs relevance scores, not returns, so return MAE is invalid.
    """
    if bundle.family == "lambdarank":
        return None, "RANKING_METRIC_NOT_EVALUATED", 0
    if not len(train) or not len(test) or len(keys) != len(y):
        return None, "NO_VALID_HOLDOUT", 0
    if set(train.tolist()) & set(test.tolist()):
        return None, "TRAIN_TEST_OVERLAP", 0
    dates = sorted({d for _, d in keys})
    last_train = max(keys[int(i)][1] for i in train)
    first_test = min(keys[int(i)][1] for i in test)
    if dates.index(first_test) - dates.index(last_train) <= purge:
        return None, "INSUFFICIENT_PURGE", 0
    prediction = np.asarray(_predict(bundle, x[test]), dtype=float).reshape(-1)
    actual = np.asarray(y[test], dtype=float).reshape(-1)
    if (
        prediction.shape != actual.shape
        or not np.isfinite(prediction).all()
        or not np.isfinite(actual).all()
    ):
        return None, "INVALID_EVALUATION_VALUES", 0
    return float(np.mean(np.abs(prediction - actual))), "CHRONOLOGICAL_HOLDOUT", len(test)


def train_batch_models(
    conn: duckdb.DuckDBPyConnection,
    artifacts_dir: Path,
) -> dict[tuple[str, int, str], FittedBundle]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out: dict[tuple[str, int, str], FittedBundle] = {}
    journal: list[ModelChangeJournalEvent] = []
    for asset in (AssetClass.US_EQUITY.value, AssetClass.BTC.value):
        for h in HORIZONS:
            x, y, keys = load_matrix(conn, asset_class=asset, horizon=h, matured_only=True)
            if len(y) < 8:
                continue
            tr, te = chronological_split(len(y), purge=h, keys=keys)
            xtr, ytr = x[tr], y[tr]
            scaler, ridge = _fit_ridge(xtr, ytr)
            ridge_b = FittedBundle("ridge", h, asset, scaler, ridge, "1")
            lgbm = _fit_lgbm(xtr, ytr, rank=False, groups=None)
            lgbm_b = FittedBundle("lightgbm_reg", h, asset, scaler, lgbm, "1")
            seq, seq_y, seq_keys = load_sequences(
                conn, asset_class=asset, horizon=h, lookback=LSTM_LOOKBACK, matured_only=True
            )
            if len(seq_y) >= 8:
                str_, ste = chronological_split(len(seq_y), purge=h, keys=seq_keys)
                lstm = _fit_lstm(seq[str_], seq_y[str_])
                lstm_lookback = LSTM_LOOKBACK
                lstm_eval = (seq, seq_y, seq_keys, str_, ste)
            else:
                lstm = _fit_lstm(xtr[:, None, :], ytr)
                lstm_lookback = 1
                lstm_eval = (x[:, None, :], y, keys, tr, te)
            lstm_b = FittedBundle("torch_sequence", h, asset, scaler, lstm, "1", lookback=lstm_lookback)
            out[(asset, h, "ridge")] = ridge_b
            out[(asset, h, "lightgbm_reg")] = lgbm_b
            out[(asset, h, "torch_sequence")] = lstm_b
            if asset == AssetClass.US_EQUITY.value:
                order, rel, groups = _rank_labels(y, keys, tr)
                if len(groups) >= 2:
                    ranker = _fit_lgbm(x[order], rel, rank=True, groups=groups)
                else:
                    ranker = _fit_lgbm(xtr, ytr, rank=False, groups=None)
                out[(asset, h, "lambdarank")] = FittedBundle("lambdarank", h, asset, scaler, ranker, "1")
            families_here = ["ridge", "lightgbm_reg", "torch_sequence"]
            if asset == AssetClass.US_EQUITY.value:
                families_here.append("lambdarank")
            for fam in families_here:
                b = out[(asset, h, fam)]
                ex, ey, ekeys, etrain, etest = lstm_eval if fam == "torch_sequence" else (x, y, keys, tr, te)
                mae, evaluation_status, eval_count = _holdout_mae(b, ex, ey, ekeys, etrain, etest, purge=h)
                train_window = _training_window(ekeys, etrain)
                eval_window = _training_window(ekeys, etest) if mae is not None else None
                prev_version, _ = last_promoted(conn, asset=asset, horizon=h, family=fam)
                estimator_kind = (
                    "ridge_fallback"
                    if isinstance(b.model, tuple) and b.model[0] == "ridge_fallback"
                    else fam
                )
                journal.append(
                    ModelChangeJournalEvent(
                        event_id=f"mcj_{uuid4().hex[:12]}",
                        kind=ModelChangeKind.PROMOTED,
                        model_family=family_enum(fam),
                        asset_class=asset,
                        horizon=h,
                        previous_version=prev_version,
                        new_version=b.version,
                        reason_codes=["batch_fit_applied", evaluation_status],
                        matured_label_count=int(len(ey)),
                        sample_count=int(len(etrain)),
                        metric_name="chronological_holdout_mae" if mae is not None else None,
                        # Previous generations may use a different evaluation window
                        # or the legacy shared-Ridge metric: no like-for-like delta.
                        metric_before=None,
                        metric_after=mae,
                        evaluation_status=evaluation_status,
                        evaluation_sample_count=eval_count,
                        estimator_kind=estimator_kind,
                        promotion_result="accepted",
                        training_period=train_window,
                        evaluation_period=eval_window,
                    )
                )
                try:
                    import joblib

                    joblib.dump(b, artifacts_dir / f"{asset}_{h}_{fam}.joblib")
                except Exception:
                    pass
    _persist_journal(conn, journal)
    return out


def expected_bundle_keys() -> list[tuple[str, int, str]]:
    keys: list[tuple[str, int, str]] = []
    for asset in (AssetClass.US_EQUITY.value, AssetClass.BTC.value):
        for h in HORIZONS:
            for fam in ("ridge", "lightgbm_reg", "torch_sequence"):
                keys.append((asset, h, fam))
            if asset == AssetClass.US_EQUITY.value:
                keys.append((asset, h, "lambdarank"))
    return keys


def load_fitted_bundles(artifacts_dir: Path) -> dict[tuple[str, int, str], FittedBundle]:
    """Load previously promoted joblibs. Missing files are omitted."""
    out: dict[tuple[str, int, str], FittedBundle] = {}
    if not artifacts_dir.exists():
        return out
    try:
        import joblib
    except Exception:
        return out
    for asset, h, fam in expected_bundle_keys():
        path = artifacts_dir / f"{asset}_{h}_{fam}.joblib"
        if not path.is_file():
            continue
        try:
            bundle = joblib.load(path)
        except Exception:
            continue
        if isinstance(bundle, FittedBundle):
            out[(asset, h, fam)] = bundle
    return out


def bundles_ready(bundles: dict[tuple[str, int, str], FittedBundle]) -> bool:
    """Enough to score: ridge / LightGBM / LSTM for both sleeves. Ranker is optional."""
    for asset in (AssetClass.US_EQUITY.value, AssetClass.BTC.value):
        for h in HORIZONS:
            for fam in ("ridge", "lightgbm_reg", "torch_sequence"):
                if (asset, h, fam) not in bundles:
                    return False
    return True


def _persist_journal(conn: duckdb.DuckDBPyConnection, events: list[ModelChangeJournalEvent]) -> None:
    for ev in events:
        conn.execute(
            """
            INSERT INTO model_change_journal (event_id, kind, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [ev.event_id, ev.kind.value, ev.model_dump_json(), ev.created_at],
        )


def persist_decision_epoch(conn: duckdb.DuckDBPyConnection, epoch: DecisionEpochManifest) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO decision_epochs
        (decision_epoch_id, created_at, preprocess_version, ensemble_weights_hash,
         thresholds_hash, formula_versions_json, parent_epoch_id, fingerprint, manifest_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(epoch.decision_epoch_id),
            epoch.created_at,
            epoch.preprocess_version,
            epoch.ensemble_weights_hash,
            epoch.thresholds_hash,
            json.dumps(epoch.formula_versions),
            str(epoch.parent_epoch_id) if epoch.parent_epoch_id else None,
            epoch.dependency_fingerprint(),
            epoch.model_dump_json(),
        ],
    )


def persist_prediction_rows(
    conn: duckdb.DuckDBPyConnection,
    rows: list[dict],
) -> None:
    now = datetime.now(timezone.utc)
    for row in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO prediction_rows
            (prediction_id, instrument_id, as_of_date, horizon, asset_class, family, value,
             decision_epoch_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["prediction_id"],
                row["instrument_id"],
                row["as_of_date"],
                row["horizon"],
                row["asset_class"],
                row["family"],
                row["value"],
                row.get("decision_epoch_id"),
                now,
            ],
        )


class OnlineHorizonModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = SGDRegressor(random_state=42, max_iter=5, tol=None)
        self.fitted = False

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(y) == 0:
            return
        if not self.fitted:
            xs = self.scaler.fit_transform(x)
            self.model.partial_fit(xs, y)
            self.fitted = True
        else:
            xs = self.scaler.transform(x)
            self.model.partial_fit(xs, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted or len(x) == 0:
            return np.zeros(len(x))
        return self.model.predict(self.scaler.transform(x))


def online_path(artifacts_dir: Path, asset: str, horizon: int) -> Path:
    return artifacts_dir / f"online_{asset}_{horizon}.joblib"


def save_online_model(path: Path, model: OnlineHorizonModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump({"scaler": model.scaler, "coef": model.model, "fitted": model.fitted}, path)
    except Exception:
        pass


def load_online_model(path: Path) -> OnlineHorizonModel:
    om = OnlineHorizonModel()
    if not path.exists():
        return om
    try:
        import joblib

        payload = joblib.load(path)
        om.scaler = payload["scaler"]
        om.model = payload["coef"]
        om.fitted = bool(payload["fitted"])
    except Exception:
        return OnlineHorizonModel()
    return om


def apply_online_updates(
    conn: duckdb.DuckDBPyConnection,
    *,
    asset: str,
    horizon: int,
    model: OnlineHorizonModel,
    x: np.ndarray,
    y: np.ndarray,
    keys: list[tuple[str, date]],
    parent_epoch_id: str,
) -> OnlineHorizonModel:
    """Fit only matured rows not yet consumed in the online ledger (crash-idempotent)."""
    if len(y) == 0:
        return model
    stream = f"{asset}:{horizon}"
    # One batch query for this stream's already-consumed ids instead of a per-row
    # query inside the loop below -- see load_sequences' docstring in features.py
    # for why that pattern turns into 100,000+ round trips at current data volumes.
    consumed_ids = {
        row[0]
        for row in conn.execute(
            "SELECT prediction_label_id FROM online_update_ledger WHERE model_stream = ? AND status = 'consumed'",
            [stream],
        ).fetchall()
    }
    pending: list[int] = []
    for i, (inst, as_of) in enumerate(keys):
        label_id = f"{inst}:{as_of.isoformat()}:{horizon}"
        if label_id not in consumed_ids:
            pending.append(i)
    for i in pending[-16:]:
        inst, as_of = keys[i]
        label_id = f"{inst}:{as_of.isoformat()}:{horizon}"
        yi = float(y[i])
        batch_hash = hashlib.sha256(f"{label_id}:{yi:.8f}".encode()).hexdigest()[:16]
        model.partial_fit(x[i : i + 1], np.asarray([yi], dtype=float))
        conn.execute(
            """
            INSERT OR REPLACE INTO online_update_ledger
            (model_stream, target_version, prediction_label_id, parent_epoch_id,
             update_batch_hash, status, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, 'consumed', ?, ?)
            """,
            [
                stream,
                "online_v1",
                label_id,
                parent_epoch_id,
                batch_hash,
                datetime.now(timezone.utc),
                json.dumps({"asset": asset, "horizon": horizon}),
            ],
        )
    return model


def default_weights(asset: str) -> dict[str, float]:
    if asset == AssetClass.US_EQUITY.value:
        names = ("ridge", "lightgbm_reg", "lambdarank", "torch_sequence", "online")
    else:
        names = ("ridge", "lightgbm_reg", "torch_sequence", "online")
    w = 1.0 / len(names)
    return {n: w for n in names}


def blend(preds: dict[str, float], weights: dict[str, float]) -> float:
    """Blend return-scale families only. LambdaRank must not enter expected_return."""
    num = 0.0
    den = 0.0
    for k, v in preds.items():
        if k == "lambdarank":
            continue
        w = max(0.05, min(0.5, weights.get(k, 0.0)))
        num += w * v
        den += w
    return num / den if den else 0.0


def inverse_error_target(errors: dict[str, float]) -> dict[str, float]:
    """The un-damped EWMA target update_weights() moves lr of the way toward: each
    key's share is proportional to 1/(1+|error|) (lower error -> more weight),
    normalized to sum to 1. This is what update_weights() would jump straight to at
    lr=1.0 -- callers that want to journal/display "current -> target -> applied"
    (see ModelChangeJournalEvent.target_ensemble_weight) call this directly instead
    of re-deriving it from before/after deltas.
    """
    if not errors:
        return {}
    inv = {k: 1.0 / (1.0 + abs(e)) for k, e in errors.items()}
    s = sum(inv.values()) or 1.0
    return {k: v / s for k, v in inv.items()}


# 2026-09-05: raised from 0.1 after a real-scan audit found the combination of (a)
# lr=0.1 and (b) the ~10-month evaluation window (see RECENT_EVAL_WINDOW_DAYS below)
# made ensemble weights barely move even when a family's relative skill had
# genuinely changed. lr=0.2 reaches ~90% of a stable target in ~11 daily updates
# (~2 trading weeks) instead of ~22; 0.3+ starts reacting to noise in a handful of
# updates, which is too fast for a signal this noisy. The window shrink (evaluating
# on recent data instead of the whole matured history) is the bigger lever of the
# two -- this alone does not fix a stale evaluation window.
DEFAULT_EWMA_LR = 0.2


def update_weights(
    weights: dict[str, float],
    errors: dict[str, float],
    *,
    lr: float = DEFAULT_EWMA_LR,
) -> dict[str, float]:
    """EWMA-style: better (lower abs error) components gain weight. No 100% jump."""
    if not errors:
        return weights
    target = inverse_error_target(errors)
    out = {}
    for k, w in weights.items():
        tw = target.get(k, w)
        nw = (1 - lr) * w + lr * tw
        out[k] = max(0.05, min(0.5, nw))
    z = sum(out.values()) or 1.0
    return {k: v / z for k, v in out.items()}


def _restrict_to_recent_window(
    idx: np.ndarray, keys: list[tuple[str, date]], *, window_days: int = RECENT_EVAL_WINDOW_DAYS
) -> np.ndarray:
    """Keep only rows whose as_of date is among the most recent window_days unique
    dates present in idx. See RECENT_EVAL_WINDOW_DAYS for why."""
    if len(idx) == 0:
        return idx
    dates = sorted({keys[i][1] for i in idx.tolist()})
    recent = set(dates[-window_days:])
    kept = [i for i in idx.tolist() if keys[i][1] in recent]
    return np.array(kept, dtype=int)


def adapt_ensemble_from_matured(
    conn: duckdb.DuckDBPyConnection,
    *,
    asset: str,
    bundles: dict[tuple[str, int, str], FittedBundle],
    online: dict[tuple[str, int], OnlineHorizonModel],
    prev_weights: dict[str, float],
) -> tuple[dict[str, float], bool, dict[str, object]]:
    """Update ensemble weights from matured walk-forward errors.

    Returns (weights, changed, diagnostics). Diagnostics carry the factual inputs of the
    update (matured label count, per-family mean error) so the change journal can record
    why weights moved instead of leaving those fields null.
    """
    errors: dict[str, list[float]] = {}
    errors_by_horizon: dict[int, dict[str, float]] = {}
    naive_mae_by_horizon: dict[int, float] = {}
    matured_count = 0
    eval_row_count = 0
    eval_dates: list[date] = []
    for h in HORIZONS:
        x, y, keys = load_matrix(conn, asset_class=asset, horizon=h, matured_only=True)
        if len(y) < 10:
            continue
        _tr, te = chronological_split(len(y), purge=h, keys=keys)
        if len(te) == 0:
            continue
        te = _restrict_to_recent_window(te, keys)
        if len(te) == 0:
            continue
        xt, yt = x[te], y[te]
        matured_count += int(len(y))
        eval_row_count += int(len(te))
        eval_dates.extend(keys[i][1] for i in te.tolist() if 0 <= i < len(keys))
        # MAE of "always predict zero excess return" on this same test slice -- the
        # scale-free denominator adapt_horizon_influence uses to tell forecast skill
        # apart from a horizon simply having noisier (larger-magnitude) labels. Real
        # data audit (2026-09-05): raw MAE alone tracked sqrt(horizon) almost exactly
        # (5d/10d/20d MAE ~ 4.90/7.02/10.24%, matching sqrt(1)/sqrt(2)/sqrt(4)), i.e.
        # it was mostly measuring 20d labels being noisier than 5d ones, not the
        # model being more skilled at 5d.
        naive_mae_by_horizon[h] = float(np.mean(np.abs(yt)))
        fams = ["ridge", "lightgbm_reg", "torch_sequence"]
        if asset == AssetClass.US_EQUITY.value:
            fams.append("lambdarank")
        for fam in fams:
            if fam == "lambdarank":
                continue
            b = bundles.get((asset, h, fam))
            if b is None:
                continue
            if fam == "torch_sequence" and b.lookback and b.lookback > 1:
                seq, seq_y, seq_keys = load_sequences(
                    conn, asset_class=asset, horizon=h, lookback=b.lookback, matured_only=True
                )
                if len(seq_y) < 10:
                    continue
                _s_tr, s_te = chronological_split(len(seq_y), purge=h, keys=seq_keys)
                if len(s_te) == 0:
                    continue
                s_te = _restrict_to_recent_window(s_te, seq_keys)
                if len(s_te) == 0:
                    continue
                pred = _predict(b, seq[s_te])
                err = float(np.mean(np.abs(pred - seq_y[s_te])))
            else:
                pred = _predict(b, xt)
                err = float(np.mean(np.abs(pred - yt)))
            errors.setdefault(fam, []).append(err)
            errors_by_horizon.setdefault(h, {})[fam] = err
        om = online.get((asset, h))
        if om is not None and om.fitted:
            pred = om.predict(xt)
            online_err = float(np.mean(np.abs(pred - yt)))
            errors.setdefault("online", []).append(online_err)
            errors_by_horizon.setdefault(h, {})["online"] = online_err
    if not errors:
        return prev_weights, False, {
            "matured_label_count": matured_count,
            "mean_abs_error": {},
            "errors_by_horizon": {},
            "naive_mae_by_horizon": {},
        }
    mean_err = {k: float(np.mean(v)) for k, v in errors.items()}
    new_w = update_weights(prev_weights, mean_err)
    changed = any(abs(new_w.get(k, 0) - prev_weights.get(k, 0)) > 1e-6 for k in set(new_w) | set(prev_weights))
    diag: dict[str, object] = {
        "matured_label_count": matured_count,
        "mean_abs_error": mean_err,
        "errors_by_horizon": errors_by_horizon,
        "naive_mae_by_horizon": naive_mae_by_horizon,
        # Un-damped target update_weights() moved `lr` of the way toward -- see
        # ModelChangeJournalEvent.target_ensemble_weight.
        "target_weights": inverse_error_target(mean_err),
        "eval_row_count": eval_row_count,
        "evaluation_period": (
            f"{min(eval_dates).isoformat()} ~ {max(eval_dates).isoformat()}" if eval_dates else None
        ),
    }
    return new_w, changed, diag


def adapt_horizon_influence(
    prev_horizon_weights: dict[str, float],
    errors_by_horizon: dict[int, dict[str, float]],
    family_weights: dict[str, float],
    *,
    naive_mae_by_horizon: dict[int, float] | None = None,
) -> tuple[dict[str, float], bool, dict[str, float], dict[str, float]]:
    """Update per-horizon trust (5d/10d/20d) from matured walk-forward family errors.

    Returns (new_weights, changed, composite_scores, target_weights) -- the 4th
    element is the un-damped EWMA target (see inverse_error_target), for callers that
    want to journal/display "current -> target -> applied" instead of just before/after.

    A horizon's composite score is the current-family-weighted mean of that horizon's
    per-family MAE (from adapt_ensemble_from_matured's errors_by_horizon), scaled by
    naive_mae_by_horizon (that same horizon's MAE from always predicting zero excess
    return) when available. Real-data audit (2026-09-05): comparing raw MAE across
    horizons mostly measured which horizon's labels are noisier, not which one the
    model forecasts better -- 5d/10d/20d MAE tracked sqrt(horizon) almost exactly.
    Dividing by the same-horizon naive baseline turns this into model_MAE / naive_MAE
    (<1 beats the zero-forecast baseline, >1 is worse than it), which is scale-free
    and comparable across horizons. Falls back to raw MAE when no naive baseline is
    supplied, so old callers/tests keep working. Reuses update_weights()'s EWMA (same
    [0.05, 0.5] clamp) so no single horizon can dominate or vanish from one update --
    that damping is intentionally untouched here.
    """
    naive = naive_mae_by_horizon or {}
    composite: dict[str, float] = {}
    for h, fam_errors in errors_by_horizon.items():
        if not fam_errors:
            continue
        wsum = sum(family_weights.get(fam, 0.0) for fam in fam_errors)
        if wsum <= 1e-9:
            raw = float(np.mean(list(fam_errors.values())))
        else:
            raw = float(
                sum(family_weights.get(fam, 0.0) * err for fam, err in fam_errors.items()) / wsum
            )
        baseline = naive.get(h)
        composite[str(h)] = raw / baseline if baseline is not None and baseline > 1e-9 else raw
    if not composite:
        return prev_horizon_weights, False, {}, {}
    new_w = update_weights(prev_horizon_weights, composite)
    changed = any(
        abs(new_w.get(k, 0) - prev_horizon_weights.get(k, 0)) > 1e-6
        for k in set(new_w) | set(prev_horizon_weights)
    )
    return new_w, changed, composite, inverse_error_target(composite)


def next_ensemble_version(version: str) -> str:
    """v3 -> v4. Legacy suffix chains (v1_adapt_adapt) collapse to the next clean generation."""
    head = (version or "").strip()
    digits = ""
    for ch in head:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    try:
        current = int(digits) if digits else 1
    except ValueError:
        current = 1
    return f"v{current + 1}"


def ensemble_generation(version: str) -> int:
    """Human-facing generation number behind a stored ensemble version string."""
    digits = ""
    for ch in (version or "").strip():
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    try:
        return int(digits) if digits else 1
    except ValueError:
        return 1


def persist_ensemble_state(conn: duckdb.DuckDBPyConnection, stream: str, weights: dict[str, float], horizons: dict[str, float], version: str) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO ensemble_state
        (stream, weights_json, horizon_influence_json, version, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            stream,
            json.dumps(weights),
            json.dumps(horizons),
            version,
            datetime.now(timezone.utc),
        ],
    )


def load_ensemble_state(conn: duckdb.DuckDBPyConnection, stream: str) -> tuple[dict[str, float], dict[str, float], str]:
    row = conn.execute(
        "SELECT weights_json, horizon_influence_json, version FROM ensemble_state WHERE stream = ?",
        [stream],
    ).fetchone()
    if not row:
        return default_weights(stream.split(":")[0]), {"5": 1 / 3, "10": 1 / 3, "20": 1 / 3}, "v1"
    return json.loads(row[0]), json.loads(row[1]), row[2]
