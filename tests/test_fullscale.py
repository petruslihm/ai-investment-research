"""Full-scale correctness tests: deadline bound, rank labels, empty equity coverage."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pytest

from trading_system.config import Settings
from trading_system.market.service import MarketDataService
from trading_system.ml_engine import _rank_labels
from trading_system.providers.fixture import FixtureMarketDataProvider, FixtureProviderConfig
from trading_system.providers.never_block import RetryPolicy, run_with_deadline
from trading_system.storage import Store


def test_deadline_does_not_wait_for_hung_worker() -> None:
    def hang() -> str:
        time.sleep(0.25)
        return "late"

    t0 = time.monotonic()
    result = run_with_deadline(
        hang,
        policy=RetryPolicy(max_attempts=1, total_deadline_seconds=0.03),
        label="hang",
    )
    elapsed = time.monotonic() - t0
    assert result.ok is False
    assert elapsed < 0.15


def test_rank_labels_are_ints_within_date() -> None:
    import numpy as np

    y = np.array([0.2, 0.1, 0.3, -0.1], dtype=float)
    keys = [
        ("a", date(2024, 1, 2)),
        ("b", date(2024, 1, 2)),
        ("c", date(2024, 1, 3)),
        ("d", date(2024, 1, 3)),
    ]
    idx = np.arange(4)
    order, rel, groups = _rank_labels(y, keys, idx)
    assert rel.dtype.kind == "i"
    assert groups == [2, 2]
    assert set(rel[:2].tolist()) == {0, 1}


def test_empty_equity_bars_not_healthy(store: Store, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, smoke_universe=("SPY", "AAPL", "BTC/USD"))
    provider = FixtureMarketDataProvider(FixtureProviderConfig(equity_bars={}, btc_bars=[]))
    svc = MarketDataService(store.conn, settings, provider, provider_name="fixture")
    rs = svc.startup_backfill(end=date(2024, 6, 14))
    snap = svc.build_data_health_snapshot(rs)
    assert snap.equity_feed.value in {"critical", "degraded"}
    assert snap.overall.value != "ok"


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "fs.duckdb")
    s.open(acquire_writer=True)
    yield s
    s.close()
