"""Foundation and safety gate tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_system.config import Settings
from trading_system.ids import (
    AssetClass,
    EvaluationBasis,
    FeatureSnapshotRef,
    InstrumentAlias,
    InstrumentRecord,
    IssuerRecord,
    OutcomeSnapshotRef,
    PriceBasisLineage,
    new_feature_snapshot_id,
    new_instrument_id,
    new_issuer_id,
    new_outcome_snapshot_id,
    new_tick_id,
)
from trading_system.models import DecisionEpochManifest, ModelArtifactRef, ModelFamily
from trading_system.portfolio import Lot, PortfolioUnits, ValuationState
from trading_system.storage import Store, WriterLeaseBusy
from trading_system.storage.assets import (
    AliasConflictError,
    add_alias,
    close_alias,
    resolve_alias,
    upsert_instrument,
    upsert_issuer,
)
from trading_system.storage.leases import JobLease, JobLeaseError, WriterLease
from trading_system.storage.schema import FOUNDATION_TABLES
from trading_system.storage.ticks import TickCommitPayload, commit_tick
from trading_system.ui.app import NAV_ROUTES, create_app


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.duckdb"


@pytest.fixture()
def store(db_path: Path) -> Store:
    s = Store(db_path)
    s.open(acquire_writer=True, stale_seconds=30)
    yield s
    s.close()


def test_empty_schema_creates_foundation_tables(store: Store) -> None:
    tables = set(store.list_tables())
    for name in FOUNDATION_TABLES:
        assert name in tables
    counts = store.table_counts()
    # Fresh DB: only schema_meta may have a row
    for name, count in counts.items():
        if name == "schema_meta":
            assert count >= 1
        elif name == "writer_lease":
            assert count == 1  # held by store fixture
        else:
            assert count == 0


def test_stale_owner_resume_writer_lease(db_path: Path) -> None:
    store = Store(db_path)
    store.open(acquire_writer=False)
    conn = store.conn

    stale = WriterLease(conn, stale_seconds=1, owner_token="stale_owner")
    stale.acquire()
    # Force expiry in the past
    past = datetime.now(timezone.utc) - timedelta(seconds=10)
    conn.execute(
        """
        UPDATE writer_lease
        SET expires_at = ?, heartbeat_at = ?
        WHERE lease_name = 'duckdb_writer'
        """,
        [past, past],
    )
    stale._held = False  # simulate crash without release

    fresh = WriterLease(conn, stale_seconds=30, owner_token="fresh_owner")
    token = fresh.acquire()
    assert token == "fresh_owner"
    row = conn.execute(
        "SELECT owner_token FROM writer_lease WHERE lease_name = 'duckdb_writer'"
    ).fetchone()
    assert row[0] == "fresh_owner"
    fresh.release()
    store.close()


def test_close_logs_when_writer_lease_release_fails(
    db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Store.close() must never raise on a failed lease release (a crash on shutdown
    would be worse than a stale lease), but silently swallowing it left no trace at
    all for why a subsequent writer had to wait out the full stale_seconds window."""
    import logging

    store = Store(db_path)
    store.open(acquire_writer=True)

    def _boom() -> None:
        raise RuntimeError("simulated release failure")

    store.writer_lease.release = _boom  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger="trading_system.storage"):
        store.close()  # must not raise

    assert store.writer_lease is None
    assert any("writer_lease.release() failed" in r.message for r in caplog.records)


def test_open_releases_lease_when_apply_schema_fails_after_acquire(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The already-initialized-DB path acquires the writer lease before apply_schema()
    (to avoid an unleased migration race). If apply_schema() then fails, the lease row
    must be released immediately -- not left "held" by a dead owner for the full
    stale_seconds window, which would block every other writer for an otherwise-instant
    DDL failure."""
    first = Store(db_path)
    first.open(acquire_writer=True)  # schema now exists
    first.close()

    failing = Store(db_path)
    original_apply_schema = Store.apply_schema

    def _boom(self: Store) -> None:
        raise RuntimeError("simulated bad migration")

    monkeypatch.setattr(Store, "apply_schema", _boom)
    with pytest.raises(RuntimeError):
        failing.open(acquire_writer=True, stale_seconds=30)
    assert failing.writer_lease is None
    monkeypatch.setattr(Store, "apply_schema", original_apply_schema)

    # A fresh writer must be able to acquire immediately -- not wait out stale_seconds.
    second = Store(db_path)
    second.open(acquire_writer=True, stale_seconds=30)  # must not raise WriterLeaseBusy
    assert second.writer_lease is not None and second.writer_lease.held
    second.close()


class _CloseFailsWrapper:
    """Delegates everything to the real DuckDB connection except close(), which
    raises -- without mutating the real connection object at all."""

    def __init__(self, real: object) -> None:
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name: str) -> object:
        return getattr(object.__getattribute__(self, "_real"), name)

    def close(self) -> None:
        raise RuntimeError("simulated connection close failure")


def test_open_conn_close_failure_does_not_mask_the_original_error(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both apply_schema() and the cleanup conn.close() fail, the caller must
    still see the ORIGINAL apply_schema error -- not have it replaced by the close()
    failure, which would hide the real cause of the open() failure."""
    import duckdb

    first = Store(db_path)
    first.open(acquire_writer=True)  # schema now exists
    first.close()

    def _boom_schema(self: Store) -> None:
        raise RuntimeError("original migration failure")

    monkeypatch.setattr(Store, "apply_schema", _boom_schema)

    original_connect = duckdb.connect

    def fake_connect(*a: object, **k: object) -> object:
        return _CloseFailsWrapper(original_connect(*a, **k))

    monkeypatch.setattr("trading_system.storage.duckdb.connect", fake_connect)

    failing = Store(db_path)
    with pytest.raises(RuntimeError, match="original migration failure"):
        failing.open(acquire_writer=True, stale_seconds=30)
    assert failing.writer_lease is None
    assert failing._conn is None


def test_stale_owner_resume_job_lease(store: Store) -> None:
    conn = store.conn
    first = JobLease(conn, "job_backfill", stale_seconds=1, owner_token="worker_a")
    first.acquire()
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    conn.execute(
        """
        UPDATE job_lease SET expires_at = ?, heartbeat_at = ?
        WHERE job_key = 'job_backfill'
        """,
        [past, past],
    )
    first._held = False

    second = JobLease(conn, "job_backfill", stale_seconds=30, owner_token="worker_b")
    assert second.acquire() == "worker_b"
    # Stale owner must not be able to finish
    first._held = True
    with pytest.raises(JobLeaseError):
        first.finish()
    second.finish()


def test_concurrent_second_writer_rejected(db_path: Path) -> None:
    import duckdb

    first = Store(db_path)
    first.open(acquire_writer=True, stale_seconds=60)
    try:
        rejected = False
        try:
            second_conn = duckdb.connect(str(db_path))
            try:
                lease = WriterLease(second_conn, stale_seconds=60, owner_token="intruder")
                with pytest.raises(WriterLeaseBusy):
                    lease.acquire()
                rejected = True
            finally:
                second_conn.close()
        except Exception:
            # DuckDB may refuse a second writer connection (file lock) — also rejection.
            rejected = True
        assert rejected, "second writer was neither lease-rejected nor connection-locked"
    finally:
        first.close()


def test_tick_transactional_commit_watermark_last(store: Store) -> None:
    tick_id = new_tick_id()
    payload = TickCommitPayload(
        tick_id=tick_id,
        decision_epoch_id="epoch_test",
        feature_snapshot_id="fs_test",
        observations=[{"instrument_id": "inst_a", "price": 1.0}],
        predictions=[{"instrument_id": "inst_a", "score": 0.5}],
        recommendations=[
            {
                "recommendation_id": "rec_1",
                "source": "quant_only",
                "instrument_id": "inst_a",
            }
        ],
        watermarks=[
            {
                "watermark_key": "live:inst_a",
                "instrument_id": "inst_a",
                "value": "2024-01-02T00:00:00Z",
            }
        ],
        alert_outbox=[
            {
                "outbox_id": "out_1",
                "alert_id": "alert_1",
                "channel": "console",
                "idempotency_key": f"{tick_id}:alert_1:console",
                "status": "pending",
                "attempts": 0,
                "payload": {"msg": "hi"},
            }
        ],
    )
    committed = commit_tick(store.conn, payload)
    assert committed == tick_id

    status = store.conn.execute(
        "SELECT status FROM ticks WHERE tick_id = ?", [tick_id]
    ).fetchone()
    assert status[0] == "committed"

    wm = store.conn.execute(
        "SELECT tick_id, value FROM watermarks WHERE watermark_key = 'live:inst_a'"
    ).fetchone()
    assert wm[0] == tick_id

    # Idempotent re-commit
    commit_tick(store.conn, payload)
    outbox_count = store.conn.execute("SELECT COUNT(*) FROM alert_outbox").fetchone()[0]
    assert outbox_count == 1


def test_decision_epoch_schema() -> None:
    manifest = DecisionEpochManifest(
        preprocess_version="prep_v1",
        ensemble_weights_hash="ew_abc",
        thresholds_hash="th_xyz",
        formula_versions={"stop_atr": "v1", "excess_return": "v1"},
        batch_models=[
            ModelArtifactRef(
                family=ModelFamily.LIGHTGBM_REG,
                horizon=5,
                version="1",
                artifact_hash="h1",
            )
        ],
        online_models=[
            ModelArtifactRef(
                family=ModelFamily.ONLINE_SGD,
                horizon=5,
                version="1",
                artifact_hash="h2",
            )
        ],
    )
    fp = manifest.dependency_fingerprint()
    assert "prep_v1" in fp
    assert "lightgbm_reg:5:1:h1" in fp
    assert manifest.decision_epoch_id.startswith("epoch_")

    # Persist via store
    # (schema round-trip in dedicated store test below)


def test_decision_epoch_persists(store: Store) -> None:
    import json

    manifest = DecisionEpochManifest(
        preprocess_version="prep_v1",
        ensemble_weights_hash="ew",
        thresholds_hash="th",
        formula_versions={"f": "1"},
    )
    store.conn.execute(
        """
        INSERT INTO decision_epochs
        (decision_epoch_id, created_at, preprocess_version, ensemble_weights_hash,
         thresholds_hash, formula_versions_json, parent_epoch_id, fingerprint, manifest_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            manifest.decision_epoch_id,
            manifest.created_at,
            manifest.preprocess_version,
            manifest.ensemble_weights_hash,
            manifest.thresholds_hash,
            json.dumps(manifest.formula_versions),
            manifest.parent_epoch_id,
            manifest.dependency_fingerprint(),
            manifest.model_dump_json(),
        ],
    )
    row = store.conn.execute(
        "SELECT fingerprint FROM decision_epochs WHERE decision_epoch_id = ?",
        [manifest.decision_epoch_id],
    ).fetchone()
    assert row[0] == manifest.dependency_fingerprint()


def test_instrument_alias_rename_and_reuse(store: Store) -> None:
    issuer = IssuerRecord(issuer_id=new_issuer_id(), display_name="Example Corp", cik="0001")
    inst_old = InstrumentRecord(
        instrument_id=new_instrument_id(),
        issuer_id=issuer.issuer_id,
        asset_class=AssetClass.US_EQUITY,
        display_name="OLD",
    )
    inst_new = InstrumentRecord(
        instrument_id=new_instrument_id(),
        issuer_id=issuer.issuer_id,
        asset_class=AssetClass.US_EQUITY,
        display_name="NEW",
    )
    upsert_issuer(store.conn, issuer)
    upsert_instrument(store.conn, inst_old)
    upsert_instrument(store.conn, inst_new)

    add_alias(
        store.conn,
        InstrumentAlias(
            instrument_id=inst_old.instrument_id,
            provider="alpaca",
            symbol="EXAMP",
            effective_from=date(2020, 1, 1),
            effective_to=None,
        ),
    )
    # Rename: close old, open new symbol for same instrument
    close_alias(store.conn, "alpaca", "EXAMP", inst_old.instrument_id, date(2023, 6, 1))
    add_alias(
        store.conn,
        InstrumentAlias(
            instrument_id=inst_old.instrument_id,
            provider="alpaca",
            symbol="EXAM",
            effective_from=date(2023, 6, 2),
        ),
    )
    assert resolve_alias(store.conn, "alpaca", "EXAMP", date(2022, 1, 1)) == inst_old.instrument_id
    assert resolve_alias(store.conn, "alpaca", "EXAM", date(2024, 1, 1)) == inst_old.instrument_id
    assert resolve_alias(store.conn, "alpaca", "EXAMP", date(2024, 1, 1)) is None

    # Ticker reuse by a different instrument after prior window closed
    add_alias(
        store.conn,
        InstrumentAlias(
            instrument_id=inst_new.instrument_id,
            provider="alpaca",
            symbol="EXAMP",
            effective_from=date(2023, 6, 2),
        ),
    )
    assert resolve_alias(store.conn, "alpaca", "EXAMP", date(2024, 1, 1)) == inst_new.instrument_id

    # Overlapping reuse must fail
    with pytest.raises(AliasConflictError):
        add_alias(
            store.conn,
            InstrumentAlias(
                instrument_id=inst_old.instrument_id,
                provider="alpaca",
                symbol="EXAMP",
                effective_from=date(2023, 1, 1),
            ),
        )


def test_feature_outcome_snapshot_lineage_schema() -> None:
    lineage = PriceBasisLineage(
        price_basis="raw",
        adjustment_revision="rev_1",
        evaluation_basis=EvaluationBasis.RAW_PLUS_CORPORATE_ACTIONS,
        provider="alpaca",
    )
    feature = FeatureSnapshotRef(
        feature_snapshot_id=new_feature_snapshot_id(),
        lineage=lineage,
    )
    outcome = OutcomeSnapshotRef(
        outcome_snapshot_id=new_outcome_snapshot_id(),
        feature_snapshot_id=feature.feature_snapshot_id,
        lineage=lineage.model_copy(),
    )
    assert outcome.lineage_compatible(feature)

    bad = OutcomeSnapshotRef(
        outcome_snapshot_id=new_outcome_snapshot_id(),
        feature_snapshot_id=feature.feature_snapshot_id,
        lineage=PriceBasisLineage(
            price_basis="raw",
            adjustment_revision="rev_2",  # mismatch
            evaluation_basis=EvaluationBasis.RAW_PLUS_CORPORATE_ACTIONS,
        ),
    )
    assert not bad.lineage_compatible(feature)

    # Distinct IDs: outcome snapshot must not equal feature snapshot
    assert outcome.outcome_snapshot_id != feature.feature_snapshot_id


def test_feature_outcome_snapshot_persists(store: Store) -> None:
    fs = new_feature_snapshot_id()
    os_ = new_outcome_snapshot_id()
    store.conn.execute(
        """
        INSERT INTO feature_snapshots
        (feature_snapshot_id, price_basis, adjustment_revision, evaluation_basis, provider, published_at)
        VALUES (?, 'raw', 'rev_1', 'raw_plus_corporate_actions', 'alpaca', ?)
        """,
        [fs, datetime.now(timezone.utc)],
    )
    store.conn.execute(
        """
        INSERT INTO outcome_snapshots
        (outcome_snapshot_id, feature_snapshot_id, price_basis, adjustment_revision,
         evaluation_basis, provider, published_at)
        VALUES (?, ?, 'raw', 'rev_1', 'raw_plus_corporate_actions', 'alpaca', ?)
        """,
        [os_, fs, datetime.now(timezone.utc)],
    )
    row = store.conn.execute(
        """
        SELECT o.adjustment_revision, f.adjustment_revision
        FROM outcome_snapshots o
        JOIN feature_snapshots f ON f.feature_snapshot_id = o.feature_snapshot_id
        WHERE o.outcome_snapshot_id = ?
        """,
        [os_],
    ).fetchone()
    assert row[0] == row[1] == "rev_1"


def test_unit_privacy_schema() -> None:
    lot = Lot(
        instrument_id=new_instrument_id(),
        acquisition_units=10.0,
        acquisition_price=2.5,
        acquired_on=date(2024, 1, 1),
    )
    portfolio = PortfolioUnits(total_base_units=1000.0, cash_units=100.0)
    # No currency/account-balance field required
    assert not hasattr(portfolio, "account_balance")
    assert portfolio.total_base_units == 1000.0

    from trading_system.portfolio import InstrumentPosition

    pos = InstrumentPosition(instrument_id=lot.instrument_id, lots=[lot], price_available=False)
    portfolio.positions = [pos]
    assert portfolio.recompute_valuation_state() == ValuationState.INCOMPLETE
    assert not portfolio.actionable_for_recommendations()

    pos.price_available = True
    assert portfolio.recompute_valuation_state() == ValuationState.COMPLETE
    assert portfolio.actionable_for_recommendations()

    with pytest.raises(Exception):
        Lot(
            instrument_id=lot.instrument_id,
            acquisition_units=-1,
            acquisition_price=1.0,
            acquired_on=date(2024, 1, 1),
        )


def test_config_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HISTORY_YEARS", raising=False)
    monkeypatch.delenv("LIVE_REFRESH_SECONDS", raising=False)
    monkeypatch.delenv("BTC_SYMBOL", raising=False)
    monkeypatch.delenv("RANDOM_SEED", raising=False)
    monkeypatch.delenv("UI_STACK", raising=False)
    monkeypatch.chdir(tmp_path)  # avoid loading repo .env
    s = Settings(_env_file=None)
    assert s.history_years == 3
    assert s.live_refresh_seconds == 180
    assert s.btc_symbol == "BTC/USD"
    assert s.usd_cash_asset == "USD_CASH"
    assert s.horizons == (5, 10, 20)
    assert s.random_seed == 42
    assert s.ui_stack == "fastapi_vite_react"
    assert "SPY" in s.smoke_universe


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_settings_reject_nonfinite_numbers(bad: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_daily_budget_usd=bad)


@pytest.mark.parametrize("bad", [(), (0, 5), (-1, 5), (5, 5)])
def test_settings_reject_invalid_horizons(bad: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, horizons=bad)


def test_failed_writer_open_closes_connection_and_can_retry(tmp_path: Path) -> None:
    db_path = tmp_path / "open_cleanup.duckdb"
    first = Store(db_path)
    first.open(acquire_writer=True, stale_seconds=60)
    second = Store(db_path)
    try:
        schema_calls: list[bool] = []
        original_apply_schema = second.apply_schema

        def spy_apply_schema() -> None:
            schema_calls.append(True)
            original_apply_schema()

        second.apply_schema = spy_apply_schema  # type: ignore[method-assign]
        with pytest.raises(WriterLeaseBusy):
            second.open(acquire_writer=True, stale_seconds=60)
        assert second._conn is None
        assert schema_calls == [], "a rejected writer must not run migrations before acquiring the lease"

        first.close()
        second.open(acquire_writer=True, stale_seconds=60)
        assert second.conn is not None
        assert schema_calls == [True]
    finally:
        first.close()
        second.close()


def test_ui_shell_routes() -> None:
    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    for href, *_rest in NAV_ROUTES:
        assert href in paths
    assert "/api/health" in paths
    assert "/auth/kakao/callback" in paths
    assert "/auth/kakao/connect" in paths
    # No broker/order routes
    assert not any("order" in p for p in paths)


def test_no_broker_module_importable() -> None:
    import importlib
    import pkgutil

    import trading_system as pkg

    names = {m.name for m in pkgutil.iter_modules(pkg.__path__)}
    assert "broker" not in names
    assert "orders" not in names
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("trading_system.broker")
