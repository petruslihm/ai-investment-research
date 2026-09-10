"""Regression checks for the meaning of reported metrics and source labels."""
import json
from datetime import date, timedelta

import numpy as np
import pytest

from trading_system import ml_engine as ml
from trading_system.judge_package import verify_pack_sources
from trading_system.storage import Store
from trading_system.ui.models_view import collect_models_view, render_models_page


def _keys(n):
    return [("example", date(2025, 1, 1) + timedelta(days=i)) for i in range(n)]


@pytest.mark.parametrize("n,seq_n", [(100, 60), (9, 0)])
def test_persisted_metrics_belong_to_the_estimator_and_its_evaluation_input(monkeypatch, tmp_path, n, seq_n):
    monkeypatch.setattr(ml, "HORIZONS", (5,))
    monkeypatch.setattr(ml, "load_matrix", lambda *a, **k: (np.zeros((n, 7)), np.full(n, .25), _keys(n)))
    seq = np.zeros((seq_n, ml.LSTM_LOOKBACK, 7))
    monkeypatch.setattr(ml, "load_sequences", lambda *a, **k: (seq, np.full(seq_n, .25), _keys(seq_n)))
    monkeypatch.setattr(ml, "_fit_ridge", lambda x, y: (None, "ridge-test-double"))
    monkeypatch.setattr(ml, "_fit_lgbm", lambda *a, **k: "lgbm-test-double")
    monkeypatch.setattr(ml, "_fit_lstm", lambda *a, **k: "lstm-test-double")
    calls = []

    def predict(bundle, x):
        assert bundle.family != "lambdarank", "Ranking output is not a return prediction"
        if bundle.family == "torch_sequence":
            assert x.ndim == 3 and x.shape[1] == ml.LSTM_LOOKBACK
        calls.append((bundle.family, len(x)))
        return np.full(len(x), {"ridge": 0., "lightgbm_reg": 1., "torch_sequence": 2.}[bundle.family])

    monkeypatch.setattr(ml, "_predict", predict)
    store = Store(tmp_path / "metrics.duckdb")
    store.open(acquire_writer=True)
    try:
        ml.train_batch_models(store.conn, tmp_path / "models")
        rows = [json.loads(row[0]) for row in store.conn.execute("SELECT payload_json FROM model_change_journal").fetchall()]
        assert len(rows) == 7  # Four equity families, three BTC families.
        for row in rows:
            family = row["model_family"]
            assert row["metric_before"] is None
            if family == "lambdarank" or n == 9:
                assert row["metric_after"] is None and row["evaluation_sample_count"] == 0
                assert row["evaluation_period"] is None
            else:
                expected = {"ridge": .25, "lightgbm_reg": .75, "torch_sequence": 1.75}[family]
                assert row["metric_after"] == pytest.approx(expected)
                assert row["metric_name"] == "chronological_holdout_mae"
                assert row["sample_count"] == (42 if family == "torch_sequence" else 70)
                assert row["evaluation_sample_count"] == (13 if family == "torch_sequence" else 25)
        if n == 9:
            assert calls == []  # Do not evaluate a small-sample in-sample fallback.
    finally:
        store.close()


def test_disjoint_but_unpurged_fallback_is_not_a_holdout_metric():
    bundle = ml.FittedBundle("ridge", 20, "us_equity", None, None, "test")
    metric, status, count = ml._holdout_mae(
        bundle, np.zeros((30, 7)), np.zeros(30), _keys(30), np.arange(21), np.array([29]), purge=20)
    assert (metric, status, count) == (None, "INSUFFICIENT_PURGE", 0)


def test_old_shared_metrics_are_preserved_but_not_shown_as_model_performance(tmp_path):
    store = Store(tmp_path / "legacy.duckdb")
    store.open(acquire_writer=True)
    try:
        payload = dict(model_family="torch_sequence", asset_class="us_equity", horizon=5,
                       metric_name="walkforward_mae", metric_after=.98765)
        store.conn.execute("INSERT INTO model_change_journal (event_id,kind,payload_json,created_at) VALUES ('old','promoted',?,now())", [json.dumps(payload)])
        view = collect_models_view(store.conn)
        assert all(row[3] == "—" for row in view.research_rows)
        assert all(row["metric"] == "—" for card in view.horizon_cards for row in card["rows"])
        assert "모델별 성능으로 사용 불가" in view.cards[0].metric_line
        assert "0.98765" in render_models_page(view, horizon="all", change="all", family="all")  # Original payload still inspectable.
    finally:
        store.close()


def test_source_labels_cannot_turn_supplied_urls_into_fact_verification():
    pack = verify_pack_sources({
        "sources": [{"url": "https://example.com/article?id=1"}, {"url": "javascript:alert(1)"}],
        "supporting_evidence": [
            {"claim": "listed", "source_url": "https://example.com/article?id=1", "source_status": "VERIFIED_SOURCE"},
            {"claim": "different article", "source_url": "https://example.com/article?id=2"},
            {"claim": "unfetched", "source_url": "https://www.sec.gov/nonexistent-example"},
            {"claim": "unsafe", "source_url": "javascript:alert(1)"}],
        "news_events": [{"url": "https://www.sec.gov/nonexistent-example", "source_status": "VERIFIED_SOURCE"}],
    })
    assert [row["source_status"] for row in pack["supporting_evidence"]] == [
        "CITED_URL", "UNVERIFIED", "SEC_DOMAIN_ONLY", "UNVERIFIED"]
    assert pack["news_events"][0]["source_status"] == "SEC_DOMAIN_ONLY"
