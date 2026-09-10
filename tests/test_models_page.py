"""Models / Learning page: humanized journal, no raw JSON by default, honest empty states."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading_system.ml_engine import ensemble_generation, next_ensemble_version
from trading_system.storage import Store
from trading_system.ui.models_view import (
    NO_METRIC,
    NO_OUTCOMES,
    NO_OVERRIDE,
    NO_REASON,
    collect_models_view,
    humanize_kind,
    humanize_version,
    render_models_page,
)


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "models.duckdb")
    s.open(acquire_writer=True)
    yield s
    s.close()


def _journal(conn, kind: str, payload: dict, *, at: datetime | None = None) -> None:
    body = {"event_id": payload.get("event_id", f"mcj_{kind}_{len(json.dumps(payload))}"), "kind": kind}
    body.update(payload)
    conn.execute(
        "INSERT INTO model_change_journal (event_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?)",
        [body["event_id"], kind, json.dumps(body), at or datetime.now(timezone.utc)],
    )


def _seed(conn) -> None:
    conn.execute(
        """
        INSERT INTO ensemble_state (stream, weights_json, horizon_influence_json, version, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            "us_equity",
            json.dumps(
                {"ridge": 0.21, "lightgbm_reg": 0.21, "torch_sequence": 0.21, "online": 0.21, "lambdarank": 0.16}
            ),
            json.dumps({"5": 1 / 3, "10": 1 / 3, "20": 1 / 3}),
            "v4",
            datetime.now(timezone.utc),
        ],
    )
    _journal(
        conn,
        "promoted",
        {
            "event_id": "mcj_promoted_1",
            "model_family": "ridge",
            "asset_class": "us_equity",
            "horizon": 5,
            "previous_version": "1",
            "new_version": "1",
            "reason_codes": ["batch_fit_applied", "CHRONOLOGICAL_HOLDOUT"],
            "matured_label_count": 640,
            "sample_count": 448,
            "metric_name": "chronological_holdout_mae",
            "metric_before": None,
            "metric_after": 0.024,
            "promotion_result": "accepted",
            "training_period": "2025-01-02 ~ 2026-05-30",
            "evaluation_period": "2026-06-01 ~ 2026-08-20",
        },
    )
    _journal(
        conn,
        "ensemble_weight_changed",
        {
            "event_id": "mcj_weight_1",
            "model_family": "lightgbm_reg",
            "asset_class": "us_equity",
            "previous_version": "v3",
            "new_version": "v4",
            "previous_ensemble_weight": 0.2,
            "new_ensemble_weight": 0.21,
            "reason_codes": ["matured_ewma"],
            "matured_label_count": 640,
            "promotion_result": "applied",
        },
    )
    _journal(
        conn,
        "online_updated",
        {
            "event_id": "mcj_online_1",
            "model_family": "online_sgd",
            "asset_class": "us_equity",
            "horizon": 10,
            "new_version": "online_v1",
            "reason_codes": ["matured_label_partial_fit"],
            "matured_label_count": 128,
            "training_period": "2025-01-02 ~ 2026-08-01",
        },
    )


def test_event_names_are_humanized() -> None:
    assert humanize_kind("promoted") == "새 모델 채택"
    assert humanize_kind("rejected") == "새 모델 채택 안 함"
    assert humanize_kind("online_updated") == "온라인 학습 업데이트"
    assert humanize_kind("ensemble_weight_changed") == "모델 가중치 조정"
    assert humanize_kind("retrained") == "모델 재학습"
    assert humanize_kind("rollback") == "이전 안정 버전으로 복구"
    assert humanize_kind("rolled_back") == "이전 안정 버전으로 복구"


def test_version_display_never_leaks_adapt_chain() -> None:
    assert humanize_version("v1_adapt_adapt_adapt_adapt") == "v1"
    assert humanize_version("v12") == "v12"
    assert humanize_version("online_v1") == "온라인 v1"
    assert next_ensemble_version("v1_adapt_adapt") == "v2"
    assert next_ensemble_version("v11") == "v12"
    assert ensemble_generation("v7_adapt") == 7


def test_page_shows_no_raw_json_outside_technical_details(store: Store) -> None:
    _seed(store.conn)
    view = collect_models_view(store.conn)
    html = render_models_page(view, horizon="all", change="all", family="all")
    outside = re.sub(r"<details class=\"tech\">.*?</details>", "", html, flags=re.S)
    assert "mcj_promoted_1" not in outside
    assert "reason_codes" not in outside
    assert "matured_label_count" not in outside
    # the raw payload is still preserved, just tucked away
    assert "mcj_promoted_1" in html
    assert html.count("<details class=\"tech\">") == len(view.cards)


def test_journal_cards_carry_the_facts(store: Store) -> None:
    _seed(store.conn)
    view = collect_models_view(store.conn)
    kinds = {c.kind: c for c in view.cards}

    promoted = kinds["promoted"]
    assert promoted.title == "새 모델 채택"
    assert promoted.horizon == "5일"
    assert promoted.family == "Ridge"
    assert promoted.matured == "640건"
    assert promoted.training_window == "2025-01-02 ~ 2026-05-30"
    assert promoted.evaluation_window == "2026-06-01 ~ 2026-08-20"
    assert "0.02400" in promoted.metric_line
    assert "동일 조건 비교 아님" in promoted.metric_line
    assert promoted.metric_improved is None
    assert promoted.result_label == "적용됨"
    assert "5일" in promoted.effect
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", promoted.at)

    weight = kinds["ensemble_weight_changed"]
    assert weight.what_changed == "앙상블 가중치"
    assert weight.change_detail == "20.0% → 21.0%"
    assert "평균 절대오차" in weight.reason

    online = kinds["online_updated"]
    assert online.family == "온라인 SGD"
    assert online.matured == "128건"


def test_weight_change_shows_current_target_and_applied(store: Store) -> None:
    """User-reported confusion (2026-09-05): a small before->after delta ("21.1% ->
    21.2%") looks like the model barely preferred this family, but it could equally
    mean the model wanted it much higher and lr=0.2 (or the old 0.1) only let it move
    part way there. Once target_ensemble_weight is recorded, the card must show all
    three numbers so that ambiguity is gone."""
    _journal(
        store.conn,
        "ensemble_weight_changed",
        {
            "event_id": "mcj_weight_target",
            "model_family": "ridge",
            "asset_class": "us_equity",
            "previous_ensemble_weight": 0.211,
            "new_ensemble_weight": 0.212,
            "target_ensemble_weight": 0.221,
            "sample_count": 31500,
            "matured_label_count": 1059788,
            "reason_codes": ["matured_ewma"],
            "promotion_result": "applied",
        },
    )
    view = collect_models_view(store.conn)
    card = view.cards[0]
    assert card.change_detail == "21.1% → 목표 22.1% → 적용 21.2%"
    assert "최근 평가 31,500건" in card.matured
    assert "전체 확정 1,059,788건" in card.matured


def test_missing_facts_say_so_instead_of_inventing(store: Store) -> None:
    _journal(store.conn, "promoted", {"event_id": "bare", "model_family": "ridge", "horizon": 20})
    view = collect_models_view(store.conn)
    card = view.cards[0]
    assert card.reason == NO_REASON
    assert card.metric_line == NO_METRIC
    assert card.matured == "기록 없음"
    assert card.training_window == "기록 없음"


def test_empty_outcome_and_override_messages(store: Store) -> None:
    _seed(store.conn)
    view = collect_models_view(store.conn)
    html = render_models_page(view, horizon="all", change="all", family="all")
    assert view.outcome_count == 0
    assert "recommendation_outcomes" not in html
    assert "아직 평가 가능한 추천 결과가 없습니다." in html
    assert NO_OUTCOMES.splitlines()[1] in html
    assert NO_OVERRIDE in html


def test_summary_reports_generation_and_health(store: Store) -> None:
    _seed(store.conn)
    store.conn.execute(
        """
        INSERT INTO decision_epochs
        (decision_epoch_id, created_at, preprocess_version, ensemble_weights_hash, thresholds_hash,
         formula_versions_json, parent_epoch_id, fingerprint, manifest_json)
        VALUES ('epoch_1', ?, 'feat_v1', 'h', 'th', '{}', NULL, 'fp', '{}')
        """,
        [datetime.now(timezone.utc)],
    )
    view = collect_models_view(store.conn)
    assert view.generation_label == "Epoch 1 / 앙상블 v4"
    assert "_adapt" not in view.generation_label
    assert view.matured_used == 128 and view.matured_known
    assert view.health_label == "정상"
    assert view.recent_changes == 3


def test_stale_models_are_flagged(store: Store) -> None:
    _journal(
        store.conn,
        "promoted",
        {"event_id": "old", "model_family": "ridge", "horizon": 5},
        at=datetime.now(timezone.utc) - timedelta(days=30),
    )
    view = collect_models_view(store.conn)
    assert view.health_label == "오래됨"
    assert view.recent_changes == 0


def test_horizon_cards_split_ranking_from_return_blend(store: Store) -> None:
    _seed(store.conn)
    view = collect_models_view(store.conn)
    assert [c["horizon"] for c in view.horizon_cards] == ["5", "10", "20"]
    for card in view.horizon_cards:
        assert [r["label"] for r in card["rows"]] == ["Ridge", "LightGBM", "LSTM", "온라인 SGD"]
        assert card["rank"]["label"] == "LambdaRank"
        assert card["rank"]["weight"] == "16.0%"
    html = render_models_page(view, horizon="all", change="all", family="all")
    assert "순위 전용" in html


def test_filters_narrow_the_journal(store: Store) -> None:
    _seed(store.conn)
    only_online = collect_models_view(store.conn, change="online")
    assert {c.kind for c in only_online.cards} == {"online_updated"}

    five_day = collect_models_view(store.conn, horizon="5")
    kinds = {c.kind for c in five_day.cards}
    assert "promoted" in kinds
    assert "online_updated" not in kinds  # that event is a 10d update
    # ensemble-wide changes have no horizon and stay visible for every horizon
    assert "ensemble_weight_changed" in kinds

    by_family = collect_models_view(store.conn, family="ridge")
    assert {c.family for c in by_family.cards} == {"Ridge"}


def test_research_metrics_are_labelled_separately_from_live(store: Store) -> None:
    _seed(store.conn)
    view = collect_models_view(store.conn)
    html = render_models_page(view, horizon="all", change="all", family="all")
    assert "실시간 성과 (성숙한 추천 결과)" in html
    assert "연구 진단 (시간순 분리)" in html
    assert "실현 성과가 아니며" in html
    assert view.research_rows and view.research_rows[0][0] == "미국 주식"
