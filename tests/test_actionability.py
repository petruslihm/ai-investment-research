"""User-facing deadband: tiny unit targets are 관망 / 변화 없음, not trades."""

from __future__ import annotations

from trading_system.actionability import (
    NOTE_SMALL_DELTA,
    NOTE_SMALL_TARGET,
    actionable_view,
    display_unit_quantum,
    round_units,
)
from trading_system.config import Settings
from trading_system.recommendations import RecommendationAction


def _cfg() -> Settings:
    return Settings(_env_file=None)


def test_tiny_new_target_is_watch_not_buy() -> None:
    view = actionable_view(
        {
            "action": RecommendationAction.BUY,
            "current_units": 0,
            "recommended_units": 4.28,
            "delta_units": 4.28,
        },
        settings=_cfg(),
        total_base_units=1200,
    )
    assert view.is_buy is False
    assert view.display_action == "NO_ACTION"
    assert view.display_label == "관망"
    assert view.note == NOTE_SMALL_TARGET
    assert view.raw_recommended_units == 4.28
    assert view.display_recommended_units == 0.0


def test_small_delta_on_holding_is_no_change() -> None:
    view = actionable_view(
        {
            "action": RecommendationAction.ADD,
            "current_units": 100,
            "recommended_units": 102.4,
            "delta_units": 2.4,
        },
        settings=_cfg(),
        total_base_units=1200,
    )
    assert view.display_action == "HOLD"
    assert view.display_label == "변화 없음"
    assert view.note == NOTE_SMALL_DELTA
    assert view.raw_recommended_units == 102.4
    assert view.display_delta_units == 0.0


def test_material_target_stays_actionable_and_rounds() -> None:
    view = actionable_view(
        {
            "action": RecommendationAction.BUY,
            "current_units": 0,
            "recommended_units": 48.37,
            "delta_units": 48.37,
        },
        settings=_cfg(),
        total_base_units=1200,
    )
    assert view.is_buy is True
    assert view.display_action == "BUY"
    assert view.display_recommended_units == 48.0
    assert view.raw_recommended_units == 48.37
    assert view.quantum == 1.0


def test_reduce_with_large_delta_stays_reduce() -> None:
    view = actionable_view(
        {
            "action": RecommendationAction.REDUCE,
            "current_units": 80,
            "recommended_units": 20,
            "delta_units": -60,
        },
        settings=_cfg(),
        total_base_units=1200,
    )
    assert view.display_action == "REDUCE"
    assert view.suppressed is False
    assert view.display_recommended_units == 20.0


def test_unheld_hold_does_not_get_small_target_lecture() -> None:
    view = actionable_view(
        {
            "action": RecommendationAction.HOLD,
            "current_units": 0,
            "recommended_units": 0,
            "delta_units": 0,
        },
        settings=_cfg(),
        total_base_units=1200,
    )
    assert view.display_action == "NO_ACTION"
    assert view.display_label == "관망"
    assert view.note is None
    assert view.suppressed is False


def test_relative_thresholds_scale_with_book() -> None:
    small_book = actionable_view(
        {
            "action": RecommendationAction.BUY,
            "current_units": 0,
            "recommended_units": 4.28,
            "delta_units": 4.28,
        },
        settings=_cfg(),
        total_base_units=100,
    )
    assert small_book.is_buy is True  # 4.28% of 100u book


def test_round_units_to_quantum() -> None:
    assert round_units(4.28, 1.0) == 4.0
    assert round_units(48.37, 1.0) == 48.0
    assert display_unit_quantum(_cfg(), 1200) == 1.0
