"""Score-to-weight allocation: eligibility, budget, cap/floor redistribute, action from target."""

from __future__ import annotations

import pytest

from trading_system.allocation import (
    allocate,
    allocation_score,
    cap_floor_redistribute,
    derive_stock_action,
    implied_strong_opportunity_score,
    name_conviction,
    opportunity_score,
)
from trading_system.btc_sleeve import BtcLiquidityState, BtcSleeveState
from trading_system.config import Settings
from trading_system.ids import new_tick_id
from trading_system.market.registry import stable_instrument_id
from trading_system.recommendations import RecommendationAction


def _hs(x: float) -> dict[int, float]:
    return {5: x, 10: x, 20: x}


def _settings(**over: object) -> Settings:
    kw: dict[str, object] = {
        "_env_file": None,
        "alpaca_api_key": None,
        "alpaca_secret_key": None,
        "openai_api_key": None,
        "gemini_api_key": None,
        "anthropic_api_key": None,
        "min_opportunity_score": 0.01,
        "min_position_weight": 0.01,
        "max_equity_positions": 20,
        "max_single_stock_weight": 0.25,
        "max_total_stock_weight": 0.80,
        "min_cash_weight": 0.0,
        "min_delta_weight": 0.005,
        "max_btc_weight": 0.30,
    }
    kw.update(over)
    return Settings(**kw)  # type: ignore[arg-type]


def _btc(*, units: float = 0.0, liq: BtcLiquidityState = BtcLiquidityState.UNAVAILABLE) -> BtcSleeveState:
    return BtcSleeveState(
        instrument_id=stable_instrument_id("BTC/USD"),
        liquidity=liq,
        current_units=units,
    )


def _alloc(settings: Settings, scores: dict[str, dict[int, float]], **kwargs: object) -> dict:
    stock_vol = kwargs.pop("stock_vol", None)
    vol = {k: 0.02 for k in scores}
    if isinstance(stock_vol, dict):
        vol.update(stock_vol)
    return allocate(
        stock_scores=scores,
        stock_vol=vol,
        btc_scores=kwargs.pop("btc_scores", {5: -0.1, 10: -0.1, 20: -0.1}),  # type: ignore[arg-type]
        settings=settings,
        btc_state=kwargs.pop("btc_state", _btc()),  # type: ignore[arg-type]
        tick_id=str(new_tick_id()),
        epoch_id="epoch_alloc",
        feature_snapshot_id="fs_alloc",
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_no_qualified_stocks_is_100_percent_cash() -> None:
    out = _alloc(_settings(), {"inst_aapl": _hs(-0.2)})
    assert out["payload"]["cash_weight"] == pytest.approx(1.0)
    assert out["payload"]["stock_weights"] == {}


def test_b_one_strong_stock_is_capped_rest_cash() -> None:
    inst = "inst_aapl"
    s = _settings(max_single_stock_weight=0.25, max_total_stock_weight=0.80)
    out = _alloc(s, {inst: _hs(0.08)})
    w = out["payload"]["stock_weights"][inst]
    assert w == pytest.approx(0.25)
    assert out["payload"]["cash_weight"] == pytest.approx(0.75)
    rec = next(r for r in out["recommendations"] if str(r.instrument_id) == inst)
    assert rec.action == RecommendationAction.ENTER
    assert rec.recommended_units == pytest.approx(250.0)


def test_c_three_strong_stocks_are_proportional() -> None:
    scores = {"inst_a": _hs(0.09), "inst_b": _hs(0.06), "inst_c": _hs(0.03)}
    out = _alloc(_settings(max_total_stock_weight=0.60, max_single_stock_weight=0.40), scores)
    weights = out["payload"]["stock_weights"]
    assert set(weights) == {"inst_a", "inst_b", "inst_c"}
    assert weights["inst_a"] > weights["inst_b"] > weights["inst_c"]
    assert all(w + 1e-9 >= 0.01 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(0.60)


def test_d_many_weak_positives_do_not_become_tiny_buys() -> None:
    scores = {f"inst_{i}": _hs(0.002) for i in range(40)}
    out = _alloc(_settings(), scores)
    assert out["payload"]["stock_weights"] == {}
    buys = [
        r
        for r in out["recommendations"]
        if r.action in {RecommendationAction.BUY, RecommendationAction.ENTER, RecommendationAction.ADD}
    ]
    assert buys == []
    assert all(r.action == RecommendationAction.NO_ACTION for r in out["recommendations"] if "btc" not in str(r.instrument_id))


def test_e_raw_4_28u_on_1200_is_no_action() -> None:
    # Force a pre-floor target of 4.28u (0.3567%) on a 1200 book — below 1% floor.
    s = _settings(max_total_stock_weight=4.28 / 1200, min_position_weight=0.01)
    out = _alloc(s, {"inst_pltr": _hs(0.08)}, total_base_units=1200)
    rec = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_pltr")
    assert rec.pre_floor_units == pytest.approx(4.28, rel=1e-3)
    assert rec.recommended_units == pytest.approx(0.0)
    assert rec.action == RecommendationAction.NO_ACTION
    assert rec.exclusion_reason == "BELOW_MIN_POSITION"
    assert rec.allocation_note and "0.36%" in rec.allocation_note and "1.0%" in rec.allocation_note
    assert "inst_pltr" not in out["payload"]["stock_weights"]
    diag = out["payload"]["diagnostics"]["inst_pltr"]
    assert diag["opportunity_score"] is not None
    assert diag["equity_budget"] == pytest.approx(4.28 / 1200)
    assert diag["initial_units"] == pytest.approx(4.28, rel=1e-3)
    assert diag["pre_floor_units"] == pytest.approx(4.28, rel=1e-3)
    assert diag["final_units"] == pytest.approx(0.0)
    assert diag["exclusion_reason"] == "BELOW_MIN_POSITION"


def test_f_floor_drops_are_redistributed() -> None:
    weights, trace = cap_floor_redistribute(
        {"a": 10.0, "b": 1.0, "c": 1.0},
        budget=0.05,
        max_single=0.25,
        min_position=0.01,
    )
    assert set(weights) == {"a"}
    assert weights["a"] == pytest.approx(0.05)
    assert trace["b"]["exclusion_reason"] == "BELOW_MIN_POSITION"
    assert trace["c"]["exclusion_reason"] == "BELOW_MIN_POSITION"


def test_g_single_stock_cap_is_redistributed() -> None:
    weights, trace = cap_floor_redistribute(
        {"a": 10.0, "b": 10.0},
        budget=0.80,
        max_single=0.25,
        min_position=0.01,
    )
    assert weights["a"] == pytest.approx(0.25)
    assert weights["b"] == pytest.approx(0.25)
    assert trace["a"]["cap_applied"] is True
    leftover = 0.80 - 0.50
    assert leftover == pytest.approx(0.30)


def test_h_final_book_constraints() -> None:
    s = _settings(min_cash_weight=0.10, max_total_stock_weight=0.80, max_single_stock_weight=0.25)
    scores = {f"inst_{i}": _hs(0.05 + i * 0.01) for i in range(6)}
    out = _alloc(s, scores, total_base_units=1200)
    p = out["payload"]
    stock = sum(p["stock_weights"].values())
    assert stock + p["btc_weight"] + p["cash_weight"] == pytest.approx(1.0)
    assert p["cash_weight"] + 1e-9 >= 0.10
    assert stock + p["btc_weight"] <= 1.0 + 1e-9
    for w in p["stock_weights"].values():
        assert w + 1e-9 >= s.min_position_weight
        assert w <= s.max_single_stock_weight + 1e-9


def test_i_vol_no_longer_affects_score_weight_or_sizing() -> None:
    """2026-09-04: opportunity_score() dropped its 1/(1+vol/0.02) term (a real-scan
    audit found it rejected ~97% of names technical_factors.py independently rated
    as strong leaders/breakouts -- TSLA, AAPL, CRM among them -- purely because a
    genuinely strong recent move also raises 10-day realized vol). Two names with
    identical mean/confidence/agreement but very different vol must now score,
    weight, and size identically; vol remains a stored diagnostic only."""
    s = _settings(max_total_stock_weight=0.50, max_single_stock_weight=0.40)
    scores = {"inst_lowvol": _hs(0.08), "inst_highvol": _hs(0.08)}
    out = _alloc(
        s,
        scores,
        stock_vol={"inst_lowvol": 0.02, "inst_highvol": 0.08},
        stock_confidence={"inst_lowvol": 1.0, "inst_highvol": 1.0},
    )
    adj_low = opportunity_score(0.08, 1.0, 0.02, "agree")
    adj_high = opportunity_score(0.08, 1.0, 0.08, "agree")
    assert adj_low == pytest.approx(adj_high)
    w = out["payload"]["stock_weights"]
    diag = out["payload"]["diagnostics"]
    assert w["inst_lowvol"] == pytest.approx(w["inst_highvol"])
    assert diag["inst_lowvol"]["allocation_score"] == pytest.approx(diag["inst_highvol"]["allocation_score"])
    # vol is still computed and stored per name for display (allocation_trace), just unused in scoring.
    assert diag["inst_lowvol"]["vol"] == pytest.approx(0.02)
    assert diag["inst_highvol"]["vol"] == pytest.approx(0.08)


def test_max_equity_positions_is_a_cap_not_a_fill() -> None:
    s = _settings(max_equity_positions=3, max_total_stock_weight=0.60)
    scores = {f"inst_{i}": _hs(0.08 - i * 0.005) for i in range(8)}
    out = _alloc(s, scores)
    assert len(out["payload"]["stock_weights"]) <= 3
    assert len(out["payload"]["shortlist_order"]) == 3
    skipped = [r for r in out["recommendations"] if r.exclusion_reason == "MAX_EQUITY_POSITIONS"]
    assert len(skipped) == 5


def test_derive_action_from_final_target() -> None:
    assert derive_stock_action(0, 4.28, min_position_units=12, min_delta_units=6) == RecommendationAction.NO_ACTION
    assert derive_stock_action(0, 48, min_position_units=12, min_delta_units=6) == RecommendationAction.ENTER
    assert derive_stock_action(40, 80, min_position_units=12, min_delta_units=6) == RecommendationAction.ADD
    assert derive_stock_action(40, 41, min_position_units=12, min_delta_units=6) == RecommendationAction.HOLD
    assert derive_stock_action(80, 20, min_position_units=12, min_delta_units=6) == RecommendationAction.REDUCE
    assert derive_stock_action(40, 0, min_position_units=12, min_delta_units=6) == RecommendationAction.EXIT


# ---------------------------------------------------------------------------
# technical_sizing_multiplier (2026-09-04): chart-quality reads now move real
# position size for new entries, not just a display badge.
# ---------------------------------------------------------------------------


def test_j_bad_entry_timing_shrinks_a_new_entry_relative_to_a_good_one() -> None:
    """Two names with identical opportunity_score (same mean/conf/agreement) but
    opposite chart-quality reads must NOT size equally any more -- the CRM case
    (leader_score=80, buyable_score=20) shrinks; a name with a good entry read grows."""
    s = _settings(max_total_stock_weight=0.50, max_single_stock_weight=0.40)
    scores = {"inst_bad_entry": _hs(0.05), "inst_good_entry": _hs(0.05)}
    out = _alloc(
        s,
        scores,
        stock_technical={
            "inst_bad_entry": {
                "buyable_score": 20.0, "leader_score": 80.0,
                "quality_of_trend_score": 50.0, "breakout_score": 0.0, "catalyst_score": 50.0,
            },
            "inst_good_entry": {
                "buyable_score": 90.0, "leader_score": 80.0,
                "quality_of_trend_score": 70.0, "breakout_score": 60.0, "catalyst_score": 65.0,
            },
        },
    )
    w = out["payload"]["stock_weights"]
    diag = out["payload"]["diagnostics"]
    assert w["inst_good_entry"] > w["inst_bad_entry"]
    assert diag["inst_bad_entry"]["technical_sizing_multiplier"] < 1.0
    assert diag["inst_good_entry"]["technical_sizing_multiplier"] > 1.0


def test_j_neutral_technical_reads_do_not_change_sizing_at_all() -> None:
    """All-50/no-flag technical inputs must map to exactly 1.0x (see
    technical_sizing_multiplier's docstring) -- providing them changes nothing."""
    s = _settings(max_total_stock_weight=0.50, max_single_stock_weight=0.40)
    scores = {"inst_a": _hs(0.05), "inst_b": _hs(0.05)}
    baseline = _alloc(s, scores)
    with_neutral_tech = _alloc(
        s,
        scores,
        stock_technical={
            "inst_a": {
                "buyable_score": 50.0, "leader_score": 50.0,
                "quality_of_trend_score": 50.0, "breakout_score": 0.0, "catalyst_score": 50.0,
            },
            "inst_b": {
                "buyable_score": 50.0, "leader_score": 50.0,
                "quality_of_trend_score": 50.0, "breakout_score": 0.0, "catalyst_score": 50.0,
            },
        },
    )
    assert with_neutral_tech["payload"]["stock_weights"] == pytest.approx(baseline["payload"]["stock_weights"])


def test_j_held_position_sizing_is_never_touched_by_technical_reads() -> None:
    """A bad technical read on a name the user already owns must not shrink its
    ADD/HOLD sizing -- only NEW entries are ever nudged (see allocate()'s
    _tech_multiplier_for, which returns 1.0 whenever current holdings > 0)."""
    s = _settings(max_total_stock_weight=0.50, max_single_stock_weight=0.40)
    scores = {"inst_held": _hs(0.05)}
    held_units = {"inst_held": 100.0}
    without_tech = _alloc(s, scores, stock_units=held_units)
    with_bad_tech = _alloc(
        s,
        scores,
        stock_units=held_units,
        stock_technical={
            "inst_held": {
                "buyable_score": 0.0, "leader_score": 0.0,
                "quality_of_trend_score": 0.0, "breakout_score": 0.0, "catalyst_score": 0.0,
            },
        },
    )
    assert with_bad_tech["payload"]["stock_weights"] == pytest.approx(without_tech["payload"]["stock_weights"])


def test_j_technical_sizing_multiplier_is_neutral_at_midpoint_and_bounded() -> None:
    from trading_system.allocation import technical_sizing_multiplier as calc_mult

    assert calc_mult(None) == pytest.approx(1.0)
    assert calc_mult(
        {"buyable_score": 50.0, "leader_score": 50.0, "quality_of_trend_score": 50.0,
         "breakout_score": 0.0, "catalyst_score": 50.0}
    ) == pytest.approx(1.0)
    worst = calc_mult(
        {"buyable_score": 0.0, "leader_score": 0.0, "quality_of_trend_score": 0.0,
         "breakout_score": 0.0, "catalyst_score": 0.0}
    )
    best = calc_mult(
        {"buyable_score": 100.0, "leader_score": 100.0, "quality_of_trend_score": 100.0,
         "breakout_score": 100.0, "catalyst_score": 100.0}
    )
    assert 0.5 <= worst < 1.0 < best <= 1.6


# ---------------------------------------------------------------------------
# research_sizing_multiplier (2026-09-04): Gemini research+adversarial evidence pack
# now moves real new-entry sizing too, in a narrower band than the technical read
# because it is an LLM's own scoring of its own prose, not a verifiable indicator.
# ---------------------------------------------------------------------------


def _pack(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "web_search_used": True,
        "data_quality_score": 80.0,
        "rerating_score": 50.0,
        "valuation_support_score": 50.0,
        "cash_relative_score": 50.0,
        "bearish_severity_score": 0.0,
    }
    base.update(over)
    return base


def test_j_research_sizing_multiplier_is_neutral_by_default_and_bounded() -> None:
    from trading_system.allocation import research_sizing_multiplier as calc_mult

    assert calc_mult(None) == pytest.approx(1.0)
    # Not researched / Gemini not configured / no live web search -> no opinion.
    assert calc_mult({}) == pytest.approx(1.0)
    assert calc_mult(_pack(web_search_used=False)) == pytest.approx(1.0)
    # Self-reported low confidence -> no opinion, even with a bullish score.
    assert calc_mult(_pack(data_quality_score=10.0, rerating_score=100.0)) == pytest.approx(1.0)
    # All-neutral (50/50/50, no bearish objections) -> exactly 1.0x.
    assert calc_mult(_pack()) == pytest.approx(1.0)

    worst = calc_mult(_pack(rerating_score=0.0, valuation_support_score=0.0, cash_relative_score=0.0))
    best = calc_mult(_pack(rerating_score=100.0, valuation_support_score=100.0, cash_relative_score=100.0))
    assert 0.75 <= worst < 1.0 < best <= 1.25

    # Strong adversarial objections must discount a bullish read back toward neutral.
    bullish_uncontested = calc_mult(_pack(rerating_score=100.0, valuation_support_score=100.0, cash_relative_score=100.0, bearish_severity_score=0.0))
    bullish_contested = calc_mult(_pack(rerating_score=100.0, valuation_support_score=100.0, cash_relative_score=100.0, bearish_severity_score=100.0))
    assert bullish_contested == pytest.approx(1.0)
    assert bullish_contested < bullish_uncontested


def test_j_research_reads_move_sizing_and_never_touch_held_positions() -> None:
    s = _settings(max_total_stock_weight=0.50, max_single_stock_weight=0.40)
    scores = {"inst_bull": _hs(0.05), "inst_bear": _hs(0.05)}
    out = _alloc(
        s,
        scores,
        stock_research={
            "inst_bull": _pack(rerating_score=100.0, valuation_support_score=100.0, cash_relative_score=100.0),
            "inst_bear": _pack(rerating_score=0.0, valuation_support_score=0.0, cash_relative_score=0.0),
        },
    )
    w = out["payload"]["stock_weights"]
    diag = out["payload"]["diagnostics"]
    assert w["inst_bull"] > w["inst_bear"]
    assert diag["inst_bull"]["research_sizing_multiplier"] > 1.0
    assert diag["inst_bear"]["research_sizing_multiplier"] < 1.0

    held_units = {"inst_bear": 100.0}
    without_research = _alloc(s, {"inst_bear": _hs(0.05)}, stock_units=held_units)
    with_bad_research = _alloc(
        s,
        {"inst_bear": _hs(0.05)},
        stock_units=held_units,
        stock_research={"inst_bear": _pack(rerating_score=0.0, valuation_support_score=0.0, cash_relative_score=0.0)},
    )
    assert with_bad_research["payload"]["stock_weights"] == pytest.approx(without_research["payload"]["stock_weights"])
