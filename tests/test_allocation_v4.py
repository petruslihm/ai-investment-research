"""Equity deployment factor: strength-based book size, not a filled cap."""

from __future__ import annotations

import pytest

from trading_system.allocation import (
    allocate,
    allocation_score,
    allocation_trace_from_quant_recs,
    deployment_factor,
    implied_strong_opportunity_score,
    name_conviction,
    opportunity_score,
)
from trading_system.btc_sleeve import BtcLiquidityState, BtcSleeveState
from trading_system.config import Settings
from trading_system.ids import new_tick_id
from trading_system.recommendations import RecommendationAction
from trading_system.market.registry import stable_instrument_id


def _hs(x: float) -> dict[int, float]:
    return {5: x, 10: x, 20: x}


def _mean_for_opp(opp: float) -> float:
    """Invert opportunity_score(mean, 1, agree) = mean * 1.15 (vol no longer scores,
    see allocation.opportunity_score's 2026-09-04 docstring note)."""
    return float(opp) / 1.15


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
        "max_total_stock_weight": 1.0,
        "min_cash_weight": 0.0,
        "min_delta_weight": 0.005,
        "max_btc_weight": 0.30,
        "strong_horizon_mean_return": 0.04,
        "strong_opportunity_score": None,
        "risk_appetite": "conservative",
        "sizing_alpha": 1.0,
    }
    kw.update(over)
    return Settings(**kw)  # type: ignore[arg-type]


def _alloc(settings: Settings, scores: dict[str, dict[int, float]], **kwargs: object) -> dict:
    vol = {k: 0.02 for k in scores}
    return allocate(
        stock_scores=scores,
        stock_vol=vol,
        btc_scores=kwargs.pop("btc_scores", {5: -0.1, 10: -0.1, 20: -0.1}),  # type: ignore[arg-type]
        settings=settings,
        btc_state=kwargs.pop(
            "btc_state",
            BtcSleeveState(
                instrument_id=stable_instrument_id("BTC/USD"),
                liquidity=BtcLiquidityState.UNAVAILABLE,
                current_units=0,
            ),
        ),
        tick_id=str(new_tick_id()),
        epoch_id="epoch_v4",
        feature_snapshot_id="fs_v4",
        **kwargs,  # type: ignore[arg-type]
    )


def test_strong_score_is_derived_and_documented() -> None:
    derived = implied_strong_opportunity_score(0.04)
    assert derived == pytest.approx(opportunity_score(0.04, 1.0, 0.02, "agree"))
    out = _alloc(_settings(), {"inst_a": _hs(0.08)})
    info = out["payload"]["strong_opportunity"]
    assert info["v1_status"] == "placeholder_not_oos_calibrated"
    assert info["source"] == "derived_from_strong_horizon_mean_return"
    assert info["used_score"] == pytest.approx(derived)
    assert "not OOS-calibrated" in str(info["why"])


def test_many_barely_qualified_names_keep_equity_low() -> None:
    scores = {f"inst_{i}": _hs(_mean_for_opp(0.0105)) for i in range(11)}
    out = _alloc(_settings(), scores)
    p = out["payload"]
    used = sum(p["stock_weights"].values())
    assert p["deployment_factor"] < 0.25
    assert used < 0.25
    assert used < p["max_total_stock_weight"]
    assert p["residual_cash_reason"] == "LOW_CONVICTION"
    assert "기회 강도가 낮아 현금 유지" in p["residual_cash_label"]
    assert "최대 100% 중" in p["deployment_note"]


def test_one_barely_qualified_name_is_not_max_cap() -> None:
    out = _alloc(_settings(), {"inst_pltr": _hs(_mean_for_opp(0.0101))})
    p = out["payload"]
    used = sum(p["stock_weights"].values())
    assert p["deployment_factor"] < 0.05
    assert used < 0.05
    rec = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_pltr")
    assert rec.action.value in {"NO_ACTION", "ENTER"}
    if rec.action.value == "ENTER":
        assert float(rec.recommended_units or 0) < 50


def test_several_strong_names_deploy_more_than_barely() -> None:
    barely = _alloc(_settings(), {f"inst_{i}": _hs(_mean_for_opp(0.0105)) for i in range(4)})
    strong = _alloc(_settings(), {f"inst_{i}": _hs(0.08) for i in range(4)})
    assert strong["payload"]["deployment_factor"] > barely["payload"]["deployment_factor"] + 0.3
    assert sum(strong["payload"]["stock_weights"].values()) > sum(barely["payload"]["stock_weights"].values()) + 0.2


def test_stronger_scores_monotonically_increase_deployment() -> None:
    levels = [0.012, 0.018, 0.025, 0.04]
    factors = []
    for opp in levels:
        out = _alloc(_settings(), {f"inst_{i}": _hs(_mean_for_opp(opp)) for i in range(3)})
        factors.append(out["payload"]["deployment_factor"])
    assert factors == sorted(factors)
    assert factors[-1] >= factors[0]


def test_threshold_crossing_is_not_a_huge_jump() -> None:
    below = _alloc(_settings(), {"inst_a": _hs(_mean_for_opp(0.0099))})
    above = _alloc(_settings(), {"inst_a": _hs(_mean_for_opp(0.0101))})
    w_below = sum(below["payload"]["stock_weights"].values())
    w_above = sum(above["payload"]["stock_weights"].values())
    assert w_below == pytest.approx(0.0)
    assert w_above < 0.03
    assert above["payload"]["deployment_factor"] < 0.05


def test_equity_never_exceeds_max_total_stock_weight() -> None:
    out = _alloc(_settings(max_total_stock_weight=0.50), {f"inst_{i}": _hs(0.12) for i in range(8)})
    used = sum(out["payload"]["stock_weights"].values())
    assert used <= 0.50 + 1e-9
    assert out["payload"]["equity_budget"] <= 0.50 + 1e-9


def test_min_cash_survives_high_deployment() -> None:
    s = _settings(min_cash_weight=0.20, max_total_stock_weight=0.90)
    out = _alloc(s, {f"inst_{i}": _hs(0.10) for i in range(6)})
    assert out["payload"]["cash_weight"] + 1e-9 >= 0.20
    assert sum(out["payload"]["stock_weights"].values()) <= 0.80 + 1e-9


def test_conviction_is_zero_at_threshold_and_one_at_strong() -> None:
    strong = implied_strong_opportunity_score(0.04)
    excess0, conv0 = name_conviction(0.01, 0.01, strong)
    _, conv1 = name_conviction(strong, 0.01, strong)
    assert excess0 == pytest.approx(0.0)
    assert conv0 == pytest.approx(0.0)
    assert conv1 == pytest.approx(1.0)
    agg, factor = deployment_factor([0.02] * 20, equivalent_names=3.2)
    assert factor < 0.2
    _, strong_factor = deployment_factor([1.0, 1.0, 1.0, 1.0], equivalent_names=3.2)
    assert strong_factor == pytest.approx(1.0)
    assert agg == pytest.approx(0.4)


def test_allocation_trace_has_preds_cutoff_and_percentiles() -> None:
    scores = {
        "inst_nvda": _hs(_mean_for_opp(0.018)),
        "inst_msft": _hs(_mean_for_opp(0.0097)),
        "inst_aapl": _hs(_mean_for_opp(0.012)),
    }
    out = _alloc(_settings(), scores)
    trace = out["payload"]["allocation_trace"]
    assert trace["pred_meaning"] == "expected_return_fraction"
    assert trace["cutoff"] == pytest.approx(0.01)
    assert trace["universe"] == 3
    assert trace["passed_cutoff"] == 2
    by_t = {r["ticker"]: r for r in trace["rows"]}
    assert by_t["NVDA"]["pass_cutoff"] is True
    assert by_t["MSFT"]["pass_cutoff"] is False
    assert by_t["NVDA"]["pred_5d"] == pytest.approx(_mean_for_opp(0.018))
    assert "=== ALLOCATION TRACE ===" in trace["log_text"]
    recs = [r.model_dump(mode="json") for r in out["recommendations"]]
    again = allocation_trace_from_quant_recs(recs, out["payload"])
    assert again["passed_cutoff"] == 2
    assert again["universe"] == 3


def test_nine_barely_qualified_names_compress_equity_budget() -> None:
    scores = {f"inst_{i}": _hs(_mean_for_opp(0.0109)) for i in range(9)}
    out = _alloc(_settings(), scores)
    p = out["payload"]
    trace = p["allocation_trace"]
    assert trace["passed_cutoff"] == 9
    # 2026-09-04: opportunity_score() dropped its vol term, which raises every score's
    # scale (see allocation.opportunity_score's docstring) -- these "barely qualified"
    # names now clear the min_opportunity_score cutoff by less relative headroom
    # against the (also un-rescaled) strong_horizon_mean_return reference, so
    # deployment compresses harder than the pre-fix ~0.16. The compression behavior
    # this test exists to verify (barely-qualified names stay far from full
    # deployment) still holds -- only the exact magnitude moved.
    assert p["deployment_factor"] == pytest.approx(0.056, abs=0.01)
    used = sum(p["stock_weights"].values())
    assert used == pytest.approx(0.056, abs=0.01)
    assert used == pytest.approx(p["equity_budget"], abs=0.02)
    assert p["residual_cash_reason"] == "LOW_CONVICTION"
    assert trace["diagnosis"]["sizing_compression"] is True
    recs = [r.model_dump(mode="json") for r in out["recommendations"]]
    again = allocation_trace_from_quant_recs(recs, {k: v for k, v in p.items() if k != "allocation_trace"})
    assert again["passed_cutoff"] == 9
    assert again["deployment_factor"] == pytest.approx(p["deployment_factor"], abs=1e-9)
    assert again["stock_weight"] == pytest.approx(used, abs=1e-9)
    held = next(r.model_dump(mode="json") for r in out["recommendations"])
    held["instrument_id"] = "inst_hubb"
    held["recommended_units"] = 400
    held["current_units"] = 400
    held["exclusion_reason"] = "HELD_WITHOUT_SCORE"
    held["opportunity_score"] = 0.0
    bloated = allocation_trace_from_quant_recs(
        recs + [held],
        {k: v for k, v in p.items() if k != "allocation_trace"},
    )
    assert bloated["stock_weight"] == pytest.approx(used, abs=1e-9)


def test_aggressive_deploys_more_than_conservative() -> None:
    scores = {f"inst_{i}": _hs(_mean_for_opp(0.0109)) for i in range(9)}
    cons = _alloc(_settings(risk_appetite="conservative", sizing_alpha=1.0), scores)
    agg = _alloc(_settings(risk_appetite="aggressive", sizing_alpha=1.5), scores)
    assert agg["payload"]["full_deployment_equivalent_names"] == pytest.approx(1.0)
    assert cons["payload"]["full_deployment_equivalent_names"] == pytest.approx(4.0)
    # 2026-09-04: absolute deployment_factor values shrank after opportunity_score()
    # dropped its vol term (see the compress_equity_budget test above) -- the
    # aggressive/conservative gap this test checks is still a clean 4x, just off a
    # smaller base, so the margin is scaled down accordingly.
    assert agg["payload"]["deployment_factor"] > cons["payload"]["deployment_factor"] + 0.1
    assert sum(agg["payload"]["stock_weights"].values()) > sum(cons["payload"]["stock_weights"].values()) + 0.1


def test_sizing_uses_excess_not_raw_opportunity() -> None:
    scores = {
        "inst_dy": _hs(_mean_for_opp(0.0125)),
        "inst_cde": _hs(_mean_for_opp(0.0101)),
    }
    out = _alloc(_settings(risk_appetite="aggressive", sizing_alpha=1.5, min_position_weight=0.0), scores)
    w = out["payload"]["stock_weights"]
    assert w["inst_dy"] > 8.0 * w["inst_cde"]
    assert allocation_score(0.0025, 1.5) > 20 * allocation_score(0.0001, 1.5)
    assert out["payload"]["risk_appetite"] == "aggressive"
    assert out["payload"]["allocation_trace"]["vol_unit"]
    assert out["payload"]["formula_version"] == "quant_alloc_v8"


def test_stock_horizon_influence_actually_moves_sizing() -> None:
    """End-to-end: two names whose flat 5/10/20 mean (and thus opportunity_score) is
    identical must diverge once horizon_influence says 5d is more trustworthy than
    10d/20d -- proving allocate() actually consumes the weight instead of just
    plumbing ensemble_state's horizon_influence_json through unused, which was the bug
    (allocation.py's blend used to be a flat mean() no matter what horizon_influence said).
    Read from diagnostics (pre-position-cap) since both names are strong enough here to
    saturate max_single_stock_weight, which would otherwise mask the difference."""
    scores = {
        "inst_five_strong": {5: 0.05, 10: 0.01, 20: 0.01},
        "inst_twenty_strong": {5: 0.01, 10: 0.01, 20: 0.05},
    }
    settings = _settings(risk_appetite="aggressive", sizing_alpha=1.0, min_position_weight=0.0)

    flat = _alloc(settings, scores)
    diag_flat = flat["payload"]["diagnostics"]
    assert diag_flat["inst_five_strong"]["horizon_mean"] == pytest.approx(
        diag_flat["inst_twenty_strong"]["horizon_mean"], rel=1e-6
    )
    assert diag_flat["inst_five_strong"]["opportunity_score"] == pytest.approx(
        diag_flat["inst_twenty_strong"]["opportunity_score"], rel=1e-6
    )

    favor_5d = _alloc(
        settings,
        scores,
        stock_horizon_influence={"5": 0.6, "10": 0.2, "20": 0.2},
    )
    diag_5d = favor_5d["payload"]["diagnostics"]
    assert diag_5d["inst_five_strong"]["horizon_mean"] > diag_5d["inst_twenty_strong"]["horizon_mean"]
    assert diag_5d["inst_five_strong"]["opportunity_score"] > diag_5d["inst_twenty_strong"]["opportunity_score"]


def test_held_below_buy_gate_is_kept_not_exited() -> None:
    scores = {
        "inst_goog": _hs(0.005),
        "inst_iag": _hs(_mean_for_opp(0.012)),
    }
    out = _alloc(
        _settings(risk_appetite="aggressive"),
        scores,
        stock_units={"inst_goog": 221.0},
        total_base_units=1200.0,
    )
    goog = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_goog")
    iag = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_iag")
    assert goog.action == RecommendationAction.HOLD
    assert goog.recommended_units == pytest.approx(221.0)
    assert goog.exclusion_reason == "HOLD_NOT_NEW_BUY"
    assert "보유를 유지" in (goog.allocation_note or "")
    assert iag.recommended_units > 0
    assert out["payload"]["stock_weights"]["inst_goog"] == pytest.approx(221.0 / 1200.0)
    assert out["payload"]["held_keep_weight"] == pytest.approx(221.0 / 1200.0)
    assert out["payload"]["stock_weights"]["inst_goog"] + out["payload"]["stock_weights"]["inst_iag"] == pytest.approx(
        sum(out["payload"]["stock_weights"].values())
    )
    goog_row = next(
        r for r in out["payload"]["allocation_trace"]["rows"] if r["ticker"] == "GOOG"
    )
    assert goog_row["exclusion_reason"] == "HOLD_NOT_NEW_BUY"
    assert goog_row["pass_cutoff"] is False
    assert goog_row["final_weight"] == pytest.approx(221.0 / 1200.0)
    recs = [r.model_dump(mode="json") for r in out["recommendations"]]
    again = allocation_trace_from_quant_recs(
        recs,
        {k: v for k, v in out["payload"].items() if k != "allocation_trace"},
    )
    again_goog = next(r for r in again["rows"] if r["ticker"] == "GOOG")
    assert again_goog["exclusion_reason"] == "HOLD_NOT_NEW_BUY"
    assert again["held_keep_weight"] == pytest.approx(221.0 / 1200.0)


def test_tiny_held_below_min_position_is_still_kept() -> None:
    out = _alloc(
        _settings(risk_appetite="aggressive"),
        {"inst_pltr": _hs(0.005)},
        stock_units={"inst_pltr": 5.0},
        total_base_units=1200.0,
    )
    rec = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_pltr")
    assert rec.action == RecommendationAction.HOLD
    assert rec.recommended_units == pytest.approx(5.0)
    assert out["payload"]["stock_weights"]["inst_pltr"] == pytest.approx(5.0 / 1200.0)
    assert rec.exclusion_reason == "HOLD_NOT_NEW_BUY"


def test_held_negative_outlook_exits() -> None:
    out = _alloc(
        _settings(risk_appetite="aggressive"),
        {"inst_amd": _hs(-0.02)},
        stock_units={"inst_amd": 32.0},
        total_base_units=1200.0,
    )
    rec = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_amd")
    assert rec.action == RecommendationAction.EXIT
    assert rec.recommended_units == pytest.approx(0.0)
    assert rec.exclusion_reason == "EXIT_NEGATIVE_OUTLOOK"
    assert "inst_amd" not in out["payload"]["stock_weights"]


def test_held_with_incomplete_horizons_is_not_forced_to_exit() -> None:
    """A single negative horizon out of 3, with the other 2 missing (not zero), must not
    average to a fabricated negative mean and force-liquidate a held position -- that data
    is INVALID_HORIZONS, not a genuine negative-outlook signal."""
    out = _alloc(
        _settings(risk_appetite="aggressive"),
        {"inst_amd": {10: -0.05, 20: -0.05}},  # horizon 5 missing entirely
        stock_units={"inst_amd": 32.0},
        total_base_units=1200.0,
    )
    rec = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_amd")
    assert rec.action == RecommendationAction.HOLD
    assert rec.recommended_units == pytest.approx(32.0)
    assert rec.exclusion_reason != "EXIT_NEGATIVE_OUTLOOK"
    assert out["payload"]["stock_weights"]["inst_amd"] == pytest.approx(32.0 / 1200.0)


def test_max_equity_positions_overflow_fails_trace_cutoff() -> None:
    scores = {
        "inst_a": _hs(_mean_for_opp(0.02)),
        "inst_b": _hs(_mean_for_opp(0.018)),
    }
    out = _alloc(_settings(max_equity_positions=1), scores)
    trace = out["payload"]["allocation_trace"]
    by_t = {r["ticker"]: r for r in trace["rows"]}
    assert by_t["A"]["pass_cutoff"] is True
    assert by_t["B"]["exclusion_reason"] == "MAX_EQUITY_POSITIONS"
    assert by_t["B"]["pass_cutoff"] is False
    assert trace["passed_cutoff"] == 1


def test_kept_holding_overflow_evicts_via_cash_floor_not_fabricated_weights() -> None:
    """When kept_total + btc_w > 1 and the BTC sleeve is NOT transfer-blocked, the honest
    fix is to let the pre-existing CASH_FLOOR eviction (kept positions evicted last, see
    the sort key in allocation.py) restore cash -- not to silently shrink the reported
    stock weight for a position whose real recommendation (rec_u=cur) never changes.
    This asserts the two stay consistent: whatever recs_out says is exactly what the
    payload weights show, with no phantom weight left behind after eviction."""
    out = _alloc(
        _settings(risk_appetite="aggressive", max_btc_weight=0.30, min_cash_weight=0.0),
        {},
        stock_units={"inst_x": 1150.0},
        total_base_units=1200.0,
        btc_scores=_hs(0.5),
        btc_state=BtcSleeveState(
            instrument_id=stable_instrument_id("BTC/USD"),
            liquidity=BtcLiquidityState.AVAILABLE,
            current_units=0.0,
        ),
    )
    p = out["payload"]
    assert p["btc_weight"] == pytest.approx(0.30)
    rec = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_x")
    reported_units = p["stock_weights"].get("inst_x", 0.0) * p["total_base_units"]
    assert reported_units == pytest.approx(float(rec.recommended_units or 0.0))
    if rec.action == RecommendationAction.EXIT:
        assert "inst_x" not in p["stock_weights"]
    else:
        assert p["stock_weights"]["inst_x"] == pytest.approx(1150.0 / 1200.0)
    total = sum(p["stock_weights"].values()) + p["btc_weight"] + p["cash_weight"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_kept_holding_overflow_blocked_btc_forces_honest_exit_not_fabricated_cash() -> None:
    """A transfer-blocked BTC sleeve locks btc_w (can't be rebalanced), but that must
    never excuse skipping stock cash-floor eviction when cash_w is genuinely negative
    (kept_total + btc_w > 1.0) -- the payload identity stock_w + btc_w + cash_w == 1.0
    is a hard invariant (asserted in allocate()), not a preference. The locked BTC
    units stay untouched; the over-committed stock position is honestly recommended
    for exit (the same weakest-first CASH_FLOOR mechanism used in the unblocked case),
    and cash is never fabricated to paper over the shortfall."""
    out = _alloc(
        _settings(risk_appetite="aggressive", min_cash_weight=0.0),
        {},
        stock_units={"inst_x": 900.0},
        total_base_units=1200.0,
        btc_scores=_hs(0.5),
        btc_state=BtcSleeveState(
            instrument_id=stable_instrument_id("BTC/USD"),
            liquidity=BtcLiquidityState.TRANSFER_PENDING,
            current_units=400.0,
        ),
    )
    p = out["payload"]
    # Locked BTC units are untouched by the blocked sleeve.
    assert p["btc_weight"] == pytest.approx(400.0 / 1200.0)
    btc_rec = next(r for r in out["recommendations"] if "btc" in str(r.instrument_id).lower())
    assert btc_rec.recommended_units == pytest.approx(400.0)
    rec = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_x")
    assert rec.action == RecommendationAction.EXIT
    assert rec.recommended_units == pytest.approx(0.0)
    assert rec.exclusion_reason == "CASH_FLOOR"
    assert "inst_x" not in p["stock_weights"]
    assert p["held_keep_weight"] == pytest.approx(0.0)
    assert p["allocation_trace"]["held_keep_weight"] == pytest.approx(0.0)
    assert p["cash_weight"] == pytest.approx(1.0 - 400.0 / 1200.0)
    total = sum(p["stock_weights"].values()) + p["btc_weight"] + p["cash_weight"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_soft_min_cash_preference_still_skipped_while_btc_transfer_blocked() -> None:
    """The blocked-BTC bypass above must stay scoped to genuine invariant emergencies
    (cash_w < 0) -- it must not also force eviction merely because cash_w sits below
    the soft min_cash preference while still non-negative. That softer skip-while-
    blocked behavior is untouched: kept 0.25 (inst_x) + locked btc 0.25 leaves a
    non-negative cash_w of 0.50, which is below min_cash_weight=0.6 (a preference,
    not an emergency) but must NOT trigger eviction while blocked."""
    out = _alloc(
        _settings(risk_appetite="aggressive", min_cash_weight=0.6, max_total_stock_weight=1.0),
        {},
        stock_units={"inst_x": 300.0},
        total_base_units=1200.0,
        btc_scores=_hs(0.5),
        btc_state=BtcSleeveState(
            instrument_id=stable_instrument_id("BTC/USD"),
            liquidity=BtcLiquidityState.TRANSFER_PENDING,
            current_units=300.0,
        ),
    )
    p = out["payload"]
    assert p["btc_weight"] == pytest.approx(0.25)
    rec = next(r for r in out["recommendations"] if str(r.instrument_id) == "inst_x")
    assert rec.action == RecommendationAction.HOLD
    assert rec.recommended_units == pytest.approx(300.0)
    assert p["stock_weights"]["inst_x"] == pytest.approx(300.0 / 1200.0)
    assert p["cash_weight"] == pytest.approx(0.5)
    total = sum(p["stock_weights"].values()) + p["btc_weight"] + p["cash_weight"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_unblocked_btc_respects_min_cash_weight_with_no_stocks() -> None:
    """An unblocked BTC sleeve is being freshly sized, so it must never claim more
    than 1 - min_cash_weight on its own -- otherwise a name-less book (no stock
    candidates qualify at all) could leave cash below the floor with nothing left
    to evict, since BTC itself was never subject to the cash-floor mechanism."""
    out = _alloc(
        _settings(risk_appetite="aggressive", min_cash_weight=0.4, max_btc_weight=1.0),
        {},
        total_base_units=1200.0,
        btc_scores=_hs(0.5),
        btc_state=BtcSleeveState(
            instrument_id=stable_instrument_id("BTC/USD"),
            liquidity=BtcLiquidityState.AVAILABLE,
            current_units=0.0,
        ),
    )
    p = out["payload"]
    # Uncapped-by-min-cash, btc_conf=1.0 (agree) * mean(0.5)*2 = 1.0, only the
    # max_btc_weight=1.0 cap would otherwise bind -- min_cash_weight=0.4 must bind
    # first, capping btc_weight at 1 - 0.4 = 0.6.
    assert p["btc_weight"] == pytest.approx(0.6)
    assert p["cash_weight"] == pytest.approx(0.4)
    btc_rec = next(r for r in out["recommendations"] if "btc" in str(r.instrument_id).lower())
    assert btc_rec.recommended_units == pytest.approx(0.6 * 1200.0)
    total = sum(p["stock_weights"].values()) + p["btc_weight"] + p["cash_weight"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_unblocked_btc_min_cash_cap_does_not_bind_when_signal_is_weaker() -> None:
    """The new min_cash cap on unblocked BTC must not change sizing for the common
    case where the signal-derived weight is already below 1 - min_cash_weight."""
    out = _alloc(
        _settings(risk_appetite="aggressive", min_cash_weight=0.4, max_btc_weight=1.0),
        {},
        total_base_units=1200.0,
        btc_scores=_hs(0.1),  # mean*2*conf = 0.2, well under 1 - 0.4 = 0.6
        btc_state=BtcSleeveState(
            instrument_id=stable_instrument_id("BTC/USD"),
            liquidity=BtcLiquidityState.AVAILABLE,
            current_units=0.0,
        ),
    )
    p = out["payload"]
    assert p["btc_weight"] == pytest.approx(0.2)
    assert p["cash_weight"] == pytest.approx(0.8)


def test_blocked_btc_weight_is_unaffected_by_min_cash_weight() -> None:
    """A transfer-blocked sleeve's units are locked -- min_cash_weight must never
    resize it (only the cash-floor eviction path may respond, by shedding stock,
    which this scenario has none of to shed)."""
    out = _alloc(
        _settings(risk_appetite="aggressive", min_cash_weight=0.4, max_btc_weight=1.0),
        {},
        total_base_units=1200.0,
        btc_scores=_hs(0.5),
        btc_state=BtcSleeveState(
            instrument_id=stable_instrument_id("BTC/USD"),
            liquidity=BtcLiquidityState.TRANSFER_PENDING,
            current_units=900.0,  # 0.75 of base -- exceeds 1 - min_cash (0.6)
        ),
    )
    p = out["payload"]
    assert p["btc_weight"] == pytest.approx(900.0 / 1200.0)
    btc_rec = next(r for r in out["recommendations"] if "btc" in str(r.instrument_id).lower())
    assert btc_rec.recommended_units == pytest.approx(900.0)
    assert p["cash_weight"] == pytest.approx(1.0 - 900.0 / 1200.0)
    total = sum(p["stock_weights"].values()) + p["btc_weight"] + p["cash_weight"]
    assert total == pytest.approx(1.0, abs=1e-6)
