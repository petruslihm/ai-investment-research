"""Quant-only recommendation and unit-first allocation (no orders)."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from uuid import uuid4

from trading_system.btc_sleeve import BtcLiquidityState, BtcSleeveAction, BtcSleeveState
from trading_system.config import Settings
from trading_system.ids import USD_CASH_INSTRUMENT_ID, InstrumentId
from trading_system.recommendations import (
    HorizonOutlook,
    RecommendationAction,
    RecommendationRecord,
    quant_only_record,
)
from trading_system.technical_factors import CATALYST_SIGNAL_THRESHOLD


FORMULA_VERSION = "quant_alloc_v8"
# BTC sleeve keeps the v2 positive-outlook gate. Do not reuse stock min_opportunity_score.
_BTC_MIN_OUTLOOK = 0.0
_HORIZONS = (5, 10, 20)
# opportunity_score() no longer uses vol (2026-09-04, see its docstring), so this
# value is inert -- kept only because opportunity_score()'s signature still takes a
# vol argument for implied_vol_from_score()'s historical-trace reconstruction.
_STRONG_SCORE_VOL = 0.02
_STRONG_WHY = (
    "V1 stand-in, not OOS-calibrated: opportunity_score of strong_horizon_mean_return "
    "at confidence=1, agreeing 5/10/20 horizons. Replace after matured-label calibration."
)


def _weighted_horizon_mean(hs: dict[int, float], horizon_weights: dict[str, float] | None) -> float:
    """Mean of the present 5/10/20-day scores, weighted by ensemble_state's learned
    per-horizon trust when available. Falls back to a flat mean (equal weights, or a
    horizon missing/zero total weight) so behavior is unchanged until horizon_weights
    has actually adapted away from its 1/3-1/3-1/3 default."""
    present = [(h, v) for h in _HORIZONS if (v := _finite(hs.get(h))) is not None]
    if not present:
        return 0.0
    if not horizon_weights:
        return sum(v for _h, v in present) / len(present)
    weighted = [(max(0.0, float(horizon_weights.get(str(h), 0.0))), v) for h, v in present]
    wsum = sum(w for w, _v in weighted)
    if wsum <= 1e-9:
        return sum(v for _h, v in present) / len(present)
    return sum(w * v for w, v in weighted) / wsum


def horizon_agreement(scores: dict[int, float]) -> tuple[str, float]:
    vals = [scores[h] for h in _HORIZONS if h in scores]
    if len(vals) < 2:
        return "unknown", 0.0
    signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in vals]
    if all(s == signs[0] and s != 0 for s in signs):
        return "agree", min(abs(v) for v in vals)
    if 0 in signs:
        return "mixed", float(sum(vals) / len(vals))
    return "conflict", float(sum(vals) / len(vals))


def agreement_confidence(agr: str) -> float:
    return {"agree": 1.0, "mixed": 0.55, "conflict": 0.25, "unknown": 0.4}.get(agr, 0.4)


def opportunity_score(mean: float, conf: float, vol: float, agr: str) -> float:
    """Single score used for eligibility and sizing. Conf/agreement live here.

    `vol` is accepted (and still computed/stored/displayed by callers as a
    diagnostic -- see the `vol` field in allocate()'s per-name trace) but no
    longer divides the score. Real-scan audit (2026-09-04) found the old
    1/(1+vol/0.02) term rejected ~97% of names the technical-factor layer
    (technical_factors.py) independently rated as strong leaders/breakouts --
    TSLA, AAPL, CRM, SNOW, CRWD, VRTX, TEAM, DELL, HOOD among them -- purely
    because a genuinely strong recent move also raises 10-day realized vol.
    That penalty was filtering out exactly the momentum names this system is
    supposed to catch. technical_factors.py's own risk read (top_risk_score,
    the two hard gates) is what should catch a name that is volatile *because
    it is breaking down*, not this formula penalizing volatility on its own.
    """
    adj = float(mean) * _clamp01(conf)
    if agr == "agree":
        adj *= 1.15
    elif agr == "conflict":
        adj *= 0.7
    return adj


def implied_strong_opportunity_score(mean_return: float) -> float:
    """Map a 5/10/20 mean expected return onto the opportunity_score scale."""
    return opportunity_score(float(mean_return), 1.0, _STRONG_SCORE_VOL, "agree")


def resolve_strong_opportunity(settings: Settings, min_opp: float) -> dict[str, object]:
    mean = float(settings.strong_horizon_mean_return)
    derived = implied_strong_opportunity_score(mean)
    override = settings.strong_opportunity_score
    if override is None:
        used = derived
        source = "derived_from_strong_horizon_mean_return"
    else:
        used = float(override)
        source = "explicit_strong_opportunity_score"
    if used <= min_opp:
        used = min_opp + 1e-6
        source = f"{source}_raised_above_min"
    return {
        "strong_horizon_mean_return": mean,
        "derived_score": derived,
        "used_score": used,
        "source": source,
        "v1_status": "placeholder_not_oos_calibrated",
        "why": _STRONG_WHY,
    }


def name_conviction(score: float, min_opp: float, strong: float) -> tuple[float, float]:
    """Excess over the eligibility gate, scaled so a 'strong' score is conviction 1."""
    excess = max(0.0, float(score) - float(min_opp))
    span = max(float(strong) - float(min_opp), 1e-12)
    return excess, min(1.0, excess / span)


def deployment_factor(convictions: list[float], equivalent_names: float) -> tuple[float, float]:
    """Sum of convictions vs how many fully-strong names fill the equity cap.

    Barely-qualified names contribute near-zero, so count of weak names does not
    fill the book. Breadth of strong names does.
    """
    aggregate = float(sum(max(0.0, c) for c in convictions))
    k = max(float(equivalent_names), 1e-9)
    return aggregate, min(1.0, aggregate / k)


def resolve_risk_appetite(settings: Settings) -> dict[str, object]:
    name = str(getattr(settings, "risk_appetite", "aggressive") or "aggressive").strip().lower()
    if name not in {"aggressive", "balanced", "conservative"}:
        name = "aggressive"
    equiv = {"aggressive": 1.0, "balanced": 2.0, "conservative": None}[name]
    alpha = {"aggressive": 1.5, "balanced": 1.0, "conservative": 1.0}[name]
    return {
        "name": name,
        "equivalent_names": equiv,
        "sizing_alpha": alpha,
        "note": (
            "aggressive: one fully-strong name can fill the book; "
            "balanced: two; conservative: cap ratio (usually 4). "
            "min_opp/strong_score stay placeholders until matured-label calibration."
        ),
    }


def resolve_sizing_alpha(settings: Settings) -> tuple[float, str]:
    override = getattr(settings, "sizing_alpha", None)
    if override is not None:
        return max(0.0, float(override)), "explicit_sizing_alpha"
    appetite = resolve_risk_appetite(settings)
    return float(appetite["sizing_alpha"]), f"risk_appetite:{appetite['name']}"


def allocation_score(excess: float, alpha: float) -> float:
    """Cross-sectional size from threshold excess. alpha=0 equal-weights passers."""
    e = max(0.0, float(excess))
    a = float(alpha)
    if e <= 0.0:
        return 0.0
    if a <= 1e-12:
        return 1.0
    return e**a


_TECH_SIZING_MIN = 0.5
_TECH_SIZING_MAX = 1.6


def technical_sizing_multiplier(tech: dict[str, float] | None) -> float:
    """Scale a NEW entry's size by legacy-b style chart-quality reads (technical_
    factors.py) that opportunity_score's expected-return/confidence/agreement inputs
    never see. 2026-09-04: previously leader/buyable/breakout/quality_of_trend/
    catalyst scores were computed and shown but never touched sizing or ranking --
    only momentum_score and top_risk_score (via the two hard gates) had any real
    effect. This is the fix: a genuinely bad entry-timing read (e.g. CRM's
    buyable_score=20 on 2026-09-04, despite a leader_score of 80) now shrinks the
    position instead of only showing a badge nobody's money reacts to.

    Neutral inputs (buyable/leader/quality all 50, no breakout/catalyst flag) map to
    1.0x -- this only nudges the size opportunity_score already decided on, it never
    substitutes for it, and it is NEVER applied to an existing holding (see allocate()
    -- ADD/HOLD/EXIT sizing for names already owned is untouched).
    """
    if not tech:
        return 1.0

    def _get(key: str, default: float) -> float:
        # `x or default` would silently treat a genuine 0.0 (e.g. buyable_score=0,
        # the worst possible real reading) as "missing" and replace it with the
        # neutral default -- exactly backwards. Only an actually-missing/None/
        # non-numeric key should fall back.
        val = tech.get(key, default)
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    buyable = _get("buyable_score", 50.0)
    leader = _get("leader_score", 50.0)
    quality = _get("quality_of_trend_score", 50.0)
    breakout = _get("breakout_score", 0.0)
    catalyst = _get("catalyst_score", 50.0)
    mult = 1.0
    mult += (buyable - 50.0) / 100.0 * 0.5
    mult += (leader - 50.0) / 100.0 * 0.25
    mult += (quality - 50.0) / 100.0 * 0.25
    if breakout >= 50.0:
        mult += 0.1
    if catalyst >= CATALYST_SIGNAL_THRESHOLD:
        mult += 0.1
    return max(_TECH_SIZING_MIN, min(_TECH_SIZING_MAX, mult))


_RESEARCH_SIZING_MIN = 0.75
_RESEARCH_SIZING_MAX = 1.25
_RESEARCH_MIN_DATA_QUALITY = 40.0


def research_sizing_multiplier(pack: dict[str, object] | None) -> float:
    """Scale a NEW entry's size by the Gemini research + adversarial evidence pack
    (research_agent.research_ticker), the same way technical_sizing_multiplier scales
    it from chart-shape reads. Deliberately a much narrower band ([0.75, 1.25] vs
    technical's [0.5, 1.6]): the pack is an LLM's own scoring of its own prose, not an
    independently computed numeric indicator, so a hallucinated or low-confidence read
    must not be able to swing real position sizing as hard as a verifiable chart stat
    can. Neutral (1.0, no effect) unless the pack actually used live web search AND
    self-reported a data_quality_score of at least 40 -- anything else (not researched,
    Gemini not configured, budget-skipped, low-confidence pass) must default to no
    opinion, not a guess. Never applied to an existing holding (mirrors
    technical_sizing_multiplier -- see allocate()'s _research_multiplier_for).
    """
    if not isinstance(pack, dict):
        return 1.0
    if not bool(pack.get("web_search_used")):
        return 1.0

    def _get(key: str) -> float | None:
        val = pack.get(key)
        if val is None:
            return None
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        return max(0.0, min(100.0, v))

    quality = _get("data_quality_score")
    if quality is None or quality < _RESEARCH_MIN_DATA_QUALITY:
        return 1.0

    components = [
        v
        for v in (_get("rerating_score"), _get("valuation_support_score"), _get("cash_relative_score"))
        if v is not None
    ]
    if not components:
        return 1.0
    bullish = sum(components) / len(components)  # 0..100, 50 = neutral
    bearish = _get("bearish_severity_score") or 0.0

    # Bearish evidence may attenuate a positive tilt, never erase a negative one.
    # Preserve neutral=1 and the existing [0.75, 1.25] sizing scale.
    tilt = ((bullish - 50.0) / 50.0) * (quality / 100.0)
    if tilt > 0:
        tilt *= 1.0 - bearish / 100.0
    half_range = _RESEARCH_SIZING_MAX - 1.0
    raw = 1.0 + tilt * half_range
    return max(_RESEARCH_SIZING_MIN, min(_RESEARCH_SIZING_MAX, raw))


def resolve_equivalent_names(settings: Settings, max_stock: float, max_single: float) -> tuple[float, str]:
    override = settings.full_deployment_equivalent_names
    if override is not None and float(override) > 0:
        return float(override), "explicit_full_deployment_equivalent_names"
    appetite = resolve_risk_appetite(settings)
    preset = appetite["equivalent_names"]
    if preset is not None:
        return float(preset), f"risk_appetite:{appetite['name']}"
    if max_single > 1e-12:
        return max(max_stock / max_single, 1e-9), "max_total_stock_weight / max_single_stock_weight"
    return 1.0, "fallback_one_name"


def deployment_note_ko(
    *,
    qualified: int,
    max_stock: float,
    equity_budget: float,
    deployment: float,
    kept_total: float = 0.0,
) -> str:
    keep_bit = ""
    if kept_total > 1e-9:
        keep_bit = (
            f" 매수 기준 미달 보유 {kept_total * 100:.1f}%는 "
            "전망이 음수가 아니면 청산하지 않고 유지합니다."
        )
    if qualified <= 0:
        if kept_total > 1e-9:
            return "새 매수 후보는 없습니다." + keep_bit
        return "자격 있는 주식 기회가 없어 현금을 유지합니다."
    if deployment + 1e-9 < 1.0:
        return (
            "현재 후보들은 매수 기준은 통과했지만 신호 강도가 높지 않아 "
            f"최대 {max_stock * 100:.0f}% 중 {equity_budget * 100:.0f}%만 새 매수 예산으로 씁니다."
            + keep_bit
        )
    return f"기회 강도가 높아 주식 한도 {max_stock * 100:.0f}%까지 배분합니다." + keep_bit


def stock_action(score: float, threshold: float) -> RecommendationAction:
    """Legacy score-only mapping. Sizing uses derive_stock_action instead."""
    if score >= threshold + 0.01:
        return RecommendationAction.BUY
    if score <= -threshold - 0.01:
        return RecommendationAction.REDUCE
    return RecommendationAction.HOLD


def derive_stock_action(
    current: float,
    target: float,
    *,
    min_position_units: float,
    min_delta_units: float,
) -> RecommendationAction:
    """User-facing action from the final target, not from the raw score."""
    has_pos = current > 1e-12
    if not has_pos:
        return RecommendationAction.ENTER if target + 1e-12 >= min_position_units else RecommendationAction.NO_ACTION
    if target <= 1e-12:
        return RecommendationAction.EXIT
    delta = target - current
    if abs(delta) < min_delta_units:
        return RecommendationAction.HOLD
    if delta > 0:
        return RecommendationAction.ADD
    return RecommendationAction.REDUCE


def held_should_exit(row: dict[str, object]) -> bool:
    """Buy cutoff is not a sell cutoff. Exit held names only when expected return is negative
    and the horizon data backing that mean is actually complete -- missing horizons must not
    silently count as 0.0 and tip an incomplete mean negative."""
    hs = row.get("hs")
    if not isinstance(hs, dict) or not horizons_valid(hs):
        return False
    mean = _finite(row.get("mean"))
    if mean is None:
        return False
    return mean < 0.0


def btc_action(
    current: float,
    recommended: float,
    state: BtcSleeveState,
    settings: Settings,
    outlook: float,
) -> BtcSleeveAction:
    if state.transfer_blocks_immediate_rebalance():
        return BtcSleeveAction.NO_ACTION
    delta = recommended - current
    if abs(delta) < settings.btc_min_delta_units or abs(delta) < settings.btc_hysteresis * max(1.0, current):
        return BtcSleeveAction.HOLD if current > 0 else BtcSleeveAction.NO_ACTION
    if current <= 0 and recommended > 0 and outlook > 0:
        return BtcSleeveAction.ENTER
    if delta > 0:
        return BtcSleeveAction.ADD
    if recommended <= 0:
        return BtcSleeveAction.EXIT
    return BtcSleeveAction.REDUCE


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def _finite(value: object) -> float | None:
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(n):
        return None
    return n


def horizons_valid(hs: dict[int, float]) -> bool:
    for h in _HORIZONS:
        if h not in hs or _finite(hs.get(h)) is None:
            return False
    return True


_CUTOFF_FAIL_REASONS = frozenset(
    {
        "BELOW_MIN_OPPORTUNITY",
        "INVALID_HORIZONS",
        "ZERO_CONFIDENCE",
        "HELD_WITHOUT_SCORE",
        "HOLD_NOT_NEW_BUY",
        "EXIT_NEGATIVE_OUTLOOK",
        "MAX_EQUITY_POSITIONS",
    }
)
_SIZE_ONLY_REASONS = frozenset({"BELOW_MIN_POSITION", "CASH_FLOOR", "NO_EQUITY_BUDGET"})

_VOL_UNIT = (
    "vol_10 = 10-session std of daily close-to-close returns "
    "(not annualized; not matched to 5/10/20 holding-period mean)"
)
_SCORE_FORMULA_V4 = (
    "opp = mean * conf / (1 + vol/0.02) * {agree: 1.15, conflict: 0.7, else: 1.0}; "
    "mean = (pred_5d + pred_10d + pred_20d) / 3; "
    "pred_* = expected_return fraction (not probability, not z-score); "
    "pass if opp >= min_opportunity_score; "
    "conviction = clip((opp - min_opp) / (strong_score - min_opp), 0, 1); "
    "deploy = min(1, sum(shortlist convictions) / equivalent_names); "
    "equity_budget = max_equity_budget * deploy; "
    "stock_weight_i = equity_budget * opp_i / sum(shortlist opp), then single-name cap/floor"
)
_SCORE_FORMULA = (
    "opp = mean * conf / (1 + vol/0.02) * {agree: 1.15, conflict: 0.7, else: 1.0}; "
    "mean = (pred_5d + pred_10d + pred_20d) / 3; "
    "pred_* = expected_return fraction (not probability, not z-score); "
    "pass if opp >= min_opportunity_score; "
    "conviction = clip((opp - min_opp) / (strong_score - min_opp), 0, 1); "
    "deploy = min(1, sum(shortlist convictions) / equivalent_names); "
    "equity_budget = max_equity_budget * deploy; "
    "allocation_score = excess ** sizing_alpha; "
    "stock_weight_i = equity_budget * allocation_score_i / sum(allocation_score), then cap/floor; "
    "held names that fail the buy gate are kept unless 5/10/20 mean expected return is negative"
)


def _agreement_mult(agr: str) -> float:
    if agr == "agree":
        return 1.15
    if agr == "conflict":
        return 0.7
    return 1.0


def implied_vol_from_score(mean: float, conf: float, agr: str, adj: float) -> float | None:
    """Invert opportunity_score() when vol was not persisted."""
    if adj <= 1e-15:
        return None
    ratio = float(mean) * _clamp01(conf) * _agreement_mult(agr) / float(adj)
    vol = 0.02 * (ratio - 1.0)
    if not math.isfinite(vol):
        return None
    return max(1e-4, vol)


def _instrument_ticker(inst: str) -> str:
    s = str(inst or "")
    if s.startswith("inst_"):
        s = s[5:]
    return s.upper()


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * max(0.0, min(1.0, p))
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(sorted_vals[lo])
    w = k - lo
    return float(sorted_vals[lo]) * (1.0 - w) + float(sorted_vals[hi]) * w


def _pct_map(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(float(v) for v in values)
    return {
        "min": ordered[0] if ordered else None,
        "p50": _percentile(ordered, 0.50),
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1] if ordered else None,
        "n": len(ordered),
    }


def _horizons_from_rec(rec: dict) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in rec.get("horizons") or []:
        if not isinstance(item, dict):
            continue
        h = item.get("horizon")
        v = _finite(item.get("expected_return"))
        if h in _HORIZONS and v is not None:
            out[int(h)] = v
    return out


def _eligibility_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    if reason in _SIZE_ONLY_REASONS:
        return None
    return reason


def _diagnose_trace(
    *,
    score_p: dict[str, float | None],
    mean_p: dict[str, float | None],
    min_opp: float,
    n_passed: int,
    n_universe: int,
    deploy: float,
    equity_budget: float,
) -> dict[str, object]:
    p50 = score_p.get("p50")
    p90 = score_p.get("p90")
    p95 = score_p.get("p95")
    mean50 = mean_p.get("p50")
    notes: list[str] = []
    if mean50 is not None and mean50 < 0.005:
        model_returns = "low"
        notes.append("원시 예측 중앙값이 낮습니다 (5/10/20 평균 기대수익).")
    elif mean50 is not None and mean50 >= 0.015:
        model_returns = "high"
        notes.append("원시 예측 중앙값은 낮지 않습니다.")
    else:
        model_returns = "mixed"
        notes.append("원시 예측 중앙값은 중간 수준입니다.")
    score_max = score_p.get("max")
    mean75 = mean_p.get("p75")
    if score_max is not None and score_max + 1e-12 < min_opp:
        cutoff_vs = "above_max"
        notes.append("컷오프가 최대 점수보다 높아 사실상 전 종목이 탈락합니다.")
    elif p95 is not None and p95 + 1e-12 < min_opp:
        cutoff_vs = "cuts_top_5pct"
        notes.append("0.01 컷오프가 기회점수 상위 약 5% 이하만 통과시키는 위치입니다.")
    elif p50 is not None and p50 >= min_opp and n_passed <= max(12, n_universe // 20):
        cutoff_vs = "hidden_filter"
        notes.append("중앙값이 컷오프 이상인데 통과 종목이 적어 추가 필터를 의심할 수 있습니다.")
    elif n_passed <= max(12, n_universe // 30):
        cutoff_vs = "strict"
        notes.append("컷오프가 분포 대비 엄격해 통과 종목이 적습니다.")
    else:
        cutoff_vs = "many_pass"
        notes.append("컷오프를 통과한 종목이 적지 않습니다.")
    if mean75 is not None and mean75 >= min_opp and (p90 is None or p90 < min_opp):
        notes.append(
            "원시 기대수익이 1%를 넘는 종목이 꽤 있어도, vol/신뢰도 변환 뒤 기회점수는 0.01 아래로 떨어집니다. "
            "컷오프 0.01은 +1% 기대수익이 아닙니다."
        )
    compression = bool(n_passed >= 5 and deploy + 1e-9 < 0.35)
    if compression:
        notes.append(
            f"통과 {n_passed}개여도 확신도 합이 작아 주식 예산이 {equity_budget * 100:.1f}%로 압축됩니다. "
            "통과 개수와 투자 가능한 강한 후보는 다릅니다."
        )
    return {
        "model_expected_returns": model_returns,
        "cutoff_vs_distribution": cutoff_vs,
        "sizing_compression": compression,
        "notes": notes,
    }


def build_allocation_trace(
    scored: dict[str, dict[str, object]],
    *,
    shortlist: list[str],
    stock_w: dict[str, float],
    size_trace: dict[str, dict[str, object]],
    min_opp: float,
    strong_score: float,
    strong_info: dict[str, object],
    equiv_names: float,
    equiv_source: str,
    aggregate_conv: float,
    deploy: float,
    max_equity_budget: float,
    equity_budget: float,
    btc_w: float,
    cash_w: float,
    residual_code: str,
    risk_appetite: str = "aggressive",
    sizing_alpha: float = 1.0,
    sizing_alpha_source: str = "",
    held_keep_weight: float = 0.0,
    new_name_budget: float | None = None,
) -> dict[str, object]:
    """One-scan reverse-trace: preds → opportunity_score → cutoff → conviction → equity_budget."""
    rows: list[dict[str, object]] = []
    opp_vals: list[float] = []
    mean_vals: list[float] = []
    short_set = set(shortlist)
    for inst, row in scored.items():
        reason = str(row.get("reason") or "") or None
        if reason == "HELD_WITHOUT_SCORE":
            hs = row.get("hs") if isinstance(row.get("hs"), dict) else {}
            st = size_trace.get(inst) or {}
            rows.append(
                {
                    "ticker": _instrument_ticker(inst),
                    "instrument_id": inst,
                    "pred_5d": None,
                    "pred_10d": None,
                    "pred_20d": None,
                    "mean": 0.0,
                    "vol": _finite(row.get("vol")),
                    "conf": _finite(row.get("conf")),
                    "agreement": str(row.get("agr") or ""),
                    "opp_score": float(row.get("adj") or 0.0),
                    "pass_cutoff": False,
                    "exclusion_reason": reason,
                    "excess": float(row.get("excess") or 0.0),
                    "conviction": float(row.get("conviction") or 0.0),
                    "allocation_score": float(row.get("allocation_score") or 0.0),
                    "shortlist": False,
                    "initial_weight": st.get("initial_weight"),
                    "final_weight": float(stock_w.get(inst, 0.0)),
                }
            )
            continue
        hs = row.get("hs") if isinstance(row.get("hs"), dict) else {}
        p5 = _finite(hs.get(5))
        p10 = _finite(hs.get(10))
        p20 = _finite(hs.get(20))
        mean = _finite(row.get("mean"))
        if mean is None and p5 is not None and p10 is not None and p20 is not None:
            mean = (p5 + p10 + p20) / 3.0
        adj = float(row.get("adj") or 0.0)
        passed = reason not in _CUTOFF_FAIL_REASONS
        st = size_trace.get(inst) or {}
        if mean is not None:
            mean_vals.append(mean)
        opp_vals.append(adj)
        rows.append(
            {
                "ticker": _instrument_ticker(inst),
                "instrument_id": inst,
                "pred_5d": p5,
                "pred_10d": p10,
                "pred_20d": p20,
                "mean": mean,
                "vol": _finite(row.get("vol")),
                "conf": _finite(row.get("conf")),
                "agreement": str(row.get("agr") or ""),
                "opp_score": adj,
                "pass_cutoff": passed,
                "exclusion_reason": reason or (str(st.get("exclusion_reason") or "") or None),
                "excess": float(row.get("excess") or 0.0),
                "conviction": float(row.get("conviction") or 0.0),
                "allocation_score": float(row.get("allocation_score") or 0.0),
                "shortlist": inst in short_set,
                "initial_weight": st.get("initial_weight"),
                "final_weight": float(stock_w.get(inst, 0.0)),
            }
        )
    rows.sort(key=lambda r: float(r.get("opp_score") or 0.0), reverse=True)
    score_p = _pct_map(opp_vals)
    mean_p = _pct_map(mean_vals)
    passed_rows = [r for r in rows if r.get("pass_cutoff")]
    short_rows = [r for r in rows if r.get("shortlist")]
    convs = [float(r.get("conviction") or 0.0) for r in short_rows]
    top4 = sorted(convs, reverse=True)[:4]
    strong_count = sum(1 for r in short_rows if float(r.get("conviction") or 0.0) + 1e-12 >= 1.0)
    diagnosis = _diagnose_trace(
        score_p=score_p,
        mean_p=mean_p,
        min_opp=min_opp,
        n_passed=len(passed_rows),
        n_universe=len(opp_vals),
        deploy=deploy,
        equity_budget=equity_budget,
    )
    trace: dict[str, object] = {
        "formula": _SCORE_FORMULA,
        "pred_meaning": "expected_return_fraction",
        "vol_unit": _VOL_UNIT,
        "risk_appetite": risk_appetite,
        "sizing_alpha": sizing_alpha,
        "sizing_alpha_source": sizing_alpha_source,
        "cutoff": min_opp,
        "strong_score": strong_score,
        "strong_opportunity": strong_info,
        "equivalent_names": equiv_names,
        "equivalent_names_source": equiv_source,
        "universe": len(opp_vals),
        "passed_cutoff": len(passed_rows),
        "shortlist_n": len(shortlist),
        "strong_count_conviction_1": strong_count,
        "aggregate_conviction": aggregate_conv,
        "top4_conviction_sum": float(sum(top4)),
        "deployment_factor": deploy,
        "max_equity_budget": max_equity_budget,
        "equity_budget": equity_budget,
        "held_keep_weight": held_keep_weight,
        "new_name_budget": equity_budget if new_name_budget is None else new_name_budget,
        "btc_weight": btc_w,
        "stock_weight": float(sum(stock_w.values())),
        "cash_weight": cash_w,
        "residual_cash_reason": residual_code,
        "score_percentiles": score_p,
        "mean_return_percentiles": mean_p,
        "rows": rows,
        "rows_passed": passed_rows,
        "diagnosis": diagnosis,
    }
    trace["log_text"] = format_allocation_trace_log(trace)
    return trace


def format_allocation_trace_log(trace: dict[str, object]) -> str:
    score_p = trace.get("score_percentiles") if isinstance(trace.get("score_percentiles"), dict) else {}
    mean_p = (
        trace.get("mean_return_percentiles")
        if isinstance(trace.get("mean_return_percentiles"), dict)
        else {}
    )

    def _n(v: object, digits: int = 4) -> str:
        n = _finite(v)
        return "n/a" if n is None else f"{n:.{digits}f}"

    def _pct(v: object) -> str:
        n = _finite(v)
        return "n/a" if n is None else f"{n * 100:.1f}%"

    lines = [
        "=== ALLOCATION TRACE ===",
        "",
        f"Universe                 {int(trace.get('universe') or 0)}",
        f"pred_* meaning           {trace.get('pred_meaning')}",
        f"vol unit                 {trace.get('vol_unit') or _VOL_UNIT}",
        f"risk appetite            {trace.get('risk_appetite')}",
        f"sizing alpha             {_n(trace.get('sizing_alpha'), 2)}",
        f"formula                  {trace.get('formula')}",
        "",
        "Raw 5/10/20 mean expected return:",
        f"  min                    {_n(mean_p.get('min'))}",
        f"  median                 {_n(mean_p.get('p50'))}",
        f"  p75                    {_n(mean_p.get('p75'))}",
        f"  p90                    {_n(mean_p.get('p90'))}",
        f"  p95                    {_n(mean_p.get('p95'))}",
        f"  max                    {_n(mean_p.get('max'))}",
        "",
        "Opportunity score:",
        f"  min                    {_n(score_p.get('min'))}",
        f"  median                 {_n(score_p.get('p50'))}",
        f"  p75                    {_n(score_p.get('p75'))}",
        f"  p90                    {_n(score_p.get('p90'))}",
        f"  p95                    {_n(score_p.get('p95'))}",
        f"  max                    {_n(score_p.get('max'))}",
        "",
        f"Cutoff                   {_n(trace.get('cutoff'))}",
        f"Passed cutoff            {int(trace.get('passed_cutoff') or 0)}",
        f"Shortlist                {int(trace.get('shortlist_n') or 0)}",
        "",
        "Top opportunities:",
        f"{'Ticker':<8} {'5D':>8} {'10D':>8} {'20D':>8} {'Score':>8} {'Conv':>6} {'Pass':>4} {'Wgt':>7}",
    ]
    shown = list(trace.get("rows_passed") or [])[:12]
    if not shown:
        shown = list(trace.get("rows") or [])[:12]
    for row in shown:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"{str(row.get('ticker') or ''):<8} "
            f"{_n(row.get('pred_5d')):>8} "
            f"{_n(row.get('pred_10d')):>8} "
            f"{_n(row.get('pred_20d')):>8} "
            f"{_n(row.get('opp_score')):>8} "
            f"{_n(row.get('conviction'), 2):>6} "
            f"{'Y' if row.get('pass_cutoff') else 'N':>4} "
            f"{_pct(row.get('final_weight')):>7}"
        )
    lines.extend(
        [
            "",
            f"Strong (conviction=1)    {int(trace.get('strong_count_conviction_1') or 0)}",
            f"Shortlist conviction sum {_n(trace.get('aggregate_conviction'))}",
            f"Top-4 conviction sum     {_n(trace.get('top4_conviction_sum'))}",
            f"Equivalent names         {_n(trace.get('equivalent_names'), 2)}",
            f"Deployment factor        {_n(trace.get('deployment_factor'))}",
            f"Max equity budget        {_pct(trace.get('max_equity_budget'))}",
            f"Equity budget            {_pct(trace.get('equity_budget'))}",
            f"Held keep (not a sell)   {_pct(trace.get('held_keep_weight'))}",
            f"New-name budget          {_pct(trace.get('new_name_budget'))}",
            "",
            "Final:",
            f"Stocks                   {_pct(trace.get('stock_weight'))}",
            f"BTC                      {_pct(trace.get('btc_weight'))}",
            f"Cash                     {_pct(trace.get('cash_weight'))}",
            f"Residual reason          {trace.get('residual_cash_reason')}",
        ]
    )
    diag = trace.get("diagnosis") if isinstance(trace.get("diagnosis"), dict) else {}
    notes = diag.get("notes") if isinstance(diag.get("notes"), list) else []
    if notes:
        lines.append("")
        lines.append("Diagnosis:")
        lines.extend(f"- {n}" for n in notes)
    return "\n".join(lines)


def allocation_trace_from_quant_recs(
    recs: list[dict],
    alloc: dict | None = None,
) -> dict[str, object]:
    """Rebuild the trace from a persisted scan (no new allocate() required)."""
    payload = alloc if isinstance(alloc, dict) else {}
    min_opp = float(payload.get("min_opportunity_score") or 0.01)
    strong_info = payload.get("strong_opportunity") if isinstance(payload.get("strong_opportunity"), dict) else {}
    strong_score = _finite(strong_info.get("used_score"))
    if strong_score is None:
        mean_ret = _finite(strong_info.get("strong_horizon_mean_return")) or 0.04
        strong_score = implied_strong_opportunity_score(mean_ret)
        strong_info = {
            **strong_info,
            "used_score": strong_score,
            "strong_horizon_mean_return": mean_ret,
            "source": strong_info.get("source") or "reconstructed",
        }
    equiv_names = _finite(payload.get("full_deployment_equivalent_names")) or 4.0
    equiv_source = str(payload.get("full_deployment_names_source") or "reconstructed")
    risk_appetite = str(payload.get("risk_appetite") or "reconstructed")
    sizing_alpha = _finite(payload.get("sizing_alpha"))
    if sizing_alpha is None:
        sizing_alpha = 1.0
    alpha_source = str(payload.get("sizing_alpha_source") or "reconstructed")
    stock_w = {
        str(k): float(v)
        for k, v in (payload.get("stock_weights") or {}).items()
        if _finite(v) is not None
    }
    diag = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    size_trace: dict[str, dict[str, object]] = {}
    for inst, d in diag.items():
        if not isinstance(d, dict):
            continue
        size_trace[str(inst)] = {
            "initial_weight": d.get("initial_weight"),
            "final_weight": d.get("final_weight"),
            "exclusion_reason": d.get("exclusion_reason"),
        }
    shortlist = [str(x) for x in (payload.get("shortlist_order") or [])]
    scored: dict[str, dict[str, object]] = {}
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        inst = str(rec.get("instrument_id") or "")
        if not inst or "btc" in inst.lower():
            continue
        hs = _horizons_from_rec(rec)
        agr, _ = horizon_agreement(hs) if hs else ("unknown", 0.0)
        conf = _finite(rec.get("confidence"))
        if conf is None:
            conf = agreement_confidence(agr)
        conf = _clamp01(conf)
        mean = sum(float(hs[h]) for h in _HORIZONS if h in hs) / 3.0 if hs else 0.0
        d = diag.get(inst) if isinstance(diag.get(inst), dict) else {}
        adj = _finite(rec.get("opportunity_score"))
        if adj is None:
            adj = _finite(d.get("opportunity_score"))
        vol = _finite(d.get("vol"))
        if vol is None and adj is not None:
            vol = implied_vol_from_score(mean, conf, agr, adj)
        if vol is None:
            vol = 0.02
        if adj is None and horizons_valid(hs):
            adj = opportunity_score(mean, conf, vol, agr)
        if adj is None:
            adj = 0.0
        merged_reason = str(rec.get("exclusion_reason") or "") or None
        if not merged_reason:
            merged_reason = str(d.get("exclusion_reason") or "") or None
        reason = _eligibility_reason(merged_reason)
        if reason is None:
            if not hs or not horizons_valid(hs):
                reason = "INVALID_HORIZONS"
            elif conf <= 1e-15:
                reason = "ZERO_CONFIDENCE"
            elif adj < min_opp:
                reason = "BELOW_MIN_OPPORTUNITY"
            else:
                reason = None
        if merged_reason in _SIZE_ONLY_REASONS:
            reason = None
        excess = _finite(rec.get("excess_score"))
        conv = _finite(rec.get("conviction"))
        if excess is None or conv is None:
            excess, conv = name_conviction(adj, min_opp, strong_score)
        scored[inst] = {
            "hs": hs,
            "agr": agr,
            "conf": conf,
            "adj": adj,
            "mean": mean,
            "vol": vol,
            "reason": reason,
            "excess": excess,
            "conviction": conv,
            "allocation_score": allocation_score(float(excess or 0.0), sizing_alpha),
        }
        st = size_trace.setdefault(inst, {})
        if st.get("exclusion_reason") is None:
            st["exclusion_reason"] = merged_reason
        if st.get("final_weight") is None:
            st["final_weight"] = stock_w.get(inst, 0.0)
    if not shortlist:
        shortlist = [
            inst
            for inst, row in scored.items()
            if row.get("reason") is None and float(stock_w.get(inst, 0.0)) > 1e-12
        ]
        shortlist.sort(key=lambda i: float(scored[i]["adj"]), reverse=True)
    convictions = [float(scored[i]["conviction"]) for i in shortlist if i in scored]
    aggregate_conv, deploy = deployment_factor(convictions, equiv_names)
    if payload.get("aggregate_conviction") is not None:
        aggregate_conv = float(payload["aggregate_conviction"])
    if payload.get("deployment_factor") is not None:
        deploy = float(payload["deployment_factor"])
    max_equity_budget = _finite(payload.get("max_equity_budget"))
    if max_equity_budget is None:
        max_equity_budget = max(0.0, 1.0 - float(payload.get("btc_weight") or 0.0))
    equity_budget = _finite(payload.get("equity_budget"))
    if equity_budget is None:
        equity_budget = max_equity_budget * deploy
    btc_w = float(payload.get("btc_weight") or 0.0)
    cash_w = _finite(payload.get("cash_weight"))
    if cash_w is None:
        cash_w = max(0.0, 1.0 - sum(stock_w.values()) - btc_w)
    residual_code = str(payload.get("residual_cash_reason") or "")
    trace = build_allocation_trace(
        scored,
        shortlist=shortlist,
        stock_w=stock_w,
        size_trace=size_trace,
        min_opp=min_opp,
        strong_score=strong_score,
        strong_info=strong_info,
        equiv_names=equiv_names,
        equiv_source=equiv_source,
        aggregate_conv=aggregate_conv,
        deploy=deploy,
        max_equity_budget=max_equity_budget,
        equity_budget=equity_budget,
        btc_w=btc_w,
        cash_w=cash_w,
        residual_code=residual_code,
        risk_appetite=risk_appetite,
        sizing_alpha=sizing_alpha,
        sizing_alpha_source=alpha_source,
        held_keep_weight=float(payload.get("held_keep_weight") or 0.0),
        new_name_budget=_finite(payload.get("new_name_budget")),
    )
    used = _finite(payload.get("equity_budget_used"))
    if used is None:
        used = _finite(payload.get("equity_budget"))
    if used is not None:
        trace["stock_weight"] = used
        trace["log_text"] = format_allocation_trace_log(trace)
    if str(payload.get("formula_version") or "") == "quant_alloc_v4":
        trace["formula"] = _SCORE_FORMULA_V4
        trace["risk_appetite"] = payload.get("risk_appetite") or "legacy_v4"
        trace["log_text"] = format_allocation_trace_log(trace)
    return trace


def resolve_allocation_trace(alloc: dict | None, recs: list[dict] | None = None) -> dict[str, object]:
    payload = alloc if isinstance(alloc, dict) else {}
    existing = payload.get("allocation_trace")
    if isinstance(existing, dict) and existing.get("score_percentiles") and existing.get("formula"):
        return existing
    return allocation_trace_from_quant_recs(list(recs or []), payload)


def cap_floor_redistribute(
    scores: dict[str, float],
    *,
    budget: float,
    max_single: float,
    min_position: float,
) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    """Proportional weights with single-name cap, min-position floor, and redistribution."""
    trace: dict[str, dict[str, object]] = {}
    pos = {k: max(0.0, float(v)) for k, v in scores.items() if float(v) > 0}
    s0 = sum(pos.values())
    for inst, sc in scores.items():
        trace[inst] = {
            "opportunity_score": float(sc),
            "initial_weight": (budget * pos[inst] / s0) if s0 > 0 and inst in pos else 0.0,
            "pre_floor_weight": None,
            "final_weight": 0.0,
            "cap_applied": False,
            "exclusion_reason": None,
            "redistributed_from": 0.0,
        }
    if budget <= 1e-15 or s0 <= 1e-15:
        for inst in pos:
            trace[inst]["exclusion_reason"] = "NO_EQUITY_BUDGET"
        return {}, trace

    cap = min(max_single, budget)
    floor = min_position
    if budget + 1e-15 < floor:
        for inst in pos:
            trace[inst]["pre_floor_weight"] = trace[inst]["initial_weight"]
            trace[inst]["exclusion_reason"] = "BELOW_MIN_POSITION"
        return {}, trace

    active = set(pos)
    locked: dict[str, float] = {}
    for _ in range(len(pos) + 8):
        if not active:
            break
        pool = budget - sum(locked.values())
        if pool + 1e-15 < floor:
            for inst in list(active):
                trace[inst]["exclusion_reason"] = "BELOW_MIN_POSITION"
                if trace[inst]["pre_floor_weight"] is None:
                    trace[inst]["pre_floor_weight"] = trace[inst]["initial_weight"]
                active.discard(inst)
            break
        s = sum(pos[i] for i in active)
        if s <= 1e-15:
            break
        raw = {i: pool * pos[i] / s for i in active}
        small = [i for i in active if raw[i] + 1e-12 < floor]
        if small:
            i = min(small, key=lambda n: raw[n])
            trace[i]["pre_floor_weight"] = raw[i]
            trace[i]["exclusion_reason"] = "BELOW_MIN_POSITION"
            active.discard(i)
            continue
        big = [i for i in active if raw[i] > cap + 1e-12]
        if big:
            i = max(big, key=lambda n: raw[n])
            locked[i] = cap
            trace[i]["cap_applied"] = True
            trace[i]["pre_floor_weight"] = raw[i]
            trace[i]["redistributed_from"] = raw[i] - cap
            active.discard(i)
            continue
        for i, w in raw.items():
            locked[i] = w
            if trace[i]["pre_floor_weight"] is None:
                trace[i]["pre_floor_weight"] = w
        active.clear()
        break

    out = {k: v for k, v in locked.items() if v + 1e-12 >= floor}
    for inst, w in out.items():
        trace[inst]["final_weight"] = w
        trace[inst]["exclusion_reason"] = None
    return out, trace


def _note_for_exclusion(
    ticker: str,
    reason: str | None,
    *,
    pre_floor_weight: float | None,
    min_position: float,
    max_positions: int,
) -> str | None:
    if reason == "BELOW_MIN_POSITION" and pre_floor_weight is not None:
        return (
            f"{ticker}은(는) 계산된 목표가 {pre_floor_weight * 100:.2f}%로, "
            f"최소 의미 비중 {min_position * 100:.1f}%에 미달해 추천하지 않습니다."
        )
    if reason == "BELOW_MIN_OPPORTUNITY":
        return f"{ticker}은(는) 기회 점수가 최소 기준에 미달해 배분 대상이 아닙니다."
    if reason == "HOLD_NOT_NEW_BUY":
        return (
            f"{ticker}은(는) 새 매수 기준에는 못 미치지만, "
            "전망이 나빠 청산할 정도는 아니라 보유를 유지합니다."
        )
    if reason == "EXIT_NEGATIVE_OUTLOOK":
        return f"{ticker}은(는) 5/10/20 평균 기대수익이 음수라 보유를 줄이라고 권고합니다."
    if reason == "INVALID_HORIZONS":
        return f"{ticker}은(는) 5/10/20일 점수가 불완전해 배분하지 않습니다."
    if reason == "ZERO_CONFIDENCE":
        return f"{ticker}은(는) 신뢰도가 0이라 배분하지 않습니다."
    if reason == "MAX_EQUITY_POSITIONS":
        return f"{ticker}은(는) 주식 종목 수 상한({max_positions}) 밖의 후보라 배분하지 않습니다."
    if reason == "CASH_FLOOR":
        return f"{ticker}은(는) 최소 현금 비중을 지키기 위해 배분에서 제외했습니다."
    if reason == "HELD_WITHOUT_SCORE":
        return f"{ticker}은(는) 점수가 없어 보유를 유지합니다."
    return None


def allocate(
    *,
    stock_scores: dict[str, dict[int, float]],
    stock_vol: dict[str, float],
    btc_scores: dict[int, float],
    settings: Settings,
    btc_state: BtcSleeveState,
    tick_id: str,
    epoch_id: str,
    feature_snapshot_id: str,
    total_base_units: float = 1000.0,
    stock_units: dict[str, float] | None = None,
    stock_confidence: dict[str, float] | None = None,
    stock_rank_scores: dict[str, dict[int, float]] | None = None,
    stock_technical: dict[str, dict[str, float]] | None = None,
    stock_research: dict[str, dict[str, object]] | None = None,
    stock_horizon_influence: dict[str, float] | None = None,
    btc_horizon_influence: dict[str, float] | None = None,
) -> dict:
    """Return weights summing to 1.0; 100% cash is allowed. Units scale with total_base_units."""
    base = max(1e-9, float(total_base_units))
    holdings = stock_units or {}
    rank_scores = stock_rank_scores or {}
    technical = stock_technical or {}
    research = stock_research or {}
    min_cash = _clamp01(settings.min_cash_weight)
    max_stock = _clamp01(settings.max_total_stock_weight)
    max_btc = _clamp01(settings.max_btc_weight)
    max_single = min(max_stock, _clamp01(settings.max_single_stock_weight))
    min_pos = _clamp01(settings.min_position_weight)
    min_delta_w = _clamp01(settings.min_delta_weight)
    max_positions = max(1, int(settings.max_equity_positions))
    min_opp = float(settings.min_opportunity_score)
    if min_cash + 0.0 > 1.0:
        min_cash = 1.0

    def _rank_mean(inst: str) -> float:
        hs = rank_scores.get(inst) or {}
        vals = [float(hs[h]) for h in _HORIZONS if hs.get(h) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def _ticker(inst: str) -> str:
        s = inst[5:] if inst.startswith("inst_") else inst
        return s.upper()

    scored: dict[str, dict[str, object]] = {}
    for inst, hs in stock_scores.items():
        agr, _ = horizon_agreement(hs)
        mean = _weighted_horizon_mean(hs, stock_horizon_influence)
        vol = max(1e-4, stock_vol.get(inst, 0.02))
        conf = stock_confidence.get(inst) if stock_confidence else None
        if conf is None:
            conf = agreement_confidence(agr)
        conf = _clamp01(float(conf))
        adj = opportunity_score(mean, conf, vol, agr)
        reason = None
        if not horizons_valid(hs):
            reason = "INVALID_HORIZONS"
        elif conf <= 1e-15:
            reason = "ZERO_CONFIDENCE"
        elif adj < min_opp:
            reason = "BELOW_MIN_OPPORTUNITY"
        scored[inst] = {
            "hs": hs,
            "agr": agr,
            "conf": conf,
            "adj": adj,
            "mean": mean,
            "vol": vol,
            "reason": reason,
        }

    eligible = [inst for inst, row in scored.items() if row["reason"] is None]
    eligible.sort(key=lambda i: (float(scored[i]["adj"]), _rank_mean(i)), reverse=True)
    shortlist = eligible[:max_positions]
    overflow = eligible[max_positions:]
    for inst in overflow:
        scored[inst]["reason"] = "MAX_EQUITY_POSITIONS"

    for inst, units in holdings.items():
        if inst in scored or float(units or 0) <= 1e-12:
            continue
        scored[inst] = {
            "hs": {5: 0.0, 10: 0.0, 20: 0.0},
            "agr": "unknown",
            "conf": agreement_confidence("unknown"),
            "adj": 0.0,
            "mean": 0.0,
            "vol": 0.02,
            "reason": "HELD_WITHOUT_SCORE",
        }

    # --- BTC sleeve (v2 formula) ---
    btc_mean = _weighted_horizon_mean(btc_scores, btc_horizon_influence)
    btc_agr, _ = horizon_agreement(btc_scores)
    btc_conf = agreement_confidence(btc_agr)
    blocked = btc_state.transfer_blocks_immediate_rebalance()
    current_btc_w = min(1.0, max(0.0, btc_state.current_units / base))

    if blocked:
        # Locked units: a transfer-blocked sleeve can't be resized, so min_cash_weight
        # cannot be honored against it here -- the cash-floor eviction path (below)
        # is the only truthful way to restore cash in that case, by shedding stock.
        btc_w = current_btc_w
        rec_btc_units = btc_state.current_units
    elif btc_mean > _BTC_MIN_OUTLOOK and btc_state.liquidity == BtcLiquidityState.AVAILABLE:
        # Unblocked: BTC is being freshly sized, so it must never claim more than
        # 1 - min_cash on its own -- otherwise a name-less book (no stock candidates
        # at all) could still leave cash below the floor with nothing to evict.
        btc_w = min(max_btc, max(0.0, 1.0 - min_cash), max(0.0, btc_mean * 2.0) * btc_conf)
        rec_btc_units = btc_w * base
    else:
        btc_w = 0.0
        rec_btc_units = 0.0

    strong_info = resolve_strong_opportunity(settings, min_opp)
    strong_score = float(strong_info["used_score"])
    appetite = resolve_risk_appetite(settings)
    sizing_alpha, alpha_source = resolve_sizing_alpha(settings)
    equiv_names, equiv_source = resolve_equivalent_names(settings, max_stock, max_single)
    max_equity_budget = min(max_stock, max(0.0, 1.0 - min_cash - btc_w))
    def _tech_multiplier_for(inst: str) -> float:
        # Only nudges relative sizing among names already selected by opportunity_score;
        # never applied to an existing holding (see the ADD/HOLD/EXIT paths below, which
        # never call this). conviction/deployment_factor (aggregate equity deployed) are
        # deliberately left untouched -- this shifts weight *within* the shortlist, it
        # does not change how much of the book goes into stocks at all.
        if float(holdings.get(inst, 0.0)) > 1e-12:
            return 1.0
        return technical_sizing_multiplier(technical.get(inst))

    def _research_multiplier_for(inst: str) -> float:
        # Same held-position exemption as _tech_multiplier_for -- see
        # research_sizing_multiplier's docstring for why this band is intentionally
        # narrower than the technical one.
        if float(holdings.get(inst, 0.0)) > 1e-12:
            return 1.0
        return research_sizing_multiplier(research.get(inst))

    for inst in shortlist:
        adj = float(scored[inst]["adj"])
        excess, conv = name_conviction(adj, min_opp, strong_score)
        scored[inst]["excess"] = excess
        scored[inst]["conviction"] = conv
        tech_mult = _tech_multiplier_for(inst)
        research_mult = _research_multiplier_for(inst)
        scored[inst]["technical_sizing_multiplier"] = tech_mult
        scored[inst]["research_sizing_multiplier"] = research_mult
        scored[inst]["allocation_score"] = allocation_score(excess, sizing_alpha) * tech_mult * research_mult
    for inst, row in scored.items():
        if inst in shortlist:
            continue
        adj = float(row["adj"])
        excess, conv = name_conviction(adj, min_opp, strong_score)
        row["excess"] = excess
        row["conviction"] = conv
        tech_mult = _tech_multiplier_for(inst)
        research_mult = _research_multiplier_for(inst)
        row["technical_sizing_multiplier"] = tech_mult
        row["research_sizing_multiplier"] = research_mult
        row["allocation_score"] = allocation_score(excess, sizing_alpha) * tech_mult * research_mult

    convictions = [float(scored[inst]["conviction"]) for inst in shortlist]
    aggregate_conv, deploy = deployment_factor(convictions, equiv_names)
    equity_budget = max_equity_budget * deploy
    kept_w: dict[str, float] = {}
    for inst, row in scored.items():
        cur = float(holdings.get(inst, 0.0))
        if inst in shortlist or cur <= 1e-12:
            continue
        if held_should_exit(row):
            row["exit_held"] = True
            row["reason"] = "EXIT_NEGATIVE_OUTLOOK"
            continue
        kept_w[inst] = cur / base
        row["keep_held"] = True
        if row.get("reason") in {"BELOW_MIN_OPPORTUNITY", "MAX_EQUITY_POSITIONS"}:
            row["reason"] = "HOLD_NOT_NEW_BUY"
    kept_total = sum(kept_w.values())
    new_budget = min(equity_budget, max(0.0, max_equity_budget - kept_total))
    sizing_scores = {inst: float(scored[inst]["allocation_score"]) for inst in shortlist}
    if not sizing_scores:
        stock_w, size_trace = {}, {}
        if not kept_w:
            deploy = 0.0
            aggregate_conv = 0.0
            equity_budget = 0.0
            new_budget = 0.0
    else:
        stock_w, size_trace = cap_floor_redistribute(
            sizing_scores,
            budget=new_budget,
            max_single=max_single,
            min_position=min_pos,
        )
    for inst, w in kept_w.items():
        stock_w[inst] = w

    cash_w = 1.0 - sum(stock_w.values()) - btc_w
    if cash_w < -1e-9:
        room = max(0.0, 1.0 - btc_w - kept_total)
        ss = sum(v for k, v in stock_w.items() if k not in kept_w)
        if ss > 0 and ss > room:
            factor = room / ss
            for k in list(stock_w):
                if k not in kept_w:
                    stock_w[k] *= factor
        stock_w = {
            k: v
            for k, v in stock_w.items()
            if k in kept_w or v + 1e-12 >= min_pos
        }
        cash_w = 1.0 - sum(stock_w.values()) - btc_w
        if cash_w < -1e-9:
            # kept_w positions are never rescaled here (or above): recs_out always
            # reports kept/held names at their true current units (see keep_held
            # below), so shrinking their weight in stock_w would desync the
            # reported weights/cash from the actual recommendations.
            room = max(0.0, 1.0 - btc_w - kept_total)
            ss = sum(v for k, v in stock_w.items() if k not in kept_w)
            if ss > 0 and ss > room:
                factor = room / ss
                for k in list(stock_w):
                    if k not in kept_w:
                        stock_w[k] *= factor
            cash_w = 1.0 - sum(stock_w.values()) - btc_w
    # A transfer-blocked BTC sleeve means btc_w can't move, but it must never excuse
    # skipping stock eviction when cash_w is genuinely negative (kept_total + btc_w
    # over 1.0): the payload identity (stock_w + btc_w + cash_w == 1.0) is asserted
    # below and is non-negotiable. "not blocked" only skips the *soft* min_cash
    # preference (cash_w already non-negative) while a transfer is pending.
    if cash_w + 1e-12 < min_cash and (cash_w < -1e-9 or not blocked):
        ordered = sorted(
            stock_w,
            key=lambda k: (1 if k in kept_w else 0, stock_w[k]),
        )
        for inst in ordered:
            if cash_w + 1e-12 >= min_cash:
                break
            cash_w += stock_w.pop(inst)
            st = size_trace.get(inst)
            if st is not None:
                st["exclusion_reason"] = "CASH_FLOOR"
                st["final_weight"] = 0.0
            if inst in kept_w:
                scored[inst]["keep_held"] = False
                scored[inst]["reason"] = "CASH_FLOOR"
        cash_w = 1.0 - sum(stock_w.values()) - btc_w
    cash_w = max(0.0, cash_w)
    # Cash-floor enforcement may evict a position that was initially classified
    # as kept. Diagnostics and reconstructed UI traces must report what survived,
    # not the pre-constraint amount.
    kept_total = sum(stock_w.get(inst, 0.0) for inst in kept_w)

    used_eq = sum(stock_w.values())
    if not shortlist and kept_w:
        residual_code = "HELD_BELOW_BUY_GATE"
        residual_label = "새 매수 후보는 없지만 기존 보유는 유지합니다."
    elif not shortlist:
        residual_code = "NO_QUALIFIED_OPPORTUNITIES"
        residual_label = "자격 있는 주식 기회가 없어 현금 유지"
    elif deploy + 1e-9 < 1.0:
        residual_code = "LOW_CONVICTION"
        residual_label = "기회 강도가 낮아 현금 유지"
    elif used_eq + 1e-9 < equity_budget:
        residual_code = "POSITION_CAPS"
        residual_label = "종목 상한으로 흡수하지 못한 잔여 현금"
    else:
        residual_code = "CONSTRAINT_RESIDUAL"
        residual_label = ""
    deploy_note = deployment_note_ko(
        qualified=len(shortlist),
        max_stock=max_stock,
        equity_budget=equity_budget,
        deployment=deploy,
        kept_total=kept_total,
    )

    min_pos_units = min_pos * base
    min_delta_units = min_delta_w * base
    recs_out: list[RecommendationRecord] = []
    name_diag: dict[str, dict[str, object]] = {}
    redistributed = 0.0
    for inst, row in scored.items():
        cur = float(holdings.get(inst, 0.0))
        st = size_trace.get(inst) or {}
        reason = str(st.get("exclusion_reason") or row["reason"] or "") or None
        pre_w = st.get("pre_floor_weight")
        if pre_w is None:
            pre_w = st.get("initial_weight")
        init_w = st.get("initial_weight")
        init_u = None if init_w is None else float(init_w) * base
        pre_u = None if pre_w is None else float(pre_w) * base
        rec_u = float(stock_w.get(inst, 0.0)) * base
        keep_held = bool(row.get("keep_held")) and inst in stock_w
        if keep_held:
            rec_u = cur
            if reason in {"BELOW_MIN_OPPORTUNITY", "MAX_EQUITY_POSITIONS", None}:
                reason = "HOLD_NOT_NEW_BUY"
            elif reason == "HELD_WITHOUT_SCORE":
                reason = "HELD_WITHOUT_SCORE"
            action = RecommendationAction.HOLD
        elif row.get("exit_held"):
            rec_u = 0.0
            reason = "EXIT_NEGATIVE_OUTLOOK"
            action = RecommendationAction.EXIT
        elif reason == "HELD_WITHOUT_SCORE":
            rec_u = cur
            action = RecommendationAction.HOLD
        else:
            action = derive_stock_action(
                cur,
                rec_u,
                min_position_units=min_pos_units,
                min_delta_units=min_delta_units,
            )
        note = _note_for_exclusion(
            _ticker(inst),
            reason,
            pre_floor_weight=None if pre_w is None else float(pre_w),
            min_position=min_pos,
            max_positions=max_positions,
        )
        redistributed += float(st.get("redistributed_from") or 0.0)
        hs = row["hs"]  # type: ignore[assignment]
        conf = float(row["conf"])
        agr = str(row["agr"])
        recs_out.append(
            quant_only_record(
                tick_id=tick_id,  # type: ignore[arg-type]
                decision_epoch_id=epoch_id,  # type: ignore[arg-type]
                feature_snapshot_id=feature_snapshot_id,  # type: ignore[arg-type]
                instrument_id=InstrumentId(inst),
                action=action,
                current_units=cur,
                recommended_units=rec_u,
                delta_units=rec_u - cur,
                confidence=conf,
                opportunity_score=float(row["adj"]),
                excess_score=float(row.get("excess") or 0.0),
                conviction=float(row.get("conviction") or 0.0),
                equity_budget=equity_budget,
                initial_units=init_u,
                pre_floor_units=pre_u,
                exclusion_reason=reason,
                allocation_note=note,
                horizons=[
                    HorizonOutlook(
                        horizon=h,
                        expected_return=hs.get(h),  # type: ignore[union-attr]
                        rank_score=(rank_scores.get(inst) or {}).get(h),
                        confidence=conf,
                    )
                    for h in _HORIZONS
                ],
                thesis=(
                    f"agreement={agr} formula={FORMULA_VERSION} "
                    f"lambdarank_tiebreak={_rank_mean(inst):.4f}"
                    + (f" {note}" if note else "")
                ),
            )
        )
        name_diag[inst] = {
            "opportunity_score": float(row["adj"]),
            "min_opportunity_score": min_opp,
            "excess_score": float(row.get("excess") or 0.0),
            "conviction": float(row.get("conviction") or 0.0),
            "allocation_score": float(row.get("allocation_score") or 0.0),
            "eligible": reason is None and inst in stock_w,
            "max_equity_budget": max_equity_budget,
            "equity_budget": equity_budget,
            "deployment_factor": deploy,
            "initial_weight": st.get("initial_weight"),
            "initial_units": init_u,
            "pre_floor_weight": pre_w,
            "pre_floor_units": pre_u,
            "cap_applied": bool(st.get("cap_applied")),
            "floor_applied": reason == "BELOW_MIN_POSITION",
            "final_weight": float(stock_w.get(inst, 0.0)),
            "final_units": rec_u,
            "exclusion_reason": reason,
            "allocation_note": note,
            "vol": float(row.get("vol") or 0.02),
            "horizon_mean": float(row.get("mean") or 0.0),
            "technical_sizing_multiplier": float(row.get("technical_sizing_multiplier") or 1.0),
            "research_sizing_multiplier": float(row.get("research_sizing_multiplier") or 1.0),
            "keep_held": keep_held,
            "exit_held": bool(row.get("exit_held")),
        }

    action = btc_action(btc_state.current_units, rec_btc_units, btc_state, settings, btc_mean)
    recs_out.append(
        quant_only_record(
            tick_id=tick_id,  # type: ignore[arg-type]
            decision_epoch_id=epoch_id,  # type: ignore[arg-type]
            feature_snapshot_id=feature_snapshot_id,  # type: ignore[arg-type]
            instrument_id=InstrumentId(btc_state.instrument_id),
            action=RecommendationAction(action.value)
            if action.value in RecommendationAction._value2member_map_
            else RecommendationAction.NO_ACTION,
            current_units=btc_state.current_units,
            recommended_units=rec_btc_units,
            delta_units=rec_btc_units - btc_state.current_units,
            confidence=btc_conf,
            horizons=[HorizonOutlook(horizon=h, expected_return=btc_scores.get(h), confidence=btc_conf) for h in _HORIZONS],
            thesis=f"btc_sleeve liquidity={btc_state.liquidity.value} formula={FORMULA_VERSION}",
        )
    )

    agr_by_inst = {inst: str(row["agr"]) for inst, row in scored.items()}
    payload = {
        "formula_version": FORMULA_VERSION,
        "stock_weights": stock_w,
        "btc_weight": btc_w,
        "cash_weight": cash_w,
        "cash_instrument_id": USD_CASH_INSTRUMENT_ID,
        "total_base_units": base,
        "horizon_conflict": agr_by_inst,
        "research_performance_note": "historical diagnostics are not live performance",
        "live_performance_note": "live outcomes scored separately after labels mature",
        "lambdarank_mean": {inst: _rank_mean(inst) for inst in stock_scores},
        "shortlist_order": list(shortlist),
        "rank_factor": "lambdarank_tiebreak",
        "equity_budget": equity_budget,
        "equity_budget_used": sum(stock_w.values()),
        "max_equity_budget": max_equity_budget,
        "max_total_stock_weight": max_stock,
        "deployment_factor": deploy,
        "aggregate_conviction": aggregate_conv,
        "full_deployment_equivalent_names": equiv_names,
        "full_deployment_names_source": equiv_source,
        "risk_appetite": str(appetite["name"]),
        "sizing_alpha": sizing_alpha,
        "sizing_alpha_source": alpha_source,
        "held_keep_weight": kept_total,
        "new_name_budget": new_budget,
        "strong_opportunity": strong_info,
        "min_opportunity_score": min_opp,
        "min_position_weight": min_pos,
        "max_equity_positions": max_positions,
        "redistributed_weight": redistributed,
        "residual_cash": cash_w,
        "residual_cash_reason": residual_code,
        "residual_cash_label": residual_label,
        "deployment_note": deploy_note,
        "diagnostics": name_diag,
        "sizing": "deployment_factor_then_excess_power_no_second_haircut",
        "allocation_trace": build_allocation_trace(
            scored,
            shortlist=shortlist,
            stock_w=stock_w,
            size_trace=size_trace,
            min_opp=min_opp,
            strong_score=strong_score,
            strong_info=strong_info,
            equiv_names=equiv_names,
            equiv_source=equiv_source,
            aggregate_conv=aggregate_conv,
            deploy=deploy,
            max_equity_budget=max_equity_budget,
            equity_budget=equity_budget,
            btc_w=btc_w,
            cash_w=cash_w,
            residual_code=residual_code,
            risk_appetite=str(appetite["name"]),
            sizing_alpha=sizing_alpha,
            sizing_alpha_source=alpha_source,
            held_keep_weight=kept_total,
            new_name_budget=new_budget,
        ),
    }
    assert abs(sum(stock_w.values()) + btc_w + cash_w - 1.0) < 1e-6
    for inst, w in stock_w.items():
        if inst in kept_w:
            continue
        assert w + 1e-9 >= min_pos or w <= 1e-12
    return {
        "payload": payload,
        "recommendations": recs_out,
        "btc_action": action,
        "recommended_btc_units": rec_btc_units,
    }


def persist_allocation(conn, payload: dict) -> str:
    sid = f"alloc_{uuid4().hex[:12]}"
    conn.execute(
        """
        INSERT INTO allocation_snapshots (snapshot_id, created_at, formula_version, payload_json)
        VALUES (?, ?, ?, ?)
        """,
        [sid, datetime.now(timezone.utc), str(payload.get("formula_version") or FORMULA_VERSION), json.dumps(payload)],
    )
    return sid
