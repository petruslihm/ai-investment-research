"""Validate the final judge and render/store one canonical decision. No I/O or orders."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from trading_system.actionability import actionable_view
from trading_system.portfolio_gate import apply_portfolio_gate
from trading_system.recommendations import RecommendationAction, RecommendationRecord, RecommendationSource, normalize_position
from trading_system.llm_log import ticker_from_instrument


CONTRACT_VERSION = "final_decision_v2"


def prepare_pending(package: dict, *, resume_guard: str) -> dict | None:
    """Public builds resume only the current, identical-input contract.

    Historical operating-package migration is deliberately not included.
    A matching package with inconsistent position semantics requires re-evaluation.
    """
    if package.get("contract_version") != CONTRACT_VERSION or package.get("resume_guard") != resume_guard:
        return None
    try:
        stages = package["stages"]
        research = {r["instrument_id"]: r for r in stages["research_adjusted"]}
        for row in stages["quant"] + stages["research_adjusted"]:
            if normalize_position(row) != row:
                raise ValueError("noncanonical position")
        for candidate in package["candidates"]:
            q = candidate["quant"]
            if normalize_position(q) != q:
                raise ValueError("noncanonical candidate")
            if any(q.get(k) != research[candidate["instrument_id"]].get(k) for k in ("position_held", "current_units", "acquisition_units")):
                raise ValueError("candidate position mismatch")
        return package
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("REEVALUATION_REQUIRED:INVALID_PENDING_INPUT") from exc


class JudgeChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    instrument_id: str
    action: Literal["BUY", "WATCH", "HOLD", "SELL", "ENTER", "ADD", "REDUCE", "EXIT", "NO_ACTION"]
    recommended_units: float | None = Field(strict=True)
    rank: int = Field(ge=1, strict=True)
    thesis: str
    contrary_evidence: str
    change_conditions: str


class FinalJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    input_id: str
    input_as_of: str
    recommendations: list[JudgeChoice]


def parse_final_judgment(raw: str, package: dict) -> FinalJudgment:
    """All-or-nothing: no regex interpretation of prose or partial portfolios."""
    if package.get("contract_version", CONTRACT_VERSION) != CONTRACT_VERSION:
        raise ValueError("REEVALUATION_REQUIRED:UNNORMALIZED_CONTRACT")
    result = FinalJudgment.model_validate_json(raw)
    if not package.get("input_id") or result.input_id != package["input_id"]:
        raise ValueError("INPUT_ID_MISMATCH")
    if result.input_as_of != package.get("input_as_of"):
        raise ValueError("INPUT_AS_OF_MISMATCH")
    stamp = datetime.fromisoformat(result.input_as_of)
    if stamp.tzinfo is None:
        raise ValueError("INPUT_AS_OF_REQUIRES_TIMEZONE")
    candidates = {str(c["instrument_id"]): c for c in package.get("candidates", [])}
    ids = [c.instrument_id for c in result.recommendations]
    ranks = [c.rank for c in result.recommendations]
    if len(ids) != len(set(ids)) or set(ids) != set(candidates):
        raise ValueError("CANDIDATE_COVERAGE_MISMATCH")
    if sorted(ranks) != list(range(1, len(ids) + 1)):
        raise ValueError("INVALID_RANKS")
    for choice in result.recommendations:
        quant = normalize_position(candidates[choice.instrument_id].get("quant", {}))
        current_raw = quant.get("current_units")
        held = quant["position_held"]
        unpriced_holding = held and current_raw is None
        target = choice.recommended_units
        if target is not None and (not math.isfinite(target) or target < 0):
            raise ValueError("INVALID_UNITS")
        if unpriced_holding:
            if choice.action not in {"WATCH", "NO_ACTION", "HOLD"} or target is not None:
                raise ValueError("UNPRICED_HOLDING_ACTION_CONFLICT")
            continue
        current = float(current_raw or 0)
        if choice.action in {"WATCH", "NO_ACTION", "HOLD"}:
            # WATCH is an instruction not to change exposure, including for holdings.
            if target is not None and abs(target - current) > 1e-8:
                raise ValueError("HOLD_WATCH_TARGET_CONFLICT")
        elif choice.action in {"BUY", "ENTER", "ADD"}:
            if target is None or target <= current:
                raise ValueError("BUY_TARGET_CONFLICT")
        elif target is None or current <= 0 or target >= current:
            raise ValueError("SELL_TARGET_CONFLICT")
        if choice.action == "EXIT" and target != 0:
            raise ValueError("EXIT_TARGET_CONFLICT")
    return result


def action_for_units(current: float, target: float) -> RecommendationAction:
    if abs(target - current) <= 1e-8:
        return RecommendationAction.HOLD if current > 0 else RecommendationAction.NO_ACTION
    if target > current:
        return RecommendationAction.ADD if current > 0 else RecommendationAction.ENTER
    return RecommendationAction.EXIT if target <= 1e-8 else RecommendationAction.REDUCE


def apply_final_judgment(raw: str, package: dict, research_recs: list[RecommendationRecord], *, settings, btc_state):
    result = parse_final_judgment(raw, package)
    research_recs = [RecommendationRecord.model_validate(normalize_position(r.model_dump())) for r in research_recs]
    by_id = {str(r.instrument_id): r for r in research_recs}
    candidates = {str(c["instrument_id"]): c for c in package["candidates"]}
    records = []
    for choice in sorted(result.recommendations, key=lambda c: c.rank):
        base = by_id[choice.instrument_id]
        frozen = normalize_position(candidates[choice.instrument_id].get("quant", {}))
        frozen_raw = frozen["current_units"]
        if frozen["position_held"] != base.position_held:
            raise ValueError("HOLDINGS_CHANGED")
        unpriced_holding = bool(base.position_held) and base.current_units is None
        if unpriced_holding:
            if frozen_raw is not None:
                raise ValueError("HOLDINGS_CHANGED")
            records.append(base.model_copy(update={
                "recommendation_id": f"rec_{uuid4().hex}", "source": RecommendationSource.LLM_FINAL,
                "llm_tainted": True, "action": RecommendationAction.HOLD,
                "recommended_units": None, "delta_units": None,
                "requested_units": None, "requested_action": choice.action,
                "final_rank": choice.rank, "decision_status": "VALIDATED_UNPRICED_HOLDING",
                "actionable": False,
                "override_of_recommendation_id": base.override_of_recommendation_id or base.recommendation_id,
                "previous_stage_recommendation_id": base.recommendation_id,
                "override_reasons": ["STRUCTURED_FINAL_JUDGE", "UNPRICED_HOLDING"],
                "thesis": choice.thesis, "rationale_detail": choice.thesis,
                "contrary_evidence": choice.contrary_evidence, "change_conditions": choice.change_conditions,
                "allocation_note": "확정 평가가격이 없어 보유 상태만 유지하며 목표 배분은 미확정입니다.",
                "exclusion_reason": "UNPRICED_HOLDING",
                "equity_budget": None, "initial_units": None, "pre_floor_units": None,
            }))
            continue
        current = float(base.current_units or 0)
        frozen_current = float(frozen_raw or 0)
        if abs(current - frozen_current) > 1e-8:
            raise ValueError("HOLDINGS_CHANGED")
        target = current if choice.action in {"WATCH", "NO_ACTION", "HOLD"} else float(choice.recommended_units)
        records.append(base.model_copy(update={
            "recommendation_id": f"rec_{uuid4().hex}", "source": RecommendationSource.LLM_FINAL,
            "llm_tainted": True, "action": action_for_units(current, target),
            "recommended_units": target, "delta_units": target-current,
            "requested_units": target, "requested_action": choice.action,
            "final_rank": choice.rank, "decision_status": "VALIDATED",
            "override_of_recommendation_id": base.override_of_recommendation_id or base.recommendation_id,
            "previous_stage_recommendation_id": base.recommendation_id,
            "override_reasons": ["STRUCTURED_FINAL_JUDGE"],
            "thesis": choice.thesis, "rationale_detail": choice.thesis,
            "contrary_evidence": choice.contrary_evidence, "change_conditions": choice.change_conditions,
            "allocation_note": None, "exclusion_reason": None,
            "equity_budget": None, "initial_units": None, "pre_floor_units": None,
        }))
    # Release explicitly reduced positions first, then allocate increases in GPT rank
    # order. Unjudged rows retain their clearly labelled research-stage targets.
    requested_targets = {str(r.instrument_id): r.requested_units for r in records}
    for i, rec in enumerate(records):
        if rec.current_units is None:
            continue
        view = actionable_view(rec, settings=settings, total_base_units=float(package["portfolio"]["total_base_units"]))
        if view.suppressed:
            current = float(rec.current_units or 0)
            records[i] = rec.model_copy(update={
                "action": action_for_units(current, current), "recommended_units": current,
                "delta_units": 0.0, "override_reasons": rec.override_reasons + [view.reason_code],
                "allocation_note": view.note,
            })
    incomplete = any(r.position_held and r.current_units is None for r in research_recs)
    book = {
        str(r.instrument_id): (r.current_units if str(r.instrument_id) in candidates else r.recommended_units)
        for r in research_recs if "btc" not in str(r.instrument_id).lower()
    }
    ordered = sorted(records, key=lambda r: (float(r.delta_units or 0) > 0, r.final_rank))
    gate_btc = btc_state
    unjudged_btc = next((r for r in research_recs if "btc" in str(r.instrument_id).lower()
                        and str(r.instrument_id) not in candidates), None)
    if unjudged_btc is not None and not incomplete:
        gate_btc = btc_state.model_copy(update={"current_units": float(unjudged_btc.recommended_units or 0)})
    # No capacity arithmetic is possible while part of the existing book is unknown.
    constrained = [r.model_copy(update={
        "actionable": False, "constrained_units": None,
        "allocation_note": "보유 자산 일부의 평가가 불가하여 전체 배분 제약 검증은 미확정입니다.",
        "override_reasons": r.override_reasons + ["INCOMPLETE_VALUATION"],
    }) for r in ordered] if incomplete else apply_portfolio_gate(
        ordered, settings=settings, total_base_units=float(package["portfolio"]["total_base_units"]),
        stock_marked=book, btc_state=gate_btc,
    )
    finalized = []
    for rec in constrained:
        rec = rec.model_copy(update={"requested_units": requested_targets[str(rec.instrument_id)]})
        if rec.current_units is None:
            finalized.append(rec)
            continue
        current = float(rec.current_units or 0)
        target = float(rec.recommended_units or 0)
        rec = rec.model_copy(update={"action": action_for_units(current, target), "delta_units": target-current})
        # Existing UI deadbands are part of the canonical decision too. Preserve the
        # requested/constrained amounts, but do not advertise a buy in prose and HOLD
        # on its card for the same tiny recommendation.
        view = actionable_view(rec, settings=settings, total_base_units=float(package["portfolio"]["total_base_units"]))
        if view.suppressed:
            rec = rec.model_copy(update={
                "action": action_for_units(current, current), "recommended_units": current,
                "delta_units": 0.0, "override_reasons": rec.override_reasons + [view.reason_code],
                "allocation_note": view.note,
            })
        finalized.append(rec)
    finalized.sort(key=lambda r: r.final_rank)
    final_map = {str(r.instrument_id): r for r in finalized}
    effective = finalized + [
        r.model_copy(update={"decision_status": "NOT_SELECTED_FOR_FINAL_JUDGE"})
        for r in research_recs if str(r.instrument_id) not in final_map
    ]
    if incomplete:
        effective = [r.model_copy(update={"actionable": False}) for r in effective]
    return finalized, effective


def allocation_for_records(records: list[RecommendationRecord], base_allocation: dict) -> dict:
    records = [RecommendationRecord.model_validate(normalize_position(r.model_dump())) for r in records]
    result = {k: base_allocation[k] for k in (
        "total_base_units", "cash_instrument_id", "max_total_stock_weight", "actionable", "market_data_basis",
        "research_performance_note", "live_performance_note") if k in base_allocation}
    base = float(result.get("total_base_units") or 1000)
    unpriced = [str(r.instrument_id) for r in records if bool(r.position_held) and r.current_units is None]
    if unpriced:
        result.update(
            stock_weights={}, btc_weight=None, cash_weight=None,
            decision_basis=CONTRACT_VERSION, formula_version=CONTRACT_VERSION,
            rank_factor="validated_gpt_rank",
            shortlist_order=[str(r.instrument_id) for r in records if r.final_rank is not None],
            equity_budget_used=None, actionable=False, weights_confirmed=False,
            allocation_status="INCOMPLETE_VALUATION", unpriced_holdings=sorted(unpriced),
            known_stock_units={str(r.instrument_id): r.current_units for r in records
                               if "btc" not in str(r.instrument_id).lower() and r.current_units is not None and r.position_held},
            diagnostics={str(r.instrument_id): {
                "final_units": r.recommended_units, "action": str(r.action),
                "final_rank": r.final_rank, "source": str(r.source),
                "valuation_status": r.valuation_status,
            } for r in records},
        )
        return result
    stocks = {str(r.instrument_id): float(r.recommended_units or 0)/base for r in records
              if "btc" not in str(r.instrument_id).lower() and float(r.recommended_units or 0) > 1e-9}
    btc = sum(float(r.recommended_units or 0)/base for r in records if "btc" in str(r.instrument_id).lower())
    result.update(stock_weights=stocks, btc_weight=btc, cash_weight=max(0, 1-sum(stocks.values())-btc), decision_basis=CONTRACT_VERSION)
    result["weights_confirmed"] = True
    result.update(
        formula_version=CONTRACT_VERSION, rank_factor="validated_gpt_rank",
        shortlist_order=[str(r.instrument_id) for r in records if r.final_rank is not None],
        equity_budget_used=sum(stocks.values()),
        diagnostics={str(r.instrument_id): {"final_units": r.recommended_units, "action": str(r.action),
                    "final_rank": r.final_rank, "source": str(r.source)} for r in records},
    )
    return result


def render_final_report(records: list[RecommendationRecord], *, settings, allocation: dict, input_as_of: str) -> str:
    labels = {"ENTER": "진입", "ADD": "추가", "HOLD": "보유", "NO_ACTION": "관망", "REDUCE": "축소", "EXIT": "청산"}
    def cell(value):
        return str(value or "").replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    basis = ("구조화 GPT 판단을 확인했으나 보유 평가 미확정으로 전체 배분 제약 검증과 실행은 불가합니다."
             if allocation.get("weights_confirmed") is False else "검증된 GPT 판단과 기존 배분 제약을 반영했습니다.")
    lines = [f"입력 기준: {input_as_of}. {basis}",
             "", "| 순위 | 종목 | 최종 판단 | 목표 단위 | 근거 |", "|---:|---|---|---:|---|"]
    base = float(allocation.get("total_base_units") or 1000)
    adjustments = []
    for rec in records:
        view = actionable_view(rec, settings=settings, total_base_units=base)
        units = "미확정" if view.display_recommended_units is None else f"{view.display_recommended_units:g}"
        lines.append(f"| {rec.final_rank} | {cell(ticker_from_instrument(rec.instrument_id))} | {labels.get(view.display_action, view.display_label)} | {units} | {cell(rec.thesis)} |")
        if rec.override_reasons != ["STRUCTURED_FINAL_JUDGE"]:
            requested = "미확정" if rec.requested_units is None else f"{rec.requested_units:g}"
            applied = "미확정" if rec.recommended_units is None else f"{rec.recommended_units:g}"
            adjustments.append(f"{cell(ticker_from_instrument(rec.instrument_id))}: 요청 {requested} → 적용 {applied} 단위 ({', '.join(rec.override_reasons[1:])}).")
    cash = allocation.get("cash_weight")
    cash_line = "전체 적용 배분의 잔여 현금: 평가 불가 보유 자산 때문에 미확정." if cash is None else f"전체 적용 배분의 잔여 현금: {float(cash):.2%}."
    lines += [""] + adjustments + ["", cash_line,
              "미심사 종목은 별도로 표시한 Gemini 반영 결과를 유지합니다. 주문은 실행하지 않습니다."]
    return "\n".join(lines)
