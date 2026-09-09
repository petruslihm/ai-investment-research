"""Public synthetic scenarios through the real decision, constraint and storage code.

No market/LLM provider, model fitting or credentials are used. The stage outputs
are authored examples, not model predictions or measured investment results.
"""
from __future__ import annotations

import copy
import html
import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from trading_system.btc_sleeve import BtcSleeveState
from trading_system.config import Settings
from trading_system.final_decision import CONTRACT_VERSION, allocation_for_records, apply_final_judgment, render_final_report
from trading_system.recommendations import RecommendationRecord, RecommendationSource, normalize_position
from trading_system.storage import Store
from trading_system.storage.ticks import TickCommitPayload, commit_tick
from trading_system.ui.app import render_recommendations_html
from trading_system.v1_cycle import load_last_ui_snapshot

SCENARIOS = {"validated": "최종 판단 반영", "failure": "호출 실패", "invalid": "파싱 실패", "unpriced": "평가 불가 보유"}
AS_OF = "2025-01-16T15:10:00+00:00"


def demo_settings():
    return Settings(_env_file=None, alpaca_api_key=None, alpaca_secret_key=None,
                    openai_api_key=None, gemini_api_key=None, anthropic_api_key=None,
                    max_single_stock_weight=.25)


def build_demo_snapshot(scenario: str = "validated") -> dict:
    if scenario not in SCENARIOS:
        raise ValueError("unknown demo scenario")
    cfg = demo_settings()
    tick = "tick_demo_" + uuid4().hex
    input_id = "synthetic_" + scenario
    quant = []
    for name, current, target in [("ALPHA", 0, 120), ("BETA", 60, 60), ("GAMMA", 0, 80)]:
        missing = scenario == "unpriced" and name == "BETA"
        price = {"price": None if missing else 100.0, "kind": "missing" if missing else "partial_daily_bar" if name == "GAMMA" else "final_close",
                 "provider": "SYNTHETIC fixture", "price_at": None,
                 "session_date": "2025-01-16" if name == "GAMMA" else "2025-01-15", "input_as_of": AS_OF, "received_at": AS_OF}
        row = normalize_position({"current_units": None if missing else current,
                                  "acquisition_units": 50 if current else 0,
                                  "marked_units_final": None if missing else current,
                                  "recommended_units": target, "actionable": False})
        quant.append(RecommendationRecord.model_validate({
            **row, "recommendation_id": f"{tick}_{name}_quant", "source": "quant_only",
            "tick_id": tick, "decision_epoch_id": "epoch_synthetic", "feature_snapshot_id": "fs_synthetic",
            "instrument_id": "inst_demo_" + name.lower(), "action": "HOLD" if current else "ENTER",
            "input_id": input_id, "input_as_of": AS_OF, "price_snapshot": price,
            "thesis": "SYNTHETIC: 수작업으로 정한 퀀트 예제. 실제 모델 예측이 아닙니다.",
        }))
    research = [r.model_copy(update={"recommendation_id": f"{r.recommendation_id}_research",
                 "source": RecommendationSource.RESEARCH_ADJUSTED, "llm_tainted": True,
                 "recommended_units": 150 if "alpha" in r.instrument_id else r.recommended_units,
                 "thesis": "DEMO Gemini: 가상 원문 근거를 정리한 예제. API 호출 없음."}) for r in quant]
    package = {"contract_version": CONTRACT_VERSION, "input_id": input_id, "input_as_of": AS_OF,
               "portfolio": {"total_base_units": 1000},
               "candidates": [{"instrument_id": str(r.instrument_id), "quant": r.model_dump(mode="json")} for r in research]}
    choices = []
    for rank, index in enumerate((2, 1, 0), 1):
        r = research[index]
        choices.append({"instrument_id": str(r.instrument_id), "action": "BUY" if index == 2 else "WATCH",
                        "recommended_units": 400 if index == 2 else None, "rank": rank,
                        "thesis": "DEMO GPT: 가상 근거에 따른 예제 판단. 실제 투자 의견이 아닙니다.",
                        "contrary_evidence": "SYNTHETIC: 전망 불확실성 예제", "change_conditions": "가상 근거 변경 시 재평가"})
    raw = json.dumps({"input_id": input_id, "input_as_of": AS_OF, "recommendations": choices})
    committee = {"status": "AVAILABLE", "validated": False, "demo": True}
    final = []
    effective = copy.deepcopy(research)
    if scenario == "failure":
        committee["status"] = "RATE_LIMITED"
    else:
        try:
            final, effective = apply_final_judgment("invalid example" if scenario == "invalid" else raw,
                    package, research, settings=cfg, btc_state=BtcSleeveState())
            committee["validated"] = True
        except ValueError:
            committee["status"] = "INVALID_OUTPUT"
    allocation = allocation_for_records(effective, {"total_base_units": 1000, "actionable": False})
    if committee["validated"]:
        committee["final_text"] = render_final_report(final, settings=cfg, allocation=allocation, input_as_of=AS_OF)
    else:
        committee.update(fallback="research_adjusted", final_text="",
                         note="DEMO: GPT 최종 판단 미적용. Gemini 예제 결과를 대체 표시합니다.")
        effective = [r.model_copy(update={"decision_status": "FALLBACK_RESEARCH_ADJUSTED"}) for r in effective]
    return {"tick_id": tick, "input_id": input_id, "input_as_of": AS_OF, "decision_contract": CONTRACT_VERSION,
            "mode": "DEMO / SYNTHETIC", "scenario": scenario, "no_trading": True,
            "used_synthetic_market": True, "allocation": allocation,
            "quant": [r.model_dump(mode="json") for r in quant],
            "research_adjusted": [r.model_dump(mode="json") for r in research],
            "llm_final": [r.model_dump(mode="json") for r in final],
            "effective": [r.model_dump(mode="json") for r in effective],
            "portfolio_committee": committee, "llm_judge_status": committee["status"], "quant_status": "SYNTHETIC"}


def save_demo(conn, snap):
    commit_tick(conn, TickCommitPayload(tick_id=snap["tick_id"], observations=[{
        "instrument_id": "decision_comparison_v1", "input_id": snap["input_id"], "input_as_of": snap["input_as_of"],
        "snapshot": snap}], recommendations=snap["quant"] + snap["research_adjusted"] + snap["llm_final"]))


def render_demo(snap):
    e = html.escape
    stages = [("quant", "Quant"), ("research_adjusted", "Gemini"), ("llm_final", "GPT"), ("effective", "effective")]
    table = ""
    for row in snap["quant"]:
        cells = []
        for key, _ in stages:
            r = next((v for v in snap[key] if v["instrument_id"] == row["instrument_id"]), None)
            units = "미확정" if r and r["recommended_units"] is None else f'{r["recommended_units"]:g}' if r else "—"
            cells.append(f'<td>{e(str(r["action"])) if r else "미적용"}<br><strong>{units}</strong> units</td>')
        table += f'<tr><th>{e(row["instrument_id"].removeprefix("inst_demo_"))}</th>{"".join(cells)}</tr>'
    nav = "".join(f'<a class="scenario" href="/?scenario={key}">{label}</a>' for key, label in SCENARIOS.items())
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decision Lab · SYNTHETIC</title><link rel="stylesheet" href="/assets/shell.css">
<style>body{{display:block;background:#f4f6fa;color:#172637}}.lab{{max-width:1120px;margin:auto;padding:32px 24px}}.hero{{background:#142e3f;color:white;border-radius:20px;padding:30px;margin:20px 0}}.hero p{{color:#d6e8ef}}.eyebrow{{color:#91efd2;letter-spacing:.12em;font-size:13px}}h1{{font-size:38px;margin:12px 0}}.scenario{{display:inline-block;margin:5px 6px 5px 0;padding:10px 16px;border:1px solid #ccd6df;border-radius:8px;background:white;color:#194c59;text-decoration:none}}table{{width:100%;border-collapse:collapse;background:white;border-radius:12px}}th,td{{padding:14px;text-align:left;border-bottom:1px solid #e6edf2}}.flow{{padding:18px;background:#e5f3ed;border-radius:12px}}details{{margin:20px 0}}pre{{max-height:350px;overflow:auto;background:white;padding:20px}}.lab h2{{margin-top:28px}}</style></head><body><main class="lab">
<div class="hero"><div class="eyebrow">ENGINEERING PORTFOLIO / DEMO · SYNTHETIC</div><h1>설명에서 끝나지 않는 판단</h1><p>가격의 시점, 판단의 변경, 저장된 최종 결과를 한 흐름으로 확인합니다.</p><p>모든 종목·가격·단계별 답변은 가상 예제입니다. 실제 모델 추천·API 응답·투자 성과가 아닙니다. 외부 호출·학습·주문 없음.</p></div>
<div class="flow">Quant 후보 → Gemini 근거 예제 → GPT 구조화 판단 → 제약 적용 → <strong>effective</strong> → DuckDB 저장 → 화면 복원</div>
<nav>{nav}<a class="scenario" href="/resume">저장 결과 다시 읽기</a></nav>
<h2>{e(SCENARIOS[snap['scenario']])}</h2><p>고정 입력 시각 {e(snap['input_as_of'])} · 가격 출처 SYNTHETIC · 확정 종가·장중 미완성 관측 예제</p>
<h2>단계별 비교</h2><p>아래는 비교 기록입니다. 최종 표시는 effective만 사용합니다. units는 가상 배분단위이며 주식 수나 금액이 아닙니다.</p>
<table><thead><tr><th>가상 종목</th>{''.join(f'<th>{label}</th>' for _,label in stages)}</tr></thead><tbody>{table}</tbody></table>
<h2>저장된 최종 결과</h2>{render_recommendations_html(snap, settings=demo_settings())}
<details><summary>입력·단계·effective 저장 기록 보기</summary><pre>{e(json.dumps(snap,ensure_ascii=False,indent=2))}</pre></details>
<p>연구·데모 도구 · 수익 보장 없음 · 실제 초과수익 검증 없음</p></main></body></html>'''


def create_demo_app(data_dir: Path):
    app = FastAPI(title="Synthetic Decision Lab")
    app.mount("/assets", StaticFiles(directory=Path(__file__).parent / "ui/static"), name="assets")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    def stored(scenario=None):
        store = Store(data_dir / "demo.duckdb")
        store.open(acquire_writer=True)
        try:
            snap = load_last_ui_snapshot(store.conn)
            if scenario is not None or snap is None:
                snap = build_demo_snapshot(scenario or "validated")
                save_demo(store.conn, snap)
            return load_last_ui_snapshot(store.conn)
        finally:
            store.close()

    @app.get("/", response_class=HTMLResponse)
    def home(scenario: str = Query("validated", pattern="^(validated|failure|invalid|unpriced)$")):
        return render_demo(stored(scenario))

    @app.get("/resume", response_class=HTMLResponse)
    def resume():
        return render_demo(stored())

    @app.get("/api/demo/snapshot")
    def snapshot():
        return stored()

    @app.get("/api/health")
    def health():
        return {"ok": True, "mode": "DEMO / SYNTHETIC", "provider_calls": False, "trading": False}

    return app
