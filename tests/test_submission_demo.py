"""Submission scenarios use public fixtures only and forbid external networking."""
import copy
import json
import socket

import pytest
from fastapi.testclient import TestClient

from trading_system.demo import build_demo_snapshot, create_demo_app, demo_settings, render_demo, save_demo
from trading_system.final_decision import apply_final_judgment, prepare_pending
from trading_system.btc_sleeve import BtcSleeveState
from trading_system.market.decision_data import price_description
from trading_system.recommendations import normalize_position
from trading_system.storage import Store
from trading_system.v1_cycle import load_last_ui_snapshot


@pytest.fixture(autouse=True)
def no_provider_network(monkeypatch):
    connect = socket.socket.connect
    def forbidden(sock, address):
        # Windows asyncio uses a loopback socket pair for its event loop.
        if address[0] in {"127.0.0.1", "::1"}:
            return connect(sock, address)
        raise AssertionError("Demo must not open network connections")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def test_validated_effective_controls_direction_rank_units_and_report(tmp_path):
    snap = build_demo_snapshot()
    assert [r["instrument_id"] for r in snap["effective"]] == ["inst_demo_gamma", "inst_demo_beta", "inst_demo_alpha"]
    assert [(r["action"], r["recommended_units"]) for r in snap["effective"]] == [("ENTER",250),("HOLD",60),("NO_ACTION",0)]
    assert snap["quant"][0]["recommended_units"] == 120
    assert snap["research_adjusted"][0]["recommended_units"] == 150
    assert snap["effective"][0]["requested_units"] == 400
    assert all(not r["actionable"] for r in snap["effective"])
    assert "400 → 적용 250" in snap["portfolio_committee"]["final_text"]
    store = Store(tmp_path / "test.duckdb")
    store.open(acquire_writer=True)
    try:
        save_demo(store.conn, snap)
        save_demo(store.conn, snap)
        assert store.conn.execute("SELECT count(*) FROM ticks").fetchone()[0] == 1
        assert load_last_ui_snapshot(store.conn) == snap
    finally:
        store.close()
    app = create_demo_app(tmp_path)
    # TestClient is an in-process ASGI request; no listener or external provider.
    client = TestClient(app)
    client.get("/")
    saved = client.get("/api/demo/snapshot").json()
    restored = TestClient(create_demo_app(tmp_path)).get("/api/demo/snapshot").json()
    assert restored == saved
    assert client.get("/resume").status_code == 200


@pytest.mark.parametrize("scenario,status", [("failure","RATE_LIMITED"),("invalid","INVALID_OUTPUT")])
def test_failures_are_explicit_research_fallback(scenario,status):
    snap = build_demo_snapshot(scenario)
    assert snap["portfolio_committee"]["status"] == status
    assert not snap["portfolio_committee"]["validated"] and snap["llm_final"] == []
    assert [r["recommended_units"] for r in snap["effective"]] == [r["recommended_units"] for r in snap["research_adjusted"]]
    assert "대체 표시" in render_demo(snap)
    assert "주식 29.0%" in render_demo(snap)
    assert "예측값에서 주식 0.0%" not in render_demo(snap)


def test_unpriced_is_not_zero_or_cash_and_survives_storage(tmp_path):
    snap = build_demo_snapshot("unpriced")
    for key in ("quant","research_adjusted","llm_final","effective"):
        row = next(r for r in snap[key] if r["instrument_id"] == "inst_demo_beta")
        assert row["position_held"] and row["current_units"] is None and row["recommended_units"] is None
        assert row["acquisition_units"] == 50 and not row["actionable"]
    assert snap["allocation"]["cash_weight"] is None
    page = render_demo(snap)
    assert "주식 미확정" in page and "현금 미확정" in page
    client = TestClient(create_demo_app(tmp_path))
    client.get("/?scenario=unpriced")
    assert client.get("/api/demo/snapshot").json()["allocation"]["cash_weight"] is None
    assert "전체 비중·합계 미확정" in client.get("/resume").text


def test_position_zero_null_and_price_kind_are_distinct():
    assert normalize_position({"current_units":0})["position_held"] is False
    assert normalize_position({"current_units":None,"acquisition_units":50})["current_units"] is None
    assert normalize_position({"current_units":60,"acquisition_units":50})["current_units"] == 60
    price = build_demo_snapshot()["quant"][0]["price_snapshot"]
    assert "확정 일봉 종가" in price_description(price)
    assert "장중 미완성" in price_description({**price,"kind":"partial_daily_bar"})
    assert "SYNTHETIC" in price_description(price) and "2025-01-15" in price_description(price)


def test_current_contract_resume_rejects_changed_input_and_ambiguous_position():
    snap=build_demo_snapshot()
    package={"contract_version":snap["decision_contract"],"resume_guard":"same",
             "stages":{"quant":snap["quant"],"research_adjusted":snap["research_adjusted"]},
             "candidates":[{"instrument_id":r["instrument_id"],"quant":r} for r in snap["research_adjusted"]]}
    assert prepare_pending(package,resume_guard="same") == package
    assert prepare_pending(package,resume_guard="changed") is None
    assert prepare_pending({**package,"contract_version":"older"},resume_guard="same") is None
    broken=copy.deepcopy(package)
    broken["stages"]["quant"][0]["current_units"]=None
    broken["stages"]["quant"][0]["acquisition_units"]=None
    broken["stages"]["quant"][0]["position_held"]=None
    with pytest.raises(ValueError,match="REEVALUATION_REQUIRED"):
        prepare_pending(broken,resume_guard="same")


def test_demo_ignores_configured_keys(monkeypatch,tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY","synthetic-do-not-use")
    monkeypatch.setenv("ALPACA_API_KEY","synthetic-do-not-use")
    client=TestClient(create_demo_app(tmp_path))
    assert client.get("/api/health").json()["provider_calls"] is False
    assert "DEMO · SYNTHETIC" in client.get("/").text


def test_cli_help_works_in_windows_console(monkeypatch,capsys):
    from trading_system.cli import main
    monkeypatch.setattr("sys.argv",["investassist","--help"])
    with pytest.raises(SystemExit) as out:
        main()
    assert out.value.code == 0
    help_text = capsys.readouterr().out
    help_text.encode("cp949")
    assert "--demo" in help_text and "--app" in help_text
