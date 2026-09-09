"""Completed daily bars and explicit price provenance; synthetic temporary data only."""
from datetime import date, datetime, timezone
import pytest
from trading_system.config import Settings
from trading_system.features import build_and_persist_features, features_cover_latest
from trading_system.market.decision_data import latest_price_snapshot, price_description
from trading_system.market.registry import stable_instrument_id
from trading_system.openai_judge import _legacy_b_candidate_line
from trading_system.seed import seed_synthetic_history
from trading_system.storage import Store
from trading_system.technical_factors import fetch_bars_frames

def settings(**kwargs):
    return Settings(_env_file=None, alpaca_api_key=None, alpaca_secret_key=None, gemini_api_key=None,
                    openai_api_key=None, anthropic_api_key=None, smoke_universe=("SPY", "AAPL", "MSFT", "BTC/USD"), **kwargs)

@pytest.fixture
def db(tmp_path):
    store = Store(tmp_path / "stage2.duckdb")
    store.open(acquire_writer=True)
    seed_synthetic_history(store, settings(), days=80, end=date(2024, 6, 14))
    yield store
    store.close()

def test_same_session_revision_rebuilds_but_partial_volume_is_excluded(db):
    conn = db.conn
    end = conn.execute("SELECT max(session_date) FROM equity_daily_bars WHERE finality='final'").fetchone()[0]
    btc_end = conn.execute("SELECT max(session_date) FROM btc_daily_bars WHERE finality='final'").fetchone()[0]
    build_and_persist_features(conn, last_available=end, last_available_btc=btc_end)
    assert features_cover_latest(conn, last_available=end, last_available_btc=btc_end)
    inst = str(stable_instrument_id("AAPL"))
    before = conn.execute("SELECT features_json FROM feature_rows WHERE instrument_id=? AND as_of_date=? AND horizon=5", [inst,end]).fetchone()[0]
    conn.execute("UPDATE equity_daily_bars SET close=close*1.05,volume=volume*2 WHERE instrument_id=? AND session_date=?", [inst,end])
    assert not features_cover_latest(conn, last_available=end, last_available_btc=btc_end)
    assert build_and_persist_features(conn, last_available=end, last_available_btc=btc_end) > 0
    after = conn.execute("SELECT features_json FROM feature_rows WHERE instrument_id=? AND as_of_date=? AND horizon=5", [inst,end]).fetchone()[0]
    assert before != after
    assert features_cover_latest(conn, last_available=end, last_available_btc=btc_end)
    conn.execute("UPDATE equity_daily_bars SET receive_ts=now()")
    assert features_cover_latest(conn, last_available=end, last_available_btc=btc_end)
    conn.execute("UPDATE equity_daily_bars SET finality='preliminary',volume=1 WHERE session_date=?", [end])
    completed = conn.execute("SELECT max(session_date) FROM equity_daily_bars WHERE finality='final'").fetchone()[0]
    build_and_persist_features(conn, last_available=completed, last_available_btc=btc_end)
    assert conn.execute("SELECT count(*) FROM feature_rows WHERE asset_class='us_equity' AND as_of_date>?",[completed]).fetchone()[0] == 0
    frame = fetch_bars_frames(conn, [inst])[inst]
    assert frame.iloc[-1].session_date == completed
    conn.execute("UPDATE equity_daily_bars SET volume=999999999 WHERE finality='preliminary'")
    assert features_cover_latest(conn, last_available=completed, last_available_btc=btc_end)

def test_price_input_preserves_final_vs_intraday_and_receipt_time(db):
    conn = db.conn
    inst = str(stable_instrument_id("AAPL"))
    stamp = datetime.now(timezone.utc)
    final = latest_price_snapshot(conn, inst, as_of=stamp)
    assert final["kind"] == "final_close"
    assert final["price_at"] is None
    line = _legacy_b_candidate_line({"ticker":"AAPL", "price_snapshot":final, "quant":{}})
    assert "확정 일봉 종가" in line and "수집" in line and "출처" in line
    assert "현재가" not in line
    conn.execute("UPDATE equity_daily_bars SET finality='preliminary' WHERE instrument_id=? AND session_date=?", [inst,final["session_date"]])
    partial = latest_price_snapshot(conn, inst, as_of=stamp)
    assert "장중 미완성" in price_description(partial)
    assert partial["received_at"] == final["received_at"]
    assert latest_price_snapshot(conn, "missing", as_of=stamp)["kind"] == "missing"
