"""Kakao scan digest stays within the 200-character template cap."""

from __future__ import annotations

from trading_system.alert_engine import format_scan_kakao_digest
from trading_system.kakao import KAKAO_TEXT_MAX
from trading_system.recommendations import (
    HorizonOutlook,
    RecommendationAction,
    RecommendationRecord,
    RecommendationSource,
)


def _rec(symbol: str, action: RecommendationAction) -> RecommendationRecord:
    return RecommendationRecord(
        source=RecommendationSource.LLM_FINAL,
        tick_id="tick_1",  # type: ignore[arg-type]
        decision_epoch_id="dec_1",  # type: ignore[arg-type]
        feature_snapshot_id="fs_1",  # type: ignore[arg-type]
        instrument_id=f"inst_{symbol.lower()}",  # type: ignore[arg-type]
        action=action,
        horizons=[HorizonOutlook(horizon=5, expected_return=0.01)],
        actionable=True,
    )


def test_scan_kakao_digest_lists_final_buy_and_sell() -> None:
    text = format_scan_kakao_digest(
        final_recs=[
            _rec("nvda", RecommendationAction.ENTER),
            _rec("frsh", RecommendationAction.ADD),
            _rec("pwr", RecommendationAction.EXIT),
            _rec("hubb", RecommendationAction.REDUCE),
            _rec("aapl", RecommendationAction.HOLD),
        ],
        quant_recs=[_rec("msft", RecommendationAction.ENTER)],
        llm_status="AVAILABLE",
    )
    assert text.startswith("[Stock AI]")
    assert len(text) <= KAKAO_TEXT_MAX
    assert "NVDA" in text
    assert "FRSH(추가)" in text
    assert "PWR" in text
    assert "HUBB(축소)" in text
    assert "AAPL" not in text
    assert "MSFT" not in text
    assert "(Quant)" not in text


def test_scan_kakao_digest_falls_back_to_quant() -> None:
    text = format_scan_kakao_digest(
        final_recs=[],
        quant_recs=[_rec("amd", RecommendationAction.BUY)],
        llm_status="NOT_CONFIGURED",
    )
    assert "(Quant)" in text
    assert "AMD" in text
    assert "매도:없음" in text.replace(" ", "")
    assert len(text) <= KAKAO_TEXT_MAX


def test_scan_kakao_digest_fits_many_names() -> None:
    recs = [_rec(f"t{i:02d}", RecommendationAction.ENTER) for i in range(40)]
    recs.append(_rec("zz", RecommendationAction.EXIT))
    text = format_scan_kakao_digest(final_recs=recs)
    assert len(text) <= KAKAO_TEXT_MAX
    assert "매수:" in text
    assert "매도:" in text


def test_scan_kakao_digest_uses_raw_gpt_final_without_quant_fallback() -> None:
    raw = "# 최종 판단\n\nNVDA는 매수, AAPL은 유지, 현금 비중은 20%입니다."
    text = format_scan_kakao_digest(
        final_recs=[],
        quant_recs=[_rec("amd", RecommendationAction.BUY)],
        llm_status="AVAILABLE",
        final_text=raw,
    )
    assert text.startswith("[Stock AI] GPT 최종 판단")
    assert "NVDA는 매수" in text
    assert "(Quant)" not in text
    assert len(text) <= KAKAO_TEXT_MAX
