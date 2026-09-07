"""BTC sleeve liquidity / transfer state (separate from stock cash)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field

from trading_system.ids import InstrumentId, new_instrument_id
from trading_system.recommendations import RecommendationAction


class BtcLiquidityState(StrEnum):
    AVAILABLE = "available"
    UNSETTLED = "unsettled"
    TRANSFER_PENDING = "transfer_pending"
    UNAVAILABLE = "unavailable"


class BtcSleeveAction(StrEnum):
    NO_ACTION = "NO_ACTION"
    HOLD = "HOLD"
    ENTER = "ENTER"
    ADD = "ADD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"


class BtcSleeveState(BaseModel):
    instrument_id: InstrumentId = Field(
        default_factory=lambda: new_instrument_id("btc")
    )
    symbol: str = "BTC/USD"
    liquidity: BtcLiquidityState = BtcLiquidityState.AVAILABLE
    current_units: float = 0.0
    recommended_units: float | None = None
    last_action: BtcSleeveAction = BtcSleeveAction.NO_ACTION
    transfer_caveat: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def transfer_blocks_immediate_rebalance(self) -> bool:
        return self.liquidity in {
            BtcLiquidityState.UNSETTLED,
            BtcLiquidityState.TRANSFER_PENDING,
            BtcLiquidityState.UNAVAILABLE,
        }

    def to_recommendation_action(self) -> RecommendationAction:
        return RecommendationAction(self.last_action.value)
