"""Stable instrument / issuer identifiers and effective-dated aliases."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import NewType
from uuid import uuid4

from pydantic import BaseModel, Field

InstrumentId = NewType("InstrumentId", str)
IssuerId = NewType("IssuerId", str)
RequestSetId = NewType("RequestSetId", str)
FeatureSnapshotId = NewType("FeatureSnapshotId", str)
OutcomeSnapshotId = NewType("OutcomeSnapshotId", str)
TickId = NewType("TickId", str)
DecisionEpochId = NewType("DecisionEpochId", str)


class AssetClass(StrEnum):
    US_EQUITY = "us_equity"
    BTC = "btc"
    USD_CASH = "usd_cash"


class EvaluationBasis(StrEnum):
    """Declared basis for computing realized returns / labels."""

    RAW_PLUS_CORPORATE_ACTIONS = "raw_plus_corporate_actions"
    ADJUSTED_REVISIONED = "adjusted_revisioned"


class PriceBasisLineage(BaseModel):
    """Price-basis / adjustment-revision lineage shared by feature & outcome snapshots."""

    price_basis: str
    adjustment_revision: str
    evaluation_basis: EvaluationBasis = EvaluationBasis.RAW_PLUS_CORPORATE_ACTIONS
    provider: str | None = None


class InstrumentAlias(BaseModel):
    """Effective-dated provider-symbol alias for an instrument listing."""

    instrument_id: InstrumentId
    provider: str
    symbol: str
    effective_from: date
    effective_to: date | None = None  # None = open-ended
    notes: str | None = None

    def covers(self, as_of: date) -> bool:
        if as_of < self.effective_from:
            return False
        if self.effective_to is not None and as_of > self.effective_to:
            return False
        return True


class InstrumentRecord(BaseModel):
    instrument_id: InstrumentId
    issuer_id: IssuerId
    asset_class: AssetClass
    display_name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IssuerRecord(BaseModel):
    issuer_id: IssuerId
    display_name: str | None = None
    cik: str | None = None  # PIT SEC CIK when known
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeatureSnapshotRef(BaseModel):
    feature_snapshot_id: FeatureSnapshotId
    lineage: PriceBasisLineage
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OutcomeSnapshotRef(BaseModel):
    outcome_snapshot_id: OutcomeSnapshotId
    feature_snapshot_id: FeatureSnapshotId
    lineage: PriceBasisLineage
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def lineage_compatible(self, feature: FeatureSnapshotRef) -> bool:
        if self.feature_snapshot_id != feature.feature_snapshot_id:
            return False
        fl = feature.lineage
        ol = self.lineage
        return (
            ol.price_basis == fl.price_basis
            and ol.adjustment_revision == fl.adjustment_revision
            and ol.evaluation_basis == fl.evaluation_basis
        )


def canonicalize_instrument_id(raw: object) -> InstrumentId:
    """Map a ticker (AMD) or existing id (inst_amd) onto the stable inst_* form."""
    s = str(raw or "").strip()
    if not s:
        raise ValueError("instrument_id is required")
    low = s.lower()
    if low.startswith("inst_"):
        return InstrumentId(low)
    normalized = s.upper().replace("/", "_").replace(".", "_")
    return InstrumentId(f"inst_{normalized.lower()}")


def new_instrument_id(prefix: str = "inst") -> InstrumentId:
    return InstrumentId(f"{prefix}_{uuid4().hex[:12]}")


def new_issuer_id(prefix: str = "iss") -> IssuerId:
    return IssuerId(f"{prefix}_{uuid4().hex[:12]}")


def new_request_set_id() -> RequestSetId:
    return RequestSetId(f"rs_{uuid4().hex}")


def new_feature_snapshot_id() -> FeatureSnapshotId:
    return FeatureSnapshotId(f"fs_{uuid4().hex}")


def new_outcome_snapshot_id() -> OutcomeSnapshotId:
    return OutcomeSnapshotId(f"os_{uuid4().hex}")


def new_tick_id() -> TickId:
    return TickId(f"tick_{uuid4().hex}")


def new_decision_epoch_id() -> DecisionEpochId:
    return DecisionEpochId(f"epoch_{uuid4().hex}")


# Reserved residual cash identity (not a market symbol)
USD_CASH_INSTRUMENT_ID = InstrumentId("cash_usd_residual")
USD_CASH_ISSUER_ID = IssuerId("cash_usd")
BTC_USD_DISPLAY_SYMBOL = "BTC/USD"
