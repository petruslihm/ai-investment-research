"""Decision-epoch pins, artifact refs, and Model Change Journal event shapes."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from trading_system.ids import DecisionEpochId, new_decision_epoch_id


class ModelFamily(StrEnum):
    RIDGE = "ridge"
    LIGHTGBM_REG = "lightgbm_reg"
    LAMBDARANK = "lambdarank"
    TORCH_SEQUENCE = "torch_sequence"
    ONLINE_SGD = "online_sgd"
    ENSEMBLE = "ensemble"


class ModelArtifactRef(BaseModel):
    family: ModelFamily
    horizon: int
    version: str
    artifact_hash: str
    path: str | None = None


class DecisionEpochManifest(BaseModel):
    """Immutable inference dependency pin for one tick."""

    decision_epoch_id: DecisionEpochId = Field(default_factory=new_decision_epoch_id)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    batch_models: list[ModelArtifactRef] = Field(default_factory=list)
    online_models: list[ModelArtifactRef] = Field(default_factory=list)
    preprocess_version: str
    ensemble_weights_hash: str
    thresholds_hash: str
    formula_versions: dict[str, str] = Field(default_factory=dict)
    parent_epoch_id: DecisionEpochId | None = None
    notes: str | None = None

    def dependency_fingerprint(self) -> str:
        parts = [
            self.preprocess_version,
            self.ensemble_weights_hash,
            self.thresholds_hash,
            *(f"{k}={v}" for k, v in sorted(self.formula_versions.items())),
        ]
        for m in sorted(
            self.batch_models + self.online_models,
            key=lambda x: (x.family.value, x.horizon, x.version),
        ):
            parts.append(f"{m.family}:{m.horizon}:{m.version}:{m.artifact_hash}")
        return "|".join(parts)


class OnlineUpdateLedgerEntry(BaseModel):
    """Crash-idempotent online learning ledger row."""

    model_stream: str
    target_version: str
    prediction_label_id: str
    parent_epoch_id: DecisionEpochId
    update_batch_hash: str
    status: str = "staged"  # staged | consumed | published | rolled_back
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelChangeKind(StrEnum):
    RETRAINED = "retrained"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ENSEMBLE_WEIGHT_CHANGED = "ensemble_weight_changed"
    ONLINE_UPDATED = "online_updated"
    HORIZON_WEIGHT_CHANGED = "horizon_weight_changed"
    ROLLED_BACK = "rolled_back"


class ModelChangeJournalEvent(BaseModel):
    event_id: str
    kind: ModelChangeKind
    model_family: ModelFamily | None = None
    asset_class: str | None = None
    horizon: int | None = None
    previous_version: str | None = None
    new_version: str | None = None
    previous_ensemble_weight: float | None = None
    new_ensemble_weight: float | None = None
    # Un-damped EWMA target (see ml_engine.update_weights): what the weight would
    # jump straight to at lr=1.0. new_ensemble_weight only moves lr of the way from
    # previous toward this. Exposed so a small "before -> after" delta can be told
    # apart from "the target was close" vs "the target was far but lr damped it".
    target_ensemble_weight: float | None = None
    matured_label_count: int | None = None
    matured_label_ids_hash: str | None = None
    training_period: str | None = None
    evaluation_period: str | None = None
    sample_count: int | None = None
    metric_name: str | None = None
    metric_before: float | None = None
    metric_after: float | None = None
    promotion_result: str | None = None
    decision_epoch_before: DecisionEpochId | None = None
    decision_epoch_after: DecisionEpochId | None = None
    reason_codes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


