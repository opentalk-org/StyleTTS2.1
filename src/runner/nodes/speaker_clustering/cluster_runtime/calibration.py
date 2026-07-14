from __future__ import annotations

from enum import StrEnum
import json
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from runner.nodes.models import SpeakerEmbeddingSetRef
from shared.db import database_session
from shared.db.assets import crud as asset_crud


class ThresholdMode(StrEnum):
    EXPLORATORY = "exploratory"
    CALIBRATED = "calibrated"


class SpeakerThresholdCalibration(BaseModel):
    model_config = ConfigDict(frozen=True)

    threshold_version: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)
    exact_edge_threshold: float = Field(ge=-1.0, le=1.0)
    accept_threshold: float = Field(ge=-1.0, le=1.0)
    min_margin: float = Field(ge=0.0, le=2.0)
    new_threshold: float = Field(ge=-1.0, le=1.0)
    max_cluster_dispersion: float = Field(ge=0.0, le=2.0)


def validate_calibration(
    embedding_set: SpeakerEmbeddingSetRef,
    settings: Any,
) -> None:
    if settings.threshold_mode is ThresholdMode.EXPLORATORY:
        return
    artifact_id = settings.calibration_artifact_id
    assert isinstance(artifact_id, UUID), "calibrated thresholds require an artifact"
    with database_session() as session:
        artifact = asset_crud.get_extra_file(session, artifact_id)
        if artifact.type_ != "speaker_threshold_calibration":
            raise ValueError(
                f"threshold calibration artifact {artifact_id} has type {artifact.type_}"
            )
        payload = asset_crud.read_extra_file(session, artifact_id)
    calibration = SpeakerThresholdCalibration.model_validate(json.loads(payload))
    expected = {
        "threshold_version": settings.threshold_version,
        "model_revision": embedding_set.model_revision,
        "preprocessing_version": embedding_set.preprocessing_version,
        "exact_edge_threshold": settings.exact_edge_threshold,
        "accept_threshold": settings.accept_threshold,
        "min_margin": settings.min_margin,
        "new_threshold": settings.new_threshold,
        "max_cluster_dispersion": settings.max_cluster_dispersion,
    }
    if calibration.model_dump() != expected:
        raise ValueError(
            f"threshold calibration artifact {artifact_id} does not match the "
            "embedding identity and explicitly configured thresholds"
        )
