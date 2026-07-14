from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingRunState(StrEnum):
    OPEN = "open"
    SEALED = "sealed"
    FAILED = "failed"


class EmbeddingRunCreate(BaseModel):
    dataset_id: UUID
    expected_count: int = Field(ge=0)
    dimension: int = Field(gt=0)
    model_revision: str
    preprocessing_version: str


class EmbeddingRunRead(EmbeddingRunCreate):
    id: UUID
    stored_count: int
    state: EmbeddingRunState
    failure_details: dict[str, Any] | None
    created_at: datetime
    sealed_at: datetime | None
    model_config = ConfigDict(from_attributes=True)


class EmbeddingShardCreate(BaseModel):
    artifact_id: UUID
    row_count: int = Field(gt=0)
    dimension: int = Field(gt=0)
    model_revision: str
    preprocessing_version: str


class EmbeddingShardRead(EmbeddingShardCreate):
    id: UUID
    run_id: UUID
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class EmbeddingShardCollection(BaseModel):
    run_id: UUID
    artifact_ids: list[UUID]
    stored_count: int
    expected_count: int
    dimension: int
    model_revision: str
    preprocessing_version: str
    sealed_now: bool


class ClusteringRunState(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"
    FAILED = "failed"


class SpeakerClusterStatus(StrEnum):
    ACCEPTED = "accepted"
    SUSPICIOUS = "suspicious"


class SpeakerAssignmentOutcome(StrEnum):
    ACCEPTED = "accepted"
    PROVISIONAL_NEW = "provisional_new"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


class ClusteringArtifactRole(StrEnum):
    CANDIDATE = "candidate"
    ASSIGNMENT = "assignment"
    PROTOTYPE = "prototype"
    INDEX = "index"


class ClusteringRunCreate(BaseModel):
    run_key: str = Field(min_length=1)
    embedding_run_id: UUID
    expected_count: int = Field(ge=0)
    index_factory: str
    threshold_version: str
    settings: dict[str, Any]


class ClusteringArtifactCreate(BaseModel):
    artifact_id: UUID
    role: ClusteringArtifactRole
    ordinal: int = Field(ge=0)
    row_count: int = Field(ge=0)


class ClusterSummaryCreate(BaseModel):
    cluster_key: str
    member_count: int = Field(gt=0)
    duration_seconds: float = Field(ge=0.0)
    dispersion: float = Field(ge=0.0)
    status: SpeakerClusterStatus


class ClusteringRunComplete(BaseModel):
    assignment_count: int = Field(ge=0)
    prototype_artifact_id: UUID
    index_artifact_id: UUID
