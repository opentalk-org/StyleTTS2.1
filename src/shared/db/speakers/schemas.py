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
