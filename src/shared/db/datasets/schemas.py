from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DatasetCreate(BaseModel):
    name: str


class DatasetRead(DatasetCreate):
    id: UUID
    files: int
    model_config = ConfigDict(from_attributes=True)


@dataclass(frozen=True)
class DatasetTrainingAudio:
    audio_id: UUID
    duration: float
    speaker_id: str | None
    language: str | None
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DatasetDurationBins:
    total_seconds: tuple[float, ...]
    file_count: int
