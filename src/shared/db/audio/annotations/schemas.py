import math
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class AudioAnnotationRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    datasets: tuple[str, ...]
    style_prompt: str | None
    voice_prompt: str | None
    score: float | None
    accuracy: float | None
    metadata: dict[str, Any]
    metadata_hash: str
    segments_hash: str


class AudioAnnotationUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    style_prompt: str | None
    voice_prompt: str | None
    score: float | None
    accuracy: float | None

    @field_validator("score", "accuracy")
    @classmethod
    def finite_number(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("audio annotation numbers must be finite")
        return value
