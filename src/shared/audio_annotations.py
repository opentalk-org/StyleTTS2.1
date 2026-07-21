from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class AudioAnnotations(BaseModel):
    model_config = ConfigDict(frozen=True)

    speaker_id: str | None = None
    score: float | None = None
    accuracy: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HasAudioAnnotations:
    annotations: AudioAnnotations

    @property
    def speaker_id(self) -> str | None:
        return self.annotations.speaker_id

    @property
    def score(self) -> float | None:
        return self.annotations.score

    @property
    def accuracy(self) -> float | None:
        return self.annotations.accuracy

    @property
    def metadata(self) -> dict[str, Any]:
        return self.annotations.metadata
