import json
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from shared.audio_annotations import AudioAnnotations
from shared.db.clickhouse.types import utc_datetime


class StorageKind(str, Enum):
    PACKED = "packed"
    EXTERNAL = "external"


class AudioFileRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    updated_at: datetime
    name: str
    bucket_file_id: UUID | None
    byte_offset: int
    duration: float
    byte_length: int
    score: float | None
    language: str | None
    style_prompt: str | None
    voice_prompt: str | None
    virtual: bool
    storage_kind: StorageKind
    storage_ref: dict[str, Any] | None
    metadata: dict[str, Any]

    _updated_at_utc = field_validator("updated_at")(utc_datetime)

    @field_validator("storage_ref", "metadata", mode="before")
    @classmethod
    def parse_json_fields(cls, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value

    @property
    def annotations(self) -> AudioAnnotations:
        return AudioAnnotations(score=self.score, metadata=self.metadata)

    @property
    def metadata_(self) -> dict[str, Any]:
        return self.metadata


class AudioFileUpdate(BaseModel):
    name: str
    bucket_file_id: UUID | None
    byte_offset: int
    duration: float
    byte_length: int
    score: float | None
    language: str | None
    style_prompt: str | None
    voice_prompt: str | None
    virtual: bool
    storage_kind: StorageKind
    storage_ref: dict[str, Any] | None
    metadata: dict[str, Any]
    updated_at: datetime

    _updated_at_utc = field_validator("updated_at")(utc_datetime)


class AudioSegmentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    audio_file_id: UUID
    updated_at: datetime
    position: int
    start_seconds: float
    end_seconds: float
    text: str
    phon: str
    kind: str
    accuracy: float | None
    speaker_id: str | None
    metadata: dict[str, Any]
    alignment: list[dict[str, Any]] | None

    _updated_at_utc = field_validator("updated_at")(utc_datetime)

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value

    @field_validator("alignment", mode="before")
    @classmethod
    def parse_alignment(cls, value: Any) -> Any:
        if value is None:
            return None
        if not value or isinstance(value[0], dict):
            return value
        return [
            {"word": word, "start": start, "end": end}
            for word, start, end in value
        ]

    def as_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start_seconds,
            "end": self.end_seconds,
            "text": self.text,
            "phon": self.phon,
            "type": self.kind,
            "annotations": AudioAnnotations(
                speaker_id=self.speaker_id,
                accuracy=self.accuracy,
                metadata=self.metadata,
            ).model_dump(mode="json"),
            "alignment": self.alignment,
        }
