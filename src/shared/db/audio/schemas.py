from typing import Any
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from shared.db.assets.schemas import BucketFileRead


class SegmentCreate(BaseModel):
    start: float
    end: float
    text: str
    voice_id: UUID | None
    metadata: dict[str, Any]


class SegmentUpdate(BaseModel):
    start: float
    end: float
    text: str
    voice_id: UUID | None
    metadata: dict[str, Any]


class AudioFileCreate(BaseModel):
    name: str
    bucket_file_id: UUID
    byte_offset: int
    byte_length: int
    duration: float
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    virtual: bool


class AudioFileUpdate(BaseModel):
    name: str
    byte_offset: int
    byte_length: int
    duration: float
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    virtual: bool


class AudioCreate(BaseModel):
    name: str
    wav_bytes: bytes
    duration: float
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    virtual: bool


class AudioUpdate(BaseModel):
    name: str
    wav_bytes: bytes | None
    duration: float
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    virtual: bool


PackedAudioCreate = AudioCreate
PackedAudioUpdate = AudioUpdate


class AudioPartRead(BaseModel):
    start: int
    length: int


class AudioFileRead(AudioFileCreate):
    id: UUID
    bucket_file: BucketFileRead
    updated_at: datetime
    metadata: dict[str, Any] = Field(alias="metadata_")
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
