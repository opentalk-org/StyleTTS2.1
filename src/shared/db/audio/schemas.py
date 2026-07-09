from typing import Any
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from shared.db.assets.schemas import BucketFileRead
from shared.db.waveforms.schemas import WaveformInput


class SegmentCreate(BaseModel):
    start: float
    end: float
    text: str
    phon: str
    speaker: str
    voice_id: UUID | None = None
    metadata: dict[str, Any]


class SegmentUpdate(BaseModel):
    start: float
    end: float
    text: str
    phon: str
    speaker: str
    voice_id: UUID | None = None
    metadata: dict[str, Any]


class AudioFileCreate(BaseModel):
    name: str
    bucket_file_id: UUID
    byte_offset: int
    byte_length: int
    duration: float
    score: float | None = None
    language: str | None = None
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    virtual: bool


class AudioFileUpdate(BaseModel):
    name: str
    byte_offset: int
    byte_length: int
    duration: float
    score: float | None = None
    language: str | None = None
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    virtual: bool


class AudioCreate(BaseModel):
    name: str
    wav_bytes: bytes
    duration: float
    score: float | None = None
    language: str | None = None
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    virtual: bool
    waveform: WaveformInput | None = None


class AudioUpdate(BaseModel):
    name: str
    wav_bytes: bytes | None
    duration: float
    score: float | None = None
    language: str | None = None
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    virtual: bool
    waveform: WaveformInput | None = None


PackedAudioCreate = AudioCreate
PackedAudioUpdate = AudioUpdate


class AudioPartRead(BaseModel):
    start: int
    length: int


class AudioBucketLocation(BaseModel):
    """Which bucket file an audio row lives in, and how large its slice is.

    Exposes only the bucket grouping key and the audio byte size so callers can
    order/size work by bucket without managing pack offsets themselves."""

    audio_file_id: UUID
    bucket_file_id: UUID
    byte_length: int


class AudioFileRead(AudioFileCreate):
    id: UUID
    bucket_file: BucketFileRead
    updated_at: datetime
    metadata: dict[str, Any] = Field(alias="metadata_")
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
