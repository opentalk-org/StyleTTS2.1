from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from shared.audio_annotations import AudioAnnotations


AudioSort = Literal["updated", "name", "duration", "segments"]


class WordAlignment(BaseModel):
    word: str
    start: float
    end: float


class AudioSegmentRead(BaseModel):
    id: str
    start: float
    end: float
    text: str
    phon: str
    annotations: AudioAnnotations
    type_: str
    alignment: list[WordAlignment] | None = None


class AudioSegmentWrite(BaseModel):
    id: str
    start: float
    end: float
    text: str
    phon: str
    annotations: AudioAnnotations
    type_: str
    alignment: list[WordAlignment] | None = None


class AudioRenamePayload(BaseModel):
    name: str


class AudioScorePayload(BaseModel):
    score: float | None = None


class AudioLanguagePayload(BaseModel):
    language: str | None = None


class AudioStylePromptPayload(BaseModel):
    style_prompt: str | None = None


class AudioVoicePromptPayload(BaseModel):
    voice_prompt: str | None = None


class AddToDatasetRequest(BaseModel):
    dataset_id: str
    mode: Literal["ids", "filter"]
    audio_file_ids: list[str] = []
    query: str = ""
    language: str = ""
    dataset: str = "all"


class AudioFileListItem(BaseModel):
    id: UUID
    name: str
    annotations: AudioAnnotations
    duration: float
    language: str | None
    style_prompt: str | None
    voice_prompt: str | None
    sample_rate: int | None
    byte_length: int
    size_mb: str
    segments: int
    segment_preview: list[AudioSegmentRead]
    dataset_ids: list[UUID]
    virtual: bool
    storage_kind: Literal["packed", "external"]
    updated_at: datetime


class AudioFilePage(BaseModel):
    rows: list[AudioFileListItem]
    next_cursor: str | None
    has_more: bool


class WaveformStatusRead(BaseModel):
    status: Literal["ready", "pending", "error"]
