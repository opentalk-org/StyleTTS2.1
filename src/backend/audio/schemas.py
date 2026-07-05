from typing import Any, Literal
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


AudioSort = Literal["updated", "name", "duration", "speaker", "segments"]


class AudioSegmentRead(BaseModel):
    id: str
    start: float
    end: float
    text: str
    phon: str
    speaker: str


class AudioFileListItem(BaseModel):
    id: UUID
    name: str
    speaker: str
    duration: float
    sample_rate: int | None
    byte_length: int
    size_mb: str
    segments: int
    segment_preview: list[AudioSegmentRead]
    dataset_ids: list[UUID]
    virtual: bool
    metadata: dict[str, Any]
    updated_at: datetime


class AudioFilePage(BaseModel):
    rows: list[AudioFileListItem]
    total: int
