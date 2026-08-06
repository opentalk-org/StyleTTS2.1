from typing import Any
from uuid import UUID

from pydantic import BaseModel

from shared.audio_annotations import AudioAnnotations
from shared.db.waveforms.schemas import WaveformInput


class SegmentCreate(BaseModel):
    start: float
    end: float
    text: str
    phon: str
    annotations: AudioAnnotations


class SegmentUpdate(BaseModel):
    start: float
    end: float
    text: str
    phon: str
    annotations: AudioAnnotations


class AudioCreate(BaseModel):
    name: str
    wav_bytes: bytes
    duration: float
    annotations: AudioAnnotations
    language: str | None = None
    style_prompt: str | None = None
    voice_prompt: str | None = None
    segments: list[dict[str, Any]]
    virtual: bool
    waveform: WaveformInput | None = None


class ExternalAudioLocation(BaseModel):
    provider: str
    host: str
    path: str
    item_index: int


class ExternalAudioCreate(BaseModel):
    id: UUID
    name: str
    duration: float
    annotations: AudioAnnotations
    language: str | None = None
    style_prompt: str | None = None
    voice_prompt: str | None = None
    segments: list[dict[str, Any]]
    storage_ref: ExternalAudioLocation


class AudioUpdate(BaseModel):
    name: str
    wav_bytes: bytes | None
    duration: float
    annotations: AudioAnnotations
    language: str | None = None
    style_prompt: str | None = None
    voice_prompt: str | None = None
    segments: list[dict[str, Any]]
    virtual: bool
    waveform: WaveformInput | None = None


PackedAudioCreate = AudioCreate
PackedAudioUpdate = AudioUpdate


class AudioPartRead(BaseModel):
    start: int
    length: int


class AudioStorageLocation(BaseModel):
    audio_file_id: UUID
    object_path: str
    byte_offset: int
    byte_length: int


class AudioBucketLocation(BaseModel):
    """Which bucket file an audio row lives in, and how large its slice is.

    Exposes only the bucket grouping key and the audio byte size so callers can
    order/size work by bucket without managing pack offsets themselves."""

    audio_file_id: UUID
    bucket_file_id: UUID
    byte_length: int


class AudioFileReference(BaseModel):
    id: UUID
    name: str
    duration: float
    language: str | None
    annotations: AudioAnnotations
    byte_length: int
    virtual: bool
    style_prompt: str | None
    voice_prompt: str | None
