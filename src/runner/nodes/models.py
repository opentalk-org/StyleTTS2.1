from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class AudioRecordRef:
    audio_file_id: UUID
    name: str
    duration: float
    byte_length: int
    virtual: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BucketAudio:
    audio_file_id: UUID
    name: str
    data: bytes
    sample_rate: int
    channels: int
    duration: float
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioSegment:
    source_audio_id: UUID
    start: float
    end: float
    confidence: float
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class Transcript:
    text: str
    model: str
    source_audio_id: UUID
    start: float | None
    end: float | None
    speaker: str | None
    id: str
    lineage_id: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SaveResult:
    path: Path
    kind: str
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
