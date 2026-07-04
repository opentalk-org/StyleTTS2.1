from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


@dataclass
class AudioFile:
    path: Path
    sample_rate: int
    channels: int
    duration: float
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VadSegment:
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class VadSegments:
    segments: list[VadSegment]
    source_audio_id: str
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioChunk:
    path: Path
    source_audio_id: str
    start: float
    end: float
    sample_rate: int
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str
    confidence: float | None = None


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]
    source_audio_id: str
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeakerChunk:
    path: Path
    source_audio_id: str
    speaker: str
    start: float
    end: float
    sample_rate: int
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class DenoisedAudio:
    path: Path
    source_audio_id: str
    speaker: str | None
    start: float
    end: float
    sample_rate: int
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    text: str
    model: str
    source_audio_id: str
    start: float | None
    end: float | None
    speaker: str | None
    id: str
    lineage_id: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SaveResult:
    path: Path
    kind: str
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
