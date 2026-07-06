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

    @property
    def id(self) -> str:
        return stable_id("audio_ref", self.audio_file_id)

    @property
    def lineage_id(self) -> str:
        return self.id


@dataclass(frozen=True)
class Audio:
    audio_file_id: UUID
    name: str
    data: bytes | None
    sample_rate: int
    channels: int
    start: float
    end: float
    confidence: float
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    byte_length: int = 0
    virtual: bool = False
    segments: list[AudioSegment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class AudioSegment:
    source_audio_id: UUID
    name: str
    start: float
    end: float
    sample_rate: int
    channels: int
    text: str
    phon: str
    id: str
    lineage_id: str
    segment_id: str | None = None
    speaker: str | None = None
    voice_id: UUID | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class SegmentGroup:
    name: str
    segments: list[AudioSegment]
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


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


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_id: UUID
    name: str
    path: Path
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetBundleRef:
    bundle_key: str
    name: str
    paths: list[Path]
    id: str
    lineage_id: str
    extra_file_ids: list[UUID] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingManifest:
    dataset_id: UUID
    audio_file_ids: list[UUID]
    base_checkpoint: CheckpointRef
    phoneme_alphabet: list[str]
    id: str
    lineage_id: str
    assets: AssetBundleRef | None = None
    ood_text_set_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    training_run_id: str
    checkpoint: CheckpointRef
    id: str
    lineage_id: str
    artifacts: AssetBundleRef | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SynthesisResult:
    request_id: str
    audio: Audio
    id: str
    lineage_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
