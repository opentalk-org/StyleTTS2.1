from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from shared.audio_annotations import AudioAnnotations, HasAudioAnnotations


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def typed_checkpoint(value: "CheckpointRef | dict[str, Any]") -> "CheckpointRef":
    if isinstance(value, CheckpointRef):
        return value
    raise TypeError("expected a resolved CheckpointRef")


def typed_assets(
    value: "AssetBundleRef | dict[str, Any] | None",
) -> "AssetBundleRef | None":
    if value is None or isinstance(value, AssetBundleRef):
        return value
    raise TypeError("expected a resolved AssetBundleRef")


@dataclass(frozen=True)
class AudioRecordRef(HasAudioAnnotations):
    audio_file_id: UUID
    name: str
    duration: float
    byte_length: int
    virtual: bool
    annotations: AudioAnnotations = field(default_factory=AudioAnnotations)

    @property
    def id(self) -> str:
        return stable_id("audio_ref", self.audio_file_id)

    @property
    def lineage_id(self) -> str:
        return self.id


@dataclass(frozen=True)
class Audio(HasAudioAnnotations):
    audio_file_id: UUID
    name: str
    data: bytes | None
    sample_rate: int
    channels: int
    start: float
    end: float
    annotations: AudioAnnotations
    id: str
    lineage_id: str
    byte_length: int = 0
    virtual: bool = False
    style_prompt: str | None = None
    voice_prompt: str | None = None
    segments: list[AudioSegment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class AudioSegment(HasAudioAnnotations):
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
    annotations: AudioAnnotations = field(default_factory=AudioAnnotations)
    alignment: list[dict[str, Any]] | None = None

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
    speaker_id: str | None
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
class SpeakerEmbeddingShardRef:
    run_id: UUID
    artifact_id: UUID
    row_count: int
    dimension: int
    model_revision: str
    preprocessing_version: str


@dataclass(frozen=True)
class SpeakerEmbeddingSetRef:
    run_id: UUID
    artifact_ids: list[UUID]
    dimension: int
    item_count: int
    model_revision: str
    preprocessing_version: str


@dataclass(frozen=True)
class SpeakerClusterRunRef:
    run_id: UUID
    embedding_run_id: UUID
    assignment_artifact_ids: list[UUID]
    prototype_artifact_id: UUID
    index_artifact_id: UUID


@dataclass(frozen=True)
class SpeakerAuditRef:
    audit_id: UUID
    cluster_run_id: UUID
    review_id: UUID


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
