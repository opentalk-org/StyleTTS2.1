import hashlib
import math
import os
import re
import uuid
import wave
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from .callbacks import TrainingCallbacks, TrainingMetric
from .state import StageKind

_ROLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class ArtifactKind(StrEnum):
    AUDIO = "audio"
    TENSOR = "tensor"


@dataclass(frozen=True)
class ValidationSelection:
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_ids or any(not value for value in self.source_ids):
            raise ValueError("validation selection requires non-empty source IDs")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("validation source IDs must be unique")


@dataclass(frozen=True)
class ValidationArtifact:
    role: str
    kind: ArtifactKind
    value: Tensor

    def __post_init__(self) -> None:
        if not _ROLE_PATTERN.fullmatch(self.role):
            raise ValueError(f"invalid validation artifact role: {self.role}")
        if self.value.numel() == 0 or not torch.isfinite(self.value).all():
            raise ValueError(f"validation artifact is empty or non-finite: {self.role}")


@dataclass(frozen=True)
class ValidationSample:
    source_id: str
    artifacts: tuple[ValidationArtifact, ...]

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("validation sample source ID must not be empty")
        roles = tuple(artifact.role for artifact in self.artifacts)
        if len(set(roles)) != len(roles):
            raise ValueError("validation artifact roles must be unique per sample")


@dataclass(frozen=True)
class ValidationOutput:
    samples: tuple[ValidationSample, ...]
    metrics: tuple[TrainingMetric, ...]


class ValidationRenderer(Protocol):
    def render(
        self,
        selection: ValidationSelection,
        optimizer_step: int,
    ) -> ValidationOutput: ...


class ValidationArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    role: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: ArtifactKind
    file: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    media_type: str = Field(min_length=1)


class ValidationMetricRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    value: float


class ValidationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    stage: StageKind
    optimizer_step: int = Field(gt=0)
    source_ids: tuple[str, ...] = Field(min_length=1)
    sample_rate: int = Field(gt=0)
    artifacts: tuple[ValidationArtifactRecord, ...] = Field(min_length=1)
    metrics: tuple[ValidationMetricRecord, ...] = Field(min_length=1)


def select_validation_items(
    eligible_ids: tuple[str, ...],
    sample_count: int,
    seed: int,
    restored_ids: tuple[str, ...],
) -> ValidationSelection:
    if sample_count <= 0:
        raise ValueError("validation sample_count must be positive")
    if seed < 0:
        raise ValueError("validation seed must be non-negative")
    eligible = tuple(sorted(eligible_ids))
    if any(not source_id for source_id in eligible):
        raise ValueError("eligible validation source IDs must not be empty")
    if len(set(eligible)) != len(eligible):
        raise ValueError("eligible validation source IDs must be unique")
    if restored_ids:
        restored = ValidationSelection(restored_ids)
        if len(restored.source_ids) != sample_count:
            raise ValueError("restored validation selection has a different size")
        if not set(restored.source_ids) <= set(eligible):
            raise ValueError("restored validation source is no longer eligible")
        return restored
    if sample_count > len(eligible):
        raise ValueError("validation sample_count exceeds eligible sources")
    ranked = sorted(eligible, key=lambda value: _selection_rank(seed, value))
    return ValidationSelection(tuple(ranked[:sample_count]))


def run_stage_validation(
    stage: StageKind,
    optimizer_step: int,
    selection: ValidationSelection,
    sample_rate: int,
    root: Path,
    renderer: ValidationRenderer,
    callbacks: TrainingCallbacks,
) -> tuple[TrainingMetric, ...]:
    if optimizer_step <= 0:
        raise ValueError("validation requires a completed optimizer step")
    if sample_rate <= 0:
        raise ValueError("validation sample_rate must be positive")
    parent = root / "validation"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"step_{optimizer_step}"
    if destination.exists():
        record = _load(destination, stage, optimizer_step, selection, sample_rate)
        callbacks.publish_artifact(destination, "application/vnd.beetle.validation")
        return tuple(TrainingMetric(item.name, item.value) for item in record.metrics)
    callbacks.check_cancel()
    output = renderer.render(selection, optimizer_step)
    _validate_output(stage, selection, output)
    identifier = uuid.uuid4().hex
    temporary = parent / f".step_{optimizer_step}.{identifier}.tmp"
    temporary.mkdir()
    records: list[ValidationArtifactRecord] = []
    for sample_index, sample in enumerate(output.samples):
        callbacks.check_cancel()
        for artifact in sample.artifacts:
            filename, media_type = _write_artifact(
                temporary, sample_index, artifact, sample_rate
            )
            records.append(
                ValidationArtifactRecord(
                    source_id=sample.source_id,
                    role=artifact.role,
                    kind=artifact.kind,
                    file=filename,
                    media_type=media_type,
                )
            )
    record = ValidationRecord(
        stage=stage,
        optimizer_step=optimizer_step,
        source_ids=selection.source_ids,
        sample_rate=sample_rate,
        artifacts=tuple(records),
        metrics=tuple(
            ValidationMetricRecord(name=metric.name, value=metric.value)
            for metric in output.metrics
        ),
    )
    _write_record(temporary / "record.json", record)
    _fsync_directory(temporary)
    temporary.rename(destination)
    _fsync_directory(parent)
    callbacks.publish_artifact(destination, "application/vnd.beetle.validation")
    return output.metrics


def _selection_rank(seed: int, source_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{source_id}".encode()).digest()


def _required_roles(stage: StageKind) -> frozenset[str]:
    if stage is StageKind.STAGE1:
        return frozenset(("reference", "reconstruction", "f0", "n"))
    if stage is StageKind.STAGE2:
        return frozenset(("reference", "synthesis", "duration", "alignment", "flow"))
    return frozenset(("reference", "reconstruction", "synthesis"))


def _validate_output(
    stage: StageKind,
    selection: ValidationSelection,
    output: ValidationOutput,
) -> None:
    source_ids = tuple(sample.source_id for sample in output.samples)
    if source_ids != selection.source_ids:
        raise ValueError("validation output order does not match fixed selection")
    required = _required_roles(stage)
    for sample in output.samples:
        if frozenset(artifact.role for artifact in sample.artifacts) != required:
            raise ValueError(f"validation roles do not match {stage.value}")
    metric_names = tuple(metric.name for metric in output.metrics)
    if len(set(metric_names)) != len(metric_names):
        raise ValueError("validation metric names must be unique")
    if any(not math.isfinite(metric.value) for metric in output.metrics):
        raise ValueError("validation metrics must be finite")


def _write_artifact(
    folder: Path,
    sample_index: int,
    artifact: ValidationArtifact,
    sample_rate: int,
) -> tuple[str, str]:
    prefix = f"{sample_index:04d}_{artifact.role}"
    if artifact.kind is ArtifactKind.AUDIO:
        filename = f"{prefix}.wav"
        _write_pcm_wave(folder / filename, artifact.value, sample_rate)
        return filename, "audio/wav"
    filename = f"{prefix}.pt"
    path = folder / filename
    torch.save(artifact.value.detach().cpu(), path)
    _fsync_file(path)
    return filename, "application/vnd.pytorch.tensor"


def _write_pcm_wave(path: Path, value: Tensor, sample_rate: int) -> None:
    waveform = value.detach().float().cpu().squeeze()
    if waveform.ndim != 1:
        raise ValueError("validation audio must contain one mono waveform")
    pcm = (waveform.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    _fsync_file(path)


def _write_record(path: Path, record: ValidationRecord) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(record.model_dump_json(indent=2))
        output.flush()
        os.fsync(output.fileno())


def _load(
    folder: Path,
    stage: StageKind,
    optimizer_step: int,
    selection: ValidationSelection,
    sample_rate: int,
) -> ValidationRecord:
    record = ValidationRecord.model_validate_json((folder / "record.json").read_text())
    identity = (
        record.stage,
        record.optimizer_step,
        record.source_ids,
        record.sample_rate,
    )
    if identity != (stage, optimizer_step, selection.source_ids, sample_rate):
        raise ValueError("completed validation folder does not match this run")
    if any(not (folder / artifact.file).is_file() for artifact in record.artifacts):
        raise ValueError("completed validation folder is missing an artifact")
    return record


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
