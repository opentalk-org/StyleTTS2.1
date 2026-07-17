import copy
import hashlib
import os
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from ..data.prefetch import DataPipelineState
from .state import (
    LoopState,
    NamedGradient,
    RngState,
    StageKind,
    restore_gradients,
)

CHECKPOINT_VERSION = 1
_PAYLOAD_NAME = "payload.pt"
_MANIFEST_NAME = "manifest.json"
_LATEST_NAME = "latest.json"


class StateKind(StrEnum):
    MODEL = "model"
    FROZEN_MODEL = "frozen_model"
    DISCRIMINATOR = "discriminator"
    EMA = "ema"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    SCALER = "scaler"


class Stateful(Protocol):
    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> object: ...


@dataclass(frozen=True)
class NamedState:
    name: str
    kind: StateKind
    value: dict[str, Any]


@dataclass(frozen=True)
class StateTarget:
    name: str
    kind: StateKind
    target: Stateful


@dataclass(frozen=True)
class NamedModuleGradients:
    name: str
    gradients: tuple[NamedGradient, ...]


@dataclass(frozen=True)
class GradientTarget:
    name: str
    module: nn.Module


@dataclass(frozen=True)
class LossWeight:
    name: str
    value: float


@dataclass(frozen=True)
class LossScheduleState:
    optimizer_step: int
    weights: tuple[LossWeight, ...]


@dataclass(frozen=True)
class CheckpointPayload:
    version: int
    config_fingerprint: str
    data_fingerprint: str
    loop: LoopState
    rng: RngState
    states: tuple[NamedState, ...]
    gradients: tuple[NamedModuleGradients, ...]
    sampler_state: DataPipelineState
    loss_schedule: LossScheduleState


class _FolderManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    payload: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _LatestManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    folder: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CheckpointManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, payload: CheckpointPayload) -> Path:
        _validate_payload(payload)
        identifier = uuid.uuid4().hex
        temporary = self.root / f".{identifier}.tmp"
        destination = self.root / f"checkpoint_{identifier}"
        temporary.mkdir()
        payload_path = temporary / _PAYLOAD_NAME
        with payload_path.open("xb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        digest = _sha256(payload_path)
        folder_manifest = _FolderManifest(
            version=CHECKPOINT_VERSION,
            payload=_PAYLOAD_NAME,
            sha256=digest,
        )
        _write_new_text(temporary / _MANIFEST_NAME, folder_manifest.model_dump_json())
        _fsync_directory(temporary)
        temporary.rename(destination)
        _fsync_directory(self.root)
        self._write_latest(destination.name, digest, identifier)
        return destination

    def latest(self) -> Path | None:
        manifest_path = self.root / _LATEST_NAME
        if not manifest_path.exists():
            return None
        manifest = _LatestManifest.model_validate_json(manifest_path.read_text())
        if manifest.version != CHECKPOINT_VERSION:
            raise ValueError("latest checkpoint manifest version is unsupported")
        if Path(manifest.folder).name != manifest.folder:
            raise ValueError("latest checkpoint manifest folder is invalid")
        path = self.root / manifest.folder
        self._validate_folder(path, manifest.sha256)
        return path

    def load(self, path: Path) -> CheckpointPayload:
        payload_path = self._validate_folder(path, None)
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, CheckpointPayload):
            raise TypeError("checkpoint payload has an unexpected type")
        _validate_payload(payload)
        return payload

    def _validate_folder(self, path: Path, expected_sha256: str | None) -> Path:
        manifest_path = path / _MANIFEST_NAME
        manifest = _FolderManifest.model_validate_json(manifest_path.read_text())
        if manifest.version != CHECKPOINT_VERSION:
            raise ValueError("checkpoint manifest version is unsupported")
        if manifest.payload != _PAYLOAD_NAME:
            raise ValueError("checkpoint manifest payload name is invalid")
        if expected_sha256 is not None and manifest.sha256 != expected_sha256:
            raise ValueError("latest checkpoint hash does not match folder manifest")
        payload_path = path / manifest.payload
        if _sha256(payload_path) != manifest.sha256:
            raise ValueError("checkpoint payload hash mismatch")
        return payload_path

    def _write_latest(self, folder: str, digest: str, identifier: str) -> None:
        manifest = _LatestManifest(
            version=CHECKPOINT_VERSION,
            folder=folder,
            sha256=digest,
        )
        temporary = self.root / f".{_LATEST_NAME}.{identifier}.tmp"
        _write_new_text(temporary, manifest.model_dump_json())
        os.replace(temporary, self.root / _LATEST_NAME)
        _fsync_directory(self.root)


def capture_named_state(name: str, kind: StateKind, source: Stateful) -> NamedState:
    return NamedState(name, kind, copy.deepcopy(source.state_dict()))


def restore_named_states(
    states: tuple[NamedState, ...],
    targets: tuple[StateTarget, ...],
) -> None:
    saved_keys = tuple((state.kind, state.name) for state in states)
    target_keys = tuple((target.kind, target.name) for target in targets)
    if saved_keys != target_keys:
        raise ValueError(f"checkpoint state names do not match targets: {saved_keys}")
    for state, target in zip(states, targets, strict=True):
        target.target.load_state_dict(copy.deepcopy(state.value))


def restore_checkpoint_gradients(
    gradients: tuple[NamedModuleGradients, ...],
    targets: tuple[GradientTarget, ...],
) -> None:
    saved_names = tuple(item.name for item in gradients)
    target_names = tuple(item.name for item in targets)
    if saved_names != target_names:
        raise ValueError("checkpoint gradient module names do not match targets")
    for saved, target in zip(gradients, targets, strict=True):
        restore_gradients(target.module, saved.gradients)


def validate_resume_fingerprints(
    payload: CheckpointPayload,
    stage: StageKind,
    config_fingerprint: str,
    data_fingerprint: str,
) -> None:
    if payload.loop.stage is not stage:
        raise ValueError("stage does not match checkpoint")
    if payload.config_fingerprint != config_fingerprint:
        raise ValueError("config_fingerprint does not match checkpoint")
    if payload.data_fingerprint != data_fingerprint:
        raise ValueError("data_fingerprint does not match checkpoint")


def _validate_payload(payload: CheckpointPayload) -> None:
    if payload.version != CHECKPOINT_VERSION:
        raise ValueError("checkpoint payload version is unsupported")
    if payload.loss_schedule.optimizer_step != payload.loop.optimizer_step:
        raise ValueError("loss schedule step does not match loop optimizer_step")
    if payload.sampler_state.data_fingerprint != payload.data_fingerprint:
        raise ValueError("sampler data fingerprint does not match checkpoint")


def _write_new_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
