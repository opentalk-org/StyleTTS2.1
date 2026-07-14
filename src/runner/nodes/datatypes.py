from __future__ import annotations

from dataclasses import dataclass

from runflow.core.ports import Port
from runflow.registry.type_registry import TypeRegistry
from runner.nodes.models import (
    AssetBundleRef,
    Audio,
    CheckpointRef,
    SaveResult,
    SynthesisResult,
    TrainingManifest,
    TrainingResult,
)


@dataclass(frozen=True)
class TextPort(Port):
    TYPE_NAME = "TEXT"
    python_type = str
    color = "#334155"
    description = "Plain text"


@dataclass(frozen=True)
class IntPort(Port):
    TYPE_NAME = "INT"
    python_type = int
    color = "#EA580C"
    description = "Integer"


@dataclass(frozen=True)
class BoolPort(Port):
    TYPE_NAME = "BOOL"
    python_type = bool
    color = "#0F766E"
    description = "Boolean"


@dataclass(frozen=True)
class FloatPort(Port):
    TYPE_NAME = "FLOAT"
    python_type = float
    color = "#CA8A04"
    description = "Float"


@dataclass(frozen=True)
class JsonPort(Port):
    TYPE_NAME = "JSON"
    python_type = dict
    color = "#7C3AED"
    description = "JSON-like mapping"


@dataclass(frozen=True)
class AudioPort(Port):
    TYPE_NAME = "AUDIO"
    python_type = Audio
    color = "#0891B2"
    description = "Audio payload or segment"


@dataclass(frozen=True)
class SaveResultPort(Port):
    TYPE_NAME = "SAVE_RESULT"
    python_type = SaveResult
    color = "#16A34A"
    description = "Saved output metadata"


@dataclass(frozen=True)
class CheckpointRefPort(Port):
    TYPE_NAME = "CHECKPOINT_REF"
    python_type = CheckpointRef
    color = "#4F46E5"
    description = "Checkpoint artifact reference"


@dataclass(frozen=True)
class AssetBundlePort(Port):
    TYPE_NAME = "ASSET_BUNDLE"
    python_type = AssetBundleRef
    color = "#7C3AED"
    description = "Asset bundle reference"


@dataclass(frozen=True)
class TrainingManifestPort(Port):
    TYPE_NAME = "TRAINING_MANIFEST"
    python_type = TrainingManifest
    color = "#B45309"
    description = "Training input manifest"


@dataclass(frozen=True)
class TrainingResultPort(Port):
    TYPE_NAME = "TRAINING_RESULT"
    python_type = TrainingResult
    color = "#15803D"
    description = "Training output result"


@dataclass(frozen=True)
class SynthesisResultPort(Port):
    TYPE_NAME = "SYNTHESIS_RESULT"
    python_type = SynthesisResult
    color = "#BE123C"
    description = "Synthesis output result"


ALL_PORT_TYPES: list[type[Port]] = [
    TextPort,
    IntPort,
    BoolPort,
    FloatPort,
    JsonPort,
    AudioPort,
    SaveResultPort,
    CheckpointRefPort,
    AssetBundlePort,
    TrainingManifestPort,
    TrainingResultPort,
    SynthesisResultPort,
]


def register_runner_types(registry: TypeRegistry) -> TypeRegistry:
    for port_cls in ALL_PORT_TYPES:
        registry.register(port_cls)
    return registry
