from __future__ import annotations

from runflow.core.types import DataType
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

TEXT = DataType("TEXT", str, "Plain text", "#334155")
INT = DataType("INT", int, "Integer", "#EA580C")
FLOAT = DataType("FLOAT", float, "Float", "#CA8A04")
JSON = DataType("JSON", dict, "JSON-like mapping", "#7C3AED")
AUDIO = DataType("AUDIO", Audio, "Audio payload or segment", "#0891B2")
SAVE_RESULT = DataType("SAVE_RESULT", SaveResult, "Saved output metadata", "#16A34A")
CHECKPOINT_REF = DataType("CHECKPOINT_REF", CheckpointRef, "Checkpoint artifact reference", "#4F46E5")
ASSET_BUNDLE = DataType("ASSET_BUNDLE", AssetBundleRef, "Asset bundle reference", "#7C3AED")
TRAINING_MANIFEST = DataType("TRAINING_MANIFEST", TrainingManifest, "Training input manifest", "#B45309")
TRAINING_RESULT = DataType("TRAINING_RESULT", TrainingResult, "Training output result", "#15803D")
SYNTHESIS_RESULT = DataType("SYNTHESIS_RESULT", SynthesisResult, "Synthesis output result", "#BE123C")


def register_runner_types(registry: TypeRegistry) -> TypeRegistry:
    for dtype in [
        TEXT,
        INT,
        FLOAT,
        JSON,
        AUDIO,
        SAVE_RESULT,
        CHECKPOINT_REF,
        ASSET_BUNDLE,
        TRAINING_MANIFEST,
        TRAINING_RESULT,
        SYNTHESIS_RESULT,
    ]:
        registry.register(dtype)
    return registry
