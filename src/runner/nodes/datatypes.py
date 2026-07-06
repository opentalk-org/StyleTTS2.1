from __future__ import annotations

from runflow.core.types import DataType
from runflow.registry.type_registry import TypeRegistry
from runner.nodes.models import (
    AssetBundleRef,
    Audio,
    AudioRecordRef,
    AudioSegment,
    CheckpointRef,
    SaveResult,
    SegmentGroup,
    SynthesisResult,
    TrainingManifest,
    TrainingResult,
    Transcript,
)

TEXT = DataType("TEXT", str, "Plain text", "#334155")
INT = DataType("INT", int, "Integer", "#EA580C")
FLOAT = DataType("FLOAT", float, "Float", "#CA8A04")
JSON = DataType("JSON", dict, "JSON-like mapping", "#7C3AED")
AUDIO_REF = DataType("AUDIO_REF", AudioRecordRef, "Database audio record reference", "#2563EB")
AUDIO = DataType("AUDIO", Audio, "Audio payload or segment", "#0891B2")
TRANSCRIPT = DataType("TRANSCRIPT", Transcript, "ASR transcript", "#DC2626")
SAVE_RESULT = DataType("SAVE_RESULT", SaveResult, "Saved output metadata", "#16A34A")
AUDIO_SEGMENT = DataType("AUDIO_SEGMENT", AudioSegment, "Audio segment reference", "#0E7490")
SEGMENT_GROUP = DataType("SEGMENT_GROUP", SegmentGroup, "Grouped audio segments", "#155E75")
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
        AUDIO_REF,
        AUDIO,
        TRANSCRIPT,
        SAVE_RESULT,
        AUDIO_SEGMENT,
        SEGMENT_GROUP,
        CHECKPOINT_REF,
        ASSET_BUNDLE,
        TRAINING_MANIFEST,
        TRAINING_RESULT,
        SYNTHESIS_RESULT,
    ]:
        registry.register(dtype)
    return registry
