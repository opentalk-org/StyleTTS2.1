from __future__ import annotations

from runflow.core.types import DataType
from runflow.registry.type_registry import TypeRegistry
from runner.nodes.models import Audio, AudioRecordRef, SaveResult, Transcript

TEXT = DataType("TEXT", str, "Plain text", "#334155")
INT = DataType("INT", int, "Integer", "#EA580C")
FLOAT = DataType("FLOAT", float, "Float", "#CA8A04")
JSON = DataType("JSON", dict, "JSON-like mapping", "#7C3AED")
AUDIO_REF = DataType("AUDIO_REF", AudioRecordRef, "Database audio record reference", "#2563EB")
AUDIO = DataType("AUDIO", Audio, "Audio payload or segment", "#0891B2")
TRANSCRIPT = DataType("TRANSCRIPT", Transcript, "ASR transcript", "#DC2626")
SAVE_RESULT = DataType("SAVE_RESULT", SaveResult, "Saved output metadata", "#16A34A")


def register_runner_types(registry: TypeRegistry) -> TypeRegistry:
    for dtype in [TEXT, INT, FLOAT, JSON, AUDIO_REF, AUDIO, TRANSCRIPT, SAVE_RESULT]:
        registry.register(dtype)
    return registry
