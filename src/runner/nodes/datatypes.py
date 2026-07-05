from __future__ import annotations

from runflow.core.types import DataType, UnionDataType
from runflow.registry.type_registry import TypeRegistry
from runner.nodes.models import AudioRecordRef, AudioSegment, BucketAudio, SaveResult, Transcript

TEXT = DataType("TEXT", str, "Plain text", "#FFFFFF")
INT = DataType("INT", int, "Integer", "#F2994A")
FLOAT = DataType("FLOAT", float, "Float", "#F2C94C")
JSON = DataType("JSON", dict, "JSON-like mapping", "#BDBDBD")
AUDIO_REF = DataType("AUDIO_REF", AudioRecordRef, "Database audio record reference", "#4AA3FF")
BUCKET_AUDIO = DataType("BUCKET_AUDIO", BucketAudio, "Bucket-loaded audio payload", "#4AC3FF")
AUDIO_SEGMENT = DataType("AUDIO_SEGMENT", AudioSegment, "Timed audio segment", "#F2C94C")
TRANSCRIPT = DataType("TRANSCRIPT", Transcript, "ASR transcript", "#EB5757")
SAVE_RESULT = DataType("SAVE_RESULT", SaveResult, "Saved output metadata", "#BDBDBD")
AUDIO_LIKE = UnionDataType(
    "AUDIO_LIKE",
    members=(BUCKET_AUDIO, AUDIO_SEGMENT),
    description="Loaded audio or derived audio segment",
    color="#4AA3FF",
)


def register_runner_types(registry: TypeRegistry) -> TypeRegistry:
    for dtype in [TEXT, INT, FLOAT, JSON, AUDIO_REF, BUCKET_AUDIO, AUDIO_SEGMENT, TRANSCRIPT, SAVE_RESULT, AUDIO_LIKE]:
        registry.register(dtype)
    return registry
