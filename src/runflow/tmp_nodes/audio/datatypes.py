from __future__ import annotations

from pathlib import Path
from typing import Any

from runflow.core.types import DataType, UnionDataType
from runflow.tmp_nodes.audio.models import (
    AudioChunk,
    AudioFile,
    DenoisedAudio,
    DiarizationResult,
    SaveResult,
    SpeakerChunk,
    Transcript,
    VadSegments,
)
from runflow.registry.type_registry import TypeRegistry

PATH = DataType("PATH", Path, "Filesystem path", "#56CCF2")
TEXT = DataType("TEXT", str, "Plain text", "#FFFFFF")
INT = DataType("INT", int, "Integer", "#F2994A")
FLOAT = DataType("FLOAT", float, "Float", "#F2C94C")
BOOL = DataType("BOOL", bool, "Boolean", "#6FCF97")
JSON = DataType("JSON", dict, "JSON-like mapping", "#BDBDBD")
ANY = DataType("ANY", object, "Any Python object", "#828282")

AUDIO_FILE = DataType("AUDIO_FILE", AudioFile, "Original/probed audio file", "#4AA3FF")
AUDIO_CHUNK = DataType("AUDIO_CHUNK", AudioChunk, "Speech chunk cut by VAD", "#4AC3FF")
VAD_SEGMENTS = DataType("VAD_SEGMENTS", VadSegments, "VAD timestamps", "#F2C94C")
DIARIZATION_RESULT = DataType("DIARIZATION_RESULT", DiarizationResult, "Speaker turns", "#BB6BD9")
SPEAKER_CHUNK = DataType("SPEAKER_CHUNK", SpeakerChunk, "Speaker-specific chunk", "#9B51E0")
DENOISED_AUDIO = DataType("DENOISED_AUDIO", DenoisedAudio, "Enhanced audio", "#27AE60")
TRANSCRIPT = DataType("TRANSCRIPT", Transcript, "ASR transcript", "#EB5757")
SAVE_RESULT = DataType("SAVE_RESULT", SaveResult, "Saved output metadata", "#BDBDBD")

AUDIO_LIKE = UnionDataType(
    "AUDIO_LIKE",
    members=(AUDIO_FILE, AUDIO_CHUNK, SPEAKER_CHUNK, DENOISED_AUDIO),
    description="Any audio-like artifact",
    color="#4AA3FF",
)


def register_audio_types(registry: TypeRegistry) -> TypeRegistry:
    for dtype in [
        PATH,
        TEXT,
        INT,
        FLOAT,
        BOOL,
        JSON,
        ANY,
        AUDIO_FILE,
        AUDIO_CHUNK,
        VAD_SEGMENTS,
        DIARIZATION_RESULT,
        SPEAKER_CHUNK,
        DENOISED_AUDIO,
        TRANSCRIPT,
        SAVE_RESULT,
        AUDIO_LIKE,
    ]:
        registry.register(dtype)
    return registry
