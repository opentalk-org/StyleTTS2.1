from __future__ import annotations

import base64
import binascii
import io
import wave
from uuid import UUID

from pydantic import Field, model_validator

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import JSON
from runner.nodes.models import stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud


class ResolveStyleReferenceSettings(StrictSettings):
    audio_file_id: UUID | None = Field(default=None, title="Audio file")
    wav_base64: str = Field(default="", title="WAV base64")

    @model_validator(mode="after")
    def validate_single_source(self):
        has_audio_file = self.audio_file_id is not None
        has_wav_base64 = bool(self.wav_base64)
        if has_audio_file == has_wav_base64:
            raise ValueError("ResolveStyleReference requires exactly one of audio_file_id or wav_base64")
        if has_wav_base64:
            decode_wav_base64(self.wav_base64)
        return self


class ResolveStyleReferenceNode(Node):
    NODE_TYPE = "ResolveStyleReference"
    CATEGORY = "Synthesis / Inputs"
    SETTINGS = ResolveStyleReferenceSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"style_reference": Port("style_reference", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params):
        super().__init__(node_id=node_id, **params)
        self._emitted = False

    def remaining_items(self, context):
        return 0 if self._emitted else 1

    async def execute(self, batch, context):
        assert not self._emitted, f"style reference node already emitted: {self.id}"
        self._emitted = True
        return [{"style_reference": resolve_style_reference_payload(self.settings)}]


def resolve_style_reference_payload(settings: ResolveStyleReferenceSettings) -> dict[str, object]:
    if settings.audio_file_id is not None:
        return audio_file_style_reference(settings.audio_file_id)
    return wav_base64_style_reference(settings.wav_base64)


def audio_file_style_reference(audio_file_id: UUID) -> dict[str, object]:
    with database_session() as session:
        item = audio_crud.get_audio_file(session, audio_file_id)
    return {
        "kind": "audio_file",
        "audio_file_id": str(audio_file_id),
        "name": item.name,
        "duration": item.duration,
        "byte_length": item.byte_length,
        "metadata": item.metadata_,
        "id": stable_id("style_reference", audio_file_id, item.name, item.byte_length),
    }


def wav_base64_style_reference(wav_base64: str) -> dict[str, object]:
    wav_bytes = decode_wav_base64(wav_base64)
    ref_id = stable_id("style_reference", "wav_base64", len(wav_bytes), wav_bytes[:64])
    return {
        "kind": "wav_base64",
        "wav_base64": wav_base64,
        "name": "inline_style_reference.wav",
        "byte_length": len(wav_bytes),
        "metadata": {},
        "id": ref_id,
    }


def decode_wav_base64(wav_base64: str) -> bytes:
    try:
        wav_bytes = base64.b64decode(wav_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("wav_base64 must be valid base64") from error
    if not wav_bytes:
        raise ValueError("wav_base64 decoded to empty bytes")
    _assert_wav_bytes(wav_bytes)
    return wav_bytes


def _assert_wav_bytes(wav_bytes: bytes) -> None:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            wav_file.getparams()
    except (EOFError, wave.Error) as error:
        raise ValueError("wav_base64 must decode to a valid WAV file") from error
