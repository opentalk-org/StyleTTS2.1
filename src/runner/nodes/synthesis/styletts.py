from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AUDIO, JSON, SYNTHESIS_RESULT
from runner.nodes.models import Audio, CheckpointRef, SynthesisResult, stable_id
from runner.nodes.synthesis.style_reference import decode_wav_base64
from runner.nodes.training_config import CHECKPOINT_REF_OR_JSON
from shared.db import database_session
from shared.db.audio import crud as audio_crud


class StyleTtsSynthesisSettings(StrictSettings):
    external_command: list[str] = Field(default_factory=list, title="External synthesis command")
    weights_file: str = Field(default="", title="Weights file")
    diffusion_steps: int = Field(default=5, title="Diffusion steps", ge=1, le=100)
    embedding_scale: float = Field(default=1.0, title="Embedding scale", ge=0.1, le=10)
    output_name: str = Field(default="styletts_synthesis.wav", title="Output name")


class StyleTtsSweepSynthesisSettings(StyleTtsSynthesisSettings):
    samples_per_reference: int = Field(default=1, title="Samples per reference", ge=1, le=16)


class StyleTtsSynthesisNode(Node):
    NODE_TYPE = "StyleTtsSynthesis"
    CATEGORY = "Synthesis"
    SETTINGS = StyleTtsSynthesisSettings
    INPUTS = {
        "checkpoint": Port("checkpoint", CHECKPOINT_REF_OR_JSON),
        "prompt_text": Port("prompt_text", JSON),
        "phonemes": Port("phonemes", JSON),
        "style_reference": Port("style_reference", JSON),
    }
    OUTPUTS = {"synthesis_result": Port("synthesis_result", SYNTHESIS_RESULT), "audio": Port("audio", AUDIO)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        return [synthesize_styletts(self.NODE_TYPE, self.settings, inputs, str(context.run_id), 0) for inputs in batch]


class StyleTtsSweepSynthesisNode(Node):
    NODE_TYPE = "StyleTtsSweepSynthesis"
    CATEGORY = "Synthesis"
    SETTINGS = StyleTtsSweepSynthesisSettings
    INPUTS = {
        "checkpoint": Port("checkpoint", CHECKPOINT_REF_OR_JSON),
        "prompt_text": Port("prompt_text", JSON),
        "phonemes": Port("phonemes", JSON),
        "style_reference_batch": Port("style_reference_batch", JSON),
    }
    OUTPUTS = {"synthesis_result": Port("synthesis_result", SYNTHESIS_RESULT), "audio": Port("audio", AUDIO)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            references = style_reference_batch_items(inputs["style_reference_batch"])
            samples = _sweep_sample_count(inputs["style_reference_batch"], self.settings.samples_per_reference)
            for reference_index, reference in enumerate(references):
                for sample_index in range(samples):
                    synthesis_inputs = {**inputs, "style_reference": reference}
                    output_index = reference_index * samples + sample_index
                    outputs.append(synthesize_styletts(self.NODE_TYPE, self.settings, synthesis_inputs, str(context.run_id), output_index))
        return outputs


def synthesize_styletts(
    node_type: str,
    settings: StyleTtsSynthesisSettings,
    inputs: dict[str, Any],
    run_id: str,
    output_index: int,
) -> dict[str, Audio | SynthesisResult]:
    if not settings.external_command:
        raise RuntimeError(f"{node_type} requires external synthesis command")
    request_id = stable_id("synthesis_request", node_type, run_id, output_index, _prompt_text(inputs["prompt_text"]), inputs["style_reference"])
    with tempfile.TemporaryDirectory(prefix="runflow-synthesis-") as tmp:
        tmp_path = Path(tmp)
        output_wav_path = tmp_path / "output.wav"
        style_reference = _materialize_style_reference(inputs["style_reference"], tmp_path)
        payload = _synthesis_payload(node_type, settings, inputs, request_id, style_reference, output_wav_path)
        _run_external_synthesis(node_type, settings.external_command, payload, tmp_path)
        wav_bytes = _read_output_wav(node_type, output_wav_path)
    audio = _audio_from_wav(settings.output_name, wav_bytes, request_id, payload)
    result_id = stable_id("synthesis_result", request_id, audio.id)
    result = SynthesisResult(request_id, audio, result_id, audio.lineage_id, _result_metadata(payload, audio))
    return {"synthesis_result": result, "audio": audio}


def style_reference_batch_items(value: dict[str, Any]) -> list[dict[str, Any]]:
    references = value["references"]
    assert isinstance(references, list), "style_reference_batch references must be a list"
    assert references, "style_reference_batch requires at least one reference"
    return references


def _synthesis_payload(
    node_type: str,
    settings: StyleTtsSynthesisSettings,
    inputs: dict[str, Any],
    request_id: str,
    style_reference: dict[str, Any],
    output_wav_path: Path,
) -> dict[str, Any]:
    return {
        "version": 1,
        "node_type": node_type,
        "request_id": request_id,
        "checkpoint": _checkpoint_payload(inputs["checkpoint"]),
        "prompt_text": inputs["prompt_text"],
        "text": _prompt_text(inputs["prompt_text"]),
        "phonemes": inputs["phonemes"],
        "style_reference": style_reference,
        "weights_file": settings.weights_file,
        "diffusion_steps": settings.diffusion_steps,
        "embedding_scale": settings.embedding_scale,
        "settings": settings.model_dump(mode="json"),
        "output_wav_path": str(output_wav_path),
    }


def _run_external_synthesis(node_type: str, command: list[str], payload: dict[str, Any], tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True), encoding="utf-8")
    env = {**os.environ, "RUNFLOW_SYNTHESIS_PAYLOAD": str(payload_path), "RUNFLOW_SYNTHESIS_NODE_TYPE": node_type}
    subprocess.run(command, check=True, env=env)


def _materialize_style_reference(reference: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    kind = reference["kind"]
    if kind == "audio_file":
        audio_file_id = UUID(str(reference["audio_file_id"]))
        with database_session() as session:
            wav_bytes = audio_crud.read_audio_file(session, audio_file_id)
        reference_path = tmp_path / "style_reference.wav"
        reference_path.write_bytes(wav_bytes)
        return _without_large_fields({**reference, "local_wav_path": str(reference_path)})
    if kind == "wav_base64":
        reference_path = tmp_path / "style_reference.wav"
        reference_path.write_bytes(decode_wav_base64(str(reference["wav_base64"])))
        return _without_large_fields({**reference, "local_wav_path": str(reference_path)})
    raise RuntimeError(f"StyleTTS synthesis requires resolved style reference, got kind={kind}")


def _read_output_wav(node_type: str, output_wav_path: Path) -> bytes:
    if not output_wav_path.is_file():
        raise RuntimeError(f"{node_type} external command did not produce output WAV: {output_wav_path}")
    wav_bytes = output_wav_path.read_bytes()
    if not wav_bytes:
        raise RuntimeError(f"{node_type} external command produced an empty output WAV: {output_wav_path}")
    return wav_bytes


def _audio_from_wav(output_name: str, wav_bytes: bytes, request_id: str, payload: dict[str, Any]) -> Audio:
    info = _wav_info(wav_bytes)
    audio_id = stable_id("audio", request_id)
    metadata = {"node_type": payload["node_type"], "request_id": request_id, "byte_length": len(wav_bytes)}
    return Audio(
        audio_file_id=uuid5(NAMESPACE_URL, request_id),
        name=output_name,
        data=wav_bytes,
        sample_rate=info["sample_rate"],
        channels=info["channels"],
        start=0.0,
        end=info["duration"],
        confidence=1.0,
        id=audio_id,
        lineage_id=audio_id,
        metadata=metadata,
    )


def _wav_info(wav_bytes: bytes) -> dict[str, int | float]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        return {"sample_rate": sample_rate, "channels": wav_file.getnchannels(), "duration": frame_count / sample_rate}


def _checkpoint_payload(value: CheckpointRef | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, CheckpointRef):
        return {"checkpoint_id": str(value.checkpoint_id), "name": value.name, "path": str(value.path), "metadata": value.metadata}
    raise TypeError("StyleTTS synthesis requires a resolved CheckpointRef")


def _prompt_text(value: dict[str, Any]) -> str:
    if "text" in value:
        return str(value["text"])
    if "settings" in value:
        return str(value["settings"]["text"])
    return str(value)


def _without_large_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"wav_base64"}}


def _result_metadata(payload: dict[str, Any], audio: Audio) -> dict[str, Any]:
    return {
        "node_type": payload["node_type"],
        "checkpoint": payload["checkpoint"],
        "style_reference": _without_large_fields(payload["style_reference"]),
        "settings": payload["settings"],
        "audio_id": audio.id,
    }


def _sweep_sample_count(value: dict[str, Any], default_samples: int) -> int:
    if "samples_per_voice" in value:
        samples = int(value["samples_per_voice"])
        assert samples >= 1, "style_reference_batch samples_per_voice must be at least 1"
        return samples
    if "samples_per_reference" in value:
        samples = int(value["samples_per_reference"])
        assert samples >= 1, "style_reference_batch samples_per_reference must be at least 1"
        return samples
    return default_samples
