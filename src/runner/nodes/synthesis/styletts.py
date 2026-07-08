from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode, Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AUDIO, CHECKPOINT_REF, JSON, SYNTHESIS_RESULT
from runner.nodes.models import Audio, SynthesisResult, stable_id, typed_checkpoint
from runner.nodes.synthesis.styletts_runtime.actions import (
    StyleTtsRequestSettings,
    build_styletts_payload,
    synthesize_to_wav_bytes,
    temporary_synthesis_dir,
    _prompt_text,
)
from runner.nodes.synthesis.styletts_runtime.runtime import load_synthesis_runtime

class StyleTtsSynthesisSettings(StrictSettings):
    diffusion_steps: int = Field(default=5, title="Diffusion steps", ge=1, le=100)
    embedding_scale: float = Field(default=1.0, title="Embedding scale", ge=0.1, le=10)
    phoneme_language: str = Field(default="en-us", title="Phoneme language")
    phoneme_tie: bool = Field(default=True, title="Phoneme tie")
    alpha: float = Field(default=0.7, title="Alpha", ge=0, le=1)
    beta: float = Field(default=0.3, title="Beta", ge=0, le=1)
    asr_checkpoint_id: UUID | None = Field(default=None, title="ASR checkpoint")
    f0_checkpoint_id: UUID | None = Field(default=None, title="F0 checkpoint")
    f0_inner_filename: str = Field(default="", title="F0 inner file")
    plbert_checkpoint_id: UUID | None = Field(default=None, title="PL-BERT checkpoint")
    output_name: str = Field(default="styletts_synthesis.wav", title="Output name")
    samples_per_reference: int = Field(default=1, title="Samples per reference", ge=1, le=16)


class StyleTtsSynthesisNode(Node):
    NODE_TYPE = "StyleTtsSynthesis"
    CATEGORY = "Synthesis"
    SETTINGS = StyleTtsSynthesisSettings
    INPUTS = {
        "checkpoint": Port("checkpoint", CHECKPOINT_REF, join_mode=JoinMode.BROADCAST),
        "prompt_text": Port("prompt_text", JSON),
        "style_reference": Port("style_reference", JSON),
    }
    OUTPUTS = {"synthesis_result": Port("synthesis_result", SYNTHESIS_RESULT), "audio": Port("audio", AUDIO)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, exclusive_group="accelerator")

    async def teardown(self, context) -> None:
        release_accelerator_memory()

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            outputs.extend(await asyncio.to_thread(synthesize_styletts_items, self.NODE_TYPE, self.settings, inputs, str(context.run_id)))
        return outputs


def synthesize_styletts_items(
    node_type: str,
    settings: StyleTtsSynthesisSettings,
    inputs: dict[str, Any],
    run_id: str,
) -> list[dict[str, Audio | SynthesisResult]]:
    style_reference = inputs["style_reference"]
    if isinstance(style_reference, dict) and style_reference.get("kind") == "style_reference_batch":
        outputs = []
        runtime: Any | None = None
        references = style_reference_batch_items(style_reference)
        samples = _sweep_sample_count(style_reference, settings.samples_per_reference)
        for reference_index, reference in enumerate(references):
            for sample_index in range(samples):
                output_index = reference_index * samples + sample_index
                synthesis_inputs = {**inputs, "style_reference": reference}
                if runtime is None:
                    runtime = _load_runtime_for_inputs(settings, synthesis_inputs, output_index)
                outputs.append(synthesize_styletts(node_type, settings, synthesis_inputs, run_id, output_index, runtime))
        return outputs
    return [synthesize_styletts(node_type, settings, inputs, run_id, 0, None)]


def synthesize_styletts(
    node_type: str,
    settings: StyleTtsSynthesisSettings,
    inputs: dict[str, Any],
    run_id: str,
    output_index: int,
    runtime: Any | None,
) -> dict[str, Audio | SynthesisResult]:
    request_id = stable_id("synthesis_request", node_type, run_id, output_index, _prompt_text(inputs["prompt_text"]), inputs["style_reference"])
    checkpoint = typed_checkpoint(inputs["checkpoint"])
    with temporary_synthesis_dir() as tmp:
        payload = build_styletts_payload(
            checkpoint=checkpoint,
            prompt_text=inputs["prompt_text"],
            style_reference=inputs["style_reference"],
            settings=_request_settings(settings),
            work_dir=Path(tmp),
            output_filename=_output_filename(settings.output_name, output_index),
        )
        wav_bytes = synthesize_to_wav_bytes(runtime=runtime, payload=payload)
    audio = _audio_from_wav(settings.output_name, wav_bytes, request_id, node_type)
    audio = replace(audio, metadata={**audio.metadata, "run_id": run_id})
    result_id = stable_id("synthesis_result", request_id, audio.id)
    result = SynthesisResult(request_id, audio, result_id, audio.lineage_id, _result_metadata(payload, audio, settings))
    return {"synthesis_result": result, "audio": audio}


def style_reference_batch_items(value: dict[str, Any]) -> list[dict[str, Any]]:
    references = value["references"]
    assert isinstance(references, list), "style_reference_batch references must be a list"
    assert references, "style_reference_batch requires at least one reference"
    return references


def _load_runtime_for_inputs(
    settings: StyleTtsSynthesisSettings,
    inputs: dict[str, Any],
    output_index: int,
) -> Any:
    with temporary_synthesis_dir() as tmp:
        payload = build_styletts_payload(
            checkpoint=typed_checkpoint(inputs["checkpoint"]),
            prompt_text=inputs["prompt_text"],
            style_reference=inputs["style_reference"],
            settings=_request_settings(settings),
            work_dir=Path(tmp),
            output_filename=_output_filename(settings.output_name, output_index),
        )
        return load_synthesis_runtime(payload)


def _request_settings(settings: StyleTtsSynthesisSettings) -> StyleTtsRequestSettings:
    return StyleTtsRequestSettings.model_validate(settings.model_dump(mode="python"))


def _audio_from_wav(output_name: str, wav_bytes: bytes, request_id: str, node_type: str) -> Audio:
    info = _wav_info(wav_bytes)
    audio_id = stable_id("audio", request_id)
    metadata = {"node_type": node_type, "request_id": request_id, "byte_length": len(wav_bytes)}
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


def _result_metadata(payload: dict[str, Any], audio: Audio, settings: StyleTtsSynthesisSettings) -> dict[str, Any]:
    return {
        "node_type": audio.metadata["node_type"],
        "checkpoint": {
            "bundle_root": payload["bundle_root"],
            "weights_path": payload["weights_path"],
        },
        "style_reference": _without_large_fields(payload["style_reference"]),
        "settings": settings.model_dump(mode="json"),
        "audio_id": audio.id,
    }


def _without_large_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"data", "wav_base64"}}


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


def _output_filename(output_name: str, output_index: int) -> str:
    if output_index == 0:
        return output_name
    path = Path(output_name)
    suffix = path.suffix or ".wav"
    return f"{path.stem}_{output_index}{suffix}"
