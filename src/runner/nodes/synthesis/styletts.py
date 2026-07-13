from __future__ import annotations

import asyncio
from collections.abc import Callable
import io
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runflow.runtime.output_router import INPUT_INDEX_OUTPUT
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AudioPort, CheckpointRefPort, JsonPort, SynthesisResultPort
from runner.nodes.models import Audio, SynthesisResult, stable_id, typed_checkpoint
from runner.nodes.synthesis.styletts_runtime.actions import (
    StyleTtsPayloadRequest,
    StyleTtsRequestSettings,
    build_styletts_payloads,
    synthesize_to_wav_bytes_batch,
    temporary_synthesis_dir,
    _prompt_text,
)
from shared.db import database_session
from shared.db.audio import crud as audio_crud


@dataclass(frozen=True)
class PendingStyleTts:
    inputs: dict[str, Any]
    output_index: int
    input_index: int


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
    DESCRIPTION = "Synthesize speech with StyleTTS, cloning the voice and style from a reference clip. Takes a StyleTTS checkpoint, prompt text, and a style reference (single or a batch of references), and outputs the generated audio plus a synthesis result. Tune diffusion steps, embedding scale, and the alpha/beta style blend to trade off speed against expressiveness; a reference batch fans out one sample per reference (times samples per reference)."
    CATEGORY = "Synthesis"
    SETTINGS = StyleTtsSynthesisSettings
    INPUTS = {
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "prompt_text": JsonPort(),
        "style_reference": JsonPort(),
    }
    OUTPUTS = {"synthesis_result": SynthesisResultPort(), "audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=8, max_size=16)
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 8},
        exclusive_group="accelerator",
        keep_loaded=True,
    )

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._runtime: Any | None = None
        self._loaded_checkpoint_id: UUID | None = None

    async def teardown(self, context) -> None:
        self._runtime = None
        self._loaded_checkpoint_id = None
        release_accelerator_memory()

    async def execute(self, batch, context):
        context.check_cancel()
        checkpoint = typed_checkpoint(batch[0]["checkpoint"])
        if self._loaded_checkpoint_id != checkpoint.checkpoint_id:
            self._runtime = None
        outputs, self._runtime = await asyncio.to_thread(
            synthesize_styletts_batch,
            self.NODE_TYPE,
            self.settings,
            list(batch),
            str(context.run_id),
            self._runtime,
            context.check_cancel,
        )
        self._loaded_checkpoint_id = checkpoint.checkpoint_id
        await context.report_progress(
            self.id,
            len(outputs),
            len(outputs),
            f"styletts synthesized {len(outputs)}/{len(outputs)}",
        )
        return outputs


def synthesize_styletts_batch(
    node_type: str,
    settings: StyleTtsSynthesisSettings,
    batch: list[dict[str, Any]],
    run_id: str,
    runtime: Any | None,
    check_cancel: Callable[[], None],
) -> tuple[list[dict[str, Any]], Any]:
    pending = _expand_styletts_batch(batch, settings)
    checkpoint = typed_checkpoint(pending[0].inputs["checkpoint"])
    assert all(
        typed_checkpoint(item.inputs["checkpoint"]).checkpoint_id
        == checkpoint.checkpoint_id
        for item in pending
    ), "styletts batch requires one checkpoint"
    audio_ids = list(dict.fromkeys(
        UUID(str(item.inputs["style_reference"]["audio_file_id"]))
        for item in pending
        if item.inputs["style_reference"]["kind"] == "audio_file"
    ))
    if audio_ids:
        with database_session() as session:
            audio_data = audio_crud.bulk_read_audio_files(session, audio_ids)
    else:
        audio_data = {}
    with temporary_synthesis_dir() as tmp:
        payloads = build_styletts_payloads(
            checkpoint=checkpoint,
            requests=[
                StyleTtsPayloadRequest(
                    prompt_text=item.inputs["prompt_text"],
                    style_reference=item.inputs["style_reference"],
                    output_filename=_output_filename(
                        settings.output_name,
                        item.output_index,
                    ),
                )
                for item in pending
            ],
            settings=_request_settings(settings),
            work_dir=Path(tmp),
            audio_data=audio_data,
        )
        runtime, wav_batches = synthesize_to_wav_bytes_batch(
            payloads,
            check_cancel,
            runtime,
        )
    assert len(wav_batches) == len(pending), "styletts batch output mismatch"
    assert runtime is not None, "styletts runtime was not loaded"
    outputs = [
        _styletts_output(node_type, settings, item, payload, wav_bytes, run_id)
        for item, payload, wav_bytes in zip(pending, payloads, wav_batches, strict=True)
    ]
    return outputs, runtime


def _expand_styletts_batch(
    batch: list[dict[str, Any]],
    settings: StyleTtsSynthesisSettings,
) -> list[PendingStyleTts]:
    indexed_inputs = []
    for input_index, inputs in enumerate(batch):
        style_reference = inputs["style_reference"]
        if isinstance(style_reference, dict) and style_reference.get("kind") == "style_reference_batch":
            references = style_reference_batch_items(style_reference)
            samples = _sweep_sample_count(style_reference, settings.samples_per_reference)
            indexed_inputs.extend(
                (input_index, {**inputs, "style_reference": reference})
                for reference in references
                for _sample_index in range(samples)
            )
        else:
            indexed_inputs.append((input_index, inputs))
    return [
        PendingStyleTts(inputs, output_index, input_index)
        for output_index, (input_index, inputs) in enumerate(indexed_inputs)
    ]


def _styletts_output(
    node_type: str,
    settings: StyleTtsSynthesisSettings,
    pending: PendingStyleTts,
    payload: dict[str, Any],
    wav_bytes: bytes,
    run_id: str,
) -> dict[str, Any]:
    request_id = stable_id(
        "synthesis_request",
        node_type,
        run_id,
        pending.output_index,
        _prompt_text(pending.inputs["prompt_text"]),
        pending.inputs["style_reference"],
    )
    audio = _audio_from_wav(settings.output_name, wav_bytes, request_id, node_type)
    audio = replace(audio, metadata={**audio.metadata, "run_id": run_id})
    result = SynthesisResult(
        request_id,
        audio,
        stable_id("synthesis_result", request_id, audio.id),
        audio.lineage_id,
        _result_metadata(payload, audio, settings),
    )
    return {
        INPUT_INDEX_OUTPUT: pending.input_index,
        "synthesis_result": result,
        "audio": audio,
    }


def style_reference_batch_items(value: dict[str, Any]) -> list[dict[str, Any]]:
    references = value["references"]
    assert isinstance(references, list), "style_reference_batch references must be a list"
    assert references, "style_reference_batch requires at least one reference"
    return references


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
