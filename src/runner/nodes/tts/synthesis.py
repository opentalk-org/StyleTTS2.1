from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runflow.runtime.output_router import INPUT_INDEX_OUTPUT
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import (
    AudioPort,
    CheckpointRefPort,
    JsonPort,
    SynthesisResultPort,
    TextPort,
)
from runner.nodes.languages import Language
from runner.nodes.models import SynthesisResult, stable_id, typed_checkpoint
from runner.nodes.tts.audio_out import audio_from_samples
from runner.nodes.tts.engines import load_engine
from runner.nodes.tts.engines.base import (
    EngineRuntime,
    EngineSynthesisRequest,
    EngineSynthesisResult,
)
from runner.nodes.tts.voices import TtsEngine, Voice, expand_voice_batch, parse_voice


class TtsSynthesisSettings(StrictSettings):
    language: Language = Field(default=Language.ENGLISH, title="Language")
    output_name: str = Field(default="tts.wav", title="Output name")


@dataclass(frozen=True)
class PendingSynthesis:
    request: EngineSynthesisRequest
    request_id: str
    voice_key: str
    text: str
    sample_index: int
    run_id: str
    input_index: int


class TtsSynthesisNode(Node):
    """Base synthesis node: (checkpoint, text, voice) -> audio.

    A ``voice`` may be a single voice or a ``tts_voice_batch``; a batch fans out one
    audio per (voice, sample) for each input text.
    """

    ENGINE: TtsEngine
    CATEGORY = "TTS"
    SETTINGS = TtsSynthesisSettings
    INPUTS = {
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "text": TextPort(),
        "voice": JsonPort(join_mode=JoinMode.BROADCAST),
    }
    OUTPUTS = {"audio": AudioPort(), "synthesis_result": SynthesisResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=32)
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 8},
        keep_loaded=True,
        exclusive_group="accelerator",
    )

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._runtime: EngineRuntime | None = None
        self._loaded_checkpoint_id: UUID | None = None

    async def teardown(self, context: Any) -> None:
        if self._runtime is not None:
            self._runtime.close()
        self._runtime = None
        self._loaded_checkpoint_id = None
        release_accelerator_memory()

    async def execute(self, batch, context):
        checkpoint = typed_checkpoint(batch[0]["checkpoint"])
        await self._ensure_runtime(checkpoint.checkpoint_id, checkpoint.path)
        pending: list[PendingSynthesis] = []
        run_id = str(context.run_id)
        for input_index, inputs in enumerate(batch):
            text = inputs["text"]
            voices, samples = self._resolve_voices(inputs["voice"])
            for voice in voices:
                for sample_index in range(samples):
                    context.check_cancel()
                    pending.append(
                        self._pending_synthesis(
                            text,
                            voice,
                            sample_index,
                            run_id,
                            input_index,
                        )
                    )
        assert self._runtime is not None, "runtime not loaded"
        results = await asyncio.to_thread(
            self._runtime.synthesize_batch,
            [item.request for item in pending],
            context.check_cancel,
        )
        assert len(results) == len(pending), (
            f"{self.ENGINE.value} batch output mismatch"
        )
        await context.report_progress(
            self.id,
            len(results),
            len(results),
            f"{self.ENGINE.value} synthesized {len(results)}/{len(results)}",
        )
        return [
            self._synthesis_output(item, result)
            for item, result in zip(pending, results, strict=True)
        ]

    async def _ensure_runtime(self, checkpoint_id: UUID, checkpoint_dir: Path) -> None:
        if self._runtime is not None and self._loaded_checkpoint_id == checkpoint_id:
            return
        self._runtime = await asyncio.to_thread(
            self._load_runtime_logged, checkpoint_dir
        )
        self._loaded_checkpoint_id = checkpoint_id

    def _load_runtime_logged(self, checkpoint_dir: Path) -> EngineRuntime:
        self.logger.info("loading %s engine from checkpoint", self.ENGINE.value)
        return load_engine(self.ENGINE, checkpoint_dir)

    def _resolve_voices(self, payload: dict[str, Any]) -> tuple[list[Voice], int]:
        if payload["kind"] == "tts_voice_batch":
            return expand_voice_batch(payload, self.ENGINE)
        return [parse_voice(payload, self.ENGINE)], 1

    def _pending_synthesis(
        self,
        text: str,
        voice: Voice,
        sample_index: int,
        run_id: str,
        input_index: int,
    ) -> PendingSynthesis:
        voice_key = voice.preset if voice.preset is not None else "clone"
        request_id = stable_id(
            "tts_request",
            self.NODE_TYPE,
            run_id,
            input_index,
            text,
            voice_key,
            sample_index,
        )
        request = EngineSynthesisRequest(text, voice, self.settings.language.value)
        return PendingSynthesis(
            request,
            request_id,
            voice_key,
            text,
            sample_index,
            run_id,
            input_index,
        )

    def _synthesis_output(
        self,
        pending: PendingSynthesis,
        result: EngineSynthesisResult,
    ) -> dict[str, Any]:
        metadata = {
            "engine": self.ENGINE.value,
            "voice": pending.voice_key,
            "language": self.settings.language.value,
            "text": pending.text,
            "sample_index": pending.sample_index,
            "run_id": pending.run_id,
            "sample_rate": result.sample_rate,
        }
        audio = audio_from_samples(
            node_type=self.NODE_TYPE,
            request_id=pending.request_id,
            output_name=self.settings.output_name,
            samples=result.samples,
            sample_rate=result.sample_rate,
            metadata=metadata,
        )
        synthesis_result = SynthesisResult(
            pending.request_id,
            audio,
            stable_id("tts_result", pending.request_id),
            audio.lineage_id,
            metadata,
        )
        return {
            INPUT_INDEX_OUTPUT: pending.input_index,
            "audio": audio,
            "synthesis_result": synthesis_result,
        }


class KokoroSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "KokoroSynthesis"
    DESCRIPTION = "Synthesize speech with the Kokoro TTS engine. Takes a checkpoint, input text, and a voice (a preset voice or a voice batch), and outputs the generated audio plus a synthesis result. Use it with the Kokoro preset voice nodes; a voice batch fans out one audio clip per voice and sample from the same text."
    ENGINE = TtsEngine.KOKORO


class ChatterboxSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "ChatterboxSynthesis"
    DESCRIPTION = "Synthesize speech with the Chatterbox TTS engine. Takes a checkpoint, input text, and a voice (typically a cloned voice from the Chatterbox clone node, or a voice batch), and outputs the generated audio plus a synthesis result. A voice batch fans out one audio clip per voice and sample from the same text."
    ENGINE = TtsEngine.CHATTERBOX


class F5TtsSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "F5TtsSynthesis"
    DESCRIPTION = "Synthesize speech with the F5-TTS engine. Takes a checkpoint, input text, and a voice (typically a cloned voice from the F5-TTS clone node, or a voice batch), and outputs the generated audio plus a synthesis result. A voice batch fans out one audio clip per voice and sample from the same text."
    ENGINE = TtsEngine.F5_TTS


class OrpheusSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "OrpheusSynthesis"
    DESCRIPTION = "Synthesize speech with the Orpheus TTS engine. Takes a checkpoint, input text, and a voice (typically a cloned voice from the Orpheus clone node, or a voice batch), and outputs the generated audio plus a synthesis result. A voice batch fans out one audio clip per voice and sample from the same text."
    ENGINE = TtsEngine.ORPHEUS


class DiaSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "DiaSynthesis"
    DESCRIPTION = "Synthesize speech with the Dia TTS engine. Takes a checkpoint, input text, and a voice (typically a cloned voice from the Dia clone node, or a voice batch), and outputs the generated audio plus a synthesis result. A voice batch fans out one audio clip per voice and sample from the same text."
    ENGINE = TtsEngine.DIA


class FishSpeechSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "FishSpeechSynthesis"
    DESCRIPTION = "Synthesize speech with the Fish Speech TTS engine. Takes a checkpoint, input text, and a voice (typically a cloned voice from the Fish Speech clone node, or a voice batch), and outputs the generated audio plus a synthesis result. A voice batch fans out one audio clip per voice and sample from the same text."
    ENGINE = TtsEngine.FISH_SPEECH


class RaonOpenTtsSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "RaonOpenTtsSynthesis"
    DESCRIPTION = "Synthesize speech with the Raon OpenTTS engine. Takes a checkpoint, input text, and a voice (typically a cloned voice from the Raon OpenTTS clone node, or a voice batch), and outputs the generated audio plus a synthesis result. A voice batch fans out one audio clip per voice and sample from the same text."
    ENGINE = TtsEngine.RAON_OPENTTS
