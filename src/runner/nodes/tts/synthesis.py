from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AudioPort, CheckpointRefPort, JsonPort, SynthesisResultPort, TextPort
from runner.nodes.languages import Language
from runner.nodes.models import Audio, SynthesisResult, stable_id, typed_checkpoint
from runner.nodes.tts.audio_out import audio_from_samples
from runner.nodes.tts.engines import load_engine
from runner.nodes.tts.engines.base import EngineRuntime
from runner.nodes.tts.voices import TtsEngine, Voice, expand_voice_batch, parse_voice
from shared.log_streams import route_output_to_logger


class TtsSynthesisSettings(StrictSettings):
    language: Language = Field(default=Language.ENGLISH, title="Language")
    output_name: str = Field(default="tts.wav", title="Output name")


class TtsSynthesisNode(Node):
    """Base synthesis node: (checkpoint, text, voice) -> audio.

    A ``voice`` may be a single voice or a ``tts_voice_batch``; a batch fans out one
    audio per (voice, sample) from the single input text (see AGENTS.md fan-out rule).
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
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, keep_loaded=True, exclusive_group="accelerator")

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
        outputs: list[dict[str, Any]] = []
        for index, inputs in enumerate(batch, start=1):
            text = inputs["text"]
            voices, samples = self._resolve_voices(inputs["voice"])
            await context.report_progress(self.id, index, len(batch), f"{self.ENGINE.value} synthesizing {index}/{len(batch)}")
            for voice in voices:
                for sample_index in range(samples):
                    context.check_cancel()
                    outputs.append(await asyncio.to_thread(self._synthesize_one, text, voice, sample_index, str(context.run_id)))
        return outputs

    async def _ensure_runtime(self, checkpoint_id: UUID, checkpoint_dir: Path) -> None:
        if self._runtime is not None and self._loaded_checkpoint_id == checkpoint_id:
            return
        self._runtime = await asyncio.to_thread(self._load_runtime_logged, checkpoint_dir)
        self._loaded_checkpoint_id = checkpoint_id

    def _load_runtime_logged(self, checkpoint_dir: Path) -> EngineRuntime:
        self.logger.info("loading %s engine from checkpoint", self.ENGINE.value)
        with route_output_to_logger(self.logger):
            return load_engine(self.ENGINE, checkpoint_dir)

    def _resolve_voices(self, payload: dict[str, Any]) -> tuple[list[Voice], int]:
        if payload["kind"] == "tts_voice_batch":
            return expand_voice_batch(payload, self.ENGINE)
        return [parse_voice(payload, self.ENGINE)], 1

    def _synthesize_one(self, text: str, voice: Voice, sample_index: int, run_id: str) -> dict[str, Any]:
        assert self._runtime is not None, "runtime not loaded"
        voice_key = voice.preset if voice.preset is not None else "clone"
        request_id = stable_id("tts_request", self.NODE_TYPE, run_id, text, voice_key, sample_index)
        with route_output_to_logger(self.logger):
            samples, sample_rate = self._runtime.synthesize(text, voice, self.settings.language.value)
        metadata = {
            "engine": self.ENGINE.value,
            "voice": voice_key,
            "language": self.settings.language.value,
            "text": text,
            "sample_index": sample_index,
            "run_id": run_id,
            "sample_rate": sample_rate,
        }
        audio = audio_from_samples(
            node_type=self.NODE_TYPE,
            request_id=request_id,
            output_name=self.settings.output_name,
            samples=samples,
            sample_rate=sample_rate,
            metadata=metadata,
        )
        result = SynthesisResult(request_id, audio, stable_id("tts_result", request_id), audio.lineage_id, metadata)
        return {"audio": audio, "synthesis_result": result}


class KokoroSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "KokoroSynthesis"
    ENGINE = TtsEngine.KOKORO


class ChatterboxSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "ChatterboxSynthesis"
    ENGINE = TtsEngine.CHATTERBOX


class F5TtsSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "F5TtsSynthesis"
    ENGINE = TtsEngine.F5_TTS


class OrpheusSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "OrpheusSynthesis"
    ENGINE = TtsEngine.ORPHEUS


class DiaSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "DiaSynthesis"
    ENGINE = TtsEngine.DIA


class FishSpeechSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "FishSpeechSynthesis"
    ENGINE = TtsEngine.FISH_SPEECH


class RaonOpenTtsSynthesisNode(TtsSynthesisNode):
    NODE_TYPE = "RaonOpenTtsSynthesis"
    ENGINE = TtsEngine.RAON_OPENTTS
