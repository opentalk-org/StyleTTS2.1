from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runflow.runtime.output_router import INPUT_INDEX_OUTPUT
from runner.nodes.assets.checkpoints import resolve_checkpoint_ref
from runner.nodes.datatypes import AudioPort, JsonPort, SynthesisResultPort, TextPort
from runner.nodes.models import SynthesisResult, stable_id
from runner.nodes.tts.audio_out import audio_from_samples
from runner.nodes.tts.engines.piper import PiperRuntime, PiperSynthesisOptions
from runner.nodes.tts.voices import PiperVoiceModel, TtsEngine, expand_voice_batch, parse_voice


class PiperSynthesisSettings(StrictSettings):
    output_name: str = Field(default="piper.wav", title="Output name")
    speaker_id: int | None = Field(default=None, ge=0, title="Speaker ID")
    length_scale: float = Field(default=1.0, gt=0, title="Length scale")
    noise_scale: float = Field(default=0.667, ge=0, title="Noise scale")
    noise_w_scale: float = Field(default=0.8, ge=0, title="Duration noise scale")
    volume: float = Field(default=1.0, gt=0, title="Volume")


@dataclass(frozen=True)
class PendingPiperSynthesis:
    text: str
    model: PiperVoiceModel
    sample_index: int
    input_index: int


PiperRequestGroups = OrderedDict[str, list[PendingPiperSynthesis]]


def plan_piper_requests(batch: list[dict[str, Any]]) -> PiperRequestGroups:
    groups: PiperRequestGroups = OrderedDict()
    for input_index, inputs in enumerate(batch):
        payload = inputs["voice"]
        if payload["kind"] == "tts_voice_batch":
            voices, samples_per_voice = expand_voice_batch(payload, TtsEngine.PIPER)
        else:
            voices, samples_per_voice = [parse_voice(payload, TtsEngine.PIPER)], 1
        for voice in voices:
            assert voice.piper is not None, "validated Piper voice missing model"
            for sample_index in range(samples_per_voice):
                groups.setdefault(voice.piper.checkpoint_id, []).append(
                    PendingPiperSynthesis(inputs["text"], voice.piper, sample_index, input_index)
                )
    return groups


class PiperSynthesisNode(Node):
    NODE_TYPE = "PiperSynthesis"
    DESCRIPTION = "Synthesize text with Piper voices. Each selected voice owns its downloaded ONNX checkpoint; requests are grouped by voice so one model handles its complete text batch before the next model loads."
    CATEGORY = "TTS"
    SETTINGS = PiperSynthesisSettings
    INPUTS = {"text": TextPort(), "voice": JsonPort(join_mode=JoinMode.BROADCAST)}
    OUTPUTS = {"audio": AudioPort(), "synthesis_result": SynthesisResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"cpu": 1}, keep_loaded=False)

    async def execute(self, batch, context):
        groups = plan_piper_requests(list(batch))
        total = sum(len(items) for items in groups.values())
        completed = 0
        outputs = []
        options = PiperSynthesisOptions(
            self.settings.speaker_id,
            self.settings.length_scale,
            self.settings.noise_scale,
            self.settings.noise_w_scale,
            self.settings.volume,
        )
        for checkpoint_id, pending in groups.items():
            context.check_cancel()
            checkpoint = await asyncio.to_thread(resolve_checkpoint_ref, checkpoint_id, TtsEngine.PIPER.value)
            runtime = await asyncio.to_thread(PiperRuntime, checkpoint.path)
            results = await asyncio.to_thread(
                runtime.synthesize_many,
                [item.text for item in pending],
                options,
                context.check_cancel,
            )
            outputs.extend(self._outputs(pending, results, str(context.run_id)))
            completed += len(results)
            await context.report_progress(self.id, completed, total, f"piper synthesized {completed}/{total}")
        return outputs

    def _outputs(self, pending, results, run_id: str):
        outputs = []
        for item, (samples, sample_rate) in zip(pending, results, strict=True):
            request_id = stable_id(
                "tts_request", self.NODE_TYPE, run_id, item.input_index,
                item.text, item.model.voice_id, item.sample_index,
            )
            metadata = {
                "engine": TtsEngine.PIPER.value,
                "voice": item.model.voice_id,
                "language": item.model.language,
                "locale": item.model.locale,
                "quality": item.model.quality,
                "checkpoint_id": item.model.checkpoint_id,
                "text": item.text,
                "sample_index": item.sample_index,
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
            result = SynthesisResult(
                request_id, audio, stable_id("tts_result", request_id), audio.lineage_id, metadata
            )
            outputs.append({INPUT_INDEX_OUTPUT: item.input_index, "audio": audio, "synthesis_result": result})
        return outputs
