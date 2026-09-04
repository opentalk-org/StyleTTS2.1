from __future__ import annotations

import base64
import random
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort, JsonPort, TextPort
from runner.nodes.models import Audio
from runner.nodes.tts.audio_out import samples_from_wav_bytes, wav_bytes_from_samples
from runner.nodes.tts.voices import (
    PRESET_VOICES,
    CloneReference,
    TtsEngine,
    clone_voice_payload,
    preset_voice_payload,
    voice_batch_payload,
)


class TtsSelectVoiceSettings(StrictSettings):
    engine: TtsEngine = Field(default=TtsEngine.KOKORO, title="Engine")
    voice_id: str = Field(default="af_heart", title="Preset voice")


class TtsRandomVoicesSettings(StrictSettings):
    engine: TtsEngine = Field(default=TtsEngine.KOKORO, title="Engine")
    count: int = Field(default=5, ge=1, le=64, title="Random voices")
    samples_per_voice: int = Field(default=1, ge=1, le=16, title="Samples per voice")
    seed: int = Field(default=0, title="Seed")


class _EmitOnceNode(Node):
    """Input node that emits its single output exactly once."""

    IS_INPUT = True
    INPUTS: dict[str, Any] = {}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._emitted = False

    def remaining_items(self, context) -> int:
        return 0 if self._emitted else 1


class TtsSelectVoiceNode(_EmitOnceNode):
    NODE_TYPE = "TtsSelectVoice"
    DESCRIPTION = "Pick a single preset voice for an engine that ships fixed voices. Choose the engine and one of its built-in voice ids, and it emits a single voice to wire into that engine's synthesis node. Use this when you want a specific known speaker rather than a cloned or random one."
    CATEGORY = "TTS"
    SETTINGS = TtsSelectVoiceSettings
    OUTPUTS = {"voice": JsonPort()}

    async def execute(self, batch, context):
        assert not self._emitted, f"voice node already emitted: {self.id}"
        self._emitted = True
        return [
            {
                "voice": preset_voice_payload(
                    self.settings.engine, self.settings.voice_id
                )
            }
        ]


class TtsRandomVoicesNode(_EmitOnceNode):
    NODE_TYPE = "TtsRandomVoices"
    DESCRIPTION = "Emit a batch of N random preset voices for an engine that ships fixed voices. Choose the engine, how many voices to sample, samples per voice, and a seed for reproducibility; it outputs a voice batch that fans out multiple clips per text at synthesis time. Use it to quickly compare or bulk-generate across many speakers."
    CATEGORY = "TTS"
    SETTINGS = TtsRandomVoicesSettings
    OUTPUTS = {"voice": JsonPort()}

    async def execute(self, batch, context):
        assert not self._emitted, f"voice node already emitted: {self.id}"
        self._emitted = True
        presets = PRESET_VOICES.get(self.settings.engine, ())
        if not presets:
            raise ValueError(f"{self.settings.engine.value}_has_no_preset_voices")
        take = min(self.settings.count, len(presets))
        chosen = random.Random(self.settings.seed).sample(list(presets), take)
        voices = [
            preset_voice_payload(self.settings.engine, voice_id) for voice_id in chosen
        ]
        return [
            {
                "voice": voice_batch_payload(
                    self.settings.engine, voices, self.settings.samples_per_voice
                )
            }
        ]


class TtsCloneVoiceSettings(StrictSettings):
    default_transcript: str = Field(default="", title="Reference transcript")


class TtsCloneVoiceNode(Node):
    """Base voice-cloning node: reference audio (+ optional transcript) -> voice.

    Load the reference with the generic ``LoadAudio``/``AudioSource`` nodes and wire
    it into ``audio``. ``ENGINE`` is fixed per subclass.
    """

    ENGINE: TtsEngine
    CATEGORY = "TTS"
    SETTINGS = TtsCloneVoiceSettings
    INPUTS = {"audio": AudioPort(), "transcript": TextPort(optional=True)}
    OUTPUTS = {"voice": JsonPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        voices = clone_voice_payloads(
            list(batch),
            self.ENGINE,
            self.settings.default_transcript,
        )
        return [{"voice": voice} for voice in voices]


def clone_voice_payloads(
    batch: list[dict[str, Any]],
    engine: TtsEngine,
    default_transcript: str,
) -> list[dict[str, Any]]:
    outputs = []
    for inputs in batch:
        audio: Audio = inputs["audio"]
        assert audio.data is not None, f"clone reference needs audio bytes: {audio.id}"
        samples, sample_rate = samples_from_wav_bytes(audio.data)
        wav_base64 = base64.b64encode(
            wav_bytes_from_samples(samples, sample_rate)
        ).decode("ascii")
        transcript = inputs.get("transcript") or default_transcript
        reference = CloneReference(
            wav_base64=wav_base64,
            sample_rate=sample_rate,
            transcript=transcript,
        )
        outputs.append(clone_voice_payload(engine, reference))
    return outputs


class ChatterboxCloneVoiceNode(TtsCloneVoiceNode):
    NODE_TYPE = "ChatterboxCloneVoice"
    DESCRIPTION = "Build a Chatterbox voice by cloning a reference recording. Wire reference audio (loaded with a generic audio source node) into the audio input, optionally with a transcript of what is said, and it outputs a cloned voice to feed into the Chatterbox synthesis node."
    ENGINE = TtsEngine.CHATTERBOX


class F5TtsCloneVoiceNode(TtsCloneVoiceNode):
    NODE_TYPE = "F5TtsCloneVoice"
    DESCRIPTION = "Build an F5-TTS voice by cloning a reference recording. Wire reference audio (loaded with a generic audio source node) into the audio input, optionally with a transcript of what is said, and it outputs a cloned voice to feed into the F5-TTS synthesis node."
    ENGINE = TtsEngine.F5_TTS


class OrpheusCloneVoiceNode(TtsCloneVoiceNode):
    NODE_TYPE = "OrpheusCloneVoice"
    DESCRIPTION = "Build an Orpheus voice by cloning a reference recording. Wire reference audio (loaded with a generic audio source node) into the audio input, optionally with a transcript of what is said, and it outputs a cloned voice to feed into the Orpheus synthesis node."
    ENGINE = TtsEngine.ORPHEUS


class DiaCloneVoiceNode(TtsCloneVoiceNode):
    NODE_TYPE = "DiaCloneVoice"
    DESCRIPTION = "Build a Dia voice by cloning a reference recording. Wire reference audio (loaded with a generic audio source node) into the audio input, optionally with a transcript of what is said, and it outputs a cloned voice to feed into the Dia synthesis node."
    ENGINE = TtsEngine.DIA


class FishSpeechCloneVoiceNode(TtsCloneVoiceNode):
    NODE_TYPE = "FishSpeechCloneVoice"
    DESCRIPTION = "Build a Fish Speech voice by cloning a reference recording. Wire reference audio (loaded with a generic audio source node) into the audio input, optionally with a transcript of what is said, and it outputs a cloned voice to feed into the Fish Speech synthesis node."
    ENGINE = TtsEngine.FISH_SPEECH


class RaonOpenTtsCloneVoiceNode(TtsCloneVoiceNode):
    NODE_TYPE = "RaonOpenTtsCloneVoice"
    DESCRIPTION = "Build a Raon OpenTTS voice by cloning a reference recording. Wire reference audio (loaded with a generic audio source node) into the audio input, optionally with a transcript of what is said, and it outputs a cloned voice to feed into the Raon OpenTTS synthesis node."
    ENGINE = TtsEngine.RAON_OPENTTS
