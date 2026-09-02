from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TtsEngine(str, Enum):
    """Engines with a synthesis node. Values double as checkpoint ``type_`` tags."""

    KOKORO = "kokoro"
    CHATTERBOX = "chatterbox"
    F5_TTS = "f5_tts"
    ORPHEUS = "orpheus"
    DIA = "dia"
    FISH_SPEECH = "fish_speech"
    RAON_OPENTTS = "raon_opentts"
    PIPER = "piper"


PRESET_VOICES: dict[TtsEngine, tuple[str, ...]] = {
    TtsEngine.KOKORO: (
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "pf_dora",
        "pm_alex",
        "pm_santa",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    ),
    TtsEngine.ORPHEUS: ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"),
}

CLONING_ENGINES: frozenset[TtsEngine] = frozenset(
    {
        TtsEngine.CHATTERBOX,
        TtsEngine.F5_TTS,
        TtsEngine.ORPHEUS,
        TtsEngine.DIA,
        TtsEngine.FISH_SPEECH,
        TtsEngine.RAON_OPENTTS,
    }
)


@dataclass(frozen=True)
class CloneReference:
    """A reference clip for zero-shot cloning, carried inline as base64 WAV."""

    wav_base64: str
    sample_rate: int
    transcript: str

    @property
    def wav_bytes(self) -> bytes:
        return base64.b64decode(self.wav_base64)


@dataclass(frozen=True)
class PiperVoiceModel:
    voice_id: str
    checkpoint_id: str
    language: str
    locale: str
    quality: str
    sample_rate: int


@dataclass(frozen=True)
class Voice:
    """A resolved voice: either a preset id or a clone reference, for one engine."""

    engine: TtsEngine
    preset: str | None
    clone: CloneReference | None
    piper: PiperVoiceModel | None = None

    def require_preset(self) -> str:
        if self.preset is None:
            raise ValueError(f"{self.engine.value}_voice_missing_preset")
        return self.preset

    def require_clone(self) -> CloneReference:
        if self.clone is None:
            raise ValueError(f"{self.engine.value}_voice_missing_clone_reference")
        return self.clone


def preset_voice_payload(engine: TtsEngine, voice_id: str) -> dict[str, Any]:
    valid = PRESET_VOICES.get(engine, ())
    if voice_id not in valid:
        raise ValueError(f"{engine.value}_unknown_preset_voice:{voice_id}")
    return {"kind": "tts_voice", "engine": engine.value, "voice_id": voice_id}


def piper_voice_payload(model: PiperVoiceModel) -> dict[str, Any]:
    return {
        "kind": "tts_voice",
        "engine": TtsEngine.PIPER.value,
        "voice_id": model.voice_id,
        "piper": {
            "voice_id": model.voice_id,
            "checkpoint_id": model.checkpoint_id,
            "language": model.language,
            "locale": model.locale,
            "quality": model.quality,
            "sample_rate": model.sample_rate,
        },
    }


def clone_voice_payload(engine: TtsEngine, reference: CloneReference) -> dict[str, Any]:
    return {
        "kind": "tts_voice",
        "engine": engine.value,
        "clone": {
            "wav_base64": reference.wav_base64,
            "sample_rate": reference.sample_rate,
            "transcript": reference.transcript,
        },
    }


def voice_batch_payload(
    engine: TtsEngine, voices: list[dict[str, Any]], samples_per_voice: int
) -> dict[str, Any]:
    return {
        "kind": "tts_voice_batch",
        "engine": engine.value,
        "voices": voices,
        "samples_per_voice": samples_per_voice,
    }


def parse_voice(payload: dict[str, Any], expected_engine: TtsEngine) -> Voice:
    if payload["kind"] != "tts_voice":
        raise ValueError(f"expected_tts_voice_got:{payload['kind']}")
    engine = TtsEngine(payload["engine"])
    if engine != expected_engine:
        raise ValueError(
            f"voice_engine_mismatch:{engine.value}!={expected_engine.value}"
        )
    clone_raw = payload.get("clone")
    clone = None
    if clone_raw is not None:
        clone = CloneReference(
            clone_raw["wav_base64"],
            int(clone_raw["sample_rate"]),
            clone_raw["transcript"],
        )
    piper_raw = payload.get("piper")
    piper = None
    if piper_raw is not None:
        piper = PiperVoiceModel(
            voice_id=piper_raw["voice_id"],
            checkpoint_id=piper_raw["checkpoint_id"],
            language=piper_raw["language"],
            locale=piper_raw["locale"],
            quality=piper_raw["quality"],
            sample_rate=int(piper_raw["sample_rate"]),
        )
    if engine is TtsEngine.PIPER and piper is None:
        raise ValueError("piper_voice_missing_model")
    return Voice(
        engine=engine, preset=payload.get("voice_id"), clone=clone, piper=piper
    )


def expand_voice_batch(
    payload: dict[str, Any], expected_engine: TtsEngine
) -> tuple[list[Voice], int]:
    """Return (voices, samples_per_voice) from a voice_batch payload."""
    if payload["kind"] != "tts_voice_batch":
        raise ValueError(f"expected_tts_voice_batch_got:{payload['kind']}")
    engine = TtsEngine(payload["engine"])
    if engine != expected_engine:
        raise ValueError(
            f"voice_batch_engine_mismatch:{engine.value}!={expected_engine.value}"
        )
    voices = [parse_voice(item, expected_engine) for item in payload["voices"]]
    samples = int(payload["samples_per_voice"])
    if not voices:
        raise ValueError(f"{engine.value}_voice_batch_empty")
    return voices, samples
