from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from runner.nodes.asr.audio import write_temp_wav
from runner.nodes.tts.engines.base import EngineRuntime, require_checkpoint_dir, resolve_device
from runner.nodes.tts.voices import Voice

F5_REPO_ID = "SWivid/F5-TTS"


class F5TtsRuntime(EngineRuntime):
    """F5-TTS: zero-shot cloning from reference audio + its transcript."""

    SAMPLE_RATE = 24000

    def __init__(self, model: Any):
        self._model = model

    def synthesize(self, text: str, voice: Voice, language: str) -> tuple[np.ndarray, int]:
        reference = voice.require_clone()
        ref_file = str(write_temp_wav(reference.wav_bytes))
        wav, sample_rate, _spectrogram = self._model.infer(
            ref_file=ref_file,
            ref_text=reference.transcript,
            gen_text=text,
            remove_silence=False,
        )
        return np.asarray(wav, dtype=np.float32).reshape(-1), int(sample_rate)


F5_MODEL_VARIANT = "F5TTS_v1_Base"


def load(checkpoint_dir: Path, device: str | None = None) -> F5TtsRuntime:
    try:
        from f5_tts.api import F5TTS
    except ImportError as exc:
        raise RuntimeError("f5_tts_not_installed") from exc
    # Point F5TTS at our downloaded weights + vocab instead of letting it re-fetch from HF.
    variant_dir = require_checkpoint_dir(checkpoint_dir) / F5_MODEL_VARIANT
    ckpt_file = next(variant_dir.glob("*.safetensors"), None) or next(variant_dir.glob("*.pt"), None)
    if ckpt_file is None:
        raise FileNotFoundError(f"f5_checkpoint_weights_missing:{variant_dir}")
    vocab_file = variant_dir / "vocab.txt"
    model = F5TTS(
        model=F5_MODEL_VARIANT,
        ckpt_file=str(ckpt_file),
        vocab_file=str(vocab_file) if vocab_file.exists() else "",
        device=device or resolve_device(),
    )
    return F5TtsRuntime(model)
