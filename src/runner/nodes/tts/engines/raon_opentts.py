from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from runner.nodes.tts.engines.base import EngineRuntime, resolve_device
from runner.nodes.tts.voices import Voice

RAON_REPO_ID = "KRAFTON/Raon-OpenTTS-1B"
RAON_SAMPLE_RATE = 16000


class RaonOpenTtsRuntime(EngineRuntime):
    """KRAFTON Raon-OpenTTS (F5-TTS-derived DiT + HiFi-GAN). English zero-shot cloning.

    Uses the ``f5_tts`` module that the Raon repo installs (``pip install -e .``);
    cloning needs a reference clip plus its transcript.
    """

    SAMPLE_RATE = RAON_SAMPLE_RATE

    def __init__(self, model: Any, vocoder: Any, infer_process: Any):
        self._model = model
        self._vocoder = vocoder
        self._infer_process = infer_process

    def synthesize(self, text: str, voice: Voice, language: str) -> tuple[np.ndarray, int]:
        from runner.nodes.asr.audio import write_temp_wav

        reference = voice.require_clone()
        ref_file = str(write_temp_wav(reference.wav_bytes))
        waveform, sample_rate, _spectrogram = self._infer_process(
            ref_file, reference.transcript, text, self._model, self._vocoder,
        )
        return np.asarray(waveform, dtype=np.float32).reshape(-1), int(sample_rate)


def load(checkpoint_dir: Path, device: str | None = None) -> RaonOpenTtsRuntime:
    try:
        from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
        from f5_tts.model import DiT
    except ImportError as exc:
        raise RuntimeError("raon_opentts_not_installed") from exc
    device = device or resolve_device()
    weights = next(checkpoint_dir.rglob("*.safetensors"), None) or next(checkpoint_dir.rglob("*.pt"))
    model = load_model(DiT, {}, str(weights), device=device)
    vocoder = load_vocoder(device=device)
    return RaonOpenTtsRuntime(model, vocoder, infer_process)
