from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from runner.nodes.asr.audio import write_temp_wav
from runner.nodes.tts.engines.base import (
    EngineRuntime,
    require_checkpoint_dir,
    resolve_device,
)
from runner.nodes.tts.voices import Voice

CHATTERBOX_REPO_ID = "ResembleAI/chatterbox"


class ChatterboxRuntime(EngineRuntime):
    """Resemble AI Chatterbox: zero-shot cloning from a ~10s reference, no transcript."""

    SAMPLE_RATE = 24000

    def __init__(self, model: Any, multilingual: bool):
        self._model = model
        self._multilingual = multilingual
        self._active_clone_digest: bytes | None = None
        self.SAMPLE_RATE = int(model.sr)

    def synthesize(
        self, text: str, voice: Voice, language: str
    ) -> tuple[np.ndarray, int]:
        kwargs: dict[str, Any] = {}
        if voice.clone is not None:
            clone_digest = hashlib.blake2b(
                voice.clone.wav_bytes,
                digest_size=16,
            ).digest()
            if clone_digest != self._active_clone_digest:
                prompt_path = write_temp_wav(voice.clone.wav_bytes)
                try:
                    self._model.prepare_conditionals(str(prompt_path))
                finally:
                    prompt_path.unlink()
                self._active_clone_digest = clone_digest
        if self._multilingual:
            kwargs["language_id"] = language
        wav = self._model.generate(text, **kwargs)
        return _to_mono_float(wav), self.SAMPLE_RATE


def load(
    checkpoint_dir: Path, device: str | None = None, multilingual: bool = True
) -> ChatterboxRuntime:
    device = device or resolve_device()
    try:
        if multilingual:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Model
        else:
            from chatterbox.tts import ChatterboxTTS as Model
    except ImportError as exc:
        raise RuntimeError("chatterbox_not_installed") from exc
    # Load the weights from our downloaded checkpoint folder rather than letting from_pretrained
    # re-fetch the whole repo into a separate HF cache.
    model = Model.from_local(require_checkpoint_dir(checkpoint_dir), device)
    return ChatterboxRuntime(model, multilingual)


def _to_mono_float(wav: Any) -> np.ndarray:
    array = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
    return array.reshape(-1).astype(np.float32)
