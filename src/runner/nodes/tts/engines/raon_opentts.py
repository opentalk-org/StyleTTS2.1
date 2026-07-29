from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from runner.nodes.asr.audio import write_temp_wav
from runner.nodes.tts.engines.base import EngineRuntime, resolve_device
from runner.nodes.tts.voices import Voice

RAON_REPO_ID = "KRAFTON/Raon-OpenTTS-1B"
RAON_SAMPLE_RATE = 16000
RAON_VOCAB_SIZE = 5555


class RaonOpenTtsRuntime(EngineRuntime):
    """KRAFTON Raon-OpenTTS (F5-TTS-derived DiT + HiFi-GAN). English zero-shot cloning.

    Uses the ``f5_tts`` module that the Raon repo installs (``pip install -e .``);
    cloning needs a reference clip plus its transcript.
    """

    SAMPLE_RATE = RAON_SAMPLE_RATE

    def __init__(
        self,
        model: Any,
        vocoder: Any,
        infer_process: Any,
        mel_spec_type: str,
    ):
        self._model = model
        self._vocoder = vocoder
        self._infer_process = infer_process
        self._mel_spec_type = mel_spec_type

    def synthesize(self, text: str, voice: Voice, language: str) -> tuple[np.ndarray, int]:
        reference = voice.require_clone()
        ref_file = str(write_temp_wav(reference.wav_bytes))
        waveform, sample_rate, _spectrogram = self._infer_process(
            ref_file,
            reference.transcript,
            text,
            self._model,
            self._vocoder,
            mel_spec_type=self._mel_spec_type,
        )
        return np.asarray(waveform, dtype=np.float32).reshape(-1), int(sample_rate)


def load(checkpoint_dir: Path, device: str | None = None) -> RaonOpenTtsRuntime:
    device = device or resolve_device()
    runtime_root = checkpoint_dir / "runtime"
    assert (runtime_root / "raon_f5_tts").is_dir(), (
        f"Raon runtime source missing from checkpoint: {checkpoint_dir}"
    )
    runtime_path = str(runtime_root)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)

    model_module = importlib.import_module("raon_f5_tts.model")
    model_utils = importlib.import_module("raon_f5_tts.model.utils")
    infer_utils = importlib.import_module("raon_f5_tts.infer.utils_infer")
    config = OmegaConf.load(checkpoint_dir / "config.yaml")
    vocab_map, vocab_size = model_utils.get_tokenizer(
        str(checkpoint_dir / "vocab.txt"),
        "custom",
    )
    vocab_map = {
        token: index
        for token, index in vocab_map.items()
        if index < RAON_VOCAB_SIZE
    }
    vocab_size = len(vocab_map)
    assert vocab_size == RAON_VOCAB_SIZE, (
        f"Raon vocabulary has {vocab_size} model tokens, expected {RAON_VOCAB_SIZE}"
    )
    mel_spec_type = str(config.model.mel_spec.mel_spec_type)
    weights = next(checkpoint_dir.glob("*.safetensors"), None)
    if weights is None:
        weights = next(checkpoint_dir.glob("*.pt"))
    model = infer_utils.load_model(
        model_module.DiT,
        OmegaConf.to_container(config.model.arch, resolve=True),
        str(weights),
        vocab_map,
        vocab_size,
        mel_spec_type=mel_spec_type,
        device=device,
    )
    vocoder = infer_utils.load_vocoder(
        vocoder_name=mel_spec_type,
        is_local=True,
        local_path=str(checkpoint_dir / "vocoder"),
        device=device,
    )
    return RaonOpenTtsRuntime(
        model,
        vocoder,
        infer_utils.infer_process,
        mel_spec_type,
    )
