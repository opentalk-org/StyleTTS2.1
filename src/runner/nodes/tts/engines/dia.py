from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from runner.nodes.tts.audio_out import samples_from_wav_bytes
from runner.nodes.tts.engines.base import EngineRuntime, resolve_device
from runner.nodes.tts.voices import Voice

DIA_REPO_ID = "nari-labs/Dia-1.6B-0626"
DIA_SAMPLE_RATE = 44100


class DiaRuntime(EngineRuntime):
    """Dia dialogue TTS via HF Transformers (DiaForConditionalGeneration).

    No fixed speaker; a clone reference (audio + transcript) conditions the voice,
    otherwise the speaker is sampled fresh each run.
    """

    SAMPLE_RATE = DIA_SAMPLE_RATE

    def __init__(self, model: Any, processor: Any, device: str):
        self._model = model
        self._processor = processor
        self._device = device

    def synthesize(self, text: str, voice: Voice, language: str) -> tuple[np.ndarray, int]:
        prompt = text if text.lstrip().startswith("[S") else f"[S1] {text}"
        processor_kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt", "padding": True}
        if voice.clone is not None:
            reference, ref_rate = samples_from_wav_bytes(voice.clone.wav_bytes)
            prompt = f"[S1] {voice.clone.transcript} {text}"
            processor_kwargs["text"] = [prompt]
            processor_kwargs["audio"] = [reference]
            processor_kwargs["sampling_rate"] = ref_rate
        inputs = self._processor(**processor_kwargs).to(self._device)
        outputs = self._model.generate(**inputs, max_new_tokens=3072, guidance_scale=3.0)
        decoded = self._processor.batch_decode(outputs)
        waveform = np.asarray(decoded[0], dtype=np.float32).reshape(-1)
        return waveform, DIA_SAMPLE_RATE


def load(checkpoint_dir: Path, device: str | None = None) -> DiaRuntime:
    try:
        from transformers import AutoProcessor, DiaForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("dia_transformers_not_installed") from exc
    device = device or resolve_device()
    processor = AutoProcessor.from_pretrained(str(checkpoint_dir))
    model = DiaForConditionalGeneration.from_pretrained(str(checkpoint_dir)).to(device)
    return DiaRuntime(model, processor, device)
