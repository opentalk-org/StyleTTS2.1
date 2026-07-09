from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from runner.nodes.tts.audio_out import samples_from_wav_bytes
from runner.nodes.tts.engines.base import EngineRuntime, resolve_device
from runner.nodes.tts.voices import Voice

FISH_REPO_ID = "fishaudio/openaudio-s1-mini"
FISH_SAMPLE_RATE = 44100


class FishSpeechRuntime(EngineRuntime):
    """Fish Speech / OpenAudio S1-mini via its TTSInferenceEngine.

    Cloning needs a reference clip plus its transcript. Driven through the
    ServeTTSRequest / ServeReferenceAudio API exposed by the fish_speech package.
    """

    SAMPLE_RATE = FISH_SAMPLE_RATE

    def __init__(self, engine: Any, request_cls: Any, reference_cls: Any):
        self._engine = engine
        self._request_cls = request_cls
        self._reference_cls = reference_cls

    def synthesize(self, text: str, voice: Voice, language: str) -> tuple[np.ndarray, int]:
        references = []
        if voice.clone is not None:
            references.append(self._reference_cls(audio=voice.clone.wav_bytes, text=voice.clone.transcript))
        request = self._request_cls(text=text, references=references, format="wav")
        chunks = [result.audio for result in self._engine.inference(request) if result.code == "segment"]
        if not chunks:
            raise RuntimeError("fish_speech_empty_audio")
        merged = [_chunk_to_samples(chunk) for chunk in chunks]
        return np.concatenate(merged).astype(np.float32), self.SAMPLE_RATE


def load(checkpoint_dir: Path, device: str | None = None) -> FishSpeechRuntime:
    try:
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
    except ImportError as exc:
        raise RuntimeError("fish_speech_not_installed") from exc
    device = device or resolve_device()
    decoder_path = next(checkpoint_dir.rglob("*.pth"))
    llama_queue = launch_thread_safe_queue(checkpoint_path=str(checkpoint_dir), device=device, precision="bfloat16")
    decoder = load_decoder(config_name="modded_dac_vq", checkpoint_path=str(decoder_path), device=device)
    engine = TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder, precision="bfloat16", compile=False)
    return FishSpeechRuntime(engine, ServeTTSRequest, ServeReferenceAudio)


def _chunk_to_samples(chunk: Any) -> np.ndarray:
    if isinstance(chunk, (bytes, bytearray)):
        samples, _rate = samples_from_wav_bytes(bytes(chunk))
        return samples
    array = np.asarray(chunk[1] if isinstance(chunk, tuple) else chunk, dtype=np.float32)
    return array.reshape(-1)
