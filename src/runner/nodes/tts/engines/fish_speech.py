from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from runner.nodes.tts.audio_out import samples_from_wav_bytes
from runner.nodes.tts.engines.base import EngineRuntime, resolve_device
from runner.nodes.tts.engines.fish_queue import (
    FishQueueDependencies,
    launch_memory_bounded_queue,
)
from runner.nodes.tts.voices import Voice

FISH_REPO_ID = "fishaudio/s2-pro"
FISH_SAMPLE_RATE = 44100


class FishSpeechRuntime(EngineRuntime):
    """Fish Audio S2-Pro (dual-AR, 80+ languages) via fish-speech's TTSInferenceEngine.

    S2-Pro ships a ``fish_qwen3_omni`` config that fish-speech maps to its ``dual_ar``
    model, so the same engine path runs it. Cloning needs a reference clip + transcript;
    inline ``[tag]`` control (e.g. ``[whisper]``) can be embedded in the text.
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
        # Non-streaming yields one "final" result with the whole waveform; streaming yields
        # "segment" chunks. "header" carries only a WAV header and "error" carries no audio.
        segments = [
            result.audio
            for result in self._engine.inference(request)
            if result.code in ("segment", "final") and result.audio is not None
        ]
        if not segments:
            raise RuntimeError("fish_speech_empty_audio")
        sample_rate = int(segments[0][0]) if isinstance(segments[0], tuple) else self.SAMPLE_RATE
        merged = [_chunk_to_samples(segment) for segment in segments]
        return np.concatenate(merged).astype(np.float32), sample_rate


def load(checkpoint_dir: Path, device: str | None = None) -> FishSpeechRuntime:
    try:
        import torch

        import fish_speech
    except ImportError as exc:
        raise RuntimeError("fish_speech_not_installed") from exc
    # fish-speech submodules call pyrootutils.setup_root(indicator=".project-root") at import
    # (they assume a source checkout). Installed as a namespace package there is no marker, so
    # drop one next to the package before importing the submodules that need it.
    package_dir = Path(list(fish_speech.__path__)[0]).resolve()
    (package_dir.parent / ".project-root").touch(exist_ok=True)
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder
    from fish_speech.models.text2semantic.inference import (
        WrappedGenerateResponse,
        generate_long,
        init_model,
    )
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    device = device or resolve_device()
    decoder_path = _codec_checkpoint(checkpoint_dir)
    queue_dependencies = FishQueueDependencies(
        init_model=init_model,
        generate_long=generate_long,
        wrapped_response=WrappedGenerateResponse,
    )
    llama_queue = launch_memory_bounded_queue(
        checkpoint_path=str(checkpoint_dir),
        device=device,
        precision=torch.bfloat16,
        dependencies=queue_dependencies,
    )
    decoder = load_decoder(config_name="modded_dac_vq", checkpoint_path=str(decoder_path), device=device)
    engine = TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder, precision=torch.bfloat16, compile=False)
    return FishSpeechRuntime(engine, ServeTTSRequest, ServeReferenceAudio)


def _codec_checkpoint(checkpoint_dir: Path) -> Path:
    """The DAC/codec weights inside an OpenAudio checkpoint folder (codec.pth)."""
    named = sorted(checkpoint_dir.rglob("*codec*.pth")) or sorted(checkpoint_dir.rglob("*firefly*.pth"))
    if named:
        return named[0]
    return next(checkpoint_dir.rglob("*.pth"))


def _chunk_to_samples(chunk: Any) -> np.ndarray:
    if isinstance(chunk, (bytes, bytearray)):
        samples, _rate = samples_from_wav_bytes(bytes(chunk))
        return samples
    raw = chunk[1] if isinstance(chunk, tuple) else chunk
    array = np.asarray(raw)
    if np.issubdtype(array.dtype, np.integer):
        return (array.astype(np.float32) / 32768.0).reshape(-1)
    return array.astype(np.float32).reshape(-1)
