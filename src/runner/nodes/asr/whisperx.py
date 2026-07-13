from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlignmentRequest:
    wav_path: Path
    spans: list[tuple[float, float, str]]


def whisperx_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_whisperx_align_model(checkpoint_dir: Path, language: str, device: str) -> tuple[Any, Any]:
    """Load a WhisperX phoneme alignment model from a downloaded HF wav2vec2 checkpoint folder."""
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError("whisperx_not_installed") from exc
    return whisperx.load_align_model(language_code=language, device=device, model_name=str(checkpoint_dir))


def align_words(
    model: Any,
    metadata: Any,
    wav_path: Path,
    spans: list[tuple[float, float, str]],
    device: str,
) -> list[dict[str, Any]]:
    """Force-align ``spans`` (clip-relative start/end/text) against the audio.

    Returns a flat list of {"word", "start", "end", "score"} in clip-relative time,
    for spans that carry text. Words WhisperX could not place (no timestamp) are dropped.
    """
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError("whisperx_not_installed") from exc
    transcript = [{"start": start, "end": end, "text": text} for start, end, text in spans if text.strip()]
    if not transcript:
        return []
    audio = whisperx.load_audio(str(wav_path))
    result = whisperx.align(transcript, model, metadata, audio, device, return_char_alignments=False)
    return [word for word in (_word_entry(item) for item in result.get("word_segments", [])) if word is not None]


def align_wavs(
    model: Any,
    metadata: Any,
    requests: list[AlignmentRequest],
    device: str,
    check_cancel: Callable[[], None],
) -> list[list[dict[str, Any]]]:
    outputs = []
    for request in requests:
        check_cancel()
        outputs.append(align_words(model, metadata, request.wav_path, request.spans, device))
    return outputs


def _word_entry(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict) or "start" not in item or "end" not in item:
        return None
    text = str(item.get("word", "")).strip()
    if not text:
        return None
    start = float(item["start"])
    end = max(start, float(item["end"]))
    score = item.get("score")
    return {"word": text, "start": start, "end": end, "score": None if score is None else float(score)}
