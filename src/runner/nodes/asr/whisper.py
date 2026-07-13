from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from runner.nodes.asr.confidence import confidence_from_avg_logprob
from runner.nodes.assets.model_downloads import single_checkpoint_file


def load_whisper_model(checkpoint_dir: Path) -> Any:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("openai_whisper_not_installed") from exc
    weights = single_checkpoint_file(checkpoint_dir, (".pt",))
    return whisper.load_model(str(weights))


def transcribe_wav_to_segments(
    model: Any, wav_path: Path, duration_sec: float, language: str
) -> list[tuple[float, float, str, float | None]]:
    result = model.transcribe(str(wav_path), language=_whisper_language(language))
    if not isinstance(result, dict):
        return []
    raw_segments = result.get("segments")
    if isinstance(raw_segments, list) and raw_segments:
        spans = [_span_from_whisper_segment(item, duration_sec) for item in raw_segments if isinstance(item, dict)]
        return [span for span in spans if span[2]]
    text = str(result.get("text", "")).strip()
    if not text:
        return []
    return [(0.0, max(0.0, duration_sec), text, None)]


def transcribe_wavs_to_segments(
    model: Any,
    wav_paths: list[Path],
    durations_sec: list[float],
    language: str,
    check_cancel: Callable[[], None],
) -> list[list[tuple[float, float, str, float | None]]]:
    assert len(wav_paths) == len(durations_sec), "whisper batch path/duration mismatch"
    outputs = []
    for path, duration in zip(wav_paths, durations_sec, strict=True):
        check_cancel()
        outputs.append(transcribe_wav_to_segments(model, path, duration, language))
    return outputs


def _span_from_whisper_segment(item: dict, duration_sec: float) -> tuple[float, float, str, float | None]:
    start = max(0.0, float(item.get("start", 0.0)))
    end = max(start, float(item.get("end", start)))
    if duration_sec > 0:
        start = min(start, duration_sec)
        end = min(max(start, end), duration_sec)
    confidence = confidence_from_avg_logprob(item.get("avg_logprob"))
    return start, end, str(item.get("text", "")).strip(), confidence


def _whisper_language(language: str) -> str | None:
    value = language.strip().lower()
    return None if not value or value == "auto" else value
