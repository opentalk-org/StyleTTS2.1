from __future__ import annotations

from pathlib import Path
from typing import Any


WHISPER_MODEL_NAME = "small"


def load_whisper_model(cache_dir: Path) -> Any:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("openai_whisper_not_installed") from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    return whisper.load_model(WHISPER_MODEL_NAME, download_root=str(cache_dir))


def transcribe_wav_to_segments(model: Any, wav_path: Path, duration_sec: float) -> list[tuple[float, float, str]]:
    result = model.transcribe(str(wav_path))
    if not isinstance(result, dict):
        return []
    raw_segments = result.get("segments")
    if isinstance(raw_segments, list) and raw_segments:
        spans = [_span_from_whisper_segment(item, duration_sec) for item in raw_segments if isinstance(item, dict)]
        return [span for span in spans if span[2]]
    text = str(result.get("text", "")).strip()
    if not text:
        return []
    return [(0.0, max(0.0, duration_sec), text)]


def transcribe_wav_to_text(model: Any, wav_path: Path) -> str:
    result = model.transcribe(str(wav_path))
    if not isinstance(result, dict):
        return ""
    return str(result.get("text", "")).strip()


def _span_from_whisper_segment(item: dict, duration_sec: float) -> tuple[float, float, str]:
    start = max(0.0, float(item.get("start", 0.0)))
    end = max(start, float(item.get("end", start)))
    if duration_sec > 0:
        start = min(start, duration_sec)
        end = min(max(start, end), duration_sec)
    return start, end, str(item.get("text", "")).strip()
