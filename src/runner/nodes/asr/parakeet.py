from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PARAKEET_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


def load_parakeet_model(cache_dir: Path) -> Any:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NEMO_CACHE_DIR"] = str(cache_dir / "nemo")
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError("nemo_toolkit_not_installed") from exc
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_MODEL_NAME)
    model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
    model.eval()
    return _maybe_cuda_half(model)


def transcribe_wavs_to_segments(model: Any, wav_paths: list[Path], durations_sec: list[float]) -> list[list[tuple[float, float, str]]]:
    import torch

    with torch.no_grad():
        outputs = model.transcribe([str(path) for path in wav_paths], timestamps=True)
    return [_segments_from_hypothesis(output, durations_sec[index]) for index, output in enumerate(outputs)]


def _maybe_cuda_half(model: Any) -> Any:
    import torch

    if torch.cuda.is_available():
        return model.cuda().half()
    return model


def _segments_from_hypothesis(output: Any, duration_sec: float) -> list[tuple[float, float, str]]:
    timestamp = getattr(output, "timestamp", None)
    if isinstance(timestamp, dict) and isinstance(timestamp.get("segment"), list):
        spans = [_span_from_parakeet_segment(item, duration_sec) for item in timestamp["segment"] if isinstance(item, dict)]
        return [span for span in spans if span[2]]
    text = str(getattr(output, "text", output)).strip()
    if not text:
        return []
    return [(0.0, max(0.0, duration_sec), text)]


def _span_from_parakeet_segment(item: dict, duration_sec: float) -> tuple[float, float, str]:
    start = max(0.0, float(item.get("start", 0.0)))
    end = max(start, float(item.get("end", start)))
    if duration_sec > 0:
        start = min(start, duration_sec)
        end = min(max(start, end), duration_sec)
    return start, end, str(item.get("segment", "")).strip()
