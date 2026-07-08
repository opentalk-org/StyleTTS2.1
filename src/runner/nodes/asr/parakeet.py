from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.nodes.accelerator_memory import maybe_cuda_half
from runner.nodes.assets.model_downloads import single_checkpoint_file


def load_parakeet_model(checkpoint_dir: Path) -> Any:
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError("nemo_toolkit_not_installed") from exc
    weights = single_checkpoint_file(checkpoint_dir, (".nemo",))
    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(weights))
    model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
    _set_cuda_graph_decoder(model, enabled=False)
    model.eval()
    return maybe_cuda_half(model)


def _set_cuda_graph_decoder(model: Any, *, enabled: bool) -> None:
    try:
        from omegaconf import open_dict
    except ImportError:
        return
    decoding = getattr(getattr(model, "cfg", None), "decoding", None)
    if decoding is None:
        return
    changed = False
    if "greedy" in decoding:
        with open_dict(decoding.greedy):
            decoding.greedy.use_cuda_graph_decoder = enabled
        changed = True
    if "beam" in decoding:
        with open_dict(decoding.beam):
            decoding.beam.allow_cuda_graphs = enabled
        changed = True
    if changed:
        model.change_decoding_strategy(decoding, verbose=False)


def transcribe_wavs_to_segments(
    model: Any,
    wav_paths: list[Path],
    durations_sec: list[float],
    *,
    batch_size: int,
) -> list[list[tuple[float, float, str]]]:
    import torch

    with torch.no_grad():
        outputs = model.transcribe([str(path) for path in wav_paths], batch_size=batch_size, timestamps=True, num_workers=0)
    return [_segments_from_hypothesis(output, durations_sec[index]) for index, output in enumerate(outputs)]


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
