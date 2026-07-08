from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.nodes.accelerator_memory import maybe_cuda_half
from runner.nodes.assets.model_downloads import single_checkpoint_file


def load_canary_model(checkpoint_dir: Path) -> Any:
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError("nemo_toolkit_not_installed") from exc
    weights = single_checkpoint_file(checkpoint_dir, (".nemo",))
    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(weights))
    model.eval()
    return maybe_cuda_half(model)


def transcribe_wavs_to_segments(
    model: Any,
    wav_paths: list[Path],
    durations_sec: list[float],
    *,
    source_language: str,
    target_language: str,
    pnc: bool,
    batch_size: int,
) -> list[list[tuple[float, float, str]]]:
    import torch

    with torch.no_grad():
        outputs = model.transcribe(
            [str(path) for path in wav_paths],
            batch_size=batch_size,
            num_workers=0,
            source_lang=source_language,
            target_lang=target_language,
            taskname="asr",
            pnc="yes" if pnc else "no",
        )
    return [_segments_from_output(output, durations_sec[index]) for index, output in enumerate(outputs)]


def _segments_from_output(output: Any, duration_sec: float) -> list[tuple[float, float, str]]:
    text = str(getattr(output, "text", output)).strip()
    if not text:
        return []
    return [(0.0, max(0.0, duration_sec), text)]
