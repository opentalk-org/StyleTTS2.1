from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch


def _wav_batch_b_t(x: torch.Tensor) -> torch.Tensor:
    t = x.detach().float()
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() == 3 and t.shape[1] == 1:
        return t.squeeze(1)
    return t


def export_finetune_val_wavs_for_studio(
    log_dir: str | Path,
    *,
    sample_rate: int,
    step: int,
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    max_utts: int = 4,
) -> list[dict[str, str]]:
    yp = _wav_batch_b_t(y_pred)
    yg = _wav_batch_b_t(y_gt)
    base = Path(log_dir) / "samples" / f"step_{int(step):09d}"
    base.mkdir(parents=True, exist_ok=True)
    b = min(int(yp.shape[0]), int(yg.shape[0]), int(max_utts))
    out: list[dict[str, str]] = []
    for i in range(b):
        rel_gt = f"samples/step_{int(step):09d}/{i}_gt.wav"
        rel_pred = f"samples/step_{int(step):09d}/{i}_pred.wav"
        gt = yg[i].cpu().reshape(-1).clamp(-1.0, 1.0).numpy().astype(np.float64)
        pr = yp[i].cpu().reshape(-1).clamp(-1.0, 1.0).numpy().astype(np.float64)
        pcm_gt = (np.clip(gt, -1.0, 1.0) * 32767.0).astype(np.int16)
        pcm_pr = (np.clip(pr, -1.0, 1.0) * 32767.0).astype(np.int16)
        for name, pcm in ((f"{i}_gt.wav", pcm_gt), (f"{i}_pred.wav", pcm_pr)):
            with wave.open(str(base / name), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(sample_rate))
                wf.writeframes(pcm.tobytes())
        out.append({"role": "gt", "path": rel_gt, "index": str(i)})
        out.append({"role": "pred", "path": rel_pred, "index": str(i)})
    return out
