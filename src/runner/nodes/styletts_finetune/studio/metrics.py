from __future__ import annotations

import math
from typing import Any

import torch


def _to_json_scalar(v: Any) -> float | int | None:
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return float(v.detach().float().mean().item()) if v.numel() else 0.0
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (float, int)):
        return v
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _metrics_for_json(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k == "time":
            out[k] = float(v) if isinstance(v, (int, float)) else _to_json_scalar(v)
            continue
        s = _to_json_scalar(v)
        if s is not None and isinstance(s, float) and (math.isnan(s) or math.isinf(s)):
            continue
        out[k] = s
    return out
