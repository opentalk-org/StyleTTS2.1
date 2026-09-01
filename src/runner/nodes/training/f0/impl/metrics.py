from __future__ import annotations

import math
from typing import Any

import torch


def _to_json_scalar(v: Any) -> float | int | None:
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return float(v.detach().float().mean().item())
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (float, int)):
        return v
    if isinstance(v, str):
        return float(v)
    raise TypeError(f"metric value is not numeric: {type(v).__name__}")


def metrics_for_json(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k == "time":
            out[k] = _to_json_scalar(v)
            continue
        s = _to_json_scalar(v)
        if s is not None and isinstance(s, float) and (math.isnan(s) or math.isinf(s)):
            raise ValueError(f"metric {k} is not finite")
        out[k] = s
    return out
