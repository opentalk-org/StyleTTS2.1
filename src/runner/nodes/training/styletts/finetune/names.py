from __future__ import annotations

import re


def slugify_segment(value: str, *, max_len: int = 64) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    collapsed = re.sub(r"-+", "-", normalized).strip("-._")
    if not collapsed:
        return "run"
    return collapsed[:max_len].strip("-._") or "run"
