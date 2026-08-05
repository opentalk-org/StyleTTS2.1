from __future__ import annotations

import os

import torch


_VRAM_HEADROOM_GB = 2.0
_MIN_VRAM_GB = 8.0


_MEMORY_BUDGET_FRACTION = 0.5
_MIN_MEMORY_BUDGET_MB = 512.0


def detect_system_ram_mb() -> float:
    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    if pages <= 0 or page_size <= 0:
        raise RuntimeError("system RAM size is unavailable")
    return pages * page_size / (1024 * 1024)


def default_memory_budget_mb() -> float:
    total = detect_system_ram_mb()
    return max(_MIN_MEMORY_BUDGET_MB, round(total * _MEMORY_BUDGET_FRACTION))


def detect_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_bytes / (1024 ** 3)
    return max(_MIN_VRAM_GB, round(total_gb - _VRAM_HEADROOM_GB, 1))


def apply_detected_resources(resources: dict[str, float]) -> dict[str, float]:
    updated = dict(resources)
    detected = detect_vram_gb()
    if detected is not None:
        configured = float(updated["vram_gb"]) if "vram_gb" in updated else 0.0
        updated["vram_gb"] = max(configured, detected)
    return updated
