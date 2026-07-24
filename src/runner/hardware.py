from __future__ import annotations

import os

import torch

# Reserve a little head-room so the scheduler's vram budget never promises the
# full physical device to concurrent nodes.
_VRAM_HEADROOM_GB = 2.0
_MIN_VRAM_GB = 8.0

# Fraction of system RAM handed to the scheduler as its in-flight payload budget, and a
# floor so tiny/undetectable machines still get a workable budget.
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
        updated["vram_gb"] = max(float(updated["vram_gb"]), detected)
    return updated
