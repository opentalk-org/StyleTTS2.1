from __future__ import annotations

import os

# Reserve a little head-room so the scheduler's vram budget never promises the
# full physical device to concurrent nodes.
_VRAM_HEADROOM_GB = 2.0
_MIN_VRAM_GB = 8.0

# Fraction of system RAM handed to the scheduler as its in-flight payload budget, and a
# floor so tiny/undetectable machines still get a workable budget.
_MEMORY_BUDGET_FRACTION = 0.5
_MIN_MEMORY_BUDGET_MB = 512.0


def detect_system_ram_mb() -> float | None:
    """Total system RAM in MB, or None when it can't be read."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):  # pragma: no cover - non-POSIX / unusual
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size / (1024 * 1024)


def default_memory_budget_mb() -> float:
    """A conservative default in-flight memory budget derived from detected RAM."""
    total = detect_system_ram_mb()
    if total is None:
        return _MIN_MEMORY_BUDGET_MB
    return max(_MIN_MEMORY_BUDGET_MB, round(total * _MEMORY_BUDGET_FRACTION))


def detect_vram_gb() -> float | None:
    """Return the accelerator's total VRAM in GB, or None when unavailable.

    The scheduler treats ``vram_gb`` as a capacity budget; the runtime default is
    a conservative 8 GB, which is smaller than what some nodes (e.g. StyleTTS2
    finetuning at 12 GB) request, so those nodes would wait forever on a machine
    whose budget was left at the default. Detecting the real device memory keeps
    the budget aligned with the hardware the runner is actually on."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_bytes / (1024 ** 3)
    return max(_MIN_VRAM_GB, round(total_gb - _VRAM_HEADROOM_GB, 1))


def apply_detected_resources(resources: dict[str, float]) -> dict[str, float]:
    """Return a copy of ``resources`` with the vram budget aligned to the device.

    Only raises the budget: if the caller already asked for more than the device
    reports (e.g. an explicit override) we keep their value."""
    updated = dict(resources)
    detected = detect_vram_gb()
    if detected is not None:
        updated["vram_gb"] = max(float(updated.get("vram_gb", 0.0)), detected)
    return updated
