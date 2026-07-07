from __future__ import annotations

from shared.logging_setup import get_logger

logger = get_logger(__name__)

# Reserve a little head-room so the scheduler's vram budget never promises the
# full physical device to concurrent nodes.
_VRAM_HEADROOM_GB = 2.0
_MIN_VRAM_GB = 8.0


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
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
    except Exception:  # pragma: no cover - driver/device edge cases
        logger.exception("failed to read cuda device memory")
        return None
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
