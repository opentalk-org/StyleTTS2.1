from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

class BatchMode(str, Enum):
    DISABLED = "disabled"
    MICRO_BATCH = "micro_batch"
    FULL_WINDOW = "full_window"


@dataclass(frozen=True)
class BatchPolicy:
    mode: BatchMode = BatchMode.DISABLED
    preferred_size: int = 1
    max_size: int = 1
    timeout_ms: int = 0
    sort_by: str | None = None

@dataclass(frozen=True)
class ResourcePolicy:
    resources: dict[str, float] = field(default_factory=lambda: {"cpu_workers": 1})
    keep_loaded: bool = True
    exclusive_group: str | None = None
    estimated_vram_gb: float | None = None
    unload_after_stage: bool = False

    def requirements(self) -> dict[str, float]:
        return dict(self.resources)

    def exclusive_key(self) -> str | None:
        return self.exclusive_group
