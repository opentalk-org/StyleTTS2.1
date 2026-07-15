from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from runflow.core.settings import settings_defaults


class RuntimeConfig(BaseModel):
    """Typed runtime controls shared by runner APIs and the scheduler."""

    model_config = ConfigDict(extra="forbid")

    resources: dict[str, float] = Field(
        default_factory=lambda: {
            "io": 2,
            "cpu_workers": 2,
            "accelerator": 1,
            "vram_gb": 8,
        }
    )
    # Total in-flight host-memory budget (MB) the scheduler spreads across node queues,
    # bounding resident payload bytes instead of item counts. None → the runner fills a
    # value derived from detected system RAM; the scheduler falls back to a safe default
    # if still unset. This replaces per-node queue_max_size tuning for memory safety.
    memory_budget_mb: float | None = Field(default=None, ge=1)


def runtime_config_defaults() -> dict[str, Any]:
    return settings_defaults(RuntimeConfig)
