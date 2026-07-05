from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from runflow.core.settings import settings_defaults


class RuntimeWindowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_items: int = 80
    max_cost: float | None = None


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
    queue_max_size: int = 128
    window: RuntimeWindowConfig = Field(default_factory=RuntimeWindowConfig)


def runtime_config_defaults() -> dict[str, Any]:
    return settings_defaults(RuntimeConfig)
