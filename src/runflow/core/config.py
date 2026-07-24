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
    memory_budget_mb: float | None = Field(default=None, ge=1)


def runtime_config_defaults() -> dict[str, Any]:
    return settings_defaults(RuntimeConfig)
