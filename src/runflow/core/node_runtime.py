from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy


class BatchPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    mode: BatchMode = BatchMode.MICRO_BATCH
    preferred_size: int = Field(default=64, ge=1)
    max_size: int = Field(default=64, ge=1)
    timeout_ms: int = Field(default=0, ge=0)
    sort_by: str | None = None

    def to_policy(self) -> BatchPolicy:
        return BatchPolicy(
            mode=BatchMode(self.mode),
            preferred_size=self.preferred_size,
            max_size=self.max_size,
            timeout_ms=self.timeout_ms,
            sort_by=self.sort_by,
        )


class ResourcePolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resources: dict[str, float] = Field(default_factory=lambda: {"cpu_workers": 1})
    keep_loaded: bool = True
    exclusive_group: str | None = None
    estimated_vram_gb: float | None = None
    unload_after_stage: bool = False

    def to_policy(self) -> ResourcePolicy:
        return ResourcePolicy(
            resources=self.resources,
            keep_loaded=self.keep_loaded,
            exclusive_group=self.exclusive_group,
            estimated_vram_gb=self.estimated_vram_gb,
            unload_after_stage=self.unload_after_stage,
        )


class NodeRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue_max_size: int = Field(default=64, ge=1)
    batch_policy: BatchPolicyConfig = Field(default_factory=BatchPolicyConfig)
    resource_policy: ResourcePolicyConfig = Field(default_factory=ResourcePolicyConfig)


def node_runtime_defaults(batch_policy: BatchPolicy, resource_policy: ResourcePolicy) -> dict[str, Any]:
    config = NodeRuntimeConfig(
        batch_policy=BatchPolicyConfig(
            mode=batch_policy.mode,
            preferred_size=batch_policy.preferred_size,
            max_size=batch_policy.max_size,
            timeout_ms=batch_policy.timeout_ms,
            sort_by=batch_policy.sort_by,
        ),
        resource_policy=ResourcePolicyConfig(
            resources=resource_policy.requirements(),
            keep_loaded=resource_policy.keep_loaded,
            exclusive_group=resource_policy.exclusive_group,
            estimated_vram_gb=resource_policy.estimated_vram_gb,
            unload_after_stage=resource_policy.unload_after_stage,
        ),
    )
    return config.model_dump(mode="json")
