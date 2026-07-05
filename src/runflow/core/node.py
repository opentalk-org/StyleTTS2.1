from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from runflow.core.ports import Port
from runflow.core.settings import NodeSettings, settings_defaults
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy


class Node(ABC):
    """Base class for all graph nodes.

    Subclasses declare typed INPUTS and OUTPUTS, then implement execute. The
    batch argument always contains one or more task input dictionaries.
    """

    NODE_TYPE: str = "BaseNode"
    CATEGORY: str = "Core"
    INPUTS: dict[str, Port] = {}
    OUTPUTS: dict[str, Port] = {}
    SETTINGS: type[NodeSettings] = NodeSettings

    BATCH_POLICY: BatchPolicy = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY: ResourcePolicy = ResourcePolicy()

    def __init__(self, node_id: str | None = None, **params: Any):
        self.id = node_id or params.pop("id", self.NODE_TYPE)
        self.settings = self.SETTINGS(**params)
        self.params = self.settings.model_dump()

    async def setup(self, context: Any) -> None:
        """Load resources. Called by NodeManager."""
        return None

    async def teardown(self, context: Any) -> None:
        """Release resources. Called by NodeManager."""
        return None

    @abstractmethod
    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        return {
            "type": cls.NODE_TYPE,
            "category": cls.CATEGORY,
            "inputs": {name: port.to_schema() for name, port in cls.INPUTS.items()},
            "outputs": {name: port.to_schema() for name, port in cls.OUTPUTS.items()},
            "settings": cls.SETTINGS.model_json_schema(),
            "settings_defaults": settings_defaults(cls.SETTINGS),
            "batch_policy": {
                "mode": cls.BATCH_POLICY.mode.value,
                "preferred_size": cls.BATCH_POLICY.preferred_size,
                "max_size": cls.BATCH_POLICY.max_size,
                "timeout_ms": cls.BATCH_POLICY.timeout_ms,
                "group_by": list(cls.BATCH_POLICY.group_by),
                "sort_by": cls.BATCH_POLICY.sort_by,
                "pad_to_multiple_of": cls.BATCH_POLICY.pad_to_multiple_of,
                "drop_last": cls.BATCH_POLICY.drop_last,
            },
            "resource_policy": {
                "resources": cls.RESOURCE_POLICY.requirements(),
                "keep_loaded": cls.RESOURCE_POLICY.keep_loaded,
                "exclusive_group": cls.RESOURCE_POLICY.exclusive_group,
                "estimated_vram_gb": cls.RESOURCE_POLICY.estimated_vram_gb,
                "unload_after_stage": cls.RESOURCE_POLICY.unload_after_stage,
            },
        }
