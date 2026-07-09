import logging
from abc import ABC, abstractmethod
from typing import Any

from runflow.core.node_runtime import NodeRuntimeConfig, node_runtime_defaults
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings, settings_defaults
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy


def _runtime_defaults_for(cls: type["Node"]) -> dict[str, Any]:
    defaults = node_runtime_defaults(cls.BATCH_POLICY, cls.RESOURCE_POLICY)
    defaults["queue_max_size"] = cls.QUEUE_MAX_SIZE
    return defaults


def _merge_runtime_defaults(defaults: dict[str, Any], overrides: Any) -> dict[str, Any]:
    if overrides is None:
        return defaults
    merged = dict(defaults)
    for key, value in dict(overrides).items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = {**current, **value}
            continue
        merged[key] = value
    return merged


class Node(ABC):
    """Base class for all graph nodes.

    Subclasses declare typed INPUTS and OUTPUTS, then implement execute. The
    batch argument always contains one or more task input dictionaries.
    """

    NODE_TYPE: str = "BaseNode"
    CATEGORY: str = "Core"
    INPUTS: dict[str, Port] = {}
    OUTPUTS: dict[str, Port] = {}
    SETTINGS: type[StrictSettings] = StrictSettings
    IS_INPUT: bool = False

    BATCH_POLICY: BatchPolicy = BatchPolicy()
    RESOURCE_POLICY: ResourcePolicy = ResourcePolicy()
    QUEUE_MAX_SIZE: int = 64

    def __init__(self, node_id: str | None = None, **params: Any):
        runtime_config = params.pop("runtime", None)
        self.id = node_id or params.pop("id", self.NODE_TYPE)
        self.logger = logging.getLogger(f"runflow.node.{self.NODE_TYPE}.{self.id}")
        self.settings = self.SETTINGS(**params)
        self.params = self.settings.model_dump()
        default_runtime = _runtime_defaults_for(type(self))
        self.runtime = NodeRuntimeConfig.model_validate(_merge_runtime_defaults(default_runtime, runtime_config))

    async def setup(self, context: Any) -> None:
        """Load resources. Called by NodeManager."""
        return None

    async def teardown(self, context: Any) -> None:
        """Release resources. Called by NodeManager."""
        return None

    def remaining_items(self, context: Any) -> int | None:
        return None

    @abstractmethod
    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        return {
            "type": cls.NODE_TYPE,
            "category": cls.CATEGORY,
            "is_input": cls.IS_INPUT,
            "inputs": {name: port.to_schema(name) for name, port in cls.INPUTS.items()},
            "outputs": {name: port.to_schema(name) for name, port in cls.OUTPUTS.items()},
            "settings": cls.SETTINGS.model_json_schema(),
            "settings_defaults": settings_defaults(cls.SETTINGS),
            "runtime": NodeRuntimeConfig.model_json_schema(),
            "runtime_defaults": _runtime_defaults_for(cls),
        }
