from __future__ import annotations

from runflow.core.config import RuntimeConfig, runtime_config_defaults
from runflow.registry.node_registry import NodeRegistry
from runflow.registry.type_registry import TypeRegistry


def export_ui_schema(node_registry: NodeRegistry, type_registry: TypeRegistry) -> dict:
    return {
        "types": type_registry.to_schema(),
        "nodes": node_registry.to_schema(),
        "runtime_config": RuntimeConfig.model_json_schema(),
        "runtime_config_defaults": runtime_config_defaults(),
    }
