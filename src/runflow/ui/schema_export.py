from __future__ import annotations

from runflow.registry.node_registry import NodeRegistry
from runflow.registry.type_registry import TypeRegistry


def export_ui_schema(node_registry: NodeRegistry, type_registry: TypeRegistry) -> dict:
    return {
        "types": type_registry.to_schema(),
        "nodes": node_registry.to_schema(),
    }
