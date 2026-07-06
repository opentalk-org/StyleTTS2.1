from __future__ import annotations

from typing import Type

from runflow.core.node import Node


class NodeRegistry:
    def __init__(self) -> None:
        self.nodes: dict[str, Type[Node]] = {}

    def register(self, node_cls: Type[Node]) -> Type[Node]:
        self.nodes[node_cls.NODE_TYPE] = node_cls
        return node_cls

    def create(self, node_type: str, node_id: str, params: dict | None = None) -> Node:
        if node_type not in self.nodes:
            raise KeyError(f"Unknown node type: {node_type}")
        return self.nodes[node_type](node_id=node_id, **(params or {}))

    def to_schema(self) -> dict:
        return {node_type: node_cls.to_schema() for node_type, node_cls in self.nodes.items()}
