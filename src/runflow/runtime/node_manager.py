from __future__ import annotations

from runflow.core.node import Node


class NodeManager:
    """Keeps track of loaded nodes and evicts exclusive resource users."""

    def __init__(self, context):
        self.context = context
        self.loaded: dict[str, Node] = {}

    def ensure_loaded(self, node: Node) -> None:
        if node.id in self.loaded:
            return

        self._evict_if_needed(node)
        node.setup(self.context)
        self.loaded[node.id] = node

    def unload(self, node: Node) -> None:
        if node.id not in self.loaded:
            return
        node.teardown(self.context)
        del self.loaded[node.id]

    def unload_all(self) -> None:
        for node in list(self.loaded.values()):
            self.unload(node)

    def _evict_if_needed(self, node: Node) -> None:
        exclusive_key = node.RESOURCE_POLICY.exclusive_key()
        if exclusive_key is None:
            return

        for loaded in list(self.loaded.values()):
            if loaded.RESOURCE_POLICY.exclusive_key() == exclusive_key and loaded.id != node.id:
                self.unload(loaded)
