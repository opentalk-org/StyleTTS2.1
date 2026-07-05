import asyncio

from runflow.core.node import Node


class NodeManager:
    """Loads a node once, guarded by per-node locks.

    Resource conflicts are handled by ResourcePool, not by hard-coded CPU/GPU
    node types. This manager only controls setup/teardown lifecycle.
    """

    def __init__(self, context):
        self.context = context
        self.loaded: dict[str, Node] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, node: Node) -> asyncio.Lock:
        if node.id not in self._locks:
            self._locks[node.id] = asyncio.Lock()
        return self._locks[node.id]

    async def ensure_loaded(self, node: Node) -> None:
        if node.id in self.loaded:
            return
        async with self._lock_for(node):
            if node.id in self.loaded:
                return
            node.logger.info("loading node")
            await node.setup(self.context)
            self.loaded[node.id] = node
            node.logger.info("node loaded")

    async def unload(self, node: Node) -> None:
        if node.id not in self.loaded:
            return
        async with self._lock_for(node):
            if node.id not in self.loaded:
                return
            node.logger.info("unloading node")
            await node.teardown(self.context)
            del self.loaded[node.id]
            node.logger.info("node unloaded")

    async def unload_all(self) -> None:
        for node in list(self.loaded.values()):
            await self.unload(node)
