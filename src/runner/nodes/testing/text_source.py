from __future__ import annotations

from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import TextPort


class TestingTextSourceSettings(StrictSettings):
    texts: list[str] = Field(default_factory=list, title="Texts")
    batch_size: int = Field(default=64, ge=1, le=1024, title="Batch size")


class TestingTextSourceNode(Node):
    NODE_TYPE = "TestingTextSource"
    DESCRIPTION = "Emit literal typed text items for exercising registered graph nodes without an external text-generation service."
    CATEGORY = "Testing"
    SETTINGS = TestingTextSourceSettings
    IS_INPUT = True
    INPUTS: dict[str, Any] = {}
    OUTPUTS = {"text": TextPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._cursor = 0

    def remaining_items(self, context) -> int:
        return len(self.settings.texts) - self._cursor

    async def execute(self, batch, context):
        end = min(self._cursor + self.settings.batch_size, len(self.settings.texts))
        selected = self.settings.texts[self._cursor:end]
        self._cursor = end
        return [{"text": text} for text in selected]
