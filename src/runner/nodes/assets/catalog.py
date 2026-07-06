from __future__ import annotations

import asyncio
from enum import Enum

from pydantic import Field

from runner.nodes.assets.catalog_runtime.tasks import run_catalog_task
from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import JSON


class CatalogKey(str, Enum):
    STYLETTS2_UTILS = "styletts2_utils"
    OFFICIAL_CHECKPOINTS = "official_checkpoints"
    PAPERCUP_MULTILINGUAL_PL_BERT = "papercup_multilingual_pl_bert"
    VOKAN_CHECKPOINT = "vokan_checkpoint"


class CatalogDownloadSettings(StrictSettings):
    catalog_key: CatalogKey = Field(title="Catalog")
    item: str = Field(default="", title="Requested item")


class CatalogDownloadNode(Node):
    NODE_TYPE = "CatalogDownload"
    CATEGORY = "Assets / Catalog"
    SETTINGS = CatalogDownloadSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"catalog_item": Port("catalog_item", JSON)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params):
        super().__init__(node_id=node_id, **params)
        self._emitted = False

    def remaining_items(self, context):
        return 0 if self._emitted else 1

    async def execute(self, batch, context):
        assert not self._emitted, f"catalog node already emitted: {self.id}"
        self._emitted = True
        result = await asyncio.to_thread(run_catalog_task, self.settings.catalog_key.value)
        return [
            {
                "catalog_item": {
                    "catalog_key": self.settings.catalog_key.value,
                    "requested_item": self.settings.item,
                    "status": "resolved_catalog_assets",
                    "result": result,
                }
            }
        ]
