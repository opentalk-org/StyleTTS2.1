from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

from pydantic import Field

from runner.nodes.assets.catalog_runtime.tasks import run_catalog_task
from runner.nodes.assets.checkpoints import resolve_checkpoint_ref
from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import CheckpointRefPort, JsonPort
from runner.nodes.models import CheckpointRef


class CatalogKey(str, Enum):
    STYLETTS2_UTILS = "styletts2_utils"
    OFFICIAL_CHECKPOINTS = "official_checkpoints"
    PAPERCUP_MULTILINGUAL_PL_BERT = "papercup_multilingual_pl_bert"
    VOKAN_CHECKPOINT = "vokan_checkpoint"
    ASR_MODELS = "asr_models"
    MOS_MODELS = "mos_models"
    TTS_MODELS = "tts_models"
    TURN_MODELS = "turn_models"


class CatalogDownloadSettings(StrictSettings):
    catalog_key: CatalogKey = Field(title="Catalog")
    item: str = Field(default="", title="Requested item")


class CatalogDownloadNode(Node):
    NODE_TYPE = "CatalogDownload"
    DESCRIPTION = "Download a named asset from a built-in catalog (StyleTTS2 utilities, official checkpoints, PL-BERT, ASR/MOS/TTS/turn models, etc.) so it is available locally. Pick a catalog and type the item you want; it emits the resolved catalog entry as JSON, plus a checkpoint reference when the item is a single checkpoint. Use it as a starting node to pull in pretrained weights and support assets before wiring them into downstream nodes."
    CATEGORY = "Assets"
    SETTINGS = CatalogDownloadSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {
        "checkpoint": CheckpointRefPort(optional=True),
        "catalog_item": JsonPort(),
    }
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params):
        super().__init__(node_id=node_id, **params)
        self._emitted = False

    def remaining_items(self, context):
        return 0 if self._emitted else 1

    async def execute(self, batch, context):
        if self._emitted:
            raise RuntimeError(f"catalog node already emitted: {self.id}")
        self._emitted = True
        if not self.settings.item.strip():
            raise ValueError("catalog_download_requires_item")
        self.logger.info(
            "catalog download requested catalog=%s item=%s",
            self.settings.catalog_key.value, self.settings.item,
        )
        result, checkpoint = await asyncio.to_thread(self._run_task)
        self.logger.info("catalog download resolved catalog=%s", self.settings.catalog_key.value)
        output: dict[str, Any] = {
            "catalog_item": {
                "catalog_key": self.settings.catalog_key.value,
                "requested_item": self.settings.item,
                "status": "resolved_catalog_assets",
                "result": result,
            }
        }
        if checkpoint is not None:
            output["checkpoint"] = checkpoint
        return [output]

    def _run_task(self) -> tuple[dict[str, Any], CheckpointRef | None]:
        result = run_catalog_task(self.settings.catalog_key.value, self.settings.item, logger=self.logger)
        return result, _single_checkpoint_ref(result)


def _single_checkpoint_ref(result: dict[str, Any]) -> CheckpointRef | None:
    ids = _collect_checkpoint_ids(result)
    if len(ids) > 1:
        raise RuntimeError(f"catalog_download_multiple_checkpoints:{len(ids)}")
    if not ids:
        return None
    return resolve_checkpoint_ref(ids[0])


def _collect_checkpoint_ids(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "checkpoint_id" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_collect_checkpoint_ids(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_collect_checkpoint_ids(value))
    return found
