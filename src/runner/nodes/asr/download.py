from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.assets.model_downloads import (
    download_nemo_snapshot,
    download_whisper_model_files,
    ensure_model_checkpoint,
)
from runner.nodes.datatypes import CHECKPOINT_REF
from shared.log_streams import route_output_to_logger


class WhisperModel(str, Enum):
    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    LARGE = "large"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    TURBO = "turbo"


class ParakeetModel(str, Enum):
    TDT_0_6B_V2 = "nvidia/parakeet-tdt-0.6b-v2"
    TDT_0_6B_V3 = "nvidia/parakeet-tdt-0.6b-v3"
    TDT_1_1B = "nvidia/parakeet-tdt-1.1b"
    RNNT_1_1B = "nvidia/parakeet-rnnt-1.1b"
    CTC_1_1B = "nvidia/parakeet-ctc-1.1b"


class CanaryModel(str, Enum):
    CANARY_1B_V2 = "nvidia/canary-1b-v2"
    CANARY_1B_FLASH = "nvidia/canary-1b-flash"
    CANARY_1B = "nvidia/canary-1b"
    CANARY_180M_FLASH = "nvidia/canary-180m-flash"


class ModelDownloadSettings(StrictSettings):
    kind: Literal["whisper", "parakeet", "canary", "sortformer"] = Field(default="canary", title="Model kind")
    model: str = Field(default=CanaryModel.CANARY_1B_V2.value, title="Model")


class ModelDownloadNode(Node):
    """Download an ASR model, register it as a checkpoint, and emit its CheckpointRef once."""

    NODE_TYPE = "ModelDownload"
    CATEGORY = "Assets / Models"
    SETTINGS = ModelDownloadSettings
    IS_INPUT = True
    INPUTS: dict[str, Port] = {}
    OUTPUTS = {"checkpoint_ref": Port("checkpoint_ref", CHECKPOINT_REF)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._emitted = False

    def remaining_items(self, context):
        return 0 if self._emitted else 1

    async def execute(self, batch, context):
        assert not self._emitted, f"model download node already emitted: {self.id}"
        self._emitted = True
        ref = await asyncio.to_thread(self._resolve_checkpoint)
        return [{"checkpoint_ref": ref}]

    def _resolve_checkpoint(self) -> Any:
        kind = self.settings.kind
        model_id = self.settings.model
        with route_output_to_logger(self.logger):
            if kind == "whisper":
                return ensure_model_checkpoint(kind, model_id, lambda folder: download_whisper_model_files(model_id, folder))
            return ensure_model_checkpoint(kind, model_id, lambda folder: download_nemo_snapshot(model_id, folder))
