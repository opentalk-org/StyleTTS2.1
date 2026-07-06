from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import CHECKPOINT_REF
from runner.nodes.models import CheckpointRef, stable_id
from shared.db import database_session
from shared.db.assets import crud as asset_crud


class ResolveCheckpointSettings(StrictSettings):
    checkpoint_id: str = Field(title="Checkpoint")
    expected_type: str = Field(default="", title="Expected type")


def resolve_checkpoint_ref(checkpoint_id: str, expected_type: str = "") -> CheckpointRef:
    parsed_id = UUID(checkpoint_id)
    with database_session() as session:
        checkpoint = asset_crud.get_checkpoint(session, parsed_id)
        if expected_type and checkpoint.type_ != expected_type:
            raise ValueError(f"checkpoint {parsed_id} has type {checkpoint.type_}, expected {expected_type}")
        path = asset_crud.get_checkpoint_path(session, parsed_id)

    ref_id = stable_id("checkpoint", parsed_id, checkpoint.content_hash)
    return CheckpointRef(
        checkpoint_id=parsed_id,
        name=checkpoint.name,
        path=path,
        id=ref_id,
        lineage_id=ref_id,
        metadata={
            "type": checkpoint.type_,
            "size": checkpoint.size,
            "content_hash": checkpoint.content_hash,
            "job_id": checkpoint.job_id,
            "metadata": checkpoint.metadata_,
        },
    )


def checkpoint_ref_or_stub(node_type: str, run: Any, checkpoint_id: str) -> CheckpointRef | dict[str, Any]:
    if _is_uuid(checkpoint_id):
        return resolve_checkpoint_ref(checkpoint_id)
    return {"node_type": node_type, "run": run, "source": "workflow_settings"}


def prefetch_checkpoint_ref(value: CheckpointRef | dict[str, Any]) -> CheckpointRef | dict[str, Any]:
    if isinstance(value, CheckpointRef):
        return value
    return {"source": value, "cache": "asset"}


def _is_uuid(value: str) -> bool:
    if not value:
        return False
    try:
        UUID(value)
    except ValueError:
        return False
    return True


class ResolveCheckpointNode(Node):
    NODE_TYPE = "ResolveCheckpoint"
    CATEGORY = "Assets / Checkpoints"
    SETTINGS = ResolveCheckpointSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"checkpoint_ref": Port("checkpoint_ref", CHECKPOINT_REF)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params):
        super().__init__(node_id=node_id, **params)
        self._emitted = False

    def remaining_items(self, context):
        return 0 if self._emitted else 1

    async def execute(self, batch, context):
        assert not self._emitted, f"checkpoint node already emitted: {self.id}"
        self._emitted = True
        return [
            {
                "checkpoint_ref": resolve_checkpoint_ref(
                    self.settings.checkpoint_id,
                    self.settings.expected_type,
                )
            }
        ]
