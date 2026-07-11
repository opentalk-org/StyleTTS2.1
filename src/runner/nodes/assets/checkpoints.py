from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import CheckpointRefPort
from runner.nodes.models import CheckpointRef, stable_id
from shared.db import database_session
from shared.db.assets import crud as asset_crud


# Sentinel selected in the base-checkpoint dropdown to train from random init
# instead of resuming from a stored checkpoint. It is not a real UUID, so it must
# never reach ``resolve_checkpoint_ref``; ``SelectCheckpoint`` maps it to
# ``scratch_checkpoint_ref`` before any DB lookup.
SCRATCH_CHECKPOINT_ID = "__scratch__"


def is_scratch_checkpoint(ref: CheckpointRef) -> bool:
    return bool(ref.metadata.get("scratch"))


def scratch_checkpoint_ref() -> CheckpointRef:
    """A checkpoint reference carrying no pretrained weights.

    StyleTTS from-scratch training still relies on the auxiliary ASR/F0/PL-BERT
    models supplied through the asset bundle; only the main StyleTTS modules start
    from random init. The all-zero id keeps the ref hashable and stable while
    ``is_scratch_checkpoint`` gates every path that would dereference weights."""
    ref_id = stable_id("checkpoint", "scratch")
    return CheckpointRef(
        checkpoint_id=UUID(int=0),
        name="From scratch",
        path=Path(),
        id=ref_id,
        lineage_id=ref_id,
        metadata={"scratch": True, "type": "styletts2"},
    )


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


class ResolveCheckpointNode(Node):
    NODE_TYPE = "ResolveCheckpoint"
    DESCRIPTION = "Look up a stored checkpoint by its ID and emit a checkpoint reference (name, path, and metadata) that downstream nodes can load. Optionally set an expected type to guard against wiring in the wrong kind of checkpoint. Use it as a source node whenever a later node needs a specific saved model checkpoint."
    CATEGORY = "Assets"
    SETTINGS = ResolveCheckpointSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"checkpoint": CheckpointRefPort()}
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
                "checkpoint": resolve_checkpoint_ref(
                    self.settings.checkpoint_id,
                    self.settings.expected_type,
                )
            }
        ]
