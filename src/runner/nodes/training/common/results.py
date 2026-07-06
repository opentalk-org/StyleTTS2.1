from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.nodes.models import CheckpointRef, TrainingManifest, TrainingResult, stable_id
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import CheckpointCreate
from shared.db.connection import database_session


class NoopAimRun:
    def track(self, *args, **kwargs) -> None:
        return None


def publish_training_result(
    node_type: str,
    display_name: str,
    checkpoint_type: str,
    output_checkpoint_dir: str,
    metadata: dict[str, Any],
    run_id: str,
) -> TrainingResult:
    if not output_checkpoint_dir:
        raise RuntimeError(f"{node_type} requires training output folder")
    folder_path = Path(output_checkpoint_dir)
    if not folder_path.is_dir():
        raise RuntimeError(f"{node_type} output_checkpoint_dir must be an existing folder: {folder_path}")

    with database_session() as session:
        checkpoint = asset_crud.create_checkpoint(
            session,
            CheckpointCreate(
                name=display_name,
                folder_path=folder_path,
                type_=checkpoint_type,
                metadata=metadata,
                job_id=run_id,
            ),
        )
        canonical_path = asset_crud.get_checkpoint_path(session, checkpoint.id)

    checkpoint_ref_id = stable_id("checkpoint", checkpoint.id, checkpoint.content_hash)
    checkpoint_ref = CheckpointRef(
        checkpoint_id=checkpoint.id,
        name=checkpoint.name,
        path=canonical_path,
        id=checkpoint_ref_id,
        lineage_id=checkpoint_ref_id,
        metadata={
            "type": checkpoint.type_,
            "size": checkpoint.size,
            "content_hash": checkpoint.content_hash,
            "job_id": checkpoint.job_id,
            "metadata": checkpoint.metadata_,
        },
    )
    result_id = stable_id("training_result", node_type, run_id, checkpoint.id)
    return TrainingResult(
        training_run_id=stable_id("training_run", node_type, run_id, checkpoint.id),
        checkpoint=checkpoint_ref,
        id=result_id,
        lineage_id=result_id,
        metadata={"node_type": node_type, "checkpoint_type": checkpoint_type, "checkpoint_id": str(checkpoint.id)},
    )


def training_output_dir(configured: str, manifest: TrainingManifest, name: str) -> Path:
    if configured:
        output_dir = Path(configured)
    else:
        output_dir = Path(str(manifest.metadata["train_manifest_path"])).parent.parent / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def checkpoint_weight(ref: CheckpointRef) -> str | None:
    weights = sorted(ref.path.rglob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not weights:
        return None
    return str(weights[0])


def training_manifest_metadata(inputs: dict[str, Any]) -> dict[str, Any]:
    manifest: TrainingManifest = inputs["training_manifest"]
    return {"training_manifest": {"id": manifest.id, "dataset_id": str(manifest.dataset_id), "metadata": manifest.metadata}}
