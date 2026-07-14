from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pyarrow.parquet as pq

from runner.nodes.models import SpeakerClusterRunRef, SpeakerEmbeddingSetRef, stable_id
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import ExtraFileCreate, ExtraFilePathCreate
from shared.db.speakers import crud as speaker_crud
from shared.db.speakers.schemas import (
    ClusterSummaryCreate,
    ClusteringArtifactCreate,
    ClusteringArtifactRole,
    ClusteringRunCreate,
    ClusteringRunState,
    ClusteringOutcomeCounts,
    SpeakerClusterStatus,
)


def prepare_clustering_run(
    embedding_set: SpeakerEmbeddingSetRef,
    execution_run_id: str,
    node_id: str,
    index_factory: str,
    threshold_version: str,
    settings: dict[str, Any],
) -> tuple[UUID, SpeakerClusterRunRef | None]:
    run_key = stable_id(
        "speaker_cluster_run",
        execution_run_id,
        node_id,
        embedding_set.run_id,
    )
    with database_session() as session:
        run = speaker_crud.create_clustering_run(
            session,
            ClusteringRunCreate(
                run_key=run_key,
                embedding_run_id=embedding_set.run_id,
                expected_count=embedding_set.item_count,
                index_factory=index_factory,
                threshold_version=threshold_version,
                settings=settings,
            ),
        )
        if run.state == ClusteringRunState.COMPLETED.value:
            return run.id, _completed_ref(session, run.id)
        stale_ids = speaker_crud.clear_open_clustering_artifacts(session, run.id)
        for artifact_id in stale_ids:
            asset_crud.delete_extra_file(session, artifact_id)
        return run.id, None


def fail_clustering_run(run_id: UUID, error: BaseException) -> None:
    with database_session() as session:
        speaker_crud.fail_clustering_run(
            session, run_id, clustering_failure_details(error)
        )


def clustering_failure_details(error: BaseException) -> dict[str, object]:
    return {
        "error_type": type(error).__name__,
        "message": str(error),
        "cancelled": isinstance(error, asyncio.CancelledError),
    }


def persist_clustering_outputs(
    run_id: UUID,
    assignment_paths: list[Path],
    prototype_paths: list[Path],
    index_path: Path,
    item_count: int,
    outcome_counts: ClusteringOutcomeCounts,
) -> SpeakerClusterRunRef:
    with database_session() as session:
        try:
            assignments = _upload_registered_paths(
                session, run_id, assignment_paths, "speaker_assignment_shard",
                ClusteringArtifactRole.ASSIGNMENT,
            )
            prototypes = _upload_registered_paths(
                session, run_id, prototype_paths, "speaker_prototype_shard",
                ClusteringArtifactRole.PROTOTYPE,
            )
            index = _upload_and_register_path(
                session, run_id, index_path, "speaker_candidate_index",
                ClusteringArtifactRole.INDEX, 0, item_count,
            )
            manifest = _create_manifest(session, run_id, prototypes)
            _persist_summaries(session, run_id, prototype_paths)
            speaker_crud.complete_clustering_run(
                session, run_id, item_count, manifest.id, index.id, outcome_counts
            )
            run = speaker_crud.get_clustering_run(session, run_id)
            return SpeakerClusterRunRef(
                run_id=run_id,
                embedding_run_id=run.embedding_run_id,
                assignment_artifact_ids=[artifact.id for artifact in assignments],
                prototype_artifact_id=manifest.id,
                index_artifact_id=index.id,
            )
        except BaseException:
            _delete_registered_artifacts(session, run_id)
            raise


def _upload_registered_paths(
    session: Any, run_id: UUID, paths: list[Path], type_: str,
    role: ClusteringArtifactRole,
) -> list[Any]:
    return [
        _upload_and_register_path(
            session, run_id, path, type_, role, ordinal,
            pq.ParquetFile(path).metadata.num_rows,
        )
        for ordinal, path in enumerate(paths)
    ]


def _upload_and_register_path(
    session: Any, run_id: UUID, path: Path, type_: str,
    role: ClusteringArtifactRole | str, ordinal: int, row_count: int,
) -> Any:
    artifact = asset_crud.create_extra_file_from_path(
        session,
        ExtraFilePathCreate(
            name=path.name, path=path, type_=type_,
            metadata={"clustering_run_id": str(run_id)},
        ),
    )
    try:
        _register(session, run_id, artifact.id, role, ordinal, row_count)
        return artifact
    except BaseException:
        _clear_and_delete_artifacts(session, run_id, (artifact.id,))
        raise


def _create_manifest(session: Any, run_id: UUID, prototypes: list[Any]) -> Any:
    manifest = asset_crud.create_extra_file(
        session,
        ExtraFileCreate(
            name=f"speaker-prototypes-{run_id}.json",
            data=json.dumps(
                {
                    "run_id": str(run_id),
                    "artifact_ids": [str(artifact.id) for artifact in prototypes],
                },
                sort_keys=True,
            ).encode("utf-8"),
            type_="speaker_prototype_manifest",
            metadata={"clustering_run_id": str(run_id)},
        ),
    )
    try:
        _register(
            session, run_id, manifest.id, ClusteringArtifactRole.MANIFEST, 0, 1
        )
        return manifest
    except BaseException:
        _clear_and_delete_artifacts(session, run_id, (manifest.id,))
        raise


def _persist_summaries(session: Any, run_id: UUID, paths: list[Path]) -> None:
    speaker_crud.replace_cluster_summaries(session, run_id, [])
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches():
            values = batch.to_pydict()
            payloads = [
                ClusterSummaryCreate(
                    cluster_key=str(values["cluster_id"][index]),
                    member_count=values["member_count"][index],
                    duration_seconds=values["duration_seconds"][index],
                    dispersion=values["dispersion"][index],
                    status=(
                        SpeakerClusterStatus.SUSPICIOUS
                        if values["suspicious"][index]
                        else SpeakerClusterStatus.ACCEPTED
                    ),
                )
                for index in range(batch.num_rows)
            ]
            speaker_crud.append_cluster_summaries(session, run_id, payloads)


def _delete_registered_artifacts(session: Any, run_id: UUID) -> None:
    _clear_and_delete_artifacts(session, run_id, ())


def _clear_and_delete_artifacts(
    session: Any, run_id: UUID, additional_ids: tuple[UUID, ...]
) -> None:
    artifact_ids = speaker_crud.clear_open_clustering_artifacts(session, run_id)
    deletion_ids = [
        *artifact_ids,
        *(artifact_id for artifact_id in additional_ids if artifact_id not in artifact_ids),
    ]
    for artifact_id in deletion_ids:
        asset_crud.delete_extra_file(session, artifact_id)


def _register(
    session: Any,
    run_id: UUID,
    artifact_id: UUID,
    role: ClusteringArtifactRole | str,
    ordinal: int,
    row_count: int,
) -> None:
    speaker_crud.register_clustering_artifact(
        session,
        run_id,
        ClusteringArtifactCreate(
            artifact_id=artifact_id,
            role=role,
            ordinal=ordinal,
            row_count=row_count,
        ),
    )


def _completed_ref(session: Any, run_id: UUID) -> SpeakerClusterRunRef:
    run = speaker_crud.get_clustering_run(session, run_id)
    assignments = speaker_crud.list_clustering_artifacts(
        session, run_id, ClusteringArtifactRole.ASSIGNMENT
    )
    assert run.prototype_artifact_id is not None, (
        "completed run has no prototype manifest"
    )
    assert run.index_artifact_id is not None, "completed run has no candidate index"
    return SpeakerClusterRunRef(
        run_id=run.id,
        embedding_run_id=run.embedding_run_id,
        assignment_artifact_ids=[artifact.artifact_id for artifact in assignments],
        prototype_artifact_id=run.prototype_artifact_id,
        index_artifact_id=run.index_artifact_id,
    )
