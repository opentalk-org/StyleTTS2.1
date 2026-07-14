from __future__ import annotations

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
    ClusteringArtifactCreate,
    ClusteringArtifactRole,
    ClusteringRunCreate,
    ClusteringRunState,
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


def persist_clustering_outputs(
    run_id: UUID,
    assignment_paths: list[Path],
    prototype_paths: list[Path],
    index_path: Path,
    item_count: int,
) -> SpeakerClusterRunRef:
    with database_session() as session:
        assignments = _upload_paths(
            session,
            assignment_paths,
            "speaker_assignment_shard",
            {"clustering_run_id": str(run_id)},
        )
        prototypes = _upload_paths(
            session,
            prototype_paths,
            "speaker_prototype_shard",
            {"clustering_run_id": str(run_id)},
        )
        index = asset_crud.create_extra_file_from_path(
            session,
            ExtraFilePathCreate(
                name=f"speaker-candidates-{run_id}.faiss",
                path=index_path,
                type_="speaker_candidate_index",
                metadata={"clustering_run_id": str(run_id)},
            ),
        )
        for ordinal, (artifact, path) in enumerate(
            zip(assignments, assignment_paths, strict=True)
        ):
            _register(
                session,
                run_id,
                artifact.id,
                ClusteringArtifactRole.ASSIGNMENT,
                ordinal,
                pq.ParquetFile(path).metadata.num_rows,
            )
        for ordinal, (artifact, path) in enumerate(
            zip(prototypes, prototype_paths, strict=True)
        ):
            _register(
                session,
                run_id,
                artifact.id,
                ClusteringArtifactRole.PROTOTYPE,
                ordinal,
                pq.ParquetFile(path).metadata.num_rows,
            )
        _register(
            session,
            run_id,
            index.id,
            ClusteringArtifactRole.INDEX,
            0,
            item_count,
        )
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
        speaker_crud.complete_clustering_run(
            session,
            run_id,
            item_count,
            manifest.id,
            index.id,
        )
        return SpeakerClusterRunRef(
            run_id=run_id,
            embedding_run_id=speaker_crud.get_clustering_run(
                session, run_id
            ).embedding_run_id,
            assignment_artifact_ids=[artifact.id for artifact in assignments],
            prototype_artifact_id=manifest.id,
            index_artifact_id=index.id,
        )


def _upload_paths(
    session: Any,
    paths: list[Path],
    type_: str,
    metadata: dict[str, Any],
) -> list[Any]:
    return asset_crud.bulk_create_extra_files_from_paths(
        session,
        [
            ExtraFilePathCreate(
                name=path.name,
                path=path,
                type_=type_,
                metadata=metadata,
            )
            for path in paths
        ],
    )


def _register(
    session: Any,
    run_id: UUID,
    artifact_id: UUID,
    role: ClusteringArtifactRole,
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
