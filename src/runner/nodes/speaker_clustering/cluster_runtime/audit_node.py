from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runflow.runtime.output_router import INPUT_INDEX_OUTPUT
from runner.nodes.datatypes import SpeakerAuditRefPort, SpeakerClusterRunRefPort
from runner.nodes.models import SpeakerAuditRef, SpeakerClusterRunRef
from runner.nodes.speaker_clustering.audit_report import (
    AssignmentAuditBuildResult,
    AssignmentAuditDocument,
    build_assignment_audit_report,
)
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import ExtraFilePathCreate
from shared.db.speakers import crud as speaker_crud
from shared.db.speakers.schemas import (
    ClusteringArtifactRole,
    SpeakerAuditCreate,
    SpeakerAuditMetricsRecord,
    SpeakerAuditState,
)


class AuditSpeakerClustersSettings(StrictSettings):
    seed: int = 7
    batch_rows: int = Field(default=100_000, gt=0)
    category_limit: int = Field(default=50, gt=0, le=1000)


@dataclass(frozen=True)
class AuditProgressReporter:
    loop: asyncio.AbstractEventLoop
    context: Any
    node_id: str

    def report(self, processed: int, total: int) -> None:
        message = f"scanned {processed}/{total} assignment rows"
        future = asyncio.run_coroutine_threadsafe(
            self.context.report_progress(self.node_id, processed, total, message),
            self.loop,
        )
        future.result()


class AuditSpeakerClustersNode(Node):
    NODE_TYPE = "AuditSpeakerClusters"
    DESCRIPTION = "Persist quantitative false-merge checks and deterministic listening evidence."
    CATEGORY = "Speaker Clustering"
    SETTINGS = AuditSpeakerClustersSettings
    INPUTS = {"cluster_run": SpeakerClusterRunRefPort()}
    OUTPUTS = {"audit": SpeakerAuditRefPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=False)
    QUEUE_MAX_SIZE = 1

    async def execute(
        self, batch: list[dict[str, Any]], context: Any
    ) -> list[dict[str, Any]]:
        if len(batch) != 1:
            raise ValueError(f"{self.id} requires exactly one completed cluster run")
        cluster_run = batch[0]["cluster_run"]
        assert isinstance(cluster_run, SpeakerClusterRunRef)
        completed = _prepare_audit(cluster_run, self.settings.seed)
        if completed is not None:
            return [{"audit": completed, INPUT_INDEX_OUTPUT: 0}]
        paths = _assignment_paths(cluster_run)
        reporter = AuditProgressReporter(asyncio.get_running_loop(), context, self.id)
        result = await asyncio.to_thread(
            build_assignment_audit_report,
            paths,
            context.node_dir(self.id) / f"audit-{cluster_run.run_id}",
            self.settings.batch_rows,
            self.settings.category_limit,
            context.check_cancel,
            reporter.report,
        )
        context.check_cancel()
        audit = _persist_audit(cluster_run, self.settings.seed, result, context.check_cancel)
        return [{"audit": audit, INPUT_INDEX_OUTPUT: 0}]


def _prepare_audit(cluster_run: SpeakerClusterRunRef, seed: int) -> SpeakerAuditRef | None:
    with database_session() as session:
        audit = speaker_crud.create_audit(
            session,
            SpeakerAuditCreate(cluster_run_id=cluster_run.run_id, seed=seed),
        )
        if audit.state != SpeakerAuditState.COMPLETED.value:
            return None
        assert audit.report_artifact_id is not None
        assert audit.listening_artifact_id is not None
        return SpeakerAuditRef(
            audit.id,
            audit.cluster_run_id,
            audit.report_artifact_id,
            audit.listening_artifact_id,
        )


def _assignment_paths(cluster_run: SpeakerClusterRunRef) -> list[Path]:
    with database_session() as session:
        durable = speaker_crud.list_clustering_artifacts(
            session,
            cluster_run.run_id,
            ClusteringArtifactRole.ASSIGNMENT,
        )
        artifact_ids = [artifact.artifact_id for artifact in durable]
        if artifact_ids != cluster_run.assignment_artifact_ids:
            raise ValueError("cluster run assignment manifest does not match durable artifacts")
        return [asset_crud.get_extra_file_path(session, value) for value in artifact_ids]


def _persist_audit(
    cluster_run: SpeakerClusterRunRef,
    seed: int,
    result: AssignmentAuditBuildResult,
    check_cancel: Any,
) -> SpeakerAuditRef:
    archive_path = result.json_report_path.parent / "audit-report.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        for path in (result.json_report_path, result.html_report_path):
            check_cancel()
            archive.write(path, arcname=path.name)
    document = AssignmentAuditDocument.model_validate_json(
        result.json_report_path.read_bytes()
    )
    created_ids = []
    with database_session() as session:
        audit = speaker_crud.create_audit(
            session,
            SpeakerAuditCreate(cluster_run_id=cluster_run.run_id, seed=seed),
        )
        try:
            report = _upload_audit_file(session, audit.id, archive_path, "speaker_audit_report")
            created_ids.append(report.id)
            check_cancel()
            listening = _upload_audit_file(
                session,
                audit.id,
                result.listening_manifest_path,
                "speaker_audit_listening_manifest",
            )
            created_ids.append(listening.id)
            completed = speaker_crud.complete_audit(
                session,
                audit.id,
                report.id,
                listening.id,
                SpeakerAuditMetricsRecord.model_validate(document.metrics.model_dump()),
            )
        except BaseException:
            session.rollback()
            for artifact_id in created_ids:
                asset_crud.delete_extra_file(session, artifact_id)
            raise
    assert completed.report_artifact_id is not None
    assert completed.listening_artifact_id is not None
    return SpeakerAuditRef(
        completed.id,
        completed.cluster_run_id,
        completed.report_artifact_id,
        completed.listening_artifact_id,
    )


def _upload_audit_file(session: Any, audit_id: Any, path: Path, type_: str) -> Any:
    return asset_crud.create_extra_file_from_path(
        session,
        ExtraFilePathCreate(
            name=path.name,
            path=path,
            type_=type_,
            metadata={"speaker_audit_id": str(audit_id)},
        ),
    )
