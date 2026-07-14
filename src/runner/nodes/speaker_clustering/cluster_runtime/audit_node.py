from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runflow.runtime.output_router import INPUT_INDEX_OUTPUT
from runner.nodes.datatypes import SpeakerAuditRefPort, SpeakerClusterRunRefPort
from runner.nodes.models import SpeakerAuditRef, SpeakerClusterRunRef
from runner.nodes.speaker_clustering.audit_report import (
    AssignmentAuditDocument,
    ListeningEntry,
    build_assignment_audit,
)
from runner.nodes.speaker_clustering.audit_report.review import speaker_review_payload
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.audio import crud as audio_crud
from shared.db.reviews import crud as review_crud
from shared.db.reviews.schemas import (
    AudioSegmentReviewMedia,
    ReviewContinuation,
    ReviewCreate,
)
from shared.db.speakers import crud as speaker_crud
from shared.db.speakers.schemas import (
    ClusteringArtifactRole,
    ClusteringOutcomeCounts,
    SpeakerAuditCreate,
    SpeakerAuditMetricsRecord,
    SpeakerAuditState,
)
from shared.schemas import GraphEdgeRequest, GraphNodeRequest, InlineGraphRunRequest


class AuditSpeakerClustersSettings(StrictSettings):
    batch_rows: int = Field(default=100_000, gt=0)
    category_limit: int = Field(default=50, gt=0, le=1000)


@dataclass(frozen=True)
class AuditProgressReporter:
    loop: asyncio.AbstractEventLoop
    context: Any
    node_id: str

    def report(self, processed: int, total: int) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self.context.report_progress(
                self.node_id,
                processed,
                total,
                f"scanned {processed}/{total} assignment rows",
            ),
            self.loop,
        )
        future.result()


class AuditSpeakerClustersNode(Node):
    NODE_TYPE = "AuditSpeakerClusters"
    DESCRIPTION = "Publish quantitative cluster checks and bounded audio samples for review."
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
        completed = _prepare_audit(cluster_run)
        if completed is not None:
            return [{"audit": completed, INPUT_INDEX_OUTPUT: 0}]
        reporter = AuditProgressReporter(asyncio.get_running_loop(), context, self.id)
        document = await asyncio.to_thread(
            build_assignment_audit,
            _assignment_paths(cluster_run),
            self.settings.batch_rows,
            self.settings.category_limit,
            context.check_cancel,
            reporter.report,
        )
        context.check_cancel()
        audit = _persist_audit(str(context.run_id), cluster_run, document)
        return [{"audit": audit, INPUT_INDEX_OUTPUT: 0}]


def _prepare_audit(cluster_run: SpeakerClusterRunRef) -> SpeakerAuditRef | None:
    with database_session() as session:
        audit = speaker_crud.create_audit(
            session,
            SpeakerAuditCreate(cluster_run_id=cluster_run.run_id, seed=0),
        )
        if audit.state != SpeakerAuditState.COMPLETED.value:
            return None
        assert audit.review_id is not None
        return SpeakerAuditRef(audit.id, audit.cluster_run_id, audit.review_id)


def _assignment_paths(cluster_run: SpeakerClusterRunRef) -> list[Path]:
    with database_session() as session:
        durable = speaker_crud.list_clustering_artifacts(
            session,
            cluster_run.run_id,
            ClusteringArtifactRole.ASSIGNMENT,
        )
        artifact_ids = [artifact.artifact_id for artifact in durable]
        if artifact_ids != cluster_run.assignment_artifact_ids:
            raise ValueError(
                "cluster run assignment manifest does not match durable artifacts"
            )
        return [
            asset_crud.get_extra_file_path(session, value) for value in artifact_ids
        ]


def _persist_audit(
    producer_run_id: str,
    cluster_run: SpeakerClusterRunRef,
    document: AssignmentAuditDocument,
) -> SpeakerAuditRef:
    with database_session() as session:
        audit = speaker_crud.create_audit(
            session,
            SpeakerAuditCreate(cluster_run_id=cluster_run.run_id, seed=0),
        )
        run = speaker_crud.get_clustering_run(session, cluster_run.run_id)
        assert run.outcome_counts is not None, "completed cluster run has no outcomes"
        outcomes = ClusteringOutcomeCounts.model_validate(run.outcome_counts)
        payload = speaker_review_payload(
            document,
            _review_media(session, document),
            outcomes,
        )
        review = review_crud.create_review(
            session,
            ReviewCreate(
                producer_run_id=producer_run_id,
                kind="speaker_cluster_audit",
                source_key=str(audit.id),
                title=f"Speaker clusters · {document.total_rows:,} segments",
                payload=payload,
                continuation=_continuation(audit.id),
            ),
            commit=False,
        )
        completed = speaker_crud.complete_audit(
            session,
            audit.id,
            review.id,
            SpeakerAuditMetricsRecord.model_validate(document.metrics.model_dump()),
        )
    assert completed.review_id is not None
    return SpeakerAuditRef(completed.id, completed.cluster_run_id, completed.review_id)


def _review_media(
    session: Any, document: AssignmentAuditDocument
) -> dict[tuple[str, str], AudioSegmentReviewMedia]:
    entries = _unique_entries(document)
    audio_ids = [UUID(entry.audio_id) for entry in entries]
    audio_files = audio_crud.get_audio_files_bulk(session, audio_ids)
    result = {}
    for entry in entries:
        audio_id = UUID(entry.audio_id)
        audio = audio_files[audio_id]
        segment = next(
            value for value in audio.segments if value["id"] == entry.segment_id
        )
        start = float(segment["start"])
        end = float(segment["end"])
        result[(entry.audio_id, entry.segment_id)] = AudioSegmentReviewMedia(
            kind="audio_segment",
            audio_file_id=audio_id,
            segment_id=entry.segment_id,
            start_seconds=start,
            end_seconds=end,
            duration_seconds=end - start,
            name=audio.name,
        )
    return result


def _unique_entries(document: AssignmentAuditDocument) -> list[ListeningEntry]:
    manifest = document.listening_manifest
    entries = (
        *manifest.worst_within_cluster,
        *manifest.closest_cross_cluster,
        *manifest.low_margin_boundaries,
        *manifest.suspicious_labeled_merges,
    )
    return list({(entry.audio_id, entry.segment_id): entry for entry in entries}.values())


def _continuation(audit_id: UUID) -> ReviewContinuation:
    return ReviewContinuation(
        graph=InlineGraphRunRequest(
            nodes=[
                GraphNodeRequest(
                    id="audit", type="SpeakerAuditSource", params={"audit_id": audit_id}
                ),
                GraphNodeRequest(
                    id="apply",
                    type="ApplySpeakerClusters",
                    params={"approved_audit_id": audit_id},
                ),
            ],
            edges=[
                GraphEdgeRequest(
                    source_node="audit",
                    source_port="audit",
                    target_node="apply",
                    target_port="audit",
                )
            ],
        )
    )
