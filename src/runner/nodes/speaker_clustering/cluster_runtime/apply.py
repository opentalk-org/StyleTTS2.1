from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pyarrow.parquet as pq
from sqlalchemy.orm import Session

from runner.nodes.models import SaveResult, SpeakerAuditRef, stable_id
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.audio import crud as audio_crud
from shared.db.audio.speaker_assignment_crud import AcceptedSpeakerAssignment
from shared.db.speakers import crud as speaker_crud
from shared.db.speakers.schemas import (
    ClusteringArtifactRole,
    ClusteringRunState,
    SpeakerAssignmentOutcome,
    SpeakerAuditState,
    SpeakerClusterStatus,
)
from shared.db.voices import crud as voice_crud
from shared.db.voices.schemas import VoiceCreate


ProgressReporter = Callable[[int, int, str], None]


@dataclass(frozen=True)
class ApplyOutcomeCounts:
    accepted: int
    provisional_new: int
    ambiguous: int
    rejected: int


class AssignmentSpool:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS accepted_assignments ("
            "audio_id TEXT NOT NULL, segment_id TEXT NOT NULL, "
            "cluster_key TEXT NOT NULL, PRIMARY KEY (audio_id, segment_id))"
        )

    def __enter__(self) -> AssignmentSpool:
        return self

    def __exit__(self, *args: object) -> None:
        self._connection.close()

    def ingest(
        self,
        paths: Sequence[Path],
        batch_rows: int,
        check_cancel: Callable[[], None],
        report_progress: ProgressReporter,
    ) -> ApplyOutcomeCounts:
        if batch_rows <= 0:
            raise ValueError("assignment spool batch_rows must be positive")
        self._connection.execute("DELETE FROM accepted_assignments")
        counts = [0, 0, 0, 0]
        for shard_index, path in enumerate(paths, start=1):
            check_cancel()
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=batch_rows,
                columns=["audio_id", "segment_id", "outcome", "cluster_id"],
            ):
                check_cancel()
                self._ingest_batch(batch.to_pydict(), counts)
            report_progress(
                shard_index,
                len(paths),
                f"spooled assignment shard {shard_index}/{len(paths)}",
            )
        self._connection.commit()
        return ApplyOutcomeCounts(*counts)

    def audio_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(DISTINCT audio_id) FROM accepted_assignments"
        ).fetchone()
        assert row is not None
        return int(row[0])

    def audio_id_pages(self, page_size: int) -> Iterator[list[UUID]]:
        if page_size <= 0:
            raise ValueError("speaker assignment page_size must be positive")
        after = ""
        while True:
            rows = self._connection.execute(
                "SELECT DISTINCT audio_id FROM accepted_assignments "
                "WHERE audio_id > ? ORDER BY audio_id LIMIT ?",
                (after, page_size),
            ).fetchall()
            if not rows:
                return
            values = [UUID(row[0]) for row in rows]
            yield values
            after = rows[-1][0]

    def assignments_for(
        self,
        audio_ids: Sequence[UUID],
        cluster_voices: dict[str, UUID],
    ) -> list[AcceptedSpeakerAssignment]:
        if not audio_ids:
            return []
        placeholders = ",".join("?" for _audio_id in audio_ids)
        rows = self._connection.execute(
            "SELECT audio_id, segment_id, cluster_key FROM accepted_assignments "
            f"WHERE audio_id IN ({placeholders}) ORDER BY audio_id, segment_id",
            [str(audio_id) for audio_id in audio_ids],
        ).fetchall()
        assignments = []
        for audio_id, segment_id, cluster_key in rows:
            if cluster_key not in cluster_voices:
                raise KeyError(f"accepted cluster summary not found: {cluster_key}")
            assignments.append(
                AcceptedSpeakerAssignment(
                    UUID(audio_id), segment_id, cluster_voices[cluster_key]
                )
            )
        return assignments

    def _ingest_batch(
        self,
        values: dict[str, list[object]],
        counts: list[int],
    ) -> None:
        for index, raw_outcome in enumerate(values["outcome"]):
            outcome = SpeakerAssignmentOutcome(str(raw_outcome))
            counts[_outcome_index(outcome)] += 1
            if outcome is not SpeakerAssignmentOutcome.ACCEPTED:
                continue
            cluster_id = values["cluster_id"][index]
            if cluster_id is None:
                raise ValueError("accepted speaker assignment has no cluster ID")
            audio_id = str(UUID(str(values["audio_id"][index])))
            self._connection.execute(
                "INSERT INTO accepted_assignments VALUES (?, ?, ?)",
                (audio_id, str(values["segment_id"][index]), str(cluster_id)),
            )
        self._connection.commit()


def reconcile_cluster_voices(
    session: Session, run_id: UUID
) -> tuple[dict[str, UUID], int]:
    summaries = speaker_crud.list_cluster_summaries(
        session, run_id, SpeakerClusterStatus.ACCEPTED
    )
    cluster_voices = {
        summary.cluster_key: summary.voice_id
        for summary in summaries
        if summary.voice_id is not None
    }
    unresolved = [summary for summary in summaries if summary.voice_id is None]
    names = {
        summary.cluster_key: f"speaker-cluster-{run_id}-{summary.cluster_key}"
        for summary in unresolved
    }
    existing = voice_crud.get_voices_by_names(session, list(names.values()))
    missing_names = [name for name in names.values() if name not in existing]
    created = voice_crud.bulk_create_voices(
        session, [VoiceCreate(name=name) for name in missing_names]
    )
    voices_by_name = {**existing, **{voice.name: voice for voice in created}}
    assignments = {
        cluster_key: voices_by_name[name].id for cluster_key, name in names.items()
    }
    speaker_crud.assign_cluster_summary_voices(session, run_id, assignments)
    cluster_voices.update(assignments)
    return cluster_voices, len(created)


def apply_speaker_audit(
    audit_ref: SpeakerAuditRef,
    spool_path: Path,
    page_size: int,
    batch_rows: int,
    check_cancel: Callable[[], None],
    report_progress: ProgressReporter,
) -> SaveResult:
    paths, cluster_voices, created_voice_count, expected = _apply_inputs(audit_ref)
    with AssignmentSpool(spool_path) as spool:
        counts = spool.ingest(paths, batch_rows, check_cancel, report_progress)
        if counts != expected:
            raise ValueError(
                f"assignment artifact counts {counts} do not match durable run {expected}"
            )
        updated_audio_count = _apply_pages(
            spool,
            cluster_voices,
            page_size,
            check_cancel,
            report_progress,
        )
    metadata = {
        "audit_id": str(audit_ref.audit_id),
        "cluster_run_id": str(audit_ref.cluster_run_id),
        "accepted_count": counts.accepted,
        "provisional_new_count": counts.provisional_new,
        "ambiguous_count": counts.ambiguous,
        "rejected_count": counts.rejected,
        "created_voice_count": created_voice_count,
        "updated_audio_count": updated_audio_count,
    }
    lineage_id = stable_id("speaker_audit_apply", audit_ref.audit_id)
    return SaveResult(
        spool_path,
        "speaker_assignment_apply",
        stable_id("save", lineage_id),
        lineage_id,
        metadata,
    )


def _apply_inputs(
    audit_ref: SpeakerAuditRef,
) -> tuple[list[Path], dict[str, UUID], int, ApplyOutcomeCounts]:
    with database_session() as session:
        audit = speaker_crud.get_audit(session, audit_ref.audit_id)
        stored_ref = (
            audit.cluster_run_id,
            audit.report_artifact_id,
            audit.listening_artifact_id,
        )
        incoming_ref = (
            audit_ref.cluster_run_id,
            audit_ref.report_artifact_id,
            audit_ref.listening_artifact_id,
        )
        if audit.state != SpeakerAuditState.COMPLETED.value or stored_ref != incoming_ref:
            raise ValueError(f"speaker audit {audit.id} is not a completed durable match")
        run = speaker_crud.get_clustering_run(session, audit.cluster_run_id)
        if run.state != ClusteringRunState.COMPLETED.value:
            raise ValueError(f"speaker clustering run {run.id} is {run.state}")
        artifacts = speaker_crud.list_clustering_artifacts(
            session, run.id, ClusteringArtifactRole.ASSIGNMENT
        )
        paths = [
            asset_crud.get_extra_file_path(session, artifact.artifact_id)
            for artifact in artifacts
        ]
        cluster_voices, created_count = reconcile_cluster_voices(session, run.id)
        assert run.outcome_counts is not None, "completed run has no outcome counts"
        expected = ApplyOutcomeCounts(
            accepted=run.outcome_counts["accepted"],
            provisional_new=run.outcome_counts["provisional_new"],
            ambiguous=run.outcome_counts["ambiguous"],
            rejected=run.outcome_counts["rejected"],
        )
    return paths, cluster_voices, created_count, expected


def _apply_pages(
    spool: AssignmentSpool,
    cluster_voices: dict[str, UUID],
    page_size: int,
    check_cancel: Callable[[], None],
    report_progress: ProgressReporter,
) -> int:
    total = spool.audio_count()
    updated = 0
    for audio_ids in spool.audio_id_pages(page_size):
        check_cancel()
        assignments = spool.assignments_for(audio_ids, cluster_voices)
        with database_session() as session:
            result = audio_crud.bulk_apply_speaker_assignments(session, assignments)
        updated += result.updated_audio_count
        report_progress(updated, total, f"updated speaker assignments for {updated}/{total} audio files")
    return updated


def _outcome_index(outcome: SpeakerAssignmentOutcome) -> int:
    order = {
        SpeakerAssignmentOutcome.ACCEPTED: 0,
        SpeakerAssignmentOutcome.PROVISIONAL_NEW: 1,
        SpeakerAssignmentOutcome.AMBIGUOUS: 2,
        SpeakerAssignmentOutcome.REJECTED: 3,
    }
    return order[outcome]
