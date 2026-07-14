from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pyarrow.parquet as pq

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
)


ProgressReporter = Callable[[int, int, str], None]


@dataclass(frozen=True)
class ApplyOutcomeCounts:
    accepted: int
    provisional_new: int
    ambiguous: int
    rejected: int


@dataclass(frozen=True)
class ApplyCheckpoint:
    last_audio_id: UUID | None
    updated_audio_count: int
    state: str


class AssignmentSpool:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS accepted_assignments ("
            "audio_id TEXT NOT NULL, segment_id TEXT NOT NULL, "
            "cluster_key TEXT NOT NULL, PRIMARY KEY (audio_id, segment_id))"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cluster_voices ("
            "cluster_key TEXT PRIMARY KEY, voice_id TEXT NOT NULL)"
        )

    def __enter__(self) -> AssignmentSpool:
        return self

    def __exit__(self, *args: object) -> None:
        self._db.close()

    def ingest(
        self, paths: Sequence[Path], batch_rows: int,
        check_cancel: Callable[[], None], report_progress: ProgressReporter,
    ) -> ApplyOutcomeCounts:
        if batch_rows <= 0:
            raise ValueError("assignment spool batch_rows must be positive")
        self._db.execute("DELETE FROM accepted_assignments")
        self._db.execute("DELETE FROM cluster_voices")
        counts = [0, 0, 0, 0]
        for shard_index, path in enumerate(paths, start=1):
            check_cancel()
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=batch_rows,
                columns=["audio_id", "segment_id", "outcome", "cluster_id"],
            ):
                check_cancel()
                self._ingest_batch(batch.to_pydict(), counts)
            report_progress(shard_index, len(paths), f"validated assignment shard {shard_index}/{len(paths)}")
        self._db.commit()
        return ApplyOutcomeCounts(*counts)

    def store_cluster_voices(self, values: dict[str, UUID]) -> None:
        self._db.executemany(
            "INSERT INTO cluster_voices VALUES (?, ?)",
            [(key, str(voice_id)) for key, voice_id in values.items()],
        )
        self._db.commit()

    def require_cluster_voices(self) -> None:
        row = self._db.execute(
            "SELECT a.cluster_key FROM accepted_assignments a "
            "LEFT JOIN cluster_voices v USING (cluster_key) "
            "WHERE v.voice_id IS NULL LIMIT 1"
        ).fetchone()
        if row is not None:
            raise KeyError(f"accepted cluster summary not found: {row[0]}")

    def audio_count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(DISTINCT audio_id) FROM accepted_assignments"
        ).fetchone()
        assert row is not None
        return int(row[0])

    def audio_id_pages(
        self, page_size: int, after: UUID | None = None
    ) -> Iterator[list[UUID]]:
        if page_size <= 0:
            raise ValueError("speaker assignment page_size must be positive")
        cursor = "" if after is None else str(after)
        while True:
            rows = self._db.execute(
                "SELECT DISTINCT audio_id FROM accepted_assignments "
                "WHERE audio_id > ? ORDER BY audio_id LIMIT ?", (cursor, page_size)
            ).fetchall()
            if not rows:
                return
            yield [UUID(row[0]) for row in rows]
            cursor = rows[-1][0]

    def assignments_for(
        self, audio_ids: Sequence[UUID]
    ) -> list[AcceptedSpeakerAssignment]:
        if not audio_ids:
            return []
        placeholders = ",".join("?" for _value in audio_ids)
        rows = self._db.execute(
            "SELECT a.audio_id, a.segment_id, v.voice_id "
            "FROM accepted_assignments a JOIN cluster_voices v USING (cluster_key) "
            f"WHERE a.audio_id IN ({placeholders}) ORDER BY a.audio_id, a.segment_id",
            [str(value) for value in audio_ids],
        ).fetchall()
        return [AcceptedSpeakerAssignment(UUID(a), segment, UUID(v)) for a, segment, v in rows]

    def _ingest_batch(self, values: dict[str, list[object]], counts: list[int]) -> None:
        for index, raw_outcome in enumerate(values["outcome"]):
            outcome = SpeakerAssignmentOutcome(str(raw_outcome))
            counts[_OUTCOME_INDEX[outcome]] += 1
            if outcome is not SpeakerAssignmentOutcome.ACCEPTED:
                continue
            cluster_id = values["cluster_id"][index]
            if cluster_id is None:
                raise ValueError("accepted speaker assignment has no cluster ID")
            self._db.execute(
                "INSERT INTO accepted_assignments VALUES (?, ?, ?)",
                (str(UUID(str(values["audio_id"][index]))),
                 str(values["segment_id"][index]), str(cluster_id)),
            )
        self._db.commit()


def apply_speaker_audit(
    audit_ref: SpeakerAuditRef, spool_path: Path, page_size: int, batch_rows: int,
    check_cancel: Callable[[], None], report_progress: ProgressReporter,
) -> SaveResult:
    paths, expected, checkpoint = _apply_inputs(audit_ref)
    total = checkpoint.updated_audio_count
    created = 0
    try:
        with AssignmentSpool(spool_path) as spool:
            counts = spool.ingest(paths, batch_rows, check_cancel, report_progress)
            if counts != expected:
                raise ValueError(f"assignment artifact counts {counts} do not match durable run {expected}")
            audio_total = spool.audio_count()
            _record_apply_progress(audit_ref.audit_id, checkpoint, audio_total)
            created = reconcile_cluster_voices(
                spool, audit_ref.cluster_run_id, page_size, check_cancel, report_progress
            )
            spool.require_cluster_voices()
            total = _apply_pages(
                spool, audit_ref.audit_id, page_size, checkpoint,
                check_cancel, report_progress,
            )
    except BaseException as error:
        _record_apply_terminal(audit_ref.audit_id, error)
        raise
    _record_apply_terminal(audit_ref.audit_id, None)
    metadata = {
        "audit_id": str(audit_ref.audit_id), "cluster_run_id": str(audit_ref.cluster_run_id),
        "accepted_count": counts.accepted, "provisional_new_count": counts.provisional_new,
        "ambiguous_count": counts.ambiguous, "rejected_count": counts.rejected,
        "created_voice_count": created, "updated_audio_count": total,
    }
    lineage = stable_id("speaker_audit_apply", audit_ref.audit_id)
    return SaveResult(spool_path, "speaker_assignment_apply", stable_id("save", lineage), lineage, metadata)


def reconcile_cluster_voices(
    spool: AssignmentSpool, run_id: UUID, page_size: int,
    check_cancel: Callable[[], None], report_progress: ProgressReporter,
) -> int:
    after = None
    created = 0
    pages = 0
    while True:
        check_cancel()
        with database_session() as session:
            page = speaker_crud.reconcile_cluster_summary_voice_page(
                session, run_id, after, page_size
            )
        if page.after is None:
            return created
        spool.store_cluster_voices(page.voice_ids)
        after = page.after
        created += page.created_count
        pages += 1
        report_progress(pages, pages + 1, f"reconciled speaker voice page {pages}")


def _apply_inputs(
    audit_ref: SpeakerAuditRef,
) -> tuple[list[Path], ApplyOutcomeCounts, ApplyCheckpoint]:
    with database_session() as session:
        audit = speaker_crud.get_audit(session, audit_ref.audit_id)
        stored = (audit.cluster_run_id, audit.report_artifact_id, audit.listening_artifact_id)
        incoming = (audit_ref.cluster_run_id, audit_ref.report_artifact_id, audit_ref.listening_artifact_id)
        if audit.state != SpeakerAuditState.COMPLETED.value or stored != incoming:
            raise ValueError(f"speaker audit {audit.id} is not a completed durable match")
        run = speaker_crud.get_clustering_run(session, audit.cluster_run_id)
        if run.state != ClusteringRunState.COMPLETED.value:
            raise ValueError(f"speaker clustering run {run.id} is {run.state}")
        artifacts = speaker_crud.list_clustering_artifacts(
            session, run.id, ClusteringArtifactRole.ASSIGNMENT
        )
        paths = [asset_crud.get_extra_file_path(session, item.artifact_id) for item in artifacts]
        assert run.outcome_counts is not None, "completed run has no outcome counts"
        expected = ApplyOutcomeCounts(**run.outcome_counts)
        progress = speaker_crud.get_audit_apply_progress(audit)
    checkpoint = ApplyCheckpoint(None, 0, "pending") if progress is None else ApplyCheckpoint(
        progress.last_audio_id, progress.updated_audio_count, progress.state.value
    )
    return paths, expected, checkpoint


def _apply_pages(
    spool: AssignmentSpool, audit_id: UUID, page_size: int, checkpoint: ApplyCheckpoint,
    check_cancel: Callable[[], None], report_progress: ProgressReporter,
) -> int:
    total = spool.audio_count()
    updated = checkpoint.updated_audio_count
    for audio_ids in spool.audio_id_pages(page_size, checkpoint.last_audio_id):
        check_cancel()
        with database_session() as session:
            result = audio_crud.bulk_apply_speaker_assignments(
                session, spool.assignments_for(audio_ids)
            )
        updated += result.updated_audio_count
        current = ApplyCheckpoint(audio_ids[-1], updated, "running")
        _record_apply_progress(audit_id, current, total)
        report_progress(updated, total, f"updated speaker assignments for {updated}/{total} audio files")
    return updated


def _record_apply_progress(audit_id: UUID, value: ApplyCheckpoint, total: int) -> None:
    with database_session() as session:
        speaker_crud.record_audit_apply_progress(
            session, audit_id,
            speaker_crud.SpeakerAuditApplyProgress(
                speaker_crud.SpeakerAuditApplyState.RUNNING, value.last_audio_id,
                value.updated_audio_count, total, None,
            ),
        )


def _record_apply_terminal(audit_id: UUID, error: BaseException | None) -> None:
    with database_session() as session:
        audit = speaker_crud.get_audit(session, audit_id)
        progress = speaker_crud.get_audit_apply_progress(audit)
        if progress is None:
            progress = speaker_crud.SpeakerAuditApplyProgress(
                speaker_crud.SpeakerAuditApplyState.RUNNING, None, 0, 0, None
            )
        state = (
            speaker_crud.SpeakerAuditApplyState.COMPLETED if error is None else
            speaker_crud.SpeakerAuditApplyState.CANCELLED
            if isinstance(error, asyncio.CancelledError) else
            speaker_crud.SpeakerAuditApplyState.FAILED
        )
        speaker_crud.record_audit_apply_progress(
            session, audit_id,
            speaker_crud.SpeakerAuditApplyProgress(
                state, progress.last_audio_id, progress.updated_audio_count,
                progress.total_audio_count, None if error is None else str(error),
            ),
        )


_OUTCOME_INDEX = {
    SpeakerAssignmentOutcome.ACCEPTED: 0,
    SpeakerAssignmentOutcome.PROVISIONAL_NEW: 1,
    SpeakerAssignmentOutcome.AMBIGUOUS: 2,
    SpeakerAssignmentOutcome.REJECTED: 3,
}
