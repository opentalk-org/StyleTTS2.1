from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from shared.db.audio.models import AudioFile
from shared.db.audio.timestamp_cleanup.schemas import (
    CleanupAudit,
    CleanupBatchResult,
    TimestampSnapshot,
)
from shared.db.datasets.models import dataset_audio_files


PARAKEET_TIMESTAMPS = """
(
    SELECT segments.metadata -> 'text_timestamps'
    FROM segments
    WHERE segments.audio_file_id = audio_files.id
      AND segments.kind = 'parakeet'
      AND segments.metadata ? 'text_timestamps'
    ORDER BY segments.position
    LIMIT 1
)
"""


def count_dataset_rows_after(
    session: Session,
    dataset_id: UUID,
    after_id: UUID | None,
) -> int:
    statement = (
        select(func.count())
        .select_from(AudioFile)
        .join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        )
        .where(
            dataset_audio_files.c.dataset_id == dataset_id,
        )
    )
    if after_id is not None:
        statement = statement.where(AudioFile.id > after_id)
    return int(session.scalar(statement))


def list_timestamp_candidate_ids(
    session: Session,
    dataset_id: UUID,
    after_id: UUID | None,
    limit: int,
) -> list[UUID]:
    statement = (
        select(AudioFile.id)
        .join(
            dataset_audio_files,
            dataset_audio_files.c.audio_file_id == AudioFile.id,
        )
        .where(
            dataset_audio_files.c.dataset_id == dataset_id,
        )
        .order_by(AudioFile.id)
        .limit(limit)
    )
    if after_id is not None:
        statement = statement.where(AudioFile.id > after_id)
    return list(session.scalars(statement))


def prune_timestamp_batch(
    session: Session,
    audio_file_ids: Sequence[UUID],
    capture_audit: bool,
) -> CleanupBatchResult:
    if not audio_file_ids:
        raise ValueError("timestamp cleanup batch must not be empty")
    before = _snapshots(session, audio_file_ids)
    already_pruned = [item for item in before if item.audio_timestamps is None]
    missing = [
        item
        for item in before
        if item.audio_timestamps is not None and item.parakeet_timestamps is None
    ]
    mismatched = [
        item
        for item in before
        if item.audio_timestamps is not None
        and item.parakeet_timestamps is not None
        and item.audio_timestamps != item.parakeet_timestamps
    ]
    if mismatched:
        ids = ", ".join(str(item.audio_file_id) for item in mismatched[:10])
        raise RuntimeError(f"conflicting audio and Parakeet timestamps: {ids}")
    eligible_ids = [
        item.audio_file_id
        for item in before
        if item.audio_timestamps is not None
        and item.parakeet_timestamps is not None
    ]
    if eligible_ids:
        _prune(session, eligible_ids)
        after = _snapshots(session, eligible_ids)
        _verify(eligible_ids, before, after)
    else:
        after = []
    session.commit()
    before_by_id = {item.audio_file_id: item for item in before}
    audits = tuple(
        CleanupAudit(item.audio_file_id, before_by_id[item.audio_file_id], item)
        for item in after
    ) if capture_audit else ()
    return CleanupBatchResult(
        last_audio_file_id=audio_file_ids[-1],
        examined=len(audio_file_ids),
        pruned=len(eligible_ids),
        already_pruned=len(already_pruned),
        missing_parakeet=len(missing),
        audits=audits,
    )


def read_timestamp_snapshots(
    session: Session,
    audio_file_ids: Sequence[UUID],
) -> list[TimestampSnapshot]:
    return _snapshots(session, audio_file_ids)


def _snapshots(
    session: Session,
    audio_file_ids: Sequence[UUID],
) -> list[TimestampSnapshot]:
    statement = text(f"""
        SELECT id,
               metadata -> 'text_timestamps' AS audio_timestamps,
               {PARAKEET_TIMESTAMPS} AS parakeet_timestamps
        FROM audio_files
        WHERE id = ANY(:audio_file_ids)
        ORDER BY id
    """)
    rows = session.execute(statement, {"audio_file_ids": list(audio_file_ids)}).all()
    if len(rows) != len(audio_file_ids):
        raise KeyError("timestamp cleanup batch contains missing audio files")
    return [TimestampSnapshot(row.id, row.audio_timestamps, row.parakeet_timestamps) for row in rows]


def _prune(session: Session, audio_file_ids: Sequence[UUID]) -> None:
    statement = text(f"""
        UPDATE audio_files
        SET metadata = metadata - 'text_timestamps'
        WHERE id = ANY(:audio_file_ids)
          AND metadata ? 'text_timestamps'
          AND metadata -> 'text_timestamps' = {PARAKEET_TIMESTAMPS}
    """)
    result = session.execute(statement, {"audio_file_ids": list(audio_file_ids)})
    if result.rowcount != len(audio_file_ids):
        raise RuntimeError(
            f"guarded timestamp prune updated {result.rowcount} of {len(audio_file_ids)} rows"
        )


def _verify(
    eligible_ids: Sequence[UUID],
    before: Sequence[TimestampSnapshot],
    after: Sequence[TimestampSnapshot],
) -> None:
    before_by_id = {item.audio_file_id: item for item in before}
    after_by_id = {item.audio_file_id: item for item in after}
    for audio_file_id in eligible_ids:
        original = before_by_id[audio_file_id]
        saved = after_by_id[audio_file_id]
        if saved.audio_timestamps is not None:
            raise RuntimeError(f"audio timestamp key survived pruning: {audio_file_id}")
        if saved.parakeet_timestamps != original.parakeet_timestamps:
            raise RuntimeError(f"Parakeet timestamps changed during pruning: {audio_file_id}")
