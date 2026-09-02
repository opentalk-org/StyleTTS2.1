import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from shared.audio_annotations import AudioAnnotations
from shared.db.audio.clickhouse.files import create_audio_files, get_audio_files
from shared.db.audio.clickhouse.models import AudioFileRecord, AudioFileUpdate
from shared.db.audio.clickhouse.segments import (
    insert_audio_segments,
    list_audio_segments,
    list_audio_segments_bulk,
)
from shared.db.clickhouse import clickhouse_client


@dataclass(frozen=True)
class AcceptedSpeakerAssignment:
    audio_id: UUID
    segment_id: str
    speaker_id: str


@dataclass(frozen=True)
class SpeakerAssignmentWriteCounts:
    accepted_assignment_count: int
    updated_audio_count: int


def audio_file_annotations(item: AudioFileRecord) -> AudioAnnotations:
    speakers = {
        segment.speaker_id
        for segment in list_audio_segments(item.id)
        if segment.speaker_id is not None
    }
    return AudioAnnotations(
        speaker_id=next(iter(speakers)) if len(speakers) == 1 else None,
        score=item.score,
        metadata=item.metadata,
    )


def bulk_update_audio_scores(scores: dict[UUID, float]) -> dict[UUID, AudioFileRecord]:
    if not scores or not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("audio scores must be non-empty and finite")
    records = {item.id: item for item in get_audio_files(list(scores))}
    missing = set(scores).difference(records)
    if missing:
        raise KeyError(f"Audio files not found: {sorted(map(str, missing))}")
    now = datetime.now(UTC)
    latest = max(item.updated_at for item in records.values())
    if now <= latest:
        now = latest + timedelta(microseconds=1)
    updated = [
        AudioFileRecord(
            id=item.id,
            **AudioFileUpdate.model_validate(
                item.model_copy(
                    update={"score": scores[item.id], "updated_at": now}
                ).model_dump()
            ).model_dump(),
        )
        for item in records.values()
    ]
    create_audio_files(updated)
    return {item.id: item for item in updated}


def iter_dataset_audio_scores(dataset_id: UUID) -> Iterator[tuple[UUID, float]]:
    result = clickhouse_client().query(
        """
        SELECT audio.id, audio.score
        FROM (SELECT id, argMax(score, updated_at) AS score FROM audio_files GROUP BY id) AS audio
        INNER JOIN dataset_audio_files AS membership FINAL ON membership.audio_file_id = audio.id
        WHERE membership.dataset_id = {dataset_id:UUID} AND audio.score IS NOT NULL
        ORDER BY audio.id
        """,
        parameters={"dataset_id": dataset_id},
    )
    yield from ((row[0], float(row[1])) for row in result.result_rows)


def list_audio_segment_accuracies(
    audio_file_ids: Sequence[UUID],
) -> dict[UUID, list[tuple[int, str, float | None]]]:
    grouped = list_audio_segments_bulk(audio_file_ids)
    return {
        audio_id: [
            (item.position, item.kind, item.accuracy)
            for item in grouped[audio_id]
        ]
        for audio_id in audio_file_ids
    }


def bulk_apply_speaker_assignments(
    assignments: Iterable[AcceptedSpeakerAssignment],
) -> SpeakerAssignmentWriteCounts:
    grouped: dict[UUID, dict[str, str]] = {}
    assignment_count = 0
    for assignment in assignments:
        values = grouped.setdefault(assignment.audio_id, {})
        if assignment.segment_id in values:
            raise ValueError(
                f"duplicate speaker assignment: {assignment.audio_id}/{assignment.segment_id}"
            )
        values[assignment.segment_id] = assignment.speaker_id
        assignment_count += 1
    now = datetime.now(UTC)
    updates = []
    segments_by_audio = list_audio_segments_bulk(list(grouped))
    for audio_id, values in grouped.items():
        segments = segments_by_audio[audio_id]
        missing = set(values).difference(item.id for item in segments)
        if missing:
            raise KeyError(
                f"audio file {audio_id} has missing segment IDs: {sorted(missing)}"
            )
        latest = max(item.updated_at for item in segments)
        if now <= latest:
            now = latest + timedelta(microseconds=1)
        updates.extend(
            item.model_copy(update={"speaker_id": values[item.id], "updated_at": now})
            for item in segments
            if item.id in values
        )
    insert_audio_segments(updates)
    return SpeakerAssignmentWriteCounts(assignment_count, len(grouped))
