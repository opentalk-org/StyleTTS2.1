from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from shared.db.audio.clickhouse.annotations import (
    AcceptedSpeakerAssignment,
    SpeakerAssignmentWriteCounts,
    bulk_apply_speaker_assignments as _apply_speakers,
    bulk_update_audio_scores as _update_scores,
    iter_dataset_audio_scores as _iter_scores,
    list_audio_segment_accuracies as _segment_accuracies,
)
from shared.db.audio.clickhouse.catalog import list_audio_files as _list_audio_files
from shared.db.audio.clickhouse.catalog import search_audio_file_ids as _search_ids
from shared.db.audio.clickhouse.conversion import segment_records
from shared.db.audio.clickhouse.files import delete_audio_files
from shared.db.audio.clickhouse.files import get_audio_file as _get_audio_file
from shared.db.audio.clickhouse.files import get_audio_files
from shared.db.audio.clickhouse.models import AudioFileRecord
from shared.db.audio.clickhouse.segments import count_audio_segments
from shared.db.audio.clickhouse.references import (
    SegmentCursor,
    SegmentReference,
    count_audio_file_references as _count_audio_refs,
    count_segment_references as _count_segment_refs,
    list_audio_file_references_page as _list_audio_refs,
    list_segment_references_page as _list_segment_refs,
)
from shared.db.audio.clickhouse.segments import list_audio_segments_bulk as _list_segments_bulk
from shared.db.audio.clickhouse.segments import (
    replace_audio_segments_bulk as _replace_segments_bulk,
)
from shared.db.audio.clickhouse.storage import bulk_create_audio_files as _create_files
from shared.db.audio.clickhouse.storage import bulk_update_audio_files as _update_files
from shared.db.audio.catalog_pagination import AudioCursor, cursor_for_row
from shared.db.audio.schemas import AudioBucketLocation, AudioCreate, AudioUpdate
from shared.db.assets.crud import delete_unreferenced_bucket_files
from shared.db.clickhouse import clickhouse_client
from shared.db.waveforms.clickhouse import get_waveforms


@dataclass(frozen=True)
class AudioPackConfig:
    target_pack_bytes: int = 128 * 1024 * 1024
    path_prefix: str = "audio-packs"
    folder_target_files: int = 256
    prune_used_ratio: float = 0.5
    prune_size_ratio: float = 0.2
    remote_workers: int = 9


def get_audio_file(_session: Session, audio_file_id: UUID) -> AudioFileRecord:
    return _get_audio_file(audio_file_id)


def get_audio_files_bulk(
    _session: Session, audio_file_ids: Sequence[UUID]
) -> dict[UUID, AudioFileRecord]:
    rows = {item.id: item for item in get_audio_files(audio_file_ids)}
    missing = set(audio_file_ids).difference(rows)
    if missing:
        raise KeyError(f"Audio files not found: {sorted(map(str, missing))}")
    return rows


def bulk_create_audio_files(
    session: Session,
    payloads: Sequence[AudioCreate],
    config: AudioPackConfig = AudioPackConfig(),
    commit: bool = True,
) -> list[AudioFileRecord]:
    return _create_files(session, payloads)


def bulk_update_audio_files(
    session: Session,
    payloads: dict[UUID, AudioUpdate],
    config: AudioPackConfig = AudioPackConfig(),
) -> dict[UUID, AudioFileRecord]:
    return _update_files(session, payloads)


def bulk_delete_audio_files(
    session: Session,
    audio_file_ids: Iterable[UUID],
    config: AudioPackConfig = AudioPackConfig(),
    commit: bool = True,
    prune: bool = False,
) -> None:
    ids = list(dict.fromkeys(audio_file_ids))
    files = get_audio_files(ids)
    waveforms = get_waveforms(ids)
    bucket_file_ids = [
        item.bucket_file_id for item in files if item.bucket_file_id is not None
    ]
    bucket_file_ids.extend(item.pack_id for item in waveforms)
    delete_audio_files(ids)
    delete_unreferenced_bucket_files(session, bucket_file_ids)


def list_audio_segments_bulk(
    _session: Session, audio_file_ids: Sequence[UUID]
) -> dict[UUID, list[dict[str, Any]]]:
    return {
        audio_id: [item.as_payload() for item in items]
        for audio_id, items in _list_segments_bulk(audio_file_ids).items()
    }


def bulk_replace_audio_segments(
    _session: Session,
    payloads: dict[UUID, Sequence[dict[str, Any]]],
    commit: bool = True,
    fallback_accuracy: dict[UUID, float | None] | None = None,
) -> dict[UUID, list[dict[str, Any]]]:
    now = datetime.now(UTC)
    saved = _replace_segments_bulk(
        {
            audio_id: segment_records(audio_id, segments, now)
            for audio_id, segments in payloads.items()
        }
    )
    return {
        audio_id: [item.as_payload() for item in items]
        for audio_id, items in saved.items()
    }


def count_audio_file_references(
    _session: Session,
    dataset_id: UUID | None,
    audio_file_ids: Sequence[UUID] | None,
    include_virtual: bool,
) -> int:
    return _count_audio_refs(dataset_id, audio_file_ids, include_virtual)


def list_audio_file_references_page(
    _session: Session,
    dataset_id: UUID | None,
    audio_file_ids: Sequence[UUID] | None,
    include_virtual: bool,
    after_id: UUID | None,
    limit: int,
):
    return _list_audio_refs(
        dataset_id, audio_file_ids, include_virtual, after_id, limit
    )


def count_segment_references(_session: Session, dataset_id: UUID) -> int:
    return _count_segment_refs(dataset_id)


def list_segment_references_page(
    _session: Session, dataset_id: UUID, after: SegmentCursor | None, limit: int
) -> list[SegmentReference]:
    return _list_segment_refs(dataset_id, after, limit)


def bulk_update_audio_scores(_session: Session, scores: dict[UUID, float]):
    return _update_scores(scores)


def iter_dataset_audio_scores(_session: Session, dataset_id: UUID):
    return _iter_scores(dataset_id)


def list_audio_segment_accuracies(_session: Session, audio_file_ids: list[UUID]):
    return _segment_accuracies(audio_file_ids)


def bulk_apply_speaker_assignments(
    _session: Session, assignments: Iterable[AcceptedSpeakerAssignment]
) -> SpeakerAssignmentWriteCounts:
    return _apply_speakers(assignments)


def search_audio_file_ids(
    _session: Session, query: str, dataset: str, language: str = ""
) -> list[UUID]:
    return _search_ids(query, dataset, language)


def search_audio_files(
    _session: Session,
    query: str,
    dataset: str,
    sort: str,
    limit: int,
    cursor: str | None,
    preview_limit: int = 8,
    language: str = "",
):
    if sort not in ("updated", "duration"):
        raise ValueError("Audio sort must be updated or duration")
    page_cursor = AudioCursor.decode(cursor, sort) if cursor else None
    rows = _list_audio_files(
        limit=limit + 1,
        order=sort,
        after_value=page_cursor.value if page_cursor else None,
        after_id=page_cursor.audio_file_id if page_cursor else None,
        dataset=dataset,
        query=query,
        language=language,
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    counts = count_audio_segments([item.id for item in rows])
    values = [
        (
            item,
            counts[item.id],
            int(item.metadata["sample_rate"])
            if "sample_rate" in item.metadata
            else None,
        )
        for item in rows
    ]
    next_cursor = cursor_for_row(sort, rows[-1]).encode() if has_more else None
    return values, next_cursor, has_more


def audio_bucket_locations(
    _session: Session, audio_file_ids: Sequence[UUID]
) -> list[AudioBucketLocation]:
    rows = {item.id: item for item in get_audio_files(audio_file_ids)}
    locations = []
    for audio_id in audio_file_ids:
        item = rows[audio_id]
        if item.bucket_file_id is None:
            raise ValueError(f"Audio has no bucket file: {audio_id}")
        locations.append(
            AudioBucketLocation(
                audio_file_id=audio_id,
                bucket_file_id=item.bucket_file_id,
                byte_length=item.byte_length,
            )
        )
    return locations


def list_audio_files(_session: Session):
    return _list_audio_files(limit=4_294_967_295)


def list_split_metadata(source_ids: Sequence[UUID], operation_ids: Sequence[str]) -> list[dict[str, Any]]:
    result = clickhouse_client().query(
        """
        SELECT argMax(metadata, updated_at) AS metadata
        FROM audio_files
        WHERE JSONExtractString(metadata, 'source_audio_id') IN {source_ids:Array(String)}
          AND JSONExtractString(metadata, 'split_operation_id') IN {operation_ids:Array(String)}
          AND JSONExtractString(metadata, 'mode') = 'replace_all'
        GROUP BY id
        """,
        parameters={
            "source_ids": [str(source_id) for source_id in source_ids],
            "operation_ids": list(operation_ids),
        },
    )
    return [row[0] for row in result.result_rows]
