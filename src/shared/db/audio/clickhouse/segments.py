from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID

from shared.db.audio.clickhouse.models import AudioSegmentRecord
from shared.db.clickhouse import clickhouse_client, delete_rows


def list_audio_segments(audio_file_id: UUID) -> list[AudioSegmentRecord]:
    result = clickhouse_client().query(
        """
        SELECT
            id,
            audio_file_id,
            updated_at,
            position,
            start_seconds,
            end_seconds,
            text,
            phon,
            kind,
            accuracy,
            speaker_id,
            metadata,
            alignment
        FROM audio_segments
        WHERE audio_file_id = {audio_file_id:UUID}
        QUALIFY row_number() OVER (
            PARTITION BY audio_file_id, id ORDER BY updated_at DESC
        ) = 1
        ORDER BY position, id
        """,
        parameters={"audio_file_id": audio_file_id},
    )
    return [AudioSegmentRecord.model_validate(row) for row in result.named_results()]


def list_audio_segments_bulk(
    audio_file_ids: Sequence[UUID],
) -> dict[UUID, list[AudioSegmentRecord]]:
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return {}
    result = clickhouse_client().query(
        """
        SELECT id, audio_file_id, updated_at, position, start_seconds, end_seconds,
               text, phon, kind, accuracy, speaker_id, metadata, alignment
        FROM audio_segments
        WHERE audio_file_id IN {ids:Array(UUID)}
        QUALIFY row_number() OVER (
            PARTITION BY audio_file_id, id ORDER BY updated_at DESC
        ) = 1
        ORDER BY audio_file_id, position, id
        """,
        parameters={"ids": ids},
    )
    grouped = {audio_id: [] for audio_id in ids}
    for row in result.named_results():
        item = AudioSegmentRecord.model_validate(row)
        grouped[item.audio_file_id].append(item)
    return grouped


def count_audio_segments(audio_file_ids: Sequence[UUID]) -> dict[UUID, int]:
    ids = list(dict.fromkeys(audio_file_ids))
    result = clickhouse_client().query(
        """
        SELECT audio_file_id, uniqExact(id) AS segment_count
        FROM audio_segments
        WHERE audio_file_id IN {ids:Array(UUID)}
        GROUP BY audio_file_id
        """,
        parameters={"ids": ids},
    )
    counts = {row[0]: int(row[1]) for row in result.result_rows}
    return {audio_id: counts.get(audio_id, 0) for audio_id in ids}


def list_audio_segment_previews(
    audio_file_ids: Sequence[UUID],
    limit: int,
) -> dict[UUID, list[AudioSegmentRecord]]:
    ids = list(dict.fromkeys(audio_file_ids))
    result = clickhouse_client().query(
        """
        SELECT
            id,
            audio_file_id,
            updated_at,
            position,
            start_seconds,
            end_seconds,
            text,
            phon,
            kind,
            accuracy,
            speaker_id,
            metadata,
            alignment
        FROM audio_segments
        WHERE audio_file_id IN {ids:Array(UUID)}
        QUALIFY row_number() OVER (
            PARTITION BY audio_file_id, id ORDER BY updated_at DESC
        ) = 1
        ORDER BY audio_file_id, position, id
        LIMIT {limit:UInt32} BY audio_file_id
        """,
        parameters={"ids": ids, "limit": limit},
    )
    previews = {audio_id: [] for audio_id in ids}
    for row in result.named_results():
        item = AudioSegmentRecord.model_validate(row)
        previews[item.audio_file_id].append(item)
    return previews


def insert_audio_segments(items: Sequence[AudioSegmentRecord]) -> None:
    if not items:
        return
    rows = [
        [
            item.id,
            item.audio_file_id,
            item.updated_at,
            item.position,
            item.start_seconds,
            item.end_seconds,
            item.text,
            item.phon,
            item.kind,
            item.accuracy,
            item.speaker_id,
            item.metadata,
            item.alignment,
        ]
        for item in items
    ]
    clickhouse_client().insert(
        "audio_segments",
        rows,
        column_names=[
            "id",
            "audio_file_id",
            "updated_at",
            "position",
            "start_seconds",
            "end_seconds",
            "text",
            "phon",
            "kind",
            "accuracy",
            "speaker_id",
            "metadata",
            "alignment",
        ],
    )


def replace_audio_segments(
    audio_file_id: UUID,
    items: Sequence[AudioSegmentRecord],
) -> list[AudioSegmentRecord]:
    assert all(item.audio_file_id == audio_file_id for item in items), (
        "segment belongs to another audio file"
    )
    delete_rows(
        clickhouse_client(),
        "audio_segments",
        "audio_file_id = {audio_file_id:UUID}",
        {"audio_file_id": audio_file_id},
    )
    insert_audio_segments(items)
    return list_audio_segments(audio_file_id)


def replace_audio_segments_bulk(
    items_by_audio_id: dict[UUID, Sequence[AudioSegmentRecord]],
) -> dict[UUID, list[AudioSegmentRecord]]:
    if not items_by_audio_id:
        return {}
    ids = list(items_by_audio_id)
    for audio_id, items in items_by_audio_id.items():
        assert all(item.audio_file_id == audio_id for item in items), (
            "segment belongs to another audio file"
        )
    delete_rows(
        clickhouse_client(),
        "audio_segments",
        "audio_file_id IN {ids:Array(UUID)}",
        {"ids": ids},
    )
    insert_audio_segments(
        [item for items in items_by_audio_id.values() for item in items]
    )
    return list_audio_segments_bulk(ids)


def update_audio_segment(item: AudioSegmentRecord) -> AudioSegmentRecord:
    current = list_audio_segments(item.audio_file_id)
    matches = [segment for segment in current if segment.id == item.id]
    if not matches:
        raise KeyError(f"Audio segment not found: {item.audio_file_id}/{item.id}")
    if item.updated_at <= matches[0].updated_at:
        item = item.model_copy(
            update={"updated_at": matches[0].updated_at + timedelta(microseconds=1)}
        )
    insert_audio_segments([item])
    rows = [
        segment
        for segment in list_audio_segments(item.audio_file_id)
        if segment.id == item.id
    ]
    if not rows:
        raise KeyError(f"Audio segment not found: {item.audio_file_id}/{item.id}")
    return rows[0]


def delete_audio_segment(audio_file_id: UUID, segment_id: str) -> None:
    delete_rows(
        clickhouse_client(),
        "audio_segments",
        "audio_file_id = {audio_file_id:UUID} AND id = {id:String}",
        {"audio_file_id": audio_file_id, "id": segment_id},
    )
