from datetime import UTC, datetime, timedelta

from shared.db.audio.clickhouse import AudioSegmentRecord
from shared.db.audio.clickhouse.segments import insert_audio_segments
from shared.db.clickhouse import clickhouse_client
from shared.db.speakers.schemas import SpeakerRead


def search_speakers(
    query: str,
    limit: int,
    offset: int,
) -> tuple[list[SpeakerRead], int]:
    parameters = {"query": query, "limit": limit, "offset": offset}
    result = clickhouse_client().query(
        """
        WITH current_segments AS (
            SELECT
                audio_file_id,
                id,
                argMax(tuple(speaker_id), updated_at).1 AS speaker_id
            FROM audio_segments
            GROUP BY audio_file_id, id
        )
        SELECT
            segment.speaker_id AS id,
            uniqExact(segment.audio_file_id) AS audio_files,
            count() AS segments,
            groupUniqArray(membership.dataset_id) AS datasets,
            count() OVER () AS total
        FROM current_segments AS segment
        LEFT JOIN dataset_audio_files AS membership FINAL
          ON membership.audio_file_id = segment.audio_file_id
        WHERE segment.speaker_id IS NOT NULL
          AND positionCaseInsensitiveUTF8(segment.speaker_id, {query:String}) > 0
        GROUP BY segment.speaker_id
        ORDER BY segment.speaker_id
        LIMIT {limit:UInt32} OFFSET {offset:UInt64}
        """,
        parameters=parameters,
    )
    rows = list(result.named_results())
    total = int(rows[0]["total"]) if rows else 0
    return [SpeakerRead.model_validate(row) for row in rows], total


def rename_speaker(speaker_id: str, replacement: str) -> None:
    if not replacement:
        raise ValueError("replacement speaker_id must not be empty")
    _replace_speaker(speaker_id, replacement)


def clear_speaker(speaker_id: str) -> None:
    _replace_speaker(speaker_id, None)


def clear_matching_speakers(query: str) -> None:
    rows, _ = search_speakers(query, 200, 0)
    while rows:
        for row in rows:
            _replace_speaker(row.id, None)
        rows, _ = search_speakers(query, 200, 0)


def _replace_speaker(speaker_id: str, replacement: str | None) -> None:
    result = clickhouse_client().query(
        """
        SELECT
            id,
            audio_file_id,
            latest.1 AS updated_at,
            latest.2 AS position,
            latest.3 AS start_seconds,
            latest.4 AS end_seconds,
            latest.5 AS text,
            latest.6 AS phon,
            latest.7 AS kind,
            latest.8 AS accuracy,
            latest.9 AS speaker_id,
            latest.10 AS metadata,
            latest.11 AS alignment
        FROM (
            SELECT
                id,
                audio_file_id,
                argMax(
                    tuple(updated_at, position, start_seconds, end_seconds, text,
                          phon, kind, accuracy, speaker_id, metadata, alignment),
                    updated_at
                ) AS latest
            FROM audio_segments
            GROUP BY audio_file_id, id
        )
        WHERE speaker_id = {speaker_id:String}
        """,
        parameters={"speaker_id": speaker_id},
    )
    items = [AudioSegmentRecord.model_validate(row) for row in result.named_results()]
    if not items:
        raise KeyError(f"speaker not found: {speaker_id}")
    updated_at = datetime.now(UTC)
    latest = max(item.updated_at for item in items)
    if updated_at <= latest:
        updated_at = latest + timedelta(microseconds=1)
    insert_audio_segments(
        [
            item.model_copy(
                update={"speaker_id": replacement, "updated_at": updated_at}
            )
            for item in items
        ]
    )
