from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from shared.audio_annotations import AudioAnnotations
from shared.db.audio.schemas import AudioFileReference
from shared.db.clickhouse import clickhouse_client


@dataclass(frozen=True)
class SegmentCursor:
    audio_file_id: UUID
    segment_index: int


@dataclass(frozen=True)
class SegmentReference:
    audio_file_id: UUID
    audio_name: str
    audio_duration: float
    annotations: AudioAnnotations
    audio_byte_length: int
    audio_virtual: bool
    audio_storage_kind: str
    language: str | None
    style_prompt: str | None
    voice_prompt: str | None
    segment_index: int
    segment: dict[str, Any]

    @property
    def cursor(self) -> SegmentCursor:
        return SegmentCursor(self.audio_file_id, self.segment_index)


def count_audio_file_references(
    dataset_id: UUID | None,
    audio_file_ids: Sequence[UUID] | None,
    include_virtual: bool,
) -> int:
    join, filters, parameters = _scope(dataset_id, audio_file_ids, include_virtual)
    result = clickhouse_client().query(
        f"""
        SELECT count()
        FROM ({_LATEST_AUDIO}) AS audio
        {join}
        {f"WHERE {' AND '.join(filters)}" if filters else ""}
        """,
        parameters=parameters,
    )
    return int(result.result_rows[0][0])


def list_audio_file_references_page(
    dataset_id: UUID | None,
    audio_file_ids: Sequence[UUID] | None,
    include_virtual: bool,
    after_id: UUID | None,
    limit: int,
) -> list[AudioFileReference]:
    join, filters, parameters = _scope(dataset_id, audio_file_ids, include_virtual)
    if after_id is not None:
        filters.append("audio.id > {after_id:UUID}")
        parameters["after_id"] = after_id
    parameters["limit"] = limit
    result = clickhouse_client().query(
        f"""
        SELECT audio.id, audio.name, audio.duration, segment.speaker_id, audio.score,
               audio.language, audio.metadata, audio.byte_length, audio.virtual,
               audio.style_prompt, audio.voice_prompt
        FROM ({_LATEST_AUDIO}) AS audio
        {join}
        LEFT JOIN ({_FIRST_SPEAKER}) AS segment ON segment.audio_file_id = audio.id
        {f"WHERE {' AND '.join(filters)}" if filters else ""}
        ORDER BY audio.id
        LIMIT {{limit:UInt32}}
        """,
        parameters=parameters,
    )
    return [
        AudioFileReference(
            id=row[0],
            name=row[1],
            duration=row[2],
            annotations=AudioAnnotations(
                speaker_id=row[3], score=row[4], metadata=row[6]
            ),
            language=row[5],
            byte_length=row[7],
            virtual=row[8],
            style_prompt=row[9],
            voice_prompt=row[10],
        )
        for row in result.result_rows
    ]


def count_segment_references(dataset_id: UUID) -> int:
    result = clickhouse_client().query(
        """
        SELECT uniqExact((segment.audio_file_id, segment.id))
        FROM audio_segments AS segment
        INNER JOIN dataset_audio_files AS membership FINAL
            ON membership.audio_file_id = segment.audio_file_id
        WHERE membership.dataset_id = {dataset_id:UUID}
        """,
        parameters={"dataset_id": dataset_id},
    )
    return int(result.result_rows[0][0])


def list_segment_references_page(
    dataset_id: UUID, after: SegmentCursor | None, limit: int
) -> list[SegmentReference]:
    filters = ["membership.dataset_id = {dataset_id:UUID}"]
    parameters: dict[str, object] = {"dataset_id": dataset_id, "limit": limit}
    if after is not None:
        filters.append(
            "(segment.audio_file_id, segment.position) > ({after_id:UUID}, {after_position:UInt32})"
        )
        parameters.update(
            after_id=after.audio_file_id, after_position=after.segment_index
        )
    result = clickhouse_client().query(
        f"""
        SELECT audio.id, audio.name, audio.duration, audio.score, audio.metadata,
               audio.byte_length, audio.virtual, audio.storage_kind, audio.language,
               audio.style_prompt, audio.voice_prompt, segment.position,
               segment.id, segment.start_seconds, segment.end_seconds, segment.text,
               segment.phon, segment.kind, segment.accuracy, segment.speaker_id,
               segment.metadata, segment.alignment
        FROM ({_LATEST_AUDIO}) AS audio
        INNER JOIN dataset_audio_files AS membership FINAL ON membership.audio_file_id = audio.id
        INNER JOIN ({_LATEST_SEGMENTS}) AS segment ON segment.audio_file_id = audio.id
        WHERE {" AND ".join(filters)}
        ORDER BY segment.audio_file_id, segment.position
        LIMIT {{limit:UInt32}}
        """,
        parameters=parameters,
    )
    return [_segment_reference(row) for row in result.result_rows]


def _scope(dataset_id: UUID | None, ids: Sequence[UUID] | None, include_virtual: bool):
    join = ""
    filters: list[str] = []
    parameters: dict[str, object] = {}
    if dataset_id is not None:
        join = "INNER JOIN dataset_audio_files AS membership FINAL ON membership.audio_file_id = audio.id"
        filters.append("membership.dataset_id = {dataset_id:UUID}")
        parameters["dataset_id"] = dataset_id
    if ids is not None:
        filters.append("audio.id IN {ids:Array(UUID)}")
        parameters["ids"] = list(ids)
    if not include_virtual:
        filters.append("audio.virtual = false")
    return join, filters, parameters


def _segment_reference(row: Sequence[Any]) -> SegmentReference:
    segment = {
        "id": row[12],
        "start": row[13],
        "end": row[14],
        "text": row[15],
        "phon": row[16],
        "type": row[17],
        "annotations": AudioAnnotations(
            speaker_id=row[19], accuracy=row[18], metadata=row[20]
        ).model_dump(mode="json"),
        "alignment": row[21],
    }
    return SegmentReference(
        audio_file_id=row[0],
        audio_name=row[1],
        audio_duration=row[2],
        annotations=AudioAnnotations(speaker_id=row[19], score=row[3], metadata=row[4]),
        audio_byte_length=row[5],
        audio_virtual=row[6],
        audio_storage_kind=row[7],
        language=row[8],
        style_prompt=row[9],
        voice_prompt=row[10],
        segment_index=row[11],
        segment=segment,
    )


_LATEST_AUDIO = """
SELECT id, latest.1 AS updated_at, latest.2 AS name, latest.3 AS bucket_file_id,
       latest.4 AS byte_offset, latest.5 AS duration, latest.6 AS byte_length,
       latest.7 AS score, latest.8 AS language, latest.9 AS style_prompt,
       latest.10 AS voice_prompt, latest.11 AS virtual, latest.12 AS storage_kind,
       latest.13 AS storage_ref, latest.14 AS metadata
FROM (SELECT id, argMax(tuple(updated_at, name, bucket_file_id, byte_offset,
     duration, byte_length, score, language, style_prompt, voice_prompt, virtual,
     storage_kind, storage_ref, metadata), updated_at) AS latest FROM audio_files GROUP BY id)
"""

_LATEST_SEGMENTS = """
SELECT audio_file_id, id, latest.1 AS position, latest.2 AS start_seconds,
       latest.3 AS end_seconds, latest.4 AS text, latest.5 AS phon,
       latest.6 AS kind, latest.7 AS accuracy, latest.8 AS speaker_id,
       latest.9 AS metadata, latest.10 AS alignment
FROM (SELECT audio_file_id, id, argMax(tuple(position, start_seconds, end_seconds,
     text, phon, kind, accuracy, speaker_id, metadata, alignment), updated_at) AS latest
     FROM audio_segments GROUP BY audio_file_id, id)
"""

_FIRST_SPEAKER = f"""
SELECT audio_file_id, argMin(speaker_id, position) AS speaker_id
FROM ({_LATEST_SEGMENTS}) GROUP BY audio_file_id
"""
