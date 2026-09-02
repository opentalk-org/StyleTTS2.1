from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID

from shared.db.audio.clickhouse.models import AudioFileRecord, AudioFileUpdate
from shared.db.clickhouse import clickhouse_client, delete_rows


def create_audio_files(items: Sequence[AudioFileRecord]) -> None:
    if not items:
        return
    rows = [
        [
            item.id,
            item.updated_at,
            item.name,
            item.bucket_file_id,
            item.byte_offset,
            item.duration,
            item.byte_length,
            item.score,
            item.language,
            item.style_prompt,
            item.voice_prompt,
            item.virtual,
            item.storage_kind.value,
            item.storage_ref,
            item.metadata,
        ]
        for item in items
    ]
    clickhouse_client().insert(
        "audio_files",
        rows,
        column_names=[
            "id",
            "updated_at",
            "name",
            "bucket_file_id",
            "byte_offset",
            "duration",
            "byte_length",
            "score",
            "language",
            "style_prompt",
            "voice_prompt",
            "virtual",
            "storage_kind",
            "storage_ref",
            "metadata",
        ],
    )


def get_audio_file(audio_file_id: UUID) -> AudioFileRecord:
    rows = get_audio_files([audio_file_id])
    if not rows:
        raise KeyError(f"Audio file not found: {audio_file_id}")
    return rows[0]


def get_audio_files(audio_file_ids: Sequence[UUID]) -> list[AudioFileRecord]:
    if not audio_file_ids:
        return []
    result = clickhouse_client().query(
        """
        SELECT
            id,
            latest.1 AS updated_at,
            latest.2 AS name,
            latest.3 AS bucket_file_id,
            latest.4 AS byte_offset,
            latest.5 AS duration,
            latest.6 AS byte_length,
            latest.7 AS score,
            latest.8 AS language,
            latest.9 AS style_prompt,
            latest.10 AS voice_prompt,
            latest.11 AS virtual,
            latest.12 AS storage_kind,
            latest.13 AS storage_ref,
            latest.14 AS metadata
        FROM (
            SELECT
                id,
                argMax(
                    tuple(
                        updated_at,
                        name,
                        bucket_file_id,
                        byte_offset,
                        duration,
                        byte_length,
                        score,
                        language,
                        style_prompt,
                        voice_prompt,
                        virtual,
                        storage_kind,
                        storage_ref,
                        metadata
                    ),
                    updated_at
                ) AS latest
            FROM audio_files
            WHERE id IN {ids:Array(UUID)}
            GROUP BY id
        )
        """,
        parameters={"ids": list(audio_file_ids)},
    )
    return [AudioFileRecord.model_validate(row) for row in result.named_results()]


def update_audio_file(audio_file_id: UUID, item: AudioFileUpdate) -> AudioFileRecord:
    current = get_audio_file(audio_file_id)
    if item.updated_at <= current.updated_at:
        item = item.model_copy(
            update={"updated_at": current.updated_at + timedelta(microseconds=1)}
        )
    create_audio_files([AudioFileRecord(id=audio_file_id, **item.model_dump())])
    return get_audio_file(audio_file_id)


def delete_audio_files(audio_file_ids: Sequence[UUID]) -> None:
    if not audio_file_ids:
        return
    parameters = {"ids": list(audio_file_ids)}
    client = clickhouse_client()
    delete_rows(
        client,
        "mos_comparisons",
        "audio_a_id IN {ids:Array(UUID)} OR audio_b_id IN {ids:Array(UUID)}",
        parameters,
    )
    delete_rows(
        client, "audio_segments", "audio_file_id IN {ids:Array(UUID)}", parameters
    )
    delete_rows(
        client, "audio_waveforms", "audio_file_id IN {ids:Array(UUID)}", parameters
    )
    delete_rows(
        client, "dataset_audio_files", "audio_file_id IN {ids:Array(UUID)}", parameters
    )
    delete_rows(
        client,
        "audio_files",
        "id IN {ids:Array(UUID)}",
        parameters,
    )
