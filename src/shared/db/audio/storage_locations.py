import uuid
from collections.abc import Sequence

from sqlalchemy.orm import Session

from shared.db.audio.schemas import (
    AudioBucketObject,
    AudioRecoveryReference,
    AudioStorageLocation,
)
from shared.db.clickhouse import clickhouse_client


def list_dataset_audio_bucket_objects(
    _session: Session, dataset_id: uuid.UUID
) -> list[AudioBucketObject]:
    result = clickhouse_client().query(
        """
        SELECT DISTINCT b.id, b.path, b.size
        FROM dataset_audio_files AS membership FINAL
        INNER JOIN (
            SELECT id, argMax(tuple(bucket_file_id, storage_kind), updated_at) AS latest
            FROM audio_files
            GROUP BY id
        ) AS audio ON audio.id = membership.audio_file_id
        INNER JOIN bucket_files AS b ON b.id = audio.latest.1
        WHERE membership.dataset_id = {dataset_id:UUID}
          AND audio.latest.2 = 'packed'
        ORDER BY b.id
        """,
        parameters={"dataset_id": dataset_id},
    )
    return [AudioBucketObject.model_validate(row) for row in result.named_results()]


def list_audio_recovery_references(
    _session: Session, bucket_file_ids: Sequence[uuid.UUID]
) -> list[AudioRecoveryReference]:
    result = clickhouse_client().query(
        """
        SELECT
            id,
            latest.1 AS bucket_file_id,
            latest.2 AS byte_length,
            JSONExtractString(latest.3, 'source_parquet_path') AS source_parquet_path,
            JSONExtractUInt(latest.3, 'source_row_index') AS source_row_index
        FROM (
            SELECT id, argMax(tuple(bucket_file_id, byte_length, metadata), updated_at) AS latest
            FROM audio_files
            GROUP BY id
        )
        WHERE bucket_file_id IN {bucket_ids:Array(UUID)}
        ORDER BY source_parquet_path, source_row_index
        """,
        parameters={"bucket_ids": list(bucket_file_ids)},
    )
    return [
        AudioRecoveryReference.model_validate(row) for row in result.named_results()
    ]


def audio_storage_locations(
    _session: Session, audio_file_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, AudioStorageLocation]:
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return {}
    result = clickhouse_client().query(
        """
        SELECT
            audio.id AS audio_file_id,
            audio.latest.1 AS storage_kind,
            bucket.path AS object_path,
            audio.latest.3 AS byte_offset,
            audio.latest.4 AS byte_length
        FROM (
            SELECT
                id,
                argMax(tuple(storage_kind, bucket_file_id, byte_offset, byte_length), updated_at) AS latest
            FROM audio_files
            WHERE id IN {ids:Array(UUID)}
            GROUP BY id
        ) AS audio
        LEFT JOIN bucket_files AS bucket ON bucket.id = audio.latest.2
        """,
        parameters={"ids": ids},
    )
    rows = {row["audio_file_id"]: row for row in result.named_results()}
    missing = set(ids).difference(rows)
    if missing:
        raise KeyError(f"Audio files not found: {sorted(map(str, missing))}")
    locations = {}
    for audio_file_id in ids:
        row = rows[audio_file_id]
        if row["storage_kind"] != "packed":
            raise ValueError(
                f"Audio {audio_file_id} contains metadata only; no stored audio bytes are available"
            )
        if row["object_path"] is None:
            raise ValueError(f"packed audio has no object path: {audio_file_id}")
        locations[audio_file_id] = AudioStorageLocation(
            audio_file_id=audio_file_id,
            object_path=row["object_path"],
            byte_offset=row["byte_offset"],
            byte_length=row["byte_length"],
        )
    return locations
