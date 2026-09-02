from typing import Literal
from uuid import UUID

from shared.db.audio.clickhouse.models import AudioFileRecord
from shared.db.clickhouse import clickhouse_client

AudioOrder = Literal["updated", "duration"]


def list_audio_files(
    *,
    limit: int,
    order: AudioOrder = "updated",
    after_value: str | float | None = None,
    after_id: UUID | None = None,
    dataset_id: UUID | None = None,
) -> list[AudioFileRecord]:
    order_column, order_type = _ORDER_COLUMNS[order]
    filters = []
    parameters: dict[str, object] = {"limit": limit}
    join = ""
    if dataset_id is not None:
        join = "INNER JOIN dataset_audio_files AS d FINAL ON d.audio_file_id = a.id"
        filters.append("d.dataset_id = {dataset_id:UUID}")
        parameters["dataset_id"] = dataset_id
    if after_value is not None:
        assert after_id is not None, "catalog cursor requires an audio ID"
        filters.append(
            f"({order_column}, a.id) < ({{after_value:{order_type}}}, {{after_id:UUID}})"
        )
        parameters.update(after_value=after_value, after_id=after_id)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    result = clickhouse_client().query(
        f"""
        SELECT
            a.id,
            a.updated_at,
            a.name,
            a.bucket_file_id,
            a.byte_offset,
            a.duration,
            a.byte_length,
            a.score,
            a.language,
            a.style_prompt,
            a.voice_prompt,
            a.virtual,
            a.storage_ref,
            a.metadata
        FROM (
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
                latest.12 AS storage_ref,
                latest.13 AS metadata
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
                            storage_ref,
                            metadata
                        ),
                        updated_at
                    ) AS latest
                FROM audio_files
                GROUP BY id
            )
        ) AS a
        {join}
        {where}
        ORDER BY {order_column} DESC, a.id DESC
        LIMIT {{limit:UInt32}}
        """,
        parameters=parameters,
    )
    return [AudioFileRecord.model_validate(row) for row in result.named_results()]


_ORDER_COLUMNS = {
    "updated": ("a.updated_at", "DateTime64(9)"),
    "duration": ("a.duration", "Float64"),
}
