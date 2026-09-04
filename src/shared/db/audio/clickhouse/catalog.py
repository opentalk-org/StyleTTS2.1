from datetime import datetime
from typing import Literal
from uuid import UUID

from shared.db.audio.clickhouse.models import AudioFileRecord
from shared.db.clickhouse import clickhouse_client

AudioOrder = Literal["updated", "duration"]


def list_audio_files(
    *,
    limit: int,
    order: AudioOrder = "updated",
    after_value: datetime | float | None = None,
    after_id: UUID | None = None,
    dataset: str = "all",
    query: str = "",
    language: str = "",
    run_id: str | None = None,
) -> list[AudioFileRecord]:
    order_column, order_type = _ORDER_COLUMNS[order]
    filters = []
    parameters: dict[str, object] = {"limit": limit}
    join = ""
    if dataset == "unassigned":
        join = "LEFT JOIN dataset_audio_files AS d FINAL ON d.audio_file_id = a.id"
        filters.append("d.audio_file_id IS NULL")
    elif dataset != "all":
        join = "INNER JOIN dataset_audio_files AS d FINAL ON d.audio_file_id = a.id"
        filters.append("d.dataset_id = {dataset_id:UUID}")
        parameters["dataset_id"] = UUID(dataset)
    if query:
        filters.append(
            "(positionCaseInsensitiveUTF8(a.name, {query:String}) > 0 OR "
            "positionCaseInsensitiveUTF8(toString(a.metadata), {query:String}) > 0)"
        )
        parameters["query"] = query
    if language.strip():
        filters.append("lower(a.language) = lower({language:String})")
        parameters["language"] = language.strip()
    if run_id is not None:
        filters.append("JSONExtractString(a.metadata, 'run_id') = {run_id:String}")
        parameters["run_id"] = run_id
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
            a.storage_kind,
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


def search_audio_file_ids(query: str, dataset: str, language: str) -> list[UUID]:
    return [
        item.id
        for item in list_audio_files(
            limit=4_294_967_295,
            dataset=dataset,
            query=query,
            language=language,
        )
    ]


def list_audio_files_by_run(run_id: str) -> list[AudioFileRecord]:
    return list_audio_files(limit=4_294_967_295, run_id=run_id)


_ORDER_COLUMNS = {
    "updated": ("a.updated_at", "DateTime64(9)"),
    "duration": ("a.duration", "Float64"),
}
