from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID

from shared.db.assets.clickhouse.models import (
    AssetKind,
    AssetRecord,
    BucketFileRecord,
    ConfigRecord,
)
from shared.db.clickhouse import clickhouse_client, delete_rows


def create_bucket_files(items: Sequence[BucketFileRecord]) -> None:
    if not items:
        return
    clickhouse_client().insert(
        "bucket_files",
        [[item.id, item.kind.value, item.path, item.size] for item in items],
        column_names=["id", "kind", "path", "size"],
    )


def get_bucket_file(bucket_file_id: UUID) -> BucketFileRecord:
    result = clickhouse_client().query(
        """
        SELECT id, kind, path, size
        FROM bucket_files
        WHERE id = {id:UUID}
        """,
        parameters={"id": bucket_file_id},
    )
    rows = list(result.named_results())
    if not rows:
        raise KeyError(f"Bucket file not found: {bucket_file_id}")
    return BucketFileRecord.model_validate(rows[0])


def get_bucket_files(bucket_file_ids: Sequence[UUID]) -> list[BucketFileRecord]:
    ids = list(dict.fromkeys(bucket_file_ids))
    if not ids:
        return []
    result = clickhouse_client().query(
        """
        SELECT id, kind, path, size
        FROM bucket_files
        WHERE id IN {ids:Array(UUID)}
        """,
        parameters={"ids": ids},
    )
    return [BucketFileRecord.model_validate(row) for row in result.named_results()]


def list_bucket_files() -> list[BucketFileRecord]:
    result = clickhouse_client().query(
        """
        SELECT id, kind, path, size
        FROM bucket_files
        ORDER BY id
        """
    )
    return [BucketFileRecord.model_validate(row) for row in result.named_results()]


def delete_bucket_file(bucket_file_id: UUID) -> None:
    delete_bucket_files([bucket_file_id])


def delete_bucket_files(bucket_file_ids: Sequence[UUID]) -> None:
    ids = list(dict.fromkeys(bucket_file_ids))
    if not ids:
        return
    delete_rows(
        clickhouse_client(),
        "bucket_files",
        "id IN {ids:Array(UUID)}",
        {"ids": ids},
    )


def referenced_bucket_file_ids(bucket_file_ids: Sequence[UUID]) -> set[UUID]:
    ids = list(dict.fromkeys(bucket_file_ids))
    if not ids:
        return set()
    result = clickhouse_client().query(
        """
        SELECT bucket_file_id
        FROM (
            SELECT argMax(bucket_file_id, updated_at) AS bucket_file_id
            FROM audio_files
            WHERE id IN (
                SELECT DISTINCT id
                FROM audio_files
                WHERE bucket_file_id IN {ids:Array(UUID)}
            )
            GROUP BY id
            HAVING bucket_file_id IN {ids:Array(UUID)}
            UNION DISTINCT
            SELECT argMax(pack_id, updated_at) AS bucket_file_id
            FROM audio_waveforms
            WHERE audio_file_id IN (
                SELECT DISTINCT audio_file_id
                FROM audio_waveforms
                WHERE pack_id IN {ids:Array(UUID)}
            )
            GROUP BY audio_file_id
            HAVING bucket_file_id IN {ids:Array(UUID)}
        )
        """,
        parameters={"ids": ids},
    )
    return {row[0] for row in result.result_rows}


def create_assets(items: Sequence[AssetRecord]) -> None:
    if not items:
        return
    rows = [
        [
            item.id,
            item.updated_at,
            item.kind.value,
            item.name,
            item.path,
            item.size,
            item.content_hash,
            item.type,
            item.metadata,
            item.run_id,
        ]
        for item in items
    ]
    clickhouse_client().insert(
        "assets",
        rows,
        column_names=[
            "id",
            "updated_at",
            "kind",
            "name",
            "path",
            "size",
            "content_hash",
            "type",
            "metadata",
            "run_id",
        ],
    )


def update_asset(item: AssetRecord) -> AssetRecord:
    current = get_asset(item.id)
    if item.updated_at <= current.updated_at:
        item = item.model_copy(
            update={"updated_at": current.updated_at + timedelta(microseconds=1)}
        )
    create_assets([item])
    return get_asset(item.id)


def get_asset(asset_id: UUID) -> AssetRecord:
    result = clickhouse_client().query(
        """
        SELECT id, updated_at, kind, name, path, size, content_hash, type, metadata, run_id
        FROM assets FINAL
        WHERE id = {id:UUID}
        """,
        parameters={"id": asset_id},
    )
    rows = list(result.named_results())
    if not rows:
        raise KeyError(f"Asset not found: {asset_id}")
    return AssetRecord.model_validate(rows[0])


def list_assets(
    kind: AssetKind | None = None, type_: str | None = None
) -> list[AssetRecord]:
    filters = []
    parameters: dict[str, object] = {}
    if kind is not None:
        filters.append("kind = {kind:String}")
        parameters["kind"] = kind.value
    if type_ is not None:
        filters.append("type = {type:String}")
        parameters["type"] = type_
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    result = clickhouse_client().query(
        f"""
        SELECT id, updated_at, kind, name, path, size, content_hash, type, metadata, run_id
        FROM assets FINAL
        {where}
        ORDER BY name, id
        """,
        parameters=parameters,
    )
    return [AssetRecord.model_validate(row) for row in result.named_results()]


def delete_asset(asset_id: UUID) -> None:
    delete_rows(clickhouse_client(), "assets", "id = {id:UUID}", {"id": asset_id})


def create_config(item: ConfigRecord) -> ConfigRecord:
    clickhouse_client().insert(
        "configs",
        [[item.id, item.updated_at, item.name, item.type, item.metadata]],
        column_names=["id", "updated_at", "name", "type", "metadata"],
    )
    return get_config(item.id)


def update_config(item: ConfigRecord) -> ConfigRecord:
    current = get_config(item.id)
    if item.updated_at <= current.updated_at:
        item = item.model_copy(
            update={"updated_at": current.updated_at + timedelta(microseconds=1)}
        )
    return create_config(item)


def get_config(config_id: UUID) -> ConfigRecord:
    result = clickhouse_client().query(
        """
        SELECT id, updated_at, name, type, metadata
        FROM configs FINAL
        WHERE id = {id:UUID}
        """,
        parameters={"id": config_id},
    )
    rows = list(result.named_results())
    if not rows:
        raise KeyError(f"Config not found: {config_id}")
    return ConfigRecord.model_validate(rows[0])


def list_configs(type_: str | None = None) -> list[ConfigRecord]:
    where = "WHERE type = {type:String}" if type_ is not None else ""
    result = clickhouse_client().query(
        f"""
        SELECT id, updated_at, name, type, metadata
        FROM configs FINAL
        {where}
        ORDER BY name, id
        """,
        parameters={"type": type_} if type_ is not None else None,
    )
    return [ConfigRecord.model_validate(row) for row in result.named_results()]


def delete_config(config_id: UUID) -> None:
    delete_rows(clickhouse_client(), "configs", "id = {id:UUID}", {"id": config_id})
