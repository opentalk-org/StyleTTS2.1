from uuid import UUID, uuid4

from shared.db.clickhouse import clickhouse_client, delete_rows
from shared.db.mos.clickhouse.models import MosComparisonRecord, MosPairIds

def create_comparison(item: MosComparisonRecord) -> MosComparisonRecord:
    if item.audio_a_id == item.audio_b_id:
        raise ValueError("MOS comparison requires two distinct audio files")
    if item.preferred_audio_id not in (item.audio_a_id, item.audio_b_id):
        raise ValueError("MOS preferred audio must belong to the comparison")
    clickhouse_client().insert(
        "mos_comparisons",
        [
            [
                item.id,
                item.updated_at,
                item.dataset_id,
                item.audio_a_id,
                item.audio_b_id,
                item.preferred_audio_id,
                item.score_a,
                item.score_b,
                item.created_at,
            ]
        ],
        column_names=[
            "id",
            "updated_at",
            "dataset_id",
            "audio_a_id",
            "audio_b_id",
            "preferred_audio_id",
            "score_a",
            "score_b",
            "created_at",
        ],
    )
    return get_comparison(item.id)


def update_comparison(item: MosComparisonRecord) -> MosComparisonRecord:
    current = get_comparison(item.id)
    if current.dataset_id != item.dataset_id or current.created_at != item.created_at:
        raise ValueError("MOS dataset and creation time are immutable")
    return create_comparison(item)


def get_comparison(comparison_id: UUID) -> MosComparisonRecord:
    result = clickhouse_client().query(
        """
        SELECT
            id,
            updated_at,
            dataset_id,
            audio_a_id,
            audio_b_id,
            preferred_audio_id,
            score_a,
            score_b,
            created_at
        FROM mos_comparisons FINAL
        WHERE id = {id:UUID}
        """,
        parameters={"id": comparison_id},
    )
    rows = list(result.named_results())
    if not rows:
        raise KeyError(f"MOS comparison not found: {comparison_id}")
    return MosComparisonRecord.model_validate(rows[0])


def list_comparisons(
    dataset_id: UUID,
    limit: int,
    offset: int = 0,
) -> list[MosComparisonRecord]:
    result = clickhouse_client().query(
        """
        SELECT
            id,
            updated_at,
            dataset_id,
            audio_a_id,
            audio_b_id,
            preferred_audio_id,
            score_a,
            score_b,
            created_at
        FROM mos_comparisons FINAL
        WHERE dataset_id = {dataset_id:UUID}
        ORDER BY created_at DESC, id DESC
        LIMIT {limit:UInt32} OFFSET {offset:UInt64}
        """,
        parameters={"dataset_id": dataset_id, "limit": limit, "offset": offset},
    )
    return [MosComparisonRecord.model_validate(row) for row in result.named_results()]


def delete_comparison(comparison_id: UUID) -> None:
    delete_rows(
        clickhouse_client(),
        "mos_comparisons",
        "id = {id:UUID}",
        {"id": comparison_id},
    )


def sample_pair(dataset_id: UUID) -> MosPairIds:
    threshold = uuid4()
    audio_ids = _pair_ids(dataset_id, threshold, ">=")
    if len(audio_ids) < 2:
        audio_ids.extend(_pair_ids(dataset_id, threshold, "<", 2 - len(audio_ids)))
    if len(audio_ids) != 2:
        raise ValueError(
            f"Dataset does not contain two eligible audio files: {dataset_id}"
        )
    return MosPairIds(
        dataset_id=dataset_id, audio_a_id=audio_ids[0], audio_b_id=audio_ids[1]
    )


def _pair_ids(
    dataset_id: UUID,
    threshold: UUID,
    operator: str,
    limit: int = 2,
) -> list[UUID]:
    assert operator in ("<", ">="), "unsupported UUID comparison"
    result = clickhouse_client().query(
        f"""
        SELECT d.audio_file_id
        FROM dataset_audio_files AS d FINAL
        INNER JOIN (
            SELECT
                id,
                argMax(virtual, updated_at) AS virtual
            FROM audio_files
            GROUP BY id
        ) AS a ON a.id = d.audio_file_id
        WHERE d.dataset_id = {{dataset_id:UUID}}
          AND a.virtual = false
          AND d.audio_file_id {operator} {{threshold:UUID}}
        ORDER BY d.audio_file_id
        LIMIT {{limit:UInt32}}
        """,
        parameters={"dataset_id": dataset_id, "threshold": threshold, "limit": limit},
    )
    return [row[0] for row in result.result_rows]
