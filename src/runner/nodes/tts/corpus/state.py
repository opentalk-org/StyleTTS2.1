from uuid import UUID

from shared.db import database_session
from shared.db.datasets import crud as dataset_crud


def completed_source_keys(
    dataset_id: UUID,
    expected_name: str,
) -> set[str]:
    with database_session() as session:
        datasets = {
            dataset.id: dataset
            for dataset in dataset_crud.list_datasets(session)
        }
        if dataset_id not in datasets:
            raise KeyError(f"Dataset not found: {dataset_id}")
        dataset = datasets[dataset_id]
        if dataset.name != expected_name:
            raise ValueError(
                f"dataset {dataset_id} is {dataset.name!r}, "
                f"expected {expected_name!r}"
            )
        return dataset_crud.list_dataset_metadata_values(
            session,
            dataset_id,
            "tts_source_key",
        )
