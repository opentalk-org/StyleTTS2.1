from uuid import UUID

from shared.db.datasets import crud as dataset_crud


def completed_source_keys(
    dataset_id: UUID,
    expected_name: str,
) -> set[str]:
    datasets = {dataset.id: dataset for dataset in dataset_crud.list_datasets()}
    if dataset_id not in datasets:
        raise KeyError(f"Dataset not found: {dataset_id}")
    dataset = datasets[dataset_id]
    if dataset.name != expected_name:
        raise ValueError(
            f"dataset {dataset_id} is {dataset.name!r}, expected {expected_name!r}"
        )
    return dataset_crud.list_dataset_metadata_values(dataset_id, "tts_source_key")
