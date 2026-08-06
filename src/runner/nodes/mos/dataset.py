from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
from uuid import UUID

from pydantic import BaseModel, Field
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from runner.nodes.mos.audio import MosFeatureExtractor, MosInputs, prepare_audio_batch
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.mos import crud as mos_crud
from shared.db.mos.models import MosComparison


class MosPairRow(BaseModel):
    comparison_id: UUID
    audio_a_id: UUID
    audio_b_id: UUID
    score_a: float = Field(allow_inf_nan=False)
    score_b: float = Field(allow_inf_nan=False)
    preferred_audio_id: UUID

    @property
    def preferred_sign(self) -> float:
        if self.preferred_audio_id == self.audio_a_id:
            return 1.0
        if self.preferred_audio_id == self.audio_b_id:
            return -1.0
        raise ValueError(
            f"MOS preference is outside pair: {self.comparison_id}"
        )


class MosPairIterableDataset(IterableDataset[MosPairRow]):
    def __init__(
        self,
        dataset_id: UUID,
        validation_comparisons: int,
        validation: bool,
    ):
        super().__init__()
        self.dataset_id = dataset_id
        self.validation = validation
        with database_session() as session:
            comparison_count = mos_crud.count_comparisons(session, dataset_id)
        if comparison_count < 2:
            raise ValueError(
                f"MOS training requires at least two comparisons: {dataset_id}"
            )
        self.validation_count = min(
            validation_comparisons,
            comparison_count - 1,
        )
        self.train_count = comparison_count - self.validation_count
        self.count = self.validation_count if validation else self.train_count

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[MosPairRow]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        with database_session() as session:
            rows = mos_crud.iter_comparisons(session, self.dataset_id)
            for index, comparison in enumerate(rows):
                selected = index >= self.train_count
                if selected != self.validation:
                    continue
                if index % worker_count == worker_id:
                    yield _pair_row(comparison)


@dataclass(frozen=True)
class MosPairBatch:
    inputs_a: MosInputs
    inputs_b: MosInputs
    score_a: torch.Tensor
    score_b: torch.Tensor
    preferred_sign: torch.Tensor

    def to(self, device: torch.device) -> MosPairBatch:
        return MosPairBatch(
            inputs_a=self.inputs_a.to(device),
            inputs_b=self.inputs_b.to(device),
            score_a=self.score_a.to(device),
            score_b=self.score_b.to(device),
            preferred_sign=self.preferred_sign.to(device),
        )


class MosPairCollator:
    def __init__(self, feature_extractor: MosFeatureExtractor):
        self.feature_extractor = feature_extractor

    def __call__(self, rows: list[MosPairRow]) -> MosPairBatch:
        audio_ids = list(dict.fromkeys(
            [row.audio_a_id for row in rows] + [row.audio_b_id for row in rows]
        ))
        with database_session() as session:
            audio_bytes = audio_crud.bulk_read_audio_files(session, audio_ids)
        combined = prepare_audio_batch(
            self.feature_extractor,
            [audio_bytes[row.audio_a_id] for row in rows] + [audio_bytes[row.audio_b_id] for row in rows],
        )
        count = len(rows)
        return MosPairBatch(
            inputs_a=MosInputs(combined.input_values[:count], combined.attention_mask[:count]),
            inputs_b=MosInputs(combined.input_values[count:], combined.attention_mask[count:]),
            score_a=torch.tensor([row.score_a for row in rows], dtype=torch.float32),
            score_b=torch.tensor([row.score_b for row in rows], dtype=torch.float32),
            preferred_sign=torch.tensor([row.preferred_sign for row in rows], dtype=torch.float32),
        )


def build_mos_dataloader(
    dataset_id: UUID,
    validation_comparisons: int,
    validation: bool,
    feature_extractor: MosFeatureExtractor,
    batch_size: int,
    workers: int,
) -> DataLoader[MosPairBatch]:
    return DataLoader(
        MosPairIterableDataset(
            dataset_id,
            validation_comparisons,
            validation,
        ),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=MosPairCollator(feature_extractor),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _pair_row(comparison: MosComparison) -> MosPairRow:
    return MosPairRow(
        comparison_id=comparison.id,
        audio_a_id=comparison.audio_a_id,
        audio_b_id=comparison.audio_b_id,
        score_a=comparison.score_a,
        score_b=comparison.score_b,
        preferred_audio_id=comparison.preferred_audio_id,
    )
