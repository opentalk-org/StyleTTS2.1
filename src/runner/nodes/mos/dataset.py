from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from runner.nodes.mos.audio import MosFeatureExtractor, MosInputs, prepare_audio_batch
from runner.nodes.mos.manifest import MosManifestRow
from shared.db import database_session
from shared.db.audio import crud as audio_crud


class MosPairIterableDataset(IterableDataset[MosManifestRow]):
    def __init__(self, path: Path, count: int):
        super().__init__()
        self.path = path
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[MosManifestRow]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        with self.path.open("r", encoding="utf-8") as source:
            for index, line in enumerate(source):
                if index % worker_count == worker_id:
                    yield MosManifestRow.model_validate_json(line)


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

    def __call__(self, rows: list[MosManifestRow]) -> MosPairBatch:
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
    path: Path,
    count: int,
    feature_extractor: MosFeatureExtractor,
    batch_size: int,
    workers: int,
) -> DataLoader[MosPairBatch]:
    return DataLoader(
        MosPairIterableDataset(path, count),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=MosPairCollator(feature_extractor),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
