import random
from uuid import UUID

from torch.utils.data import IterableDataset, get_worker_info

from shared.db.datasets import crud as dataset_crud


class DatabaseAudioDataset(IterableDataset):
    def __init__(
        self,
        dataset_id: UUID,
        validation_samples: int,
        validation: bool,
        require_phonemes: bool,
        seed: int = 1,
    ) -> None:
        self.dataset_id = dataset_id
        self.validation = validation
        self.require_phonemes = require_phonemes
        self.seed = seed
        self.epoch = 0
        self.validation_ids = self._validation_ids(validation_samples)
        total = dataset_crud.count_dataset_training_audio(dataset_id)
        self.count = (
            len(self.validation_ids)
            if validation
            else total - len(self.validation_ids)
        )

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        rows = self._selected_rows()
        if not self.validation:
            rows = self._shuffled(rows)
        for index, row in enumerate(rows):
            if index % worker_count == worker_id:
                yield row
        self.epoch += int(not self.validation)

    def _validation_ids(self, count: int) -> set[UUID]:
        if count == 0:
            return set()
        selected = set()
        rows = dataset_crud.iter_dataset_training_audio(
            self.dataset_id,
            descending=True,
        )
        for row in rows:
            if not self.require_phonemes or self.phoneme_text(row):
                selected.add(row.audio_id)
            if len(selected) == count:
                return selected
        raise ValueError("training dataset has fewer usable validation rows")

    def _selected_rows(self):
        rows = dataset_crud.iter_dataset_training_audio(
            self.dataset_id,
            descending=self.validation,
        )
        for row in rows:
            if (row.audio_id in self.validation_ids) != self.validation:
                continue
            if self.require_phonemes and not self.phoneme_text(row):
                continue
            yield row

    def _shuffled(self, rows):
        randomizer = random.Random(self.seed + self.epoch)
        buffer = []
        for row in rows:
            buffer.append(row)
            if len(buffer) == 2_048:
                randomizer.shuffle(buffer)
                yield from buffer
                buffer.clear()
        randomizer.shuffle(buffer)
        yield from buffer

    @staticmethod
    def phoneme_text(row) -> str:
        segments = sorted(
            (
                segment for segment in row.segments
                if str(segment["phon"]).strip()
                and float(segment["start"]) < float(segment["end"])
            ),
            key=lambda segment: (
                float(segment["start"]),
                float(segment["end"]),
                str(segment["id"]),
            ),
        )
        return " ".join(str(segment["phon"]).strip() for segment in segments)
