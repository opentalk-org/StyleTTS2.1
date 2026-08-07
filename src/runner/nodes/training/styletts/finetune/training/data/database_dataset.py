import hashlib
import itertools
import math
import random
from uuid import UUID

import torch
from torch.utils.data import IterableDataset

from runner.nodes.text.runtime.symbols import PAD_SYMBOL, TextCleaner
from shared.db import database_session
from shared.db.datasets import crud as dataset_crud

from .sampling import DurationBatchSampler


class DatabaseBatchDataset(IterableDataset):
    def __init__(
        self,
        dataset_id,
        validation_samples,
        validation,
        seed,
        max_audio_seconds,
        max_text_tokens,
        symbols=None,
        plbert_languages=None,
        plbert_modality_id=0,
    ):
        self.dataset_id = UUID(str(dataset_id))
        self.validation_samples = validation_samples
        self.validation = validation
        self.seed = seed
        self.epoch = 0
        self.process_index = 0
        self.process_count = 1
        self.max_audio_seconds = max_audio_seconds
        self.max_text_tokens = max_text_tokens
        self.language_ids = {
            language.lower(): index + 1
            for index, language in enumerate(plbert_languages or ())
        }
        self.modality_id = int(plbert_modality_id)
        self.text_cleaner = TextCleaner(symbols)
        self.boundary_token_id = self.text_cleaner.symbol_index[PAD_SYMBOL]
        self._cached_validation_ids = None
        self._cached_training_plan = None

    def __iter__(self):
        validation_ids = self._validation_ids()
        if self.validation:
            randomizer = random.Random(self.seed)
            batches = self._validation_batches(validation_ids, randomizer)
            yield from self._sharded(batches)
            return
        while True:
            randomizer = random.Random(self.seed + self.epoch)
            self.epoch += 1
            batches = self._training_batches(validation_ids, randomizer)
            yield from self._sharded(batches)

    def _sharded(self, batches):
        for index, batch in enumerate(batches):
            if index % self.process_count == self.process_index:
                yield batch

    def configure_shard(self, process_index: int, process_count: int) -> None:
        self.process_index = process_index
        self.process_count = process_count

    def _validation_ids(self) -> set[UUID]:
        if self._cached_validation_ids is not None:
            return self._cached_validation_ids
        selected = set()
        with database_session() as session:
            rows = dataset_crud.iter_dataset_training_audio(
                session,
                self.dataset_id,
                descending=True,
            )
            for item in rows:
                if self._has_usable_segments(item):
                    selected.add(item.audio_id)
                if len(selected) == self.validation_samples:
                    self._cached_validation_ids = selected
                    return selected
        raise ValueError("training dataset has fewer usable rows than validation_samples")

    def _selected_rows(self, validation_ids, **filters):
        with database_session() as session:
            rows = dataset_crud.iter_dataset_training_audio(
                session,
                self.dataset_id,
                descending=self.validation,
                **filters,
            )
            for item in rows:
                if (item.audio_id in validation_ids) != self.validation:
                    continue
                sample = self._sample(item)
                if sample is not None:
                    yield sample

    def _sample(self, item):
        if not 0 < item.duration <= self.max_audio_seconds:
            return None
        segments = sorted(
            (
                segment for segment in item.segments
                if str(segment["phon"]).strip()
                and float(segment["start"]) < float(segment["end"])
            ),
            key=lambda segment: (
                float(segment["start"]),
                float(segment["end"]),
                str(segment["id"]),
            ),
        )
        if not segments:
            return None
        text = self._text_to_tensor(
            " ".join(str(segment["phon"]).strip() for segment in segments)
        )
        if len(text) > self.max_text_tokens:
            return None
        speaker = item.speaker_id.strip() if item.speaker_id else "0"
        source_id = str(item.metadata.get("source_id", ""))
        source_speaker = source_id.partition("_")[0]
        if (
            item.metadata.get("repository") == "ylacombe/expresso"
            and source_speaker.startswith("ex")
            and source_speaker[2:].isdigit()
        ):
            speaker = source_speaker
        digest = hashlib.blake2b(speaker.encode(), digest_size=8).digest()
        speaker_id = int.from_bytes(digest, "big") % (2**63 - 1)
        return (
            speaker_id,
            self._resolve_language_id(item.language),
            self.modality_id,
            item.audio_id,
            text,
            item.duration,
        )

    @staticmethod
    def _has_usable_segments(item) -> bool:
        return any(
            str(segment["phon"]).strip()
            and float(segment["start"]) < float(segment["end"])
            for segment in item.segments
        )

    def _validation_batches(self, validation_ids, randomizer):
        rows = list(self._selected_rows(validation_ids))
        sampler = DurationBatchSampler(
            tuple(row[5] for row in rows),
            self.max_audio_seconds,
            shuffle=False,
            seed=randomizer.getrandbits(63),
        )
        for indices in sampler:
            yield [rows[index] for index in indices]

    def _training_batches(self, validation_ids, randomizer):
        upper_bounds, base_weights = self._training_plan(validation_ids)
        weights = list(base_weights)
        starts = [UUID(int=randomizer.getrandbits(128)) for _ in upper_bounds]
        iterators = [
            iter(
                itertools.chain(
                    self._selected_rows(
                        validation_ids,
                        duration_above=lower_bound,
                        duration_at_most=upper_bound,
                        excluded_audio_ids=validation_ids,
                        audio_id_after=start,
                    ),
                    self._selected_rows(
                        validation_ids,
                        duration_above=lower_bound,
                        duration_at_most=upper_bound,
                        excluded_audio_ids=validation_ids,
                        audio_id_at_most=start,
                    ),
                )
            )
            for lower_bound, upper_bound, start in zip(
                (0.0, *upper_bounds[:-1]), upper_bounds, starts, strict=True
            )
        ]
        pending = [None for _ in upper_bounds]
        batch_count = max(1, math.ceil(sum(weights) / self.max_audio_seconds))
        produced = 0
        while produced < batch_count and sum(weights) > 0:
            bin_index = randomizer.choices(
                range(len(weights)),
                weights=weights,
                k=1,
            )[0]
            batch = self._sample_training_batch(
                iterators[bin_index], pending, bin_index
            )
            if batch:
                produced += 1
                yield batch
            else:
                weights[bin_index] = 0.0

    def _training_plan(self, validation_ids):
        if self._cached_training_plan is not None:
            return self._cached_training_plan
        with database_session() as session:
            minimum_duration = dataset_crud.dataset_training_minimum_duration(
                session,
                self.dataset_id,
                self.max_audio_seconds,
                validation_ids,
            )
            cardinality_bounds = self._duration_upper_bounds(minimum_duration)
            stats = dataset_crud.dataset_training_duration_totals(
                session,
                self.dataset_id,
                cardinality_bounds,
                validation_ids,
            )
        self._cached_training_plan = self._merge_duration_bins(
            cardinality_bounds,
            stats.total_seconds,
            stats.file_count,
        )
        return self._cached_training_plan

    def _sample_training_batch(self, rows, pending, bin_index):
        batch = []
        remaining = self.max_audio_seconds
        while True:
            row = pending[bin_index]
            if row is None:
                try:
                    row = next(rows)
                except StopIteration:
                    return batch
            if row[5] > remaining:
                pending[bin_index] = row
                return batch
            pending[bin_index] = None
            batch.append(row)
            remaining -= row[5]

    def _duration_upper_bounds(self, minimum_duration):
        maximum_batch_size = max(
            1,
            math.floor(self.max_audio_seconds / minimum_duration),
        )
        return tuple(
            self.max_audio_seconds / batch_size
            for batch_size in range(maximum_batch_size, 0, -1)
        )

    @staticmethod
    def _merge_duration_bins(upper_bounds, weights, file_count):
        target_bin_count = min(
            len(upper_bounds), math.ceil(math.log2(file_count) + 1)
        )
        if target_bin_count == len(upper_bounds):
            return upper_bounds, list(weights)
        target_seconds = sum(weights) / target_bin_count
        merged_bounds = []
        merged_weights = []
        accumulated = 0.0
        for index, (upper_bound, weight) in enumerate(
            zip(upper_bounds, weights, strict=True)
        ):
            accumulated += weight
            last = index == len(upper_bounds) - 1
            full = accumulated >= target_seconds and (
                len(merged_bounds) < target_bin_count - 1
            )
            if full or last:
                merged_bounds.append(upper_bound)
                merged_weights.append(accumulated)
                accumulated = 0.0
        return tuple(merged_bounds), merged_weights

    def _resolve_language_id(self, language: str | None) -> int:
        if not self.language_ids:
            return 0
        if language is None:
            raise ValueError("training audio is missing its language")
        normalized = language.strip().lower().replace("_", "-")
        candidates = (normalized, normalized.split("-", 1)[0])
        for candidate in candidates:
            if candidate in self.language_ids:
                return self.language_ids[candidate]
        raise ValueError(f"training audio language {language!r} is unsupported by PLBERT")

    def _text_to_tensor(self, text: str) -> torch.LongTensor:
        tokens = self.text_cleaner(text)
        tokens.insert(0, self.boundary_token_id)
        tokens.append(self.boundary_token_id)
        return torch.LongTensor(tokens)
