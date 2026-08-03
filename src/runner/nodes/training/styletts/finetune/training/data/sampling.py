import bisect
import math
import random
from dataclasses import dataclass
from typing import Iterator

from torch.utils.data import Sampler


@dataclass(frozen=True)
class DurationBin:
    indices: tuple[int, ...]
    durations: tuple[float, ...]
    total_seconds: float


class DurationBatchSampler(Sampler[list[int]]):
    """Build variable-size batches from one duration bin at a time."""

    def __init__(
        self,
        durations: tuple[float, ...],
        max_seconds: float,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.max_seconds = max_seconds
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.bins = self._build_bins(durations, max_seconds)
        self.total_seconds = sum(duration for item in self.bins for duration in item.durations)
        self.validation_batches = self._pack_validation_batches()

    def __iter__(self) -> Iterator[list[int]]:
        if not self.shuffle:
            yield from self.validation_batches
            return
        randomizer = random.Random(self.seed + self.epoch)
        self.epoch += 1
        weights = [item.total_seconds for item in self.bins]
        for _ in range(len(self)):
            duration_bin = randomizer.choices(self.bins, weights=weights, k=1)[0]
            yield self._sample_batch(duration_bin, randomizer)

    def __len__(self) -> int:
        if not self.shuffle:
            return len(self.validation_batches)
        return max(1, math.ceil(self.total_seconds / self.max_seconds))

    @staticmethod
    def _build_bins(
        durations: tuple[float, ...],
        max_seconds: float,
    ) -> tuple[DurationBin, ...]:
        boundaries = [0.0]
        boundary = 1.0
        while boundary < max_seconds:
            boundaries.append(boundary)
            boundary *= 2
        boundaries.append(max_seconds)
        upper_bounds = boundaries[1:]
        bin_indices: list[list[int]] = [[] for _ in upper_bounds]
        bin_durations: list[list[float]] = [[] for _ in upper_bounds]
        for index, duration in enumerate(durations):
            bin_index = bisect.bisect_left(upper_bounds, duration)
            bin_indices[bin_index].append(index)
            bin_durations[bin_index].append(duration)
        return tuple(
            DurationBin(tuple(indices), tuple(values), sum(values))
            for indices, values in zip(bin_indices, bin_durations, strict=True)
            if indices
        )

    def _sample_batch(
        self,
        duration_bin: DurationBin,
        randomizer: random.Random,
    ) -> list[int]:
        available = list(zip(duration_bin.indices, duration_bin.durations, strict=True))
        batch = []
        remaining = self.max_seconds
        while True:
            eligible = [item for item in available if item[1] <= remaining]
            if not eligible:
                return batch
            selected = randomizer.choices(
                eligible,
                weights=[duration for _, duration in eligible],
                k=1,
            )[0]
            batch.append(selected[0])
            remaining -= selected[1]
            available.remove(selected)

    def _pack_validation_batches(self) -> tuple[list[int], ...]:
        batches = []
        for duration_bin in self.bins:
            batch = []
            batch_seconds = 0.0
            for index, duration in zip(
                duration_bin.indices,
                duration_bin.durations,
                strict=True,
            ):
                if batch and batch_seconds + duration > self.max_seconds:
                    batches.append(batch)
                    batch = []
                    batch_seconds = 0.0
                batch.append(index)
                batch_seconds += duration
            if batch:
                batches.append(batch)
        random.Random(self.seed).shuffle(batches)
        return tuple(batches)
