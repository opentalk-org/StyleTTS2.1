from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import UUID

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from runner.nodes.training.styletts3.testing.vocoder_training.profiles import (
    SignalGeometry,
)
from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.schemas import AudioFileReference

REFERENCE_PAGE_SIZE = 1024
SHUFFLE_BUFFER_SIZE = 2048


@dataclass(frozen=True)
class AudioEntry:
    audio_id: UUID
    path: Path
    frames: int


@dataclass(frozen=True)
class AudioSplits:
    train: list[AudioEntry]
    validation: list[AudioEntry]


def read_full_audio(entry: AudioEntry, signal: SignalGeometry) -> torch.Tensor:
    frames, sample_rate = sf.read(entry.path, dtype="float32", always_2d=True)
    assert sample_rate == signal.sample_rate, f"validation sample rate changed: {entry.path}"
    return torch.from_numpy(np.mean(frames, axis=1, dtype=np.float32))


def inspect_audio_entries(
    sources: list[tuple[UUID, Path]],
    signal: SignalGeometry,
    minimum_frames: int,
) -> list[AudioEntry]:
    entries: list[AudioEntry] = []
    for audio_id, path in sources:
        info = sf.info(path)
        if info.samplerate != signal.sample_rate:
            raise ValueError(
                f"{path} has sample rate {info.samplerate}; expected {signal.sample_rate}"
            )
        if info.frames >= minimum_frames:
            entries.append(AudioEntry(audio_id, path, info.frames))
    return entries


class StreamingCropDataset(IterableDataset[torch.Tensor]):
    """Stream every fixed-size window while opening each source file once per epoch."""

    def __init__(
        self,
        entries: list[AudioEntry],
        shuffle_buffer_size: int,
        segment_samples: int,
    ) -> None:
        self.entries = entries
        self.shuffle_buffer_size = shuffle_buffer_size
        self.segment_samples = segment_samples
        self.epoch = 0

    def __len__(self) -> int:
        return sum(math.ceil(entry.frames / self.segment_samples) for entry in self.entries)

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        base_seed = torch.initial_seed() if worker is None else worker.seed - worker.id
        epoch = self.epoch
        self.epoch += 1

        order_generator = torch.Generator().manual_seed(base_seed + epoch)
        order = torch.randperm(len(self.entries), generator=order_generator).tolist()
        worker_entries = balanced_entry_shards(self.entries, order, worker_count)[worker_id]
        shuffle_generator = torch.Generator().manual_seed(
            base_seed + epoch * worker_count + worker_id
        )
        buffer: list[torch.Tensor] = []
        for entry in worker_entries:
            for segment in _stream_entry(entry, self.segment_samples):
                if len(buffer) < self.shuffle_buffer_size:
                    buffer.append(segment)
                    continue
                index = int(torch.randint(len(buffer), (1,), generator=shuffle_generator).item())
                yield buffer[index]
                buffer[index] = segment
        while buffer:
            index = int(torch.randint(len(buffer), (1,), generator=shuffle_generator).item())
            yield buffer.pop(index)


def balanced_entry_shards(
    entries: list[AudioEntry],
    order: list[int],
    worker_count: int,
) -> list[list[AudioEntry]]:
    """Keep each worker's total audio duration close while retaining shuffled file order."""
    shard_indices: list[list[int]] = [[] for _ in range(worker_count)]
    shard_frames = [0] * worker_count
    ranked_indices = sorted(order, key=lambda index: entries[index].frames, reverse=True)
    for index in ranked_indices:
        worker_id = min(range(worker_count), key=shard_frames.__getitem__)
        shard_indices[worker_id].append(index)
        shard_frames[worker_id] += entries[index].frames

    order_rank = {index: rank for rank, index in enumerate(order)}
    for indices in shard_indices:
        indices.sort(key=order_rank.__getitem__)
    return [[entries[index] for index in indices] for indices in shard_indices]


def _stream_entry(entry: AudioEntry, segment_samples: int) -> Iterator[torch.Tensor]:
    complete_blocks, remainder = divmod(entry.frames, segment_samples)
    with sf.SoundFile(entry.path) as wav_file:
        for _ in range(complete_blocks):
            frames = wav_file.read(segment_samples, dtype="float32", always_2d=True)
            assert frames.shape[0] == segment_samples, f"short sequential read: {entry.path}"
            yield torch.from_numpy(np.mean(frames, axis=1, dtype=np.float32))
        if remainder:
            wav_file.seek(entry.frames - segment_samples)
            frames = wav_file.read(segment_samples, dtype="float32", always_2d=True)
            assert frames.shape[0] == segment_samples, f"short tail read: {entry.path}"
            yield torch.from_numpy(np.mean(frames, axis=1, dtype=np.float32))


def build_train_loader(
    entries: list[AudioEntry],
    batch_size: int,
    workers: int,
    signal: SignalGeometry,
) -> DataLoader[torch.Tensor]:
    dataset = StreamingCropDataset(entries, SHUFFLE_BUFFER_SIZE, signal.segment_samples)
    if workers == 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        multiprocessing_context="spawn",
    )


def prepare_backend_audio(
    dataset_id: UUID,
    cache_dir: Path,
    validation_samples: int,
    max_train_items: int | None,
    signal: SignalGeometry,
) -> AudioSplits:
    references = _list_dataset_references(dataset_id)
    if len(references) <= validation_samples:
        raise ValueError(
            f"dataset {dataset_id} needs more than {validation_samples} non-virtual WAVs"
        )
    train_references = references[:-validation_samples]
    validation_references = references[-validation_samples:]
    if max_train_items is not None:
        train_references = train_references[:max_train_items]

    selected = train_references + validation_references
    _cache_backend_wavs(selected, cache_dir, signal)
    train_sources = [
        (reference.id, _cache_path(cache_dir, reference.id, signal))
        for reference in train_references
    ]
    validation_sources = [
        (reference.id, _cache_path(cache_dir, reference.id, signal))
        for reference in validation_references
    ]
    train = inspect_audio_entries(train_sources, signal, signal.segment_samples)
    validation = inspect_audio_entries(validation_sources, signal, minimum_frames=1)
    if not train:
        raise ValueError(f"no training WAV is at least {signal.segment_samples} samples")
    return AudioSplits(train, validation)


def _list_dataset_references(dataset_id: UUID) -> list[AudioFileReference]:
    references: list[AudioFileReference] = []
    after_id: UUID | None = None
    with database_session() as session:
        while True:
            page = audio_crud.list_audio_file_references_page(
                dataset_id=dataset_id,
                audio_file_ids=None,
                include_virtual=False,
                after_id=after_id,
                limit=REFERENCE_PAGE_SIZE,
            )
            references.extend(page)
            if len(page) < REFERENCE_PAGE_SIZE:
                break
            after_id = page[-1].id
    return references


def _cache_backend_wavs(
    references: list[AudioFileReference],
    cache_dir: Path,
    signal: SignalGeometry,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing_ids = [
        reference.id
        for reference in references
        if not _cache_path(cache_dir, reference.id, signal).is_file()
    ]
    if not missing_ids:
        return
    with database_session() as session:
        locations = audio_crud.audio_bucket_locations(missing_ids)
    grouped: dict[UUID, list[UUID]] = {}
    for location in locations:
        grouped.setdefault(location.bucket_file_id, []).append(location.audio_file_id)
    for audio_ids in grouped.values():
        with database_session() as session:
            wavs = audio_crud.bulk_read_audio_files(session, audio_ids)
        for audio_id, wav_bytes in wavs.items():
            _write_cached_wav(wav_bytes, _cache_path(cache_dir, audio_id, signal), signal)


def _write_cached_wav(wav_bytes: bytes, path: Path, signal: SignalGeometry) -> None:
    frames, source_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    mono = torch.from_numpy(np.mean(frames, axis=1, dtype=np.float32))
    if source_rate != signal.sample_rate:
        mono = resample(mono, source_rate, signal.sample_rate)
    sf.write(path, mono.numpy(), signal.sample_rate, subtype="FLOAT")


def _cache_path(cache_dir: Path, audio_id: UUID, signal: SignalGeometry) -> Path:
    return cache_dir / f"{audio_id}-{signal.sample_rate}.wav"
