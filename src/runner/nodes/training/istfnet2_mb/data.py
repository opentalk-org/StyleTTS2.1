from __future__ import annotations

import io
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import UUID

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchaudio.functional import resample

from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.db.audio.schemas import AudioFileReference

from .config import SignalConfig

REFERENCE_PAGE_SIZE = 1_024
SHUFFLE_BUFFER_SIZE = 2_048


@dataclass(frozen=True)
class AudioEntry:
    audio_id: UUID
    path: Path
    frames: int


@dataclass(frozen=True)
class AudioSplits:
    training: list[AudioEntry]
    validation: list[AudioEntry]


class StreamingAudioDataset(IterableDataset[torch.Tensor]):
    def __init__(self, entries: list[AudioEntry], segment_samples: int) -> None:
        self.entries = entries
        self.segment_samples = segment_samples
        self.epoch = 0

    def __len__(self) -> int:
        return sum(
            math.ceil(entry.frames / self.segment_samples)
            for entry in self.entries
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        worker_seed = torch.initial_seed() if worker is None else worker.seed
        generator = torch.Generator().manual_seed(worker_seed + self.epoch)
        self.epoch += 1
        order = torch.randperm(len(self.entries), generator=generator).tolist()
        entries = [
            self.entries[index]
            for index in order[worker_id::worker_count]
        ]
        buffer: list[Tensor] = []
        for entry in entries:
            for segment in stream_segments(entry, self.segment_samples):
                if len(buffer) < SHUFFLE_BUFFER_SIZE:
                    buffer.append(segment)
                    continue
                index = int(
                    torch.randint(
                        len(buffer),
                        (1,),
                        generator=generator,
                    ).item()
                )
                yield buffer[index]
                buffer[index] = segment
        while buffer:
            index = int(
                torch.randint(
                    len(buffer),
                    (1,),
                    generator=generator,
                ).item()
            )
            yield buffer.pop(index)


def stream_segments(entry: AudioEntry, segment_samples: int) -> Iterator[Tensor]:
    complete, remainder = divmod(entry.frames, segment_samples)
    with sf.SoundFile(entry.path) as wav_file:
        for _ in range(complete):
            frames = wav_file.read(
                segment_samples,
                dtype="float32",
                always_2d=True,
            )
            assert frames.shape[0] == segment_samples
            yield torch.from_numpy(np.mean(frames, axis=1, dtype=np.float32))
        if remainder:
            wav_file.seek(max(0, entry.frames - segment_samples))
            frames = wav_file.read(
                segment_samples,
                dtype="float32",
                always_2d=True,
            )
            if frames.shape[0] < segment_samples:
                frames = np.pad(
                    frames,
                    ((0, segment_samples - frames.shape[0]), (0, 0)),
                )
            yield torch.from_numpy(np.mean(frames, axis=1, dtype=np.float32))


def training_loader(
    entries: list[AudioEntry],
    batch_size: int,
    workers: int,
    config: SignalConfig,
) -> DataLoader[Tensor]:
    dataset = StreamingAudioDataset(entries, config.segment_samples)
    if workers == 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        multiprocessing_context="spawn",
    )


def prepare_audio(
    dataset_id: UUID,
    cache_dir: Path,
    validation_samples: int,
    max_train_items: int | None,
    config: SignalConfig,
) -> AudioSplits:
    references = list_references(dataset_id)
    if len(references) <= validation_samples:
        raise ValueError(
            f"dataset {dataset_id} needs more than {validation_samples} audio files"
        )
    training = references[:-validation_samples]
    validation = references[-validation_samples:]
    if max_train_items is not None:
        training = training[:max_train_items]
    cache_audio(training + validation, cache_dir, config.sample_rate)
    return AudioSplits(
        inspect_entries(training, cache_dir, config),
        inspect_entries(validation, cache_dir, config),
    )


def list_references(dataset_id: UUID) -> list[AudioFileReference]:
    references = []
    after_id = None
    with database_session() as session:
        while True:
            page = audio_crud.list_audio_file_references_page(
                session,
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


def cache_audio(
    references: list[AudioFileReference],
    cache_dir: Path,
    sample_rate: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        reference.id
        for reference in references
        if not cache_path(cache_dir, reference.id, sample_rate).is_file()
    ]
    if not missing:
        return
    with database_session() as session:
        locations = audio_crud.audio_bucket_locations(session, missing)
    grouped: defaultdict[UUID, list[UUID]] = defaultdict(list)
    for location in locations:
        grouped[location.bucket_file_id].append(location.audio_file_id)
    for audio_ids in grouped.values():
        with database_session() as session:
            wavs = audio_crud.bulk_read_audio_files(session, audio_ids)
        for audio_id, content in wavs.items():
            write_audio(
                content,
                cache_path(cache_dir, audio_id, sample_rate),
                sample_rate,
            )


def write_audio(content: bytes, path: Path, sample_rate: int) -> None:
    frames, source_rate = sf.read(
        io.BytesIO(content),
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(np.mean(frames, axis=1, dtype=np.float32))
    if source_rate != sample_rate:
        waveform = resample(waveform, source_rate, sample_rate)
    sf.write(path, waveform.numpy(), sample_rate, subtype="FLOAT")


def inspect_entries(
    references: list[AudioFileReference],
    cache_dir: Path,
    config: SignalConfig,
) -> list[AudioEntry]:
    entries = []
    for reference in references:
        path = cache_path(cache_dir, reference.id, config.sample_rate)
        info = sf.info(path)
        entries.append(AudioEntry(reference.id, path, info.frames))
    return entries


def cache_path(cache_dir: Path, audio_id: UUID, sample_rate: int) -> Path:
    return cache_dir / f"{audio_id}-{sample_rate}.wav"
