from __future__ import annotations

import hashlib
import io
import logging
import random
from pathlib import Path
from uuid import UUID

import numpy as np
import pyworld as pw
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader

from runner.nodes.training.common.database_dataset import DatabaseAudioDataset
from shared.db import database_session
from shared.db.audio import crud as audio_crud

logger = logging.getLogger(__name__)

np.random.seed(1)
random.seed(1)

MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300,
}


class MelF0Processor:
    def __init__(
        self,
        *,
        sr: int = 24000,
        data_augmentation: bool = False,
        validation: bool = False,
        verbose: bool = False,
        f0_cache_dir: Path | None = None,
    ) -> None:
        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, **MEL_PARAMS)
        self.mean, self.std = -4.0, 4.0
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        self.verbose = verbose
        self.zero_value = -10.0
        self.bad_F0 = 5
        self._f0_cache_root = f0_cache_dir

    def _f0_cache_path(self, audio_id: UUID) -> Path:
        if self._f0_cache_root is not None:
            h = hashlib.sha256(str(audio_id).encode("utf-8")).hexdigest()[:24]
            return self._f0_cache_root / f"f0_{h}.npy"
        raise ValueError("F0 cache directory is required")

    def process(self, audio_id: UUID, wave_bytes: bytes) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wave_tensor = self._load_tensor(wave_bytes)
        out_file = self._f0_cache_path(audio_id)
        if out_file.is_file():
            f0 = np.load(str(out_file))
        else:
            if self.verbose:
                logger.info("Computing F0 for %s", audio_id)
            x = wave_tensor.numpy().astype(np.float64)
            frame_period = MEL_PARAMS["hop_length"] * 1000 / self.sr
            _f0, t = pw.harvest(x, self.sr, frame_period=frame_period)
            if np.sum(_f0 != 0) < self.bad_F0:
                _f0, t = pw.dio(x, self.sr, frame_period=frame_period)
            f0 = pw.stonemask(x, _f0, t, self.sr)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_file), f0)

        f0 = torch.from_numpy(np.asarray(f0, dtype=np.float32)).float()

        if self.data_augmentation:
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(torch.clamp(mel_tensor, min=1e-5)) - self.mean) / self.std
        mel_length = mel_tensor.size(1)

        f0_zero = f0 == 0
        is_silence = torch.zeros(f0.shape)
        is_silence[f0_zero] = 1.0

        if mel_length > self.max_mel_length:
            random_start = int(np.random.randint(0, mel_length - self.max_mel_length))
            mel_tensor = mel_tensor[:, random_start : random_start + self.max_mel_length]
            f0 = f0[random_start : random_start + self.max_mel_length]
            is_silence = is_silence[random_start : random_start + self.max_mel_length]

        if torch.any(torch.isnan(f0)):
            f0 = f0.clone()
            f0[torch.isnan(f0)] = self.zero_value

        return mel_tensor, f0, is_silence

    def _load_tensor(self, wave_bytes: bytes) -> torch.Tensor:
        with io.BytesIO(wave_bytes) as source:
            wave, sr = sf.read(source)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        wave_tensor = torch.from_numpy(wave.astype(np.float32)).float()
        if sr != self.sr:
            wave_tensor = torchaudio.functional.resample(wave_tensor, sr, self.sr)
        return wave_tensor


class F0Collater:
    def __init__(
        self,
        *,
        processor: MelF0Processor,
        max_mel_length: int = 192,
        min_mel_length: int = 192,
        mel_length_step: int = 16,
        random_time_crop: bool = True,
    ) -> None:
        self.min_mel_length = min_mel_length
        self.max_mel_length = max_mel_length
        self.mel_length_step = mel_length_step
        self.random_time_crop = random_time_crop
        self.processor = processor

    def __call__(self, rows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_ids = [row.audio_id for row in rows]
        with database_session() as session:
            audio_bytes = audio_crud.bulk_read_audio_files(session, audio_ids)
        batch = [
            self.processor.process(audio_id, audio_bytes[audio_id])
            for audio_id in audio_ids
        ]
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length), dtype=torch.float32)
        f0s = torch.zeros((batch_size, self.max_mel_length), dtype=torch.float32)
        is_silences = torch.zeros((batch_size, self.max_mel_length), dtype=torch.float32)

        for bid, (mel, f0, is_silence) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            f0s[bid, :mel_size] = f0
            is_silences[bid, :mel_size] = is_silence

        if self.random_time_crop and self.max_mel_length > self.min_mel_length:
            random_slice = (
                int(
                    np.random.randint(
                        self.min_mel_length // self.mel_length_step,
                        1 + self.max_mel_length // self.mel_length_step,
                    )
                )
                * self.mel_length_step
                + self.min_mel_length
            )
            random_slice = min(random_slice, self.max_mel_length)
            mels = mels[:, :, :random_slice]
            f0s = f0s[:, :random_slice]
            is_silences = is_silences[:, :random_slice]

        mels = mels.unsqueeze(1)
        return mels, f0s, is_silences


def build_f0_dataloaders(
    *,
    dataset_id: UUID,
    validation_samples: int,
    batch_size: int,
    num_workers: int,
    device_type: str,
    f0_cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    pin = device_type != "cpu"
    train_processor = MelF0Processor(data_augmentation=True, validation=False, f0_cache_dir=f0_cache_dir)
    val_processor = MelF0Processor(data_augmentation=False, validation=True, f0_cache_dir=f0_cache_dir)
    train_collate = F0Collater(random_time_crop=True, processor=train_processor)
    val_collate = F0Collater(random_time_crop=False, processor=val_processor)
    train_ds = DatabaseAudioDataset(dataset_id, validation_samples, False, False)
    val_ds = DatabaseAudioDataset(dataset_id, validation_samples, True, False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=train_collate,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=max(0, num_workers // 2),
        drop_last=False,
        collate_fn=val_collate,
        pin_memory=pin,
    )
    return train_loader, val_loader
