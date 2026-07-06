from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

np.random.seed(1)
random.seed(1)

MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300,
}


def load_manifest_lines(train_list_path: str) -> list[str]:
    p = Path(train_list_path)
    raw = p.read_text(encoding="utf-8").splitlines()
    return [ln + "\n" for ln in raw if ln.strip()]


class MelF0Dataset(Dataset):
    def __init__(
        self,
        data_list: list[str],
        *,
        sr: int = 24000,
        data_augmentation: bool = False,
        validation: bool = False,
        verbose: bool = False,
        f0_cache_dir: Path | None = None,
    ) -> None:
        _data_list = [l[:-1].split("|") for l in data_list]
        self.data_list = [d[0] for d in _data_list]
        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, **MEL_PARAMS)
        self.mean, self.std = -4.0, 4.0
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        self.verbose = verbose
        self.zero_value = -10.0
        self.bad_F0 = 5
        self._f0_cache_root = f0_cache_dir

    def __len__(self) -> int:
        return len(self.data_list)

    def _f0_cache_path(self, wav_path: str) -> Path:
        if self._f0_cache_root is not None:
            h = hashlib.sha256(wav_path.encode("utf-8")).hexdigest()[:24]
            return self._f0_cache_root / f"f0_{h}.npy"
        return Path(wav_path + "_f0.npy")

    def path_to_mel_and_label(self, path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wave_tensor = self._load_tensor(path)
        out_file = self._f0_cache_path(path)
        if out_file.is_file():
            f0 = np.load(str(out_file))
        else:
            if self.verbose:
                logger.info("Computing F0 for %s", path)
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.data_list[idx]
        return self.path_to_mel_and_label(data)

    def _load_tensor(self, data: str) -> torch.Tensor:
        wave, sr = sf.read(data)
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
        max_mel_length: int = 192,
        min_mel_length: int = 192,
        mel_length_step: int = 16,
        random_time_crop: bool = True,
    ) -> None:
        self.min_mel_length = min_mel_length
        self.max_mel_length = max_mel_length
        self.mel_length_step = mel_length_step
        self.random_time_crop = random_time_crop

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    train_list_path: str,
    val_list_path: str,
    batch_size: int,
    num_workers: int,
    device_type: str,
    f0_cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_lines = load_manifest_lines(train_list_path)
    val_lines = load_manifest_lines(val_list_path)
    pin = device_type != "cpu"
    train_collate = F0Collater(random_time_crop=True)
    val_collate = F0Collater(random_time_crop=False)
    train_ds = MelF0Dataset(train_lines, data_augmentation=True, validation=False, f0_cache_dir=f0_cache_dir)
    val_ds = MelF0Dataset(val_lines, data_augmentation=False, validation=True, f0_cache_dir=f0_cache_dir)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=train_collate,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        drop_last=False,
        collate_fn=val_collate,
        pin_memory=pin,
    )
    return train_loader, val_loader
