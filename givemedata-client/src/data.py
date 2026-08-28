from dataclasses import dataclass, fields

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, IterableDataset

from . import givemedata_pb2 as pb
from .client import GiveMeDataClient

to_mel = torchaudio.transforms.MelSpectrogram(
    n_fft=2048,
    win_length=1200,
    hop_length=300,
    n_mels=80,
)
MEAN, STD = -4, 4
EDGE_PAD_SAMPLES = 5000
MIN_WAVE_SAMPLES = 24_600


@dataclass(frozen=True)
class Batch:
    waves: tuple[np.ndarray, ...]
    audio_durations: tuple[float, ...]
    speaker_ids: torch.Tensor
    language_ids: torch.Tensor
    modality_ids: torch.Tensor
    texts: torch.Tensor
    input_lengths: torch.Tensor
    mels: torch.Tensor
    mel_lengths: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        # the length tensors stay on cpu, the trainer indexes with them
        cpu_fields = {"input_lengths", "mel_lengths"}
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor) and field.name not in cpu_fields
                else value
            )
        return Batch(**values)


def _pad(wave: np.ndarray) -> np.ndarray:
    edge = np.zeros(EDGE_PAD_SAMPLES, dtype=np.float32)
    padded = np.concatenate([edge, wave, edge])
    missing = max(0, MIN_WAVE_SAMPLES - padded.shape[0])
    left = missing // 2
    return np.pad(padded, (left, missing - left))


def _mel(wave: np.ndarray) -> torch.Tensor:
    mel = to_mel(torch.from_numpy(wave).float())
    mel = (torch.log(1e-5 + mel) - MEAN) / STD
    return mel[:, : mel.size(1) - mel.size(1) % 2]


def _collate(response: pb.DataResponse, modality_id: int) -> Batch:
    rows = []
    for sample in response.batch:
        # the server sends 24kHz mono int16 le pcm; scale back to -1..1 floats
        wave = _pad(np.frombuffer(sample.wave, dtype="<i2").astype(np.float32) / 32768.0)
        text = torch.from_numpy(np.frombuffer(sample.text, dtype="<i8").copy())
        rows.append((sample, wave, text, _mel(wave)))
    rows.sort(key=lambda row: row[3].size(1), reverse=True)

    batch_size = len(rows)
    max_mel_length = max(row[3].size(1) for row in rows)
    max_text_length = max(row[2].size(0) for row in rows)

    speaker_ids = torch.zeros(batch_size).long()
    language_ids = torch.zeros(batch_size).long()
    modality_ids = torch.full((batch_size,), modality_id).long()
    mels = torch.zeros((batch_size, 80, max_mel_length)).float()
    texts = torch.zeros((batch_size, max_text_length)).long()
    input_lengths = torch.zeros(batch_size).long()
    mel_lengths = torch.zeros(batch_size).long()

    for bid, (sample, _wave, text, mel) in enumerate(rows):
        speaker_ids[bid] = sample.speaker_id
        language_ids[bid] = sample.language_id
        mels[bid, :, : mel.size(1)] = mel
        texts[bid, : text.size(0)] = text
        input_lengths[bid] = text.size(0)
        mel_lengths[bid] = mel.size(1)

    return Batch(
        waves=tuple(row[1] for row in rows),
        audio_durations=tuple(row[0].duration for row in rows),
        speaker_ids=speaker_ids,
        language_ids=language_ids,
        modality_ids=modality_ids,
        texts=texts,
        input_lengths=input_lengths,
        mels=mels,
        mel_lengths=mel_lengths,
    )


class _StreamDataset(IterableDataset):
    def __init__(
        self,
        client: GiveMeDataClient,
        split: int,
        prefetch: int,
        samples_per_epoch: int | None,
    ) -> None:
        self.client = client
        self.split = split
        self.prefetch = prefetch
        self.samples_per_epoch = samples_per_epoch
        self._stream = None

    def __iter__(self):
        if self._stream is None:
            self._stream = self.client.batches(self.split, self.prefetch)
        if self.samples_per_epoch is None:
            yield from self._stream
            return
        served = 0
        while served < self.samples_per_epoch:
            response = next(self._stream)
            served += len(response.batch)
            yield response


def dataloader(
    client: GiveMeDataClient,
    validation: bool = False,
    prefetch: int = 4,
    device: str = "cpu",
    samples_per_epoch: int | None = None,
    modality_id: int = 0,
) -> DataLoader:
    split = pb.VALIDATION if validation else pb.TRAINING
    return DataLoader(
        _StreamDataset(client, split, prefetch, samples_per_epoch),
        batch_size=None,
        num_workers=0,
        collate_fn=lambda response: _collate(response, modality_id),
        pin_memory=device != "cpu",
    )
