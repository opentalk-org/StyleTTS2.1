import io
import logging
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader

from shared.db import database_session
from shared.db.audio import crud as audio_crud
from shared.storage import S3RequestMetrics

from .database_dataset import DatabaseBatchDataset
from .pipeline import PrefetchedDataPipeline
from .records import TrainingBatch

logger = logging.getLogger(__name__)

np.random.seed(1)


SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300,
}
MIN_WAVE_SAMPLES = 24_600
AUDIO_PREPROCESS_WORKERS = 8
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(**SPECT_PARAMS, **MEL_PARAMS)
mean, std = -4, 4


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


class Collater:
    def __init__(self, preprocess_workers: int) -> None:
        self._preprocess_pool = ThreadPoolExecutor(
            max_workers=preprocess_workers,
            thread_name_prefix="audio-preprocess",
        )

    def _read_wave(self, wave_bytes: bytes) -> np.ndarray:
        with io.BytesIO(wave_bytes) as source:
            wave, sr = sf.read(source)
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
        padded = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        missing_samples = max(0, MIN_WAVE_SAMPLES - padded.shape[0])
        left_padding = missing_samples // 2
        return np.pad(padded, (left_padding, missing_samples - left_padding))

    def _load_mel(self, wave: np.ndarray) -> torch.Tensor:
        mel_tensor = preprocess(wave).squeeze()
        length_feature = mel_tensor.size(1)
        return mel_tensor[:, :(length_feature - length_feature % 2)]

    def _load_batch_audio(
        self,
        audio_bytes: dict[UUID, bytes],
    ) -> dict[UUID, tuple[np.ndarray, torch.Tensor]]:
        audio_ids = list(audio_bytes)
        processed = self._preprocess_pool.map(
            self._preprocess_audio,
            audio_bytes.values(),
        )
        return dict(zip(audio_ids, processed, strict=True))

    def _preprocess_audio(
        self,
        wave_bytes: bytes,
    ) -> tuple[np.ndarray, torch.Tensor]:
        wave = self._read_wave(wave_bytes)
        return wave, self._load_mel(wave)

    def __call__(self, batch):
        batch_size = len(batch)
        audio_ids = [row[3] for row in batch]

        request_metrics = S3RequestMetrics()
        with database_session() as session:
            audio_bytes = audio_crud.bulk_read_audio_files(
                session,
                list(dict.fromkeys(audio_ids)),
                request_metrics,
            )
        cache = self._load_batch_audio(audio_bytes)
        batch = sorted(
            batch,
            key=lambda row: cache[row[3]][1].size(1),
            reverse=True,
        )
        audio_ids = [row[3] for row in batch]

        max_mel_length = max(cache[audio_id][1].size(1) for audio_id in audio_ids)
        max_text_length = max(row[4].size(0) for row in batch)

        labels = torch.zeros((batch_size)).long()
        language_ids = torch.zeros((batch_size)).long()
        modality_ids = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, 80, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        waves = [None for _ in range(batch_size)]

        for bid, (
            label,
            language_id,
            modality_id,
            audio_id,
            text,
            _duration,
        ) in enumerate(batch):
            wave, mel = cache[audio_id]
            mel_size = mel.size(1)
            text_size = text.size(0)
            labels[bid] = label
            language_ids[bid] = language_id
            modality_ids[bid] = modality_id
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            waves[bid] = wave

        return TrainingBatch(
            waves=tuple(waves),
            audio_durations=tuple(row[5] for row in batch),
            bucket_fetch_seconds=request_metrics.fetch_seconds,
            bucket_fetch_bytes=request_metrics.fetch_bytes,
            bucket_request_seconds=request_metrics.response_seconds,
            bucket_request_count=request_metrics.request_count,
            bucket_error_count=request_metrics.error_count,
            speaker_ids=labels,
            language_ids=language_ids,
            modality_ids=modality_ids,
            texts=texts,
            input_lengths=input_lengths,
            mels=mels,
            mel_lengths=output_lengths,
        )


def build_dataloader(
    dataset_id,
    validation_samples,
    validation=False,
    max_seconds=15.0,
    num_workers=1,
    device='cpu',
    dataset_config={},
    seed=1,
):
    dataset = DatabaseBatchDataset(
        dataset_id,
        validation_samples,
        validation,
        seed,
        max_audio_seconds=max_seconds,
        **dataset_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=Collater(AUDIO_PREPROCESS_WORKERS),
        pin_memory=(device != 'cpu'),
    )
    return PrefetchedDataPipeline(loader)
