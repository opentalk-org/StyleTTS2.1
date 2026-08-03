import io
import logging
from pathlib import Path
from uuid import UUID

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader

from runner.nodes.text.runtime.symbols import PAD_SYMBOL, TextCleaner
from shared.db import database_session
from shared.db.audio import crud as audio_crud

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


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_list,
        max_audio_seconds,
        max_text_tokens,
        symbols=None,
        plbert_languages=None,
        plbert_modality_id=0,
    ):
        _data_list = [l.strip().split('|') for l in data_list]
        rows = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        audio_ids = [UUID(Path(row[0]).stem) for row in rows]
        with database_session() as session:
            audio_files = audio_crud.get_audio_files_bulk(session, audio_ids)
        self.language_ids = {
            language.lower(): index + 1
            for index, language in enumerate(plbert_languages or ())
        }
        self.audio_language_ids = {
            audio_id: self._resolve_language_id(audio_files[audio_id].language)
            for audio_id in audio_ids
        }
        self.modality_id = int(plbert_modality_id)
        self.text_cleaner = TextCleaner(symbols)
        self.boundary_token_id = self.text_cleaner.symbol_index[PAD_SYMBOL]
        self.data_list = [
            row
            for row, audio_id in zip(rows, audio_ids, strict=True)
            if audio_files[audio_id].duration <= max_audio_seconds
            and len(self._text_to_tensor(row[1])) <= max_text_tokens
        ]
        skipped = len(rows) - len(self.data_list)
        logger.info(
            "training data filter max_audio_seconds=%s max_text_tokens=%s "
            "accepted=%s skipped=%s",
            max_audio_seconds,
            max_text_tokens,
            len(self.data_list),
            skipped,
        )
        if not self.data_list:
            raise ValueError(
                "no audio remains after applying max_audio_seconds="
                f"{max_audio_seconds} and max_text_tokens={max_text_tokens}; "
                f"skipped={skipped}"
        )
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_path, text, speaker_id_text = self.data_list[idx]
        audio_id = self._audio_id(data_path)
        return (
            int(speaker_id_text),
            self.audio_language_ids[audio_id],
            self.modality_id,
            audio_id,
            self._text_to_tensor(text),
        )

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
        raise ValueError(
            f"training audio language {language!r} is unsupported by PLBERT"
        )

    def _audio_id(self, wave_path: str) -> UUID:
        return UUID(Path(wave_path).stem)

    def _text_to_tensor(self, text: str) -> torch.LongTensor:
        tokens = self.text_cleaner(text)
        tokens.insert(0, self.boundary_token_id)
        tokens.append(self.boundary_token_id)
        return torch.LongTensor(tokens)


class Collater:
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

    def _load_batch_audio(self, audio_bytes: dict[UUID, bytes]) -> dict[UUID, tuple[np.ndarray, torch.Tensor]]:
        cached: dict[UUID, tuple[np.ndarray, torch.Tensor]] = {}
        for audio_id, wave_bytes in audio_bytes.items():
            wave = self._read_wave(wave_bytes)
            mel = self._load_mel(wave)
            cached[audio_id] = (wave, mel)
        return cached

    def __call__(self, batch):
        batch_size = len(batch)
        audio_ids = [row[3] for row in batch]

        with database_session() as session:
            audio_bytes = audio_crud.bulk_read_audio_files(
                session,
                list(dict.fromkeys(audio_ids)),
            )
        cache = self._load_batch_audio(audio_bytes)

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
            speaker_ids=labels,
            language_ids=language_ids,
            modality_ids=modality_ids,
            texts=texts,
            input_lengths=input_lengths,
            mels=mels,
            mel_lengths=output_lengths,
        )


def build_dataloader(
    path_list,
    validation=False,
    batch_size=4,
    num_workers=1,
    device='cpu',
    dataset_config={},
    seed=1,
):
    dataset = FilePathDataset(
        path_list,
        **dataset_config,
    )
    collate_fn = Collater()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != 'cpu'),
    )
    return PrefetchedDataPipeline(loader)
