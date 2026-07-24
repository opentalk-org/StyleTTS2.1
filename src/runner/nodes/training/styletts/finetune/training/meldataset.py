import io
import random
from pathlib import Path
from uuid import UUID

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader

from runner.nodes.text.runtime.symbols import TextCleaner
from shared.db import database_session
from shared.db.audio import crud as audio_crud

np.random.seed(1)
random.seed(1)


SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300,
}
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
        root_path,
        sr=24000,
        data_augmentation=False,
        validation=False,
        OOD_data="Data/OOD_texts.txt",
        min_length=50,
        symbols=None,
        stream_cache=None,
    ):
        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner(symbols)
        self.sr = sr

        self.rows_by_speaker: dict[int, list[list[str]]] = {}
        for row in self.data_list:
            self.rows_by_speaker.setdefault(int(row[2]), []).append(row)

        self.min_mel_length = 192
        self.max_mel_length = 192
        self.data_augmentation = data_augmentation and (not validation)
        self.min_length = min_length

        with open(OOD_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        idx = 1 if '.wav' in lines[0].split('|')[0] else 0
        self.ptexts = [line.split('|')[idx] for line in lines]

        self.root_path = root_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        speaker_id, audio_id, text, ref_audio_id, ref_text, ref_label = self._resolve_row(idx)
        return speaker_id, audio_id, text, ref_audio_id, ref_text, ref_label

    def _reference_data(self, speaker_id: int) -> list[str]:
        return random.choice(self.rows_by_speaker[speaker_id])

    def _audio_id(self, wave_path: str) -> UUID:
        return UUID(Path(wave_path).stem)

    def _text_to_tensor(self, text: str) -> torch.LongTensor:
        tokens = self.text_cleaner(text)
        tokens.insert(0, 0)
        tokens.append(0)
        return torch.LongTensor(tokens)

    def _resolve_row(self, idx):
        data_path, text, speaker_id_text = self.data_list[idx]
        speaker_id = int(speaker_id_text)
        audio_id = self._audio_id(data_path)
        reference_row = self._reference_data(speaker_id)
        ref_audio_id = self._audio_id(reference_row[0])
        ref_label = int(reference_row[2])
        ps = ""
        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]
        return (
            speaker_id,
            audio_id,
            self._text_to_tensor(text),
            ref_audio_id,
            self._text_to_tensor(ps),
            ref_label,
        )


class Collater:
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave

    def _read_wave(self, wave_bytes: bytes) -> np.ndarray:
        with io.BytesIO(wave_bytes) as source:
            wave, sr = sf.read(source)
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
        return np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)

    def _load_mel(self, wave: np.ndarray) -> torch.Tensor:
        mel_tensor = preprocess(wave).squeeze()
        length_feature = mel_tensor.size(1)
        return mel_tensor[:, :(length_feature - length_feature % 2)]

    def _load_ref_mel(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]
        return mel_tensor

    def _load_batch_audio(self, audio_bytes: dict[UUID, bytes]) -> dict[UUID, tuple[np.ndarray, torch.Tensor]]:
        cached: dict[UUID, tuple[np.ndarray, torch.Tensor]] = {}
        for audio_id, wave_bytes in audio_bytes.items():
            wave = self._read_wave(wave_bytes)
            mel = self._load_mel(wave)
            cached[audio_id] = (wave, mel)
        return cached

    def __call__(self, batch):
        batch_size = len(batch)
        audio_ids = [row[1] for row in batch]
        ref_audio_ids = [row[3] for row in batch]
        requested_ids = list(dict.fromkeys(audio_ids + ref_audio_ids))

        with database_session() as session:
            audio_bytes = audio_crud.bulk_read_audio_files(session, requested_ids)
        cache = self._load_batch_audio(audio_bytes)

        max_mel_length = max(cache[audio_id][1].size(1) for audio_id in audio_ids)
        max_text_length = max(row[2].size(0) for row in batch)
        max_rtext_length = max(row[4].size(0) for row in batch)

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, 80, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, 80, self.max_mel_length)).float()
        waves = [None for _ in range(batch_size)]

        for bid, (label, audio_id, text, ref_audio_id, ref_text, ref_label) in enumerate(batch):
            wave, mel = cache[audio_id]
            _, ref_mel = cache[ref_audio_id]
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            ref_mel = self._load_ref_mel(ref_mel)

            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            ref_mels[bid, :, : ref_mel.size(1)] = ref_mel
            waves[bid] = wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels


def build_dataloader(
    path_list,
    root_path,
    validation=False,
    OOD_data="Data/OOD_texts.txt",
    min_length=50,
    batch_size=4,
    num_workers=1,
    device='cpu',
    collate_config={},
    dataset_config={},
    stream_cache=None,
):
    dataset = FilePathDataset(
        path_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        validation=validation,
        stream_cache=stream_cache,
        **dataset_config,
    )
    collate_fn = Collater(**collate_config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != 'cpu'),
    )
