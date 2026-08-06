from __future__ import annotations

import io
from uuid import UUID

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from runner.nodes.text.runtime.symbols import TextCleaner, build_word_index_dictionary
from runner.nodes.training.common.database_dataset import DatabaseAudioDataset
from shared.db import database_session
from shared.db.audio import crud as audio_crud


class AsrPhonemeMelProcessor:
    def __init__(
        self,
        *,
        phoneme_symbols: list[str],
        mel_params: dict,
        sr: int = 24000,
        data_augmentation: bool = False,
        ctc_blank_character: str = " ",
    ) -> None:
        self.sr = sr
        sym = build_word_index_dictionary(list(phoneme_symbols))
        self.text_cleaner = TextCleaner(sym)
        self._blank_edge = int(sym[ctc_blank_character[:1] if ctc_blank_character else " "])
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**mel_params)
        self.mean, self.std = -4.0, 4.0
        self.data_augmentation = data_augmentation

    def _load_wave(self, wave_bytes: bytes) -> np.ndarray:
        with io.BytesIO(wave_bytes) as source:
            wave, file_sr = sf.read(source)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        wave_f = wave.astype(np.float32)
        if file_sr != self.sr:
            t = torch.from_numpy(wave_f)
            t = torchaudio.functional.resample(t, file_sr, self.sr)
            wave_f = t.numpy().astype(np.float32)
        return wave_f

    def process(self, row, wave_bytes: bytes) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        wave = self._load_wave(wave_bytes)
        wave_t = torch.from_numpy(wave).float()
        mel = self.to_melspec(wave_t)
        ph_str = DatabaseAudioDataset.phoneme_text(row)
        text_ix = self.text_cleaner(ph_str)
        text_ix.insert(0, self._blank_edge)
        text_ix.append(self._blank_edge)
        text_tensor = torch.LongTensor(text_ix)
        if (text_tensor.size(0) + 1) >= (mel.size(1) // 3):
            mel = F.interpolate(
                mel.unsqueeze(0),
                size=(int(text_tensor.size(0)) + 1) * 3,
                align_corners=False,
                mode="linear",
            ).squeeze(0)
        acoustic = (torch.log(torch.clamp(mel, min=1e-5)) - self.mean) / self.std
        tlen = acoustic.size(1)
        acoustic = acoustic[:, : (tlen - (tlen % 2))]
        if self.data_augmentation:
            scale = 0.5 + 0.5 * float(np.random.random())
            acoustic = acoustic * scale
        return wave_t, acoustic, text_tensor


class AsrCollater:
    def __init__(self, processor: AsrPhonemeMelProcessor) -> None:
        self.text_pad_index = 0
        self.processor = processor

    def __call__(
        self, rows
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_ids = [row.audio_id for row in rows]
        with database_session() as session:
            audio_bytes = audio_crud.bulk_read_audio_files(session, audio_ids)
        batch = [
            self.processor.process(row, audio_bytes[row.audio_id])
            for row in rows
        ]
        batch_size = len(batch)
        lengths = [b[1].shape[1] for b in batch]
        order = np.argsort(lengths)[::-1]
        batch = [batch[i] for i in order]

        nmels = batch[0][1].size(0)
        max_mel = max(int(b[1].shape[1]) for b in batch)
        max_text = max(int(b[2].shape[0]) for b in batch)

        mels = torch.zeros((batch_size, nmels, max_mel), dtype=torch.float32)
        texts = torch.zeros((batch_size, max_text), dtype=torch.long)
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        output_lengths = torch.zeros(batch_size, dtype=torch.long)

        for bid, (_, mel, text) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size

        return texts, input_lengths, mels, output_lengths


def build_asr_dataloaders(
    *,
    dataset_id: UUID,
    validation_samples: int,
    effective_config: dict,
    batch_size: int,
    num_workers: int,
    device_type: str,
) -> tuple[DataLoader, DataLoader]:
    symbols = list(effective_config.get("data_params", {}).get("phoneme_symbols") or [])
    if not symbols:
        raise ValueError("asr_phoneme_symbols_missing")
    pp = effective_config.get("preprocess_params") or {}
    sr = int(pp.get("sr", 24000))
    mel_params = dict(effective_config.get("mel_params") or {})
    dp = effective_config.get("data_params") or {}
    blank_ch = str(dp.get("ctc_blank_character") or " ")[:1]
    pin = device_type != "cpu"
    train_processor = AsrPhonemeMelProcessor(
        phoneme_symbols=symbols,
        mel_params=mel_params,
        sr=sr,
        data_augmentation=True,
        ctc_blank_character=blank_ch,
    )
    val_processor = AsrPhonemeMelProcessor(
        phoneme_symbols=symbols,
        mel_params=mel_params,
        sr=sr,
        data_augmentation=False,
        ctc_blank_character=blank_ch,
    )
    train_ds = DatabaseAudioDataset(dataset_id, validation_samples, False, True)
    val_ds = DatabaseAudioDataset(dataset_id, validation_samples, True, True)
    train_collate = AsrCollater(train_processor)
    val_collate = AsrCollater(val_processor)
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
