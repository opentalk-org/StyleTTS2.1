from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from random import Random
from typing import Iterator, Sequence

import pyarrow.parquet as pq
import torch
from torch import Tensor

from shared.db.audio import crud as audio_crud
from shared.db.connection import database_session
from shared.db.datasets import crud as dataset_crud

from .assets import BertAsset
from .config import DataConfig


@dataclass(frozen=True)
class TextPhonemeRow:
    language: str
    text: str
    phonemes: str


@dataclass(frozen=True)
class BackendAudioRow:
    audio_id: object
    language: str
    text: str
    phonemes: str
    wav_bytes: bytes


class BackendPartition(Enum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclass(frozen=True)
class TokenBatch:
    input_ids: Tensor
    attention_mask: Tensor
    decoder_input_ids: Tensor
    labels: Tensor
    language_ids: Tensor

    def to(self, device: torch.device) -> "TokenBatch":
        return TokenBatch(*(value.to(device) for value in self.__dict__.values()))


class Codec:
    def __init__(self, asset: BertAsset) -> None:
        self.symbols = asset.symbols
        self.symbol_ids = {symbol: index for index, symbol in enumerate(asset.symbols)}
        self.language_ids = {language: index + 1 for index, language in enumerate(asset.languages)}
        self.bos_id = len(asset.symbols)
        self.eos_id = len(asset.symbols) + 1
        self.pad_id = 0

    def text_ids(self, text: str) -> list[int]:
        return [3 + value for value in text.encode("utf-8")]

    def phoneme_ids(self, phonemes: str) -> list[int]:
        ids = []
        position = 0
        multichar = sorted((symbol for symbol in self.symbols if len(symbol) > 1), key=len, reverse=True)
        while position < len(phonemes):
            symbol = next((item for item in multichar if phonemes.startswith(item, position)), phonemes[position])
            ids.append(self.symbol_ids.get(symbol, self.symbol_ids["[UNK]"]))
            position += len(symbol)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        ignored = {self.pad_id, self.bos_id, self.eos_id}
        return "".join(self.symbols[index] for index in ids if index not in ignored and index < len(self.symbols))


def download_parquets(config: DataConfig, validation: bool = False) -> tuple[Path, ...]:
    remote_dir = config.remote_validation_dir if validation else config.remote_train_dir
    count = config.validation_files if validation else config.train_files
    local_dir = config.cache_dir / ("validation" if validation else "train")
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_paths = _list_remote(config.host, remote_dir)[:count]
    paths = []
    for remote in remote_paths:
        target = local_dir / PurePosixPath(remote).name
        if not target.is_file():
            _download(config.host, remote, target)
        paths.append(target)
    return tuple(paths)


def parquet_rows(paths: Sequence[Path], config: DataConfig) -> Iterator[TextPhonemeRow]:
    for path in paths:
        pending = []
        byte_count = 0
        phoneme_count = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["text", "phonemes", "language_ranges"], batch_size=4096):
            for packed in batch.to_pylist():
                for row in _unpack_row(packed, config):
                    row_bytes = len(row.text.encode("utf-8"))
                    row_phonemes = len(row.phonemes)
                    exceeds_text = byte_count + row_bytes + 1 > config.packed_text_bytes
                    exceeds_phonemes = phoneme_count + row_phonemes + 1 > config.max_phonemes
                    if pending and (exceeds_text or exceeds_phonemes):
                        yield _join_rows(pending, config.language)
                        pending = []
                        byte_count = 0
                        phoneme_count = 0
                    pending.append(row)
                    byte_count += row_bytes + int(len(pending) > 1)
                    phoneme_count += row_phonemes + int(len(pending) > 1)
        if pending:
            yield _join_rows(pending, config.language)


def shuffled_batches(rows: Iterator[TextPhonemeRow], batch_size: int, seed: int, buffer_size: int = 4096):
    rng = Random(seed)
    buffer = []
    for row in rows:
        buffer.append(row)
        if len(buffer) == buffer_size:
            rng.shuffle(buffer)
            yield from _chunks(buffer, batch_size)
            buffer.clear()
    rng.shuffle(buffer)
    yield from _chunks(buffer, batch_size)


def collate(rows: Sequence[TextPhonemeRow], codec: Codec) -> TokenBatch:
    text = [torch.tensor(codec.text_ids(row.text), dtype=torch.long) for row in rows]
    targets = [torch.tensor(codec.phoneme_ids(row.phonemes) + [codec.eos_id], dtype=torch.long) for row in rows]
    decoder = [torch.cat((torch.tensor([codec.bos_id]), target[:-1])) for target in targets]
    input_ids = torch.nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=codec.pad_id)
    decoder_ids = torch.nn.utils.rnn.pad_sequence(decoder, batch_first=True, padding_value=codec.pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
    languages = torch.tensor([codec.language_ids[row.language] for row in rows])
    return TokenBatch(input_ids, input_ids.ne(codec.pad_id), decoder_ids, labels, languages)


def backend_audio_batches(
    dataset_id,
    batch_size: int,
    language: str,
    partition: BackendPartition,
) -> Iterator[list[BackendAudioRow]]:
    cursor = None
    while True:
        with database_session() as session:
            pending = []
            rows = dataset_crud.iter_dataset_training_audio(session, dataset_id, audio_id_after=cursor)
            for row in rows:
                cursor = row.audio_id
                validation_row = row.audio_id.int % 10 == 0
                if validation_row != (partition is BackendPartition.VALIDATION):
                    continue
                if row.language != language:
                    continue
                text = " ".join(str(segment["text"]).strip() for segment in row.segments)
                phonemes = " ".join(str(segment["phon"]).strip() for segment in row.segments)
                if text and phonemes:
                    pending.append((row, text, phonemes))
                if len(pending) == batch_size:
                    break
            if not pending:
                return
            payloads = audio_crud.bulk_read_audio_files(session, [item.audio_id for item, _, _ in pending])
            batch = [
                BackendAudioRow(item.audio_id, language, text, phonemes, payloads[item.audio_id])
                for item, text, phonemes in pending
            ]
        if len(batch) < batch_size:
            return
        yield batch


def _chunks(rows: list[TextPhonemeRow], size: int):
    for offset in range(0, len(rows) - size + 1, size):
        yield rows[offset : offset + size]


def _join_rows(rows: list[TextPhonemeRow], language: str) -> TextPhonemeRow:
    return TextPhonemeRow(
        language,
        " ".join(row.text for row in rows),
        " ".join(row.phonemes for row in rows),
    )


def _unpack_row(row: dict, config: DataConfig) -> Iterator[TextPhonemeRow]:
    texts = str(row["text"]).split("<m/>")
    phonemes = str(row["phonemes"]).split("<m/>")
    if len(texts) != len(phonemes):
        raise ValueError("packed PL-BERT text and phoneme counts differ")
    ranges = row["language_ranges"]
    offset = 0
    for text, phoneme in zip(texts, phonemes, strict=True):
        language = next(item["lang"] for item in ranges if item["start"] <= offset < item["end"])
        text = text.strip()
        phoneme = phoneme.strip()
        offset += len(phoneme) + len("<m/>")
        if language == config.language and len(text.encode("utf-8")) <= config.max_text_bytes and len(phoneme) <= config.max_phonemes:
            yield TextPhonemeRow(language, text, phoneme)


def _list_remote(host: str, directory: str) -> list[str]:
    result = subprocess.run(["sftp", "-q", "-oBatchMode=yes", "-b", "-", host], input=f"ls -1 {directory}/*.parquet\n", text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"cannot list {host}:{directory}: {result.stderr.strip()}")
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".parquet"))


def _download(host: str, remote: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    result = subprocess.run(["sftp", "-q", "-oBatchMode=yes", "-b", "-", host], input=f"get {remote} {temporary}\n", text=True, capture_output=True, check=False)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"cannot download {host}:{remote}: {result.stderr.strip()}")
    temporary.replace(target)
