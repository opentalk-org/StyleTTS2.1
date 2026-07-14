from __future__ import annotations

import warnings
from io import BytesIO
from uuid import UUID

import librosa
import numpy as np

from runner.nodes.models import Audio
from shared.db import database_session
from shared.db.audio import crud as audio_crud


TARGET_SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 8 * TARGET_SAMPLE_RATE


def load_audio_bytes(audios: list[Audio]) -> list[bytes]:
    missing_ids = [audio.audio_file_id for audio in audios if audio.data is None]
    stored: dict[UUID, bytes] = {}
    if missing_ids:
        with database_session() as session:
            stored = audio_crud.bulk_read_audio_files(session, missing_ids)
    return [audio.data if audio.data is not None else stored[audio.audio_file_id] for audio in audios]


def prepare_waveforms(audios: list[Audio], payloads: list[bytes]) -> list[np.ndarray]:
    return [prepare_waveform(audio, data) for audio, data in zip(audios, payloads, strict=True)]


def prepare_waveform(audio: Audio, data: bytes) -> np.ndarray:
    if audio.duration <= 0.0:
        raise ValueError(f"SmartTurnPredict requires non-empty audio: {audio.id}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        waveform, _sample_rate = librosa.load(BytesIO(data), sr=TARGET_SAMPLE_RATE, mono=True)
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise ValueError(f"SmartTurnPredict requires non-empty audio: {audio.id}")
    samples = samples[-WINDOW_SAMPLES:]
    if samples.size < WINDOW_SAMPLES:
        samples = np.pad(samples, (WINDOW_SAMPLES - samples.size, 0))
    return np.asarray(samples, dtype=np.float32)
