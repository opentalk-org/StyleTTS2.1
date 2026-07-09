from __future__ import annotations

from io import BytesIO
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy
from runner.nodes.datatypes import AudioPort, JsonPort
from runner.nodes.models import Audio
from runner.nodes.statistics.segments import speech_segment_records


FRAME_LENGTH = 2048


class AudioFeatureSettings(StrictSettings):
    silence_threshold_db: float = Field(default=-40.0, ge=-80.0, le=0.0)
    hop_length: int = Field(default=512, ge=64, le=4096)


class AnalyzeAudioFeaturesNode(Node):
    NODE_TYPE = "AnalyzeAudioFeatures"
    CATEGORY = "Audio"
    SETTINGS = AudioFeatureSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"feature_records": JsonPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64)

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            features = analyze_audio_features(audio, self.settings.silence_threshold_db, self.settings.hop_length)
            speech = speech_segment_records(audio)
            features["segments"] = speech["segments"]
            features["duplicate_segments_collapsed"] = speech["duplicate_segments_collapsed"]
            outputs.append({"feature_records": features})
        return outputs


def analyze_audio_features(audio: Audio, silence_threshold_db: float, hop_length: int) -> dict[str, Any]:
    librosa, np = _audio_dependencies()
    y, sr = librosa.load(BytesIO(audio.data), sr=None, mono=True)
    y = np.asarray(y, dtype=np.float64)
    sr_int = int(sr)
    duration = float(len(y) / float(sr_int)) if sr_int > 0 else audio.duration
    features = _empty_features(np, audio, sr_int, duration)
    if y.size == 0:
        return features
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=hop_length)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    silent = rms_db < float(silence_threshold_db)
    silence_rms_db = rms_db[silent] if silent.any() else np.array([], dtype=np.float64)
    nonsilent = rms_db[~silent]
    frame_min, frame_max, frame_mean = _frame_min_max_mean(librosa, np, y, hop_length)
    clip_top = int(np.sum(y >= 0.999))
    clip_bottom = int(np.sum(y <= -0.999))
    features.update(
        {
            "duration": duration,
            "sample_rate": sr_int,
            "rms_db": _float_list(np, rms_db),
            "silence_ratio": float(np.mean(silent)) if silent.size else 1.0,
            "silence_rms_db": _float_list(np, silence_rms_db),
            "mean_rms_db_nonsilent": _optional_float(np, np.mean(nonsilent)) if nonsilent.size else None,
            "rms_db_nonsilent_samples": _sample_rms_db(np, y, silence_threshold_db),
            "frame_value_min": _float_list(np, frame_min),
            "frame_value_max": _float_list(np, frame_max),
            "frame_value_mean": _float_list(np, frame_mean),
            "clip_top": clip_top,
            "clip_bottom": clip_bottom,
            "has_clip": clip_top + clip_bottom > 0,
            "silence_intervals_seconds": _silence_intervals_seconds(librosa, silent, sr_int, hop_length, duration),
            "signal_duration_seconds": duration,
        }
    )
    return features


def _audio_dependencies():
    # Optional heavy imports stay inside execution so registry/schema import works in light environments.
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise ImportError("AnalyzeAudioFeatures requires optional dependencies 'librosa' and 'numpy' to execute") from exc
    return librosa, np


def _empty_features(np, audio: Audio, sample_rate: int, duration: float) -> dict[str, Any]:
    return {
        "audio_file_id": str(audio.audio_file_id),
        "name": audio.name,
        "duration": duration,
        "sample_rate": sample_rate,
        "rms_db": [],
        "silence_ratio": 1.0,
        "silence_rms_db": [],
        "mean_rms_db_nonsilent": None,
        "rms_db_nonsilent_samples": None,
        "frame_value_min": [],
        "frame_value_max": [],
        "frame_value_mean": [],
        "clip_top": 0,
        "clip_bottom": 0,
        "has_clip": False,
        "silence_intervals_seconds": [],
        "signal_duration_seconds": duration,
        "channels": audio.channels,
        "start": audio.start,
        "end": audio.end,
        "source_batch_id": audio.metadata["source_batch_id"],
        "source_batch_count": audio.metadata["source_batch_count"],
        "metadata": _json_value(audio.metadata),
    }


def _frame_min_max_mean(librosa, np, y, hop_length: int):
    if y.size < FRAME_LENGTH:
        return np.array([float(np.min(y))]), np.array([float(np.max(y))]), np.array([float(np.mean(y))])
    framed = librosa.util.frame(y, frame_length=FRAME_LENGTH, hop_length=hop_length, axis=-1)
    return np.min(framed, axis=0), np.max(framed, axis=0), np.mean(framed, axis=0)


def _silence_intervals_seconds(librosa, silent, sample_rate: int, hop_length: int, duration: float) -> list[list[float]]:
    intervals: list[list[float]] = []
    index = 0
    while index < len(silent):
        if not bool(silent[index]):
            index += 1
            continue
        end = index + 1
        while end < len(silent) and bool(silent[end]):
            end += 1
        start_time = float(librosa.frames_to_time(index, sr=sample_rate, hop_length=hop_length, n_fft=FRAME_LENGTH))
        end_time = duration if end == len(silent) else float(librosa.frames_to_time(end, sr=sample_rate, hop_length=hop_length, n_fft=FRAME_LENGTH))
        intervals.append([max(0.0, min(duration, start_time)), max(0.0, min(duration, end_time))])
        index = end
    return [interval for interval in intervals if interval[1] > interval[0]]


def _sample_rms_db(np, y, silence_threshold_db: float) -> float | None:
    threshold = 10.0 ** (float(silence_threshold_db) / 20.0)
    nonsilent = np.abs(y) >= threshold
    if not bool(np.any(nonsilent)):
        return None
    rms = float(np.sqrt(np.mean(np.square(y[nonsilent]))))
    return _optional_float(np, 20.0 * np.log10(max(rms, 1e-20)))


def _float_list(np, values) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float) if bool(np.isfinite(value))]


def _optional_float(np, value) -> float | None:
    number = float(value)
    if bool(np.isfinite(number)):
        return number
    return None


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value
