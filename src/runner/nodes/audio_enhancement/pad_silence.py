from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, stable_id

RMS_BIN_MS = 5


class PadSilenceSettings(StrictSettings):
    silence_threshold: float = Field(ge=0.0, le=1.0)
    start_silence: int = Field(ge=0, title="Start silence (ms)")
    end_silence: int = Field(ge=0, title="End silence (ms)")


@dataclass(frozen=True)
class PaddedAudio:
    wav_bytes: bytes
    duration: float
    sample_rate: int
    channels: int


class PadSilenceNode(Node):
    NODE_TYPE = "PadSilence"
    DESCRIPTION = "Trim leading and trailing silence from an audio clip and replace it with a fixed amount of silence at each end. Takes audio in and outputs the same audio with exactly the requested start and end silence padding (in milliseconds); the silence threshold controls how quiet counts as silence. Use it to give clips uniform, controlled margins."
    CATEGORY = "Audio"
    SETTINGS = PadSilenceSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64)

    async def execute(self, batch, context):
        audios = [inputs["audio"] for inputs in batch]
        assert all(isinstance(audio, Audio) for audio in audios), "pad silence inputs must be Audio"
        for audio in audios:
            assert audio.data is not None, f"audio bytes are required: {audio.id}"
        context.check_cancel()
        results = pad_silence_wav_bytes_batch(
            [audio.data for audio in audios],
            self.settings,
        )
        outputs = []
        for audio, result in zip(audios, results, strict=True):
            context.check_cancel()
            padded_id = stable_id("audio", audio.id, "pad_silence", self.settings.model_dump())
            outputs.append({
                "audio": replace(
                    audio,
                    data=result.wav_bytes,
                    sample_rate=result.sample_rate,
                    channels=result.channels,
                    start=0.0,
                    end=result.duration,
                    id=padded_id,
                    byte_length=len(result.wav_bytes),
                    metadata={
                        **audio.metadata,
                        "duration": result.duration,
                        "sample_rate": result.sample_rate,
                        "channels": result.channels,
                        "byte_length": len(result.wav_bytes),
                        "pad_silence": {
                            **self.settings.model_dump(),
                            "rms_bin_ms": RMS_BIN_MS,
                        },
                    },
                ),
            })
        return outputs


def pad_silence_wav_bytes(
    audio_bytes: bytes,
    settings: PadSilenceSettings,
    dependencies: tuple[Any, Any] | None = None,
) -> PaddedAudio:
    np, sf = dependencies if dependencies is not None else _audio_dependencies()
    samples, sample_rate = sf.read(BytesIO(audio_bytes), always_2d=True, dtype="float32")
    sample_rate = int(sample_rate)
    content = _non_silent_content(np, samples, sample_rate, settings.silence_threshold)
    start_samples = int(round(sample_rate * settings.start_silence / 1000.0))
    end_samples = int(round(sample_rate * settings.end_silence / 1000.0))
    padded = np.concatenate([
        np.zeros((start_samples, samples.shape[1]), dtype=np.float32),
        content,
        np.zeros((end_samples, samples.shape[1]), dtype=np.float32),
    ])
    output = BytesIO()
    sf.write(output, padded, sample_rate, format="WAV", subtype="PCM_16")
    return PaddedAudio(
        wav_bytes=output.getvalue(),
        duration=float(len(padded)) / float(sample_rate),
        sample_rate=sample_rate,
        channels=int(samples.shape[1]),
    )


def pad_silence_wav_bytes_batch(
    audio_batch: list[bytes],
    settings: PadSilenceSettings,
) -> list[PaddedAudio]:
    dependencies = _audio_dependencies()
    return [
        pad_silence_wav_bytes(audio_bytes, settings, dependencies)
        for audio_bytes in audio_batch
    ]


def _non_silent_content(np: Any, samples: Any, sample_rate: int, threshold: float) -> Any:
    if len(samples) == 0:
        return samples
    bin_samples = max(1, int(round(sample_rate * RMS_BIN_MS / 1000.0)))
    frame_power = np.mean(np.square(samples.astype(np.float64)), axis=1)
    bin_starts = np.arange(0, len(samples), bin_samples)
    bin_lengths = np.minimum(bin_samples, len(samples) - bin_starts)
    bin_rms = np.sqrt(np.add.reduceat(frame_power, bin_starts) / bin_lengths)
    active_bins = np.flatnonzero(bin_rms > threshold)
    if active_bins.size == 0:
        return samples[:0]
    content_start = int(active_bins[0]) * bin_samples
    content_end = min(len(samples), (int(active_bins[-1]) + 1) * bin_samples)
    return samples[content_start:content_end]


def _audio_dependencies() -> tuple[Any, Any]:
    try:
        return importlib.import_module("numpy"), importlib.import_module("soundfile")
    except ImportError as error:
        raise ImportError("PadSilence requires optional audio dependencies: numpy, soundfile") from error
