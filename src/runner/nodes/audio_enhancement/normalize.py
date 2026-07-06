from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runner.nodes.datatypes import AUDIO
from runner.nodes.models import Audio, stable_id

TARGET_SR = 24_000


class NormalizeSettings(StrictSettings):
    target_lufs: float = Field(default=-23.0, ge=-40.0, le=-6.0)
    target_rms_db: float = Field(default=-20.0, ge=-40.0, le=0.0)
    silence_threshold_db: float = Field(default=-40.0, ge=-80.0, le=0.0)
    padding_ms: int = Field(default=120, ge=0, le=1000)
    prevent_clipping: bool = True
    peak_cap_percent: int = Field(default=95, ge=50, le=100)


@dataclass(frozen=True)
class NormalizedAudio:
    wav_bytes: bytes
    duration: float
    leading_pad_seconds: float
    sample_rate: int
    channels: int


def normalize_wav_bytes(audio_bytes: bytes, settings: NormalizeSettings) -> NormalizedAudio:
    deps = _load_normalize_dependencies()
    np = deps["numpy"]
    librosa = deps["librosa"]
    sf = deps["soundfile"]

    y, sr = sf.read(BytesIO(audio_bytes), always_2d=True, dtype="float32")
    mono = np.mean(y, axis=1).astype(np.float32, copy=False)
    if mono.size == 0:
        padded = _empty_audio(np, settings.padding_ms)
        leading_pad_seconds = _empty_audio_leading_pad_seconds(settings.padding_ms)
    else:
        if int(sr) != TARGET_SR:
            mono = librosa.resample(mono, orig_sr=int(sr), target_sr=TARGET_SR).astype(np.float32)
        normalized = _normalize_rms_db(
            np,
            mono,
            settings.target_rms_db,
            silence_threshold_db=settings.silence_threshold_db,
            prevent_peak_clip=settings.prevent_clipping,
            peak_cap_percent=settings.peak_cap_percent,
        )
        padded, leading_pad_seconds = _pad_edges_silence(
            np,
            normalized,
            TARGET_SR,
            settings.padding_ms,
            settings.silence_threshold_db,
        )
    out = BytesIO()
    sf.write(out, np.asarray(padded, dtype=np.float32), TARGET_SR, format="WAV", subtype="PCM_16")
    return NormalizedAudio(
        wav_bytes=out.getvalue(),
        duration=float(len(padded)) / float(TARGET_SR),
        leading_pad_seconds=leading_pad_seconds,
        sample_rate=TARGET_SR,
        channels=1,
    )


class NormalizeLoudnessNode(Node):
    NODE_TYPE = "NormalizeLoudness"
    CATEGORY = "Audio / Enhancement"
    SETTINGS = NormalizeSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO)}

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            result = normalize_wav_bytes(audio.data, self.settings)
            metadata = {
                **audio.metadata,
                "duration": result.duration,
                "leading_pad_seconds": result.leading_pad_seconds,
                "sample_rate": result.sample_rate,
                "channels": result.channels,
                "normalize": {
                    "target_rms_db": self.settings.target_rms_db,
                    "silence_threshold_db": self.settings.silence_threshold_db,
                    "padding_ms": self.settings.padding_ms,
                    "prevent_clipping": self.settings.prevent_clipping,
                    "peak_cap_percent": self.settings.peak_cap_percent,
                },
            }
            normalized_id = stable_id("audio", audio.id, "normalize", self.settings.model_dump())
            outputs.append({
                "audio": replace(
                    audio,
                    data=result.wav_bytes,
                    sample_rate=result.sample_rate,
                    channels=result.channels,
                    start=0.0,
                    end=result.duration,
                    id=normalized_id,
                    metadata=metadata,
                ),
            })
        return outputs


def _load_normalize_dependencies() -> dict[str, Any]:
    modules = {}
    missing = []
    for name in ["numpy", "librosa", "soundfile"]:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        names = ", ".join(missing)
        raise ImportError(f"NormalizeLoudness requires optional audio dependencies: {names}") from None
    return modules


def _rms_non_silent_samples(np: Any, y: Any, silence_threshold_db: float) -> float:
    y64 = np.asarray(y, dtype=np.float64)
    if y64.size == 0:
        return 0.0
    mag = np.abs(y64)
    db = 20.0 * np.log10(np.maximum(mag, 1e-20))
    mask = db >= silence_threshold_db
    if not np.any(mask):
        return 0.0
    sel = y64[mask]
    return float(np.sqrt(np.mean(sel * sel) + 1e-20))


def _normalize_rms_db(
    np: Any,
    y: Any,
    target_rms_db: float,
    *,
    silence_threshold_db: float,
    prevent_peak_clip: bool,
    peak_cap_percent: float,
) -> Any:
    y64 = y.astype(np.float64, copy=True)
    rms = _rms_non_silent_samples(np, y64, silence_threshold_db)
    if rms < 1e-9:
        return y.astype(np.float32, copy=False)
    target_linear = float(10 ** (target_rms_db / 20.0))
    scale_rms = target_linear / rms
    scale = scale_rms
    if prevent_peak_clip:
        peak = float(np.max(np.abs(y64)))
        if peak > 1e-9:
            cap = float(peak_cap_percent) / 100.0
            peak_after = peak * scale_rms
            if peak_after > cap:
                scale = cap / peak
    return (y64 * scale).astype(np.float32, copy=False)


def _pad_edges_silence(
    np: Any,
    y: Any,
    sr: int,
    padding_silence_ms: int,
    silence_threshold_db: float,
) -> tuple[Any, float]:
    if padding_silence_ms <= 0:
        return y, 0.0
    target_s = padding_silence_ms / 1000.0
    yf = np.asarray(y, dtype=np.float32)
    n = int(yf.shape[0])
    if n == 0:
        z = _empty_audio(np, padding_silence_ms)
        return z, (len(z) / 2.0) / float(sr)
    mag = np.abs(yf.astype(np.float64, copy=False))
    db = 20.0 * np.log10(np.maximum(mag, 1e-20))
    active = db >= silence_threshold_db
    if np.any(active):
        first_i = int(np.argmax(active))
        last_i = n - 1 - int(np.argmax(active[::-1]))
    else:
        first_i = n
        last_i = -1
    leading_existing_s = first_i / float(sr)
    trailing_samples = (n - 1) - last_i if last_i >= 0 else n
    trailing_existing_s = trailing_samples / float(sr)
    pre_n = max(0, int(round(sr * max(0.0, target_s - leading_existing_s))))
    post_n = max(0, int(round(sr * max(0.0, target_s - trailing_existing_s))))
    leading_added_s = pre_n / float(sr)
    if pre_n == 0 and post_n == 0:
        return yf, 0.0
    parts = []
    if pre_n > 0:
        parts.append(np.zeros(pre_n, dtype=np.float32))
    parts.append(yf)
    if post_n > 0:
        parts.append(np.zeros(post_n, dtype=np.float32))
    return np.concatenate(parts), leading_added_s


def _empty_audio(np: Any, padding_silence_ms: int) -> Any:
    if padding_silence_ms > 0:
        need = max(1, int(round(TARGET_SR * padding_silence_ms / 1000.0)))
        return np.zeros(need * 2, dtype=np.float32)
    return np.zeros(max(1, int(TARGET_SR * 0.05)), dtype=np.float32)


def _empty_audio_leading_pad_seconds(padding_silence_ms: int) -> float:
    if padding_silence_ms <= 0:
        return 0.0
    need = max(1, int(round(TARGET_SR * padding_silence_ms / 1000.0)))
    return need / float(TARGET_SR)
