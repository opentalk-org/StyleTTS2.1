from __future__ import annotations

import io
import tempfile
import wave
from pathlib import Path


def write_temp_wav(data: bytes) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = Path(handle.name)
    with handle:
        handle.write(data)
    return path


def wav_info(data: bytes) -> dict[str, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            return {
                "sample_rate": source.getframerate(),
                "channels": source.getnchannels(),
                "sample_width": source.getsampwidth(),
                "frame_count": source.getnframes(),
            }
    except wave.Error as exc:
        raise ValueError("ASR nodes support WAV audio bytes") from exc


def extract_wav_range(data: bytes, start: float, end: float, info: dict[str, int] | None = None) -> bytes:
    wav = info if info is not None else wav_info(data)
    start_frame = int(round(max(0.0, start) * wav["sample_rate"]))
    end_frame = int(round(max(0.0, end) * wav["sample_rate"]))
    end_frame = min(max(start_frame + 1, end_frame), wav["frame_count"])
    source_buffer = io.BytesIO(data)
    output_buffer = io.BytesIO()
    with wave.open(source_buffer, "rb") as source:
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)
        with wave.open(output_buffer, "wb") as target:
            target.setnchannels(source.getnchannels())
            target.setsampwidth(source.getsampwidth())
            target.setframerate(source.getframerate())
            target.writeframes(frames)
    return output_buffer.getvalue()
