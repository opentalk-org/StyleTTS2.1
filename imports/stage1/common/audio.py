import subprocess
from pathlib import Path

import soundfile as sf


def normalize_audio(source: Path, destination: Path) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as handle:
        input_format = ["-f", "mp3"] if handle.read(3) == b"ID3" else []
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        *input_format, "-i", str(source), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s24le",
        str(destination),
    ]
    subprocess.run(command, check=True)
    return _validated_duration(destination)


def normalize_audio_bytes(source: bytes, destination: Path) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", "pipe:0", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s24le",
        str(destination),
    ]
    subprocess.run(command, input=source, check=True)
    return _validated_duration(destination)


def normalize_audio_segment(source: Path, destination: Path, start: float, end: float) -> float:
    if start < 0.0 or end <= start:
        raise ValueError("audio segment must have an ordered non-negative range")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-to", str(end), "-i", str(source),
        "-ar", "24000", "-ac", "1", "-c:a", "pcm_s24le", str(destination),
    ]
    subprocess.run(command, check=True)
    return _validated_duration(destination)


def _validated_duration(destination: Path) -> float:
    info = sf.info(destination)
    if info.samplerate != 24_000 or info.channels != 1 or info.subtype != "PCM_24":
        raise ValueError(f"{destination}: normalization produced {info}")
    return info.frames / info.samplerate
