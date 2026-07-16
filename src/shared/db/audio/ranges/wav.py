from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from shared.db.audio.pack_store import ObjectStore


@dataclass(frozen=True)
class WavTimeRange:
    start: float
    end: float


@dataclass(frozen=True)
class WavClip:
    data: bytes
    sample_rate: int
    channels: int


class BoundedObjectReader:
    """Seekable view over one packed object slice using storage range reads."""

    def __init__(
        self,
        store: ObjectStore,
        path: str,
        byte_offset: int,
        byte_length: int,
    ) -> None:
        self.store = store
        self.path = path
        self.byte_offset = byte_offset
        self.byte_length = byte_length
        self.position = 0

    def read(self, size: int = -1) -> bytes:
        available = self.byte_length - self.position
        length = available if size < 0 else min(size, available)
        if length <= 0:
            return b""
        data = self.store.read_range(
            self.path,
            self.byte_offset + self.position,
            length,
        )
        if len(data) != length:
            raise EOFError(
                f"object range returned {len(data)} byte(s); expected {length}"
            )
        self.position += length
        return data

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.byte_length + offset
        else:
            raise ValueError(f"invalid seek mode: {whence}")
        if position < 0 or position > self.byte_length:
            raise ValueError(f"seek outside object slice: {position}")
        self.position = position
        return position

    def tell(self) -> int:
        return self.position


def read_wav_ranges(
    store: ObjectStore,
    path: str,
    byte_offset: int,
    byte_length: int,
    ranges: list[WavTimeRange],
) -> list[WavClip]:
    reader = BoundedObjectReader(store, path, byte_offset, byte_length)
    with wave.open(reader, "rb") as source:
        return _read_clips(source, ranges)


def slice_wav_ranges(data: bytes, ranges: list[WavTimeRange]) -> list[WavClip]:
    with wave.open(io.BytesIO(data), "rb") as source:
        return _read_clips(source, ranges)


def _read_clips(
    source: wave.Wave_read,
    ranges: list[WavTimeRange],
) -> list[WavClip]:
    sample_rate = source.getframerate()
    channels = source.getnchannels()
    sample_width = source.getsampwidth()
    frame_count = source.getnframes()
    clips = []
    for item in ranges:
        start_frame = int(round(max(0.0, item.start) * sample_rate))
        end_frame = int(round(max(0.0, item.end) * sample_rate))
        end_frame = min(max(start_frame + 1, end_frame), frame_count)
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)
        clips.append(
            WavClip(
                _wav_bytes(frames, sample_rate, channels, sample_width),
                sample_rate,
                channels,
            )
        )
    return clips


def _wav_bytes(
    frames: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(sample_rate)
        target.writeframes(frames)
    return output.getvalue()
