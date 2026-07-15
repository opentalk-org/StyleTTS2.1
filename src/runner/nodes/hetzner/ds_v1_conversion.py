from __future__ import annotations

import subprocess
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConvertedWav:
    ordinal: int
    wav_path: Path
    sample_rate: int
    channels: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / float(self.sample_rate)


class OpusConversionPool:
    def __init__(self, output_dir: Path, workers: int):
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ds-v1-opusdec")

    def __enter__(self) -> OpusConversionPool:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def submit(self, opus_path: Path, ordinal: int) -> Future[ConvertedWav]:
        wav_path = self._output_dir / f"{ordinal:08d}.wav"
        return self._executor.submit(_convert_opus, opus_path, wav_path, ordinal)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


def _convert_opus(opus_path: Path, wav_path: Path, ordinal: int) -> ConvertedWav:
    result = subprocess.run(
        ["opusdec", "--quiet", "--no-dither", str(opus_path), str(wav_path)],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"opusdec failed for {opus_path.name}: exit={result.returncode} {detail}")
    with wave.open(str(wav_path), "rb") as reader:
        return ConvertedWav(
            ordinal,
            wav_path,
            reader.getframerate(),
            reader.getnchannels(),
            reader.getnframes(),
        )
