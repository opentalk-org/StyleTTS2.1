from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from runner.nodes.asr.audio import write_temp_wav
from runner.nodes.models import Audio


@dataclass
class TemporaryAudioBatch:
    audios: list[Audio]
    paths: list[Path] = field(default_factory=list, init=False)

    def __enter__(self) -> list[Path]:
        try:
            for audio in self.audios:
                assert audio.data is not None, f"audio bytes are required: {audio.id}"
                self.paths.append(write_temp_wav(audio.data))
        except BaseException:
            self._remove_paths()
            raise
        return self.paths

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._remove_paths()

    def _remove_paths(self) -> None:
        for path in self.paths:
            path.unlink(missing_ok=True)
