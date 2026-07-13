from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runner.nodes.tts.voices import Voice


@dataclass(frozen=True)
class EngineSynthesisRequest:
    text: str
    voice: Voice
    language: str


@dataclass(frozen=True)
class EngineSynthesisResult:
    samples: np.ndarray
    sample_rate: int


class EngineRuntime(ABC):
    """A loaded TTS engine. One instance is held per synthesis-node lifecycle."""

    SAMPLE_RATE: int

    @abstractmethod
    def synthesize(self, text: str, voice: Voice, language: str) -> tuple[np.ndarray, int]:
        """Return (float32 mono waveform in [-1, 1], sample_rate) for one utterance."""
        raise NotImplementedError

    def synthesize_batch(
        self,
        requests: list[EngineSynthesisRequest],
        check_cancel: Callable[[], None],
    ) -> list[EngineSynthesisResult]:
        outputs = []
        for request in requests:
            check_cancel()
            outputs.append(EngineSynthesisResult(*self.synthesize(request.text, request.voice, request.language)))
        return outputs

    def close(self) -> None:
        return None


def resolve_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def require_checkpoint_dir(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"tts_checkpoint_dir_missing:{checkpoint_dir}")
    return checkpoint_dir
