from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from runner.nodes.tts.voices import Voice


class EngineRuntime(ABC):
    """A loaded TTS engine. One instance is held per synthesis-node lifecycle."""

    SAMPLE_RATE: int

    @abstractmethod
    def synthesize(self, text: str, voice: Voice, language: str) -> tuple[np.ndarray, int]:
        """Return (float32 mono waveform in [-1, 1], sample_rate) for one utterance."""
        raise NotImplementedError

    def close(self) -> None:
        return None


def resolve_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def require_checkpoint_dir(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"tts_checkpoint_dir_missing:{checkpoint_dir}")
    return checkpoint_dir
