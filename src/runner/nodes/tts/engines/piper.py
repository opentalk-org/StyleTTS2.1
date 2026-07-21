from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import numpy as np
from piper import PiperVoice, SynthesisConfig


@dataclass(frozen=True)
class PiperSynthesisOptions:
    speaker_id: int | None
    length_scale: float
    noise_scale: float
    noise_w_scale: float
    volume: float


class PiperRuntime:
    def __init__(self, checkpoint_dir: Path):
        model_paths = tuple(checkpoint_dir.glob("*.onnx"))
        config_paths = tuple(checkpoint_dir.glob("*.onnx.json"))
        if len(model_paths) != 1 or len(config_paths) != 1:
            raise ValueError(f"piper_checkpoint_requires_model_and_config:{checkpoint_dir}")
        self._voice = PiperVoice.load(model_paths[0], config_path=config_paths[0])

    def synthesize_many(
        self,
        texts: list[str],
        options: PiperSynthesisOptions,
        check_cancel: Callable[[], None],
    ) -> list[tuple[np.ndarray, int]]:
        config = SynthesisConfig(
            speaker_id=options.speaker_id,
            length_scale=options.length_scale,
            noise_scale=options.noise_scale,
            noise_w_scale=options.noise_w_scale,
            volume=options.volume,
        )
        outputs = []
        for text in texts:
            check_cancel()
            chunks = list(self._voice.synthesize(text, syn_config=config))
            if not chunks:
                raise ValueError("piper_synthesis_returned_no_audio")
            samples = np.concatenate([chunk.audio_float_array for chunk in chunks]).astype(np.float32)
            outputs.append((samples, chunks[0].sample_rate))
        return outputs
