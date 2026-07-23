from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
from piper import PiperConfig, PiperVoice, SynthesisConfig


@dataclass(frozen=True)
class PiperSynthesisOptions:
    speaker_id: int | None
    length_scale: float
    noise_scale: float
    noise_w_scale: float
    volume: float


class PiperRuntime:
    def __init__(self, checkpoint_dir: Path, threads: int = 1):
        model_paths = tuple(checkpoint_dir.glob("*.onnx"))
        config_paths = tuple(checkpoint_dir.glob("*.onnx.json"))
        if len(model_paths) != 1 or len(config_paths) != 1:
            raise ValueError(f"piper_checkpoint_requires_model_and_config:{checkpoint_dir}")
        config = PiperConfig.from_dict(
            json.loads(config_paths[0].read_text(encoding="utf-8"))
        )
        session = onnxruntime.InferenceSession(
            str(model_paths[0]),
            sess_options=piper_session_options(threads),
            providers=["CPUExecutionProvider"],
        )
        self._voice = PiperVoice(session=session, config=config)

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


def piper_session_options(threads: int) -> onnxruntime.SessionOptions:
    if threads < 1:
        raise ValueError("Piper ONNX thread count must be positive")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return options
