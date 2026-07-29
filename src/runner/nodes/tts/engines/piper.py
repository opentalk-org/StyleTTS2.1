from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
from piper import PiperConfig, PiperVoice, SynthesisConfig


# Some voice maps intentionally omit combining marks emitted by espeak. Piper
# skips them safely, but warning once per mark floods long corpus-run logs.
logging.getLogger("piper.phoneme_ids").setLevel(logging.ERROR)


class PiperCudaMemoryError(RuntimeError):
    """Signals that a bounded CUDA arena needs per-batch CPU recovery."""


@dataclass(frozen=True)
class PiperSynthesisOptions:
    speaker_id: int | None
    length_scale: float
    noise_scale: float
    noise_w_scale: float
    volume: float


class PiperRuntime:
    def __init__(
        self,
        checkpoint_dir: Path,
        threads: int = 1,
        device: Literal["cpu", "cuda"] = "cpu",
        gpu_memory_mb: int = 512,
    ):
        model_paths = tuple(checkpoint_dir.glob("*.onnx"))
        config_paths = tuple(checkpoint_dir.glob("*.onnx.json"))
        if len(model_paths) != 1 or len(config_paths) != 1:
            raise ValueError(f"piper_checkpoint_requires_model_and_config:{checkpoint_dir}")
        if (
            device == "cuda"
            and "CUDAExecutionProvider"
            not in onnxruntime.get_available_providers()
        ):
            raise RuntimeError("Piper CUDA requested but ONNX Runtime has no CUDA provider")
        config = PiperConfig.from_dict(
            json.loads(config_paths[0].read_text(encoding="utf-8"))
        )
        session = onnxruntime.InferenceSession(
            str(model_paths[0]),
            sess_options=piper_session_options(threads),
            providers=piper_session_providers(device, gpu_memory_mb),
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
            try:
                chunks = list(
                    self._voice.synthesize(
                        text,
                        syn_config=config,
                    )
                )
            except RuntimeException as error:
                if "bfc_arena" not in str(error).lower():
                    raise
                raise PiperCudaMemoryError(str(error)) from error
            if not chunks:
                raise ValueError("piper_synthesis_returned_no_audio")
            samples = np.concatenate([chunk.audio_float_array for chunk in chunks]).astype(np.float32)
            outputs.append((samples, chunks[0].sample_rate))
        return outputs


def synthesize_many_with_cpu_recovery(
    primary: PiperRuntime,
    fallback: PiperRuntime,
    texts: list[str],
    options: PiperSynthesisOptions,
    check_cancel: Callable[[], None],
) -> list[tuple[np.ndarray, int]]:
    outputs: list[tuple[np.ndarray, int]] = []
    for text in texts:
        try:
            result = primary.synthesize_many(
                [text],
                options,
                check_cancel,
            )
        except PiperCudaMemoryError:
            result = fallback.synthesize_many(
                [text],
                options,
                check_cancel,
            )
        outputs.extend(result)
    return outputs


def piper_session_options(threads: int) -> onnxruntime.SessionOptions:
    if threads < 1:
        raise ValueError("Piper ONNX thread count must be positive")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return options


def piper_session_providers(
    device: Literal["cpu", "cuda"],
    gpu_memory_mb: int,
) -> list[str | tuple[str, dict[str, str]]]:
    if gpu_memory_mb < 1:
        raise ValueError("Piper CUDA memory limit must be positive")
    if device == "cpu":
        return ["CPUExecutionProvider"]
    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": "0",
                "gpu_mem_limit": str(gpu_memory_mb * 1024 * 1024),
                "arena_extend_strategy": "kSameAsRequested",
            },
        ),
        "CPUExecutionProvider",
    ]
