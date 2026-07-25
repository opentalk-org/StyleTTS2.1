import random
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from torch import Tensor

from .reporting.metrics import TrainingMetric


class TrainingPhase(StrEnum):
    READY = "ready"
    BATCH_FETCHED = "batch_fetched"
    DISCRIMINATOR_BACKWARD = "discriminator_backward"
    DISCRIMINATOR_COMPLETE = "discriminator_complete"
    GENERATOR_BACKWARD = "generator_backward"
    GENERATOR_COMPLETE = "generator_complete"
    OPTIMIZER_COMPLETE = "optimizer_complete"
    CHECKPOINTING = "checkpointing"
    CHECKPOINT_COMPLETE = "checkpoint_complete"


@dataclass(frozen=True)
class LoopState:
    optimizer_step: int
    microstep: int
    phase: TrainingPhase
    sampler_cursor: int
    cycle: int
    batch_index: int
    discriminator_metrics: tuple[TrainingMetric, ...]

@dataclass(frozen=True)
class PythonRngState:
    version: int
    internal_state: tuple[int, ...]
    gaussian: float | None


@dataclass(frozen=True)
class NumpyRngState:
    bit_generator: str
    keys: tuple[int, ...]
    position: int
    has_gaussian: int
    cached_gaussian: float


@dataclass(frozen=True)
class RngState:
    python: PythonRngState
    numpy: NumpyRngState
    torch_cpu: Tensor
    torch_cuda: tuple[Tensor, ...]


@dataclass(frozen=True)
class RankState:
    rng: RngState


def capture_rng_state() -> RngState:
    python_version, python_internal, python_gaussian = random.getstate()
    bit_generator, keys, position, has_gaussian, cached_gaussian = np.random.get_state()
    cuda_states = tuple(
        state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()
    )
    return RngState(
        python=PythonRngState(
            version=python_version,
            internal_state=python_internal,
            gaussian=python_gaussian,
        ),
        numpy=NumpyRngState(
            bit_generator=bit_generator,
            keys=tuple(int(key) for key in keys),
            position=position,
            has_gaussian=has_gaussian,
            cached_gaussian=cached_gaussian,
        ),
        torch_cpu=torch.get_rng_state().detach().cpu().clone(),
        torch_cuda=cuda_states,
    )


def restore_rng_state(state: RngState) -> None:
    random.setstate(
        (state.python.version, state.python.internal_state, state.python.gaussian)
    )
    numpy_keys = np.asarray(state.numpy.keys, dtype=np.uint32)
    np.random.set_state(
        (
            state.numpy.bit_generator,
            numpy_keys,
            state.numpy.position,
            state.numpy.has_gaussian,
            state.numpy.cached_gaussian,
        )
    )
    torch.set_rng_state(state.torch_cpu)
    torch.cuda.set_rng_state_all(list(state.torch_cuda))
