import random
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from torch import Tensor, nn

from .reporting.metrics import TrainingMetric


class StageKind(StrEnum):
    STAGE1 = "stage1"
    STAGE2 = "stage2"
    STAGE3 = "stage3"


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
    stage: StageKind
    optimizer_step: int
    microstep: int
    phase: TrainingPhase
    sampler_cursor: int
    cycle: int
    batch_index: int
    discriminator_metrics: tuple[TrainingMetric, ...]

    def __post_init__(self) -> None:
        counters = (
            ("optimizer_step", self.optimizer_step),
            ("microstep", self.microstep),
            ("sampler_cursor", self.sampler_cursor),
            ("cycle", self.cycle),
            ("batch_index", self.batch_index),
        )
        invalid = tuple(name for name, value in counters if value < 0)
        if invalid:
            raise ValueError(f"{', '.join(invalid)} must be non-negative")
        metric_names = tuple(metric.name for metric in self.discriminator_metrics)
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("pending discriminator metric names must be unique")
        metric_phases = (
            TrainingPhase.DISCRIMINATOR_COMPLETE,
            TrainingPhase.GENERATOR_BACKWARD,
        )
        if self.discriminator_metrics and self.phase not in metric_phases:
            raise ValueError(
                "pending discriminator metrics require a completed discriminator pass"
            )


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


@dataclass(frozen=True)
class NamedGradient:
    name: str
    value: Tensor | None


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
    _validate_rng_tensor(state.torch_cpu, "torch_cpu")
    if len(state.torch_cuda) != torch.cuda.device_count():
        raise ValueError(
            "torch_cuda RNG state count does not match the visible CUDA devices"
        )
    for index, cuda_state in enumerate(state.torch_cuda):
        _validate_rng_tensor(cuda_state, f"torch_cuda[{index}]")
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


def capture_gradients(module: nn.Module) -> tuple[NamedGradient, ...]:
    return tuple(
        NamedGradient(
            name=name,
            value=None
            if parameter.grad is None
            else parameter.grad.detach().cpu().clone(),
        )
        for name, parameter in module.named_parameters()
    )


def restore_gradients(
    module: nn.Module,
    gradients: tuple[NamedGradient, ...],
) -> None:
    parameters = tuple(module.named_parameters())
    expected_names = tuple(name for name, _ in parameters)
    saved_names = tuple(gradient.name for gradient in gradients)
    if saved_names != expected_names:
        raise ValueError(
            f"gradient names do not match parameters: {saved_names} != {expected_names}"
        )
    for (name, parameter), gradient in zip(parameters, gradients, strict=True):
        if gradient.value is None:
            parameter.grad = None
            continue
        if gradient.value.shape != parameter.shape:
            raise ValueError(f"gradient shape does not match parameter: {name}")
        if gradient.value.dtype != parameter.dtype:
            raise ValueError(f"gradient dtype does not match parameter: {name}")
        parameter.grad = gradient.value.to(device=parameter.device).clone()


def _validate_rng_tensor(state: Tensor, name: str) -> None:
    if state.device.type != "cpu" or state.dtype is not torch.uint8 or state.ndim != 1:
        raise ValueError(f"{name} RNG state must be a one-dimensional CPU uint8 tensor")
