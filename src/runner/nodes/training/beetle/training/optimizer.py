import math
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn

from ..config.training import OptimizerConfig, ScheduledWeight
from .callbacks import TrainingMetric
from .checkpoint import NamedState, StateKind, StateTarget, capture_named_state
from .diagnostics.clipping import (
    GradientClipping,
    GradientClipObservation,
    NamedGradientGroup,
    clip_gradient_group,
    gradient_norm,
    optimizer_clipping_metrics,
    validate_gradient_group_ownership,
)
from .distributed import DistributedRuntime


@dataclass(frozen=True)
class StepSchedule:
    start_step: int
    warmup_steps: int
    decay_steps: int
    initial_value: float
    peak_value: float
    final_value: float

    def __post_init__(self) -> None:
        if min(self.start_step, self.warmup_steps, self.decay_steps) < 0:
            raise ValueError("schedule step counts must be non-negative")
        values = (self.initial_value, self.peak_value, self.final_value)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("schedule values must be finite")

    def value(self, optimizer_step: int) -> float:
        if optimizer_step < 0:
            raise ValueError("optimizer_step must be non-negative")
        relative = optimizer_step - self.start_step
        if relative < 0:
            return self.initial_value
        if self.warmup_steps > 0 and relative < self.warmup_steps:
            ratio = relative / self.warmup_steps
            return self.initial_value + ratio * (self.peak_value - self.initial_value)
        decay_position = relative - self.warmup_steps
        if self.decay_steps == 0 or decay_position >= self.decay_steps:
            return self.final_value
        ratio = decay_position / self.decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.final_value + cosine * (self.peak_value - self.final_value)

    @classmethod
    def loss_weight(
        cls,
        value: float,
        start_step: int,
        warmup_steps: int,
    ) -> "StepSchedule":
        return cls(start_step, warmup_steps, 0, 0.0, value, value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "start_step": self.start_step,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "initial_value": self.initial_value,
            "peak_value": self.peak_value,
            "final_value": self.final_value,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict != self.state_dict():
            raise ValueError("restored schedule does not match configured schedule")


@dataclass(frozen=True)
class ScheduledOptimizer:
    name: str
    optimizer: torch.optim.Optimizer
    schedule: StepSchedule
    scaler: torch.amp.GradScaler
    maximum_gradient_norm: float
    runtime: DistributedRuntime
    gradient_groups: tuple[NamedGradientGroup, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("optimizer name must not be empty")
        if self.maximum_gradient_norm <= 0:
            raise ValueError("maximum_gradient_norm must be positive")
        if not self.gradient_groups:
            raise ValueError("optimizer must contain a gradient group")

    def parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )

    def backward(self, loss: Tensor) -> None:
        self.scaler.scale(loss).backward()

    def prepare(self, optimizer_step: int) -> float:
        learning_rate = self.schedule.value(optimizer_step)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.scaler.unscale_(self.optimizer)
        return learning_rate

    def finish(
        self,
        learning_rate: float,
        gradient_norm_value: float,
        observations: tuple[GradientClipObservation, ...],
        diagnostics: bool,
    ) -> tuple[TrainingMetric, ...]:
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = (
            TrainingMetric(
                f"optimizer/{self.name}_learning_rate",
                learning_rate,
            ),
            TrainingMetric(
                f"optimizer/{self.name}_gradient_norm",
                gradient_norm_value,
            ),
            TrainingMetric(
                f"optimizer/{self.name}_amp_scale",
                self.scaler.get_scale(),
            ),
        )
        if not diagnostics:
            return metrics
        return (
            *metrics,
            *optimizer_clipping_metrics(
                self.name,
                min(item.coefficient for item in observations),
            ),
        )


class OptimizerSet:
    def __init__(self, groups: tuple[ScheduledOptimizer, ...]) -> None:
        if not groups:
            raise ValueError("at least one optimizer group is required")
        names = tuple(group.name for group in groups)
        if len(set(names)) != len(names):
            raise ValueError("optimizer group names must be unique")
        parameter_owners: dict[int, str] = {}
        for group in groups:
            for parameter in group.parameters():
                owner = parameter_owners.setdefault(id(parameter), group.name)
                if owner != group.name:
                    raise ValueError(
                        "optimizer parameter ownership overlaps between "
                        f"{owner} and {group.name}"
                    )
        gradient_names = tuple(
            gradient_group.name
            for group in groups
            for gradient_group in group.gradient_groups
        )
        if len(set(gradient_names)) != len(gradient_names):
            raise ValueError("gradient group names must be unique")
        for group in groups:
            validate_gradient_group_ownership(
                group.name,
                group.parameters(),
                group.gradient_groups,
                parameter_owners,
            )
        self.groups = groups

    def prepare_distributed(self) -> "OptimizerSet":
        return OptimizerSet(
            tuple(
                replace(
                    group,
                    optimizer=group.runtime.prepare_optimizer(group.optimizer),
                )
                for group in self.groups
            )
        )

    def group(self, name: str) -> ScheduledOptimizer:
        matches = tuple(group for group in self.groups if group.name == name)
        if len(matches) != 1:
            raise KeyError(f"optimizer group does not exist: {name}")
        return matches[0]

    def step(
        self,
        optimizer_step: int,
        diagnostics: bool = False,
    ) -> tuple[TrainingMetric, ...]:
        learning_rates = tuple(
            group.prepare(optimizer_step) for group in self.groups
        )
        gradient_norms = tuple(
            gradient_norm(group.parameters()) for group in self.groups
        )
        owned_parameter_ids = tuple(
            {id(parameter) for parameter in group.parameters()}
            for group in self.groups
        )
        observations = tuple(
            tuple(
                clip_gradient_group(
                    gradient_group,
                    group.maximum_gradient_norm,
                    owned_parameters,
                )
                for gradient_group in group.gradient_groups
            )
            for group, owned_parameters in zip(
                self.groups,
                owned_parameter_ids,
                strict=True,
            )
        )
        module_metrics = tuple(
            metric
            for group_observations in observations
            for observation in group_observations
            for metric in observation.metrics(diagnostics)
        )
        optimizer_metrics = tuple(
            metric
            for group, learning_rate, gradient_norm_value, group_observations in zip(
                self.groups,
                learning_rates,
                gradient_norms,
                observations,
                strict=True,
            )
            for metric in group.finish(
                learning_rate,
                gradient_norm_value,
                group_observations,
                diagnostics,
            )
        )
        return (*optimizer_metrics, *module_metrics)

    def capture_states(self) -> tuple[NamedState, ...]:
        return tuple(
            state
            for group in self.groups
            for state in (
                capture_named_state(group.name, StateKind.OPTIMIZER, group.optimizer),
                capture_named_state(group.name, StateKind.SCHEDULER, group.schedule),
                capture_named_state(group.name, StateKind.SCALER, group.scaler),
            )
        )

    def state_targets(self) -> tuple[StateTarget, ...]:
        return tuple(
            target
            for group in self.groups
            for target in (
                StateTarget(group.name, StateKind.OPTIMIZER, group.optimizer),
                StateTarget(group.name, StateKind.SCHEDULER, group.schedule),
                StateTarget(group.name, StateKind.SCALER, group.scaler),
            )
        )


def learning_rate_schedule(config: OptimizerConfig) -> StepSchedule:
    return StepSchedule(
        start_step=0,
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        initial_value=0.0,
        peak_value=config.learning_rate,
        final_value=config.learning_rate * config.minimum_learning_rate_ratio,
    )


def loss_weight_schedule(config: ScheduledWeight) -> StepSchedule:
    return StepSchedule.loss_weight(
        value=config.value,
        start_step=config.start_step,
        warmup_steps=config.warmup_steps,
    )
