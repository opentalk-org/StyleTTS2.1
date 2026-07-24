import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from .callbacks import TrainingMetric
from .checkpoint import NamedState, StateKind, StateTarget, capture_named_state
from .diagnostics.clipping import (
    GradientClipObservation,
    NamedGradientGroup,
    clip_gradient_group,
    gradient_norm,
    optimizer_clipping_metrics,
    validate_gradient_group_ownership,
)
from .distributed import DistributedRuntime
from .loss_schedules import StepSchedule


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

    def discard(self) -> None:
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

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
        diagnostics: bool,
    ) -> tuple[TrainingMetric, ...]:
        learning_rates = tuple(
            group.prepare(optimizer_step) for group in self.groups
        )
        gradient_norms = tuple(
            gradient_norm(group.parameters()) for group in self.groups
        )
        invalid = tuple(
            group.name
            for group, norm in zip(self.groups, gradient_norms, strict=True)
            if not math.isfinite(norm)
        )
        if invalid:
            for group in self.groups:
                group.discard()
            raise FloatingPointError(
                f"non-finite optimizer gradients: {', '.join(invalid)}"
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
