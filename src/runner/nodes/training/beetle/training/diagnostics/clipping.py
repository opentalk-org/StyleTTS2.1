from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

from ..callbacks import TrainingMetric


class GradientClipping(str, Enum):
    CLIP = "clip"
    OBSERVE = "observe"


@dataclass(frozen=True)
class NamedGradientGroup:
    name: str
    modules: tuple[nn.Module, ...]
    clipping: GradientClipping

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gradient group name must not be empty")
        if not self.modules:
            raise ValueError("gradient group must contain a module")

    def parameters(self) -> tuple[nn.Parameter, ...]:
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for module in self.modules:
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        return tuple(parameters)


@dataclass(frozen=True)
class GradientClipObservation:
    name: str
    norm: float
    coefficient: float

    def metrics(self, diagnostics: bool) -> tuple[TrainingMetric, ...]:
        metrics = (TrainingMetric(f"gradient/{self.name}", self.norm),)
        if not diagnostics:
            return metrics
        return (
            *metrics,
            TrainingMetric(
                f"gradient/{self.name}_clip_coefficient",
                self.coefficient,
            ),
            TrainingMetric(
                f"gradient/{self.name}_was_clipped",
                float(self.coefficient < 1.0),
            ),
        )


def clip_gradient_group(
    group: NamedGradientGroup,
    maximum_norm: float,
    owned_parameters: set[int],
) -> GradientClipObservation:
    parameters = tuple(
        parameter
        for parameter in group.parameters()
        if id(parameter) in owned_parameters
    )
    if group.clipping is GradientClipping.CLIP:
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, maximum_norm))
        coefficient = min(1.0, maximum_norm / (norm + 1e-6))
    else:
        norm = gradient_norm(parameters)
        coefficient = 1.0
    return GradientClipObservation(group.name, norm, coefficient)


def gradient_norm(parameters: tuple[nn.Parameter, ...]) -> float:
    gradients = tuple(
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    )
    if not gradients:
        return 0.0
    return float(torch.stack(gradients).norm(2))


def optimizer_clipping_metrics(
    optimizer_name: str,
    coefficient: float,
) -> tuple[TrainingMetric, ...]:
    return (
        TrainingMetric(
            f"optimizer/{optimizer_name}_clip_coefficient",
            coefficient,
        ),
        TrainingMetric(
            f"optimizer/{optimizer_name}_was_clipped",
            float(coefficient < 1.0),
        ),
    )


def validate_gradient_group_ownership(
    optimizer_name: str,
    parameters: tuple[nn.Parameter, ...],
    groups: tuple[NamedGradientGroup, ...],
    owners: dict[int, str],
) -> None:
    optimizer_parameters = {id(parameter) for parameter in parameters}
    occurrences: dict[int, int] = {}
    for group in groups:
        for parameter in group.parameters():
            identifier = id(parameter)
            owner = owners.get(identifier)
            if owner is None:
                if parameter.requires_grad:
                    raise ValueError(
                        f"gradient group {group.name} contains an unowned trainable parameter"
                    )
                continue
            if owner != optimizer_name:
                raise ValueError(
                    f"gradient group {group.name} contains a foreign parameter"
                )
            occurrences[identifier] = occurrences.get(identifier, 0) + 1
    if any(count > 1 for count in occurrences.values()):
        raise ValueError("optimizer parameter belongs to more than one gradient group")
    if optimizer_parameters - occurrences.keys():
        raise ValueError("optimizer parameter is missing from gradient groups")
