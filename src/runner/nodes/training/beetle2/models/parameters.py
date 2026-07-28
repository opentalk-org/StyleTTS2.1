from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ParameterReport:
    inference: int
    frozen_helper: int
    training_only: int

    @property
    def total(self) -> int:
        return self.inference + self.frozen_helper + self.training_only


def count_unique_parameters(
    modules: tuple[nn.Module, ...],
) -> tuple[int, set[int]]:
    identities: set[int] = set()
    count = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in identities:
                identities.add(identity)
                count += parameter.numel()
    return count, identities
