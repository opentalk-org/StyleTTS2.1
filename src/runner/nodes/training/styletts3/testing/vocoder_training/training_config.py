from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    generator_learning_rate: float
    discriminator_learning_rate: float
    betas: tuple[float, float]
    max_steps_per_epoch: int | None
    validation_interval_epochs: int
    max_steps: int | None

    def effective_steps_per_epoch(self, loader_steps: int) -> int:
        if self.max_steps_per_epoch is None:
            return loader_steps
        return min(loader_steps, self.max_steps_per_epoch)

    def total_steps(self, loader_steps: int) -> int:
        if self.max_steps is not None:
            return self.max_steps
        return self.effective_steps_per_epoch(loader_steps) * self.epochs

    def training_epochs(self, loader_steps: int) -> int:
        steps_per_epoch = self.effective_steps_per_epoch(loader_steps)
        return math.ceil(self.total_steps(loader_steps) / steps_per_epoch)
