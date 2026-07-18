from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from torch import Tensor, nn

from ...data.validation_types import ValidationRecording
from ..reporting import TrainingMetric
from ..state import StageKind


SignalPair = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class ValidationSampleResult:
    audio_file_id: UUID
    losses: tuple[TrainingMetric, ...]
    ground_truth: Tensor
    prediction: Tensor
    latent: Tensor
    f0: SignalPair
    n: SignalPair
    mel: SignalPair
    alignment: Tensor | None

    def __post_init__(self) -> None:
        names = tuple(metric.name for metric in self.losses)
        if not self.losses or len(set(names)) != len(names):
            raise ValueError("validation sample losses must be nonempty and unique")


@dataclass(frozen=True)
class ValidationResult:
    stage: StageKind
    step: int
    samples: tuple[ValidationSampleResult, ...]
    aggregates: tuple[TrainingMetric, ...]

    def __post_init__(self) -> None:
        if self.step < 0 or not self.samples:
            raise ValueError("validation result requires a step and samples")


class ValidationEvaluator(Protocol):
    stage: StageKind

    def modules(self) -> tuple[nn.Module, ...]: ...

    def evaluate_samples(
        self,
        recordings: tuple[ValidationRecording, ...],
        step: int,
    ) -> tuple[ValidationSampleResult, ...]: ...


class StageValidator(Protocol):
    def evaluate(
        self,
        recordings: tuple[ValidationRecording, ...],
        step: int,
    ) -> ValidationResult: ...
