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
    seed: int

    def __post_init__(self) -> None:
        names = tuple(metric.name for metric in self.losses)
        if not self.losses or len(set(names)) != len(names):
            raise ValueError("validation sample losses must be nonempty and unique")
        if self.seed < 0:
            raise ValueError("validation seed must be non-negative")


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


def trim_waveform_pair(
    target: Tensor,
    prediction: Tensor,
    sample_count: int,
) -> SignalPair:
    if target.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("validation waveforms must have equal [B,1,S] shapes")
    if target.shape[0] != 1 or target.shape[1] != 1:
        raise ValueError("validation artifact pair requires one mono item")
    if sample_count <= 0 or sample_count > target.shape[2]:
        raise ValueError("validation waveform length is invalid")
    return (
        target[0, :, :sample_count].detach().cpu().clone(),
        prediction[0, :, :sample_count].detach().cpu().clone(),
    )
