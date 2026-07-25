from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from torch import Tensor, nn

from ...data.validation_records import ValidationRecording
from ..reporting import TrainingMetric


SignalPair = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class ValidationArtifactSet:
    ground_truth: Tensor
    prediction: Tensor
    latent: Tensor
    f0: SignalPair
    n: SignalPair
    mel: SignalPair
    alignment: Tensor | None


@dataclass(frozen=True)
class ConditionalValidationSample:
    audio_file_id: UUID
    losses: tuple[TrainingMetric, ...]
    artifacts: ValidationArtifactSet
    seed: int

@dataclass(frozen=True)
class ValidationSampleResult:
    audio_file_id: UUID
    losses: tuple[TrainingMetric, ...]
    full: ValidationArtifactSet
    audio: ValidationArtifactSet
    seed: int

@dataclass(frozen=True)
class ValidationResult:
    step: int
    samples: tuple[ValidationSampleResult, ...]
    aggregates: tuple[TrainingMetric, ...]

class ValidationEvaluator(Protocol):
    def modules(self) -> tuple[nn.Module, ...]: ...

    def evaluate_samples(
        self,
        recordings: tuple[ValidationRecording, ...],
        step: int,
    ) -> tuple[ValidationSampleResult, ...]: ...


class ValidationRunner(Protocol):
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
    return (
        target[0, :, :sample_count].detach().cpu().clone(),
        prediction[0, :, :sample_count].detach().cpu().clone(),
    )


def trim_signal_pair(
    target: Tensor,
    prediction: Tensor,
    frame_count: int,
) -> SignalPair:
    return (
        target[:frame_count].detach().cpu().clone(),
        prediction[:frame_count].detach().cpu().clone(),
    )
