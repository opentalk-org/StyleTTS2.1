from dataclasses import dataclass

import torch
from torch import nn

from ...data.validation_types import ValidationRecording
from ..reporting import TrainingMetric
from ..state import capture_rng_state, restore_rng_state
from .artifacts import ValidationArtifacts
from .types import (
    StageValidator,
    ValidationEvaluator,
    ValidationResult,
    ValidationSampleResult,
)


@dataclass(frozen=True)
class _ModuleMode:
    module: nn.Module
    training: bool


class ValidationRuntime:
    def __init__(self, evaluator: ValidationEvaluator) -> None:
        self.evaluator = evaluator

    def evaluate(self, recordings: tuple[object, ...], step: int) -> ValidationResult:
        if step < 0 or not recordings:
            raise ValueError("validation requires a nonnegative step and recordings")
        rng = capture_rng_state()
        roots = self.evaluator.modules()
        modes = _capture_modes(roots)
        try:
            for module in roots:
                module.eval()
            with torch.no_grad():
                samples = self.evaluator.evaluate_samples(recordings, step)
            if not samples:
                raise ValueError("validation evaluator returned no samples")
            aggregates = aggregate_losses(samples)
        finally:
            _restore_modes(modes)
            restore_rng_state(rng)
        return ValidationResult(self.evaluator.stage, step, samples, aggregates)


class ValidationCoordinator:
    def __init__(
        self,
        validator: StageValidator,
        recordings: tuple[ValidationRecording, ...],
        artifacts: ValidationArtifacts,
    ) -> None:
        if not recordings:
            raise ValueError("validation coordinator requires recordings")
        self.validator = validator
        self.recordings = recordings
        self.artifacts = artifacts

    def run(self, step: int) -> tuple[TrainingMetric, ...]:
        result = self.validator.evaluate(self.recordings, step)
        self.artifacts.publish(result)
        return validation_metrics(result)

    def close(self) -> None:
        self.artifacts.close()


def aggregate_losses(
    samples: tuple[ValidationSampleResult, ...],
) -> tuple[TrainingMetric, ...]:
    names = tuple(metric.name for metric in samples[0].losses)
    if any(tuple(metric.name for metric in sample.losses) != names for sample in samples):
        raise ValueError("validation loss names must match across samples")
    count = len(samples)
    return tuple(
        TrainingMetric(
            name,
            sum(sample.losses[index].value for sample in samples) / count,
        )
        for index, name in enumerate(names)
    )


def validation_metrics(result: ValidationResult) -> tuple[TrainingMetric, ...]:
    return tuple(
        TrainingMetric(f"validation/{metric.name}", metric.value)
        for metric in result.aggregates
    )


def _capture_modes(roots: tuple[nn.Module, ...]) -> tuple[_ModuleMode, ...]:
    modules: list[nn.Module] = []
    identities: set[int] = set()
    for root in roots:
        for module in root.modules():
            if id(module) not in identities:
                identities.add(id(module))
                modules.append(module)
    return tuple(_ModuleMode(module, module.training) for module in modules)


def _restore_modes(modes: tuple[_ModuleMode, ...]) -> None:
    for saved in modes:
        saved.module.train(saved.training)
