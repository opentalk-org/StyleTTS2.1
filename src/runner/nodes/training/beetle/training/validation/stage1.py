from uuid import UUID

import torch
from torch import Tensor, nn

from ...config.training import StageConfig
from ...data.sampling import derive_seed
from ...data.validation_types import ValidationRecording
from ...losses.acoustic import masked_f0_mse, masked_kl_standard_normal, masked_n_mse
from ...models.model import Stage1Models
from ..reporting import TrainingMetric
from ..stage1_setup import Stage1Schedules
from ..state import StageKind
from .types import ValidationSampleResult


class Stage1ValidationEvaluator:
    stage = StageKind.STAGE1

    def __init__(
        self,
        models: Stage1Models,
        stage_config: StageConfig,
        runtime_seed: int,
        device: torch.device,
    ) -> None:
        self.models = models
        self.schedules = Stage1Schedules.from_config(stage_config)
        self.runtime_seed = runtime_seed
        self.device = device

    @staticmethod
    def required_model_names() -> tuple[str, ...]:
        return (
            "audio_encoder",
            "feature_linear",
            "decoder",
            "generator",
            "f0_extractor",
            "reconstruction_loss",
        )

    def modules(self) -> tuple[nn.Module, ...]:
        return tuple(
            getattr(self.models, name) for name in self.required_model_names()
        )

    def evaluate_samples(
        self,
        recordings: tuple[ValidationRecording, ...],
        step: int,
    ) -> tuple[ValidationSampleResult, ...]:
        return tuple(self._evaluate(recording, step) for recording in recordings)

    def _evaluate(
        self,
        recording: ValidationRecording,
        step: int,
    ) -> ValidationSampleResult:
        values = recording.batch.to(self.device)
        latent_generator = self._generator(step, recording.audio_file_id, "latent")
        source_generator = self._generator(step, recording.audio_file_id, "source")
        synthesis = self.models.reconstruct(
            values.mel,
            values.frame_mask,
            latent_generator,
            source_generator,
        )
        targets = self.models.acoustic_targets(values.mel, values.frame_mask)
        encoder_kl = masked_kl_standard_normal(
            synthesis.posterior.mean,
            synthesis.posterior.log_scale,
            synthesis.posterior.mask,
        )
        f0 = masked_f0_mse(
            synthesis.acoustic.f0,
            targets.f0,
            synthesis.decoded.mask,
        )
        n = masked_n_mse(synthesis.acoustic.n, targets.n, synthesis.decoded.mask)
        reconstruction = self.models.reconstruction_loss(
            synthesis.waveform,
            values.waveform,
            synthesis.sample_mask,
        ).total
        weights = self.schedules.weights(step)
        total = (
            encoder_kl * weights.encoder_kl
            + f0 * weights.f0
            + n * weights.n
            + reconstruction * weights.reconstruction
        )
        target_mel, predicted_mel = self._artifact_mels(
            values.waveform,
            synthesis.waveform,
        )
        return ValidationSampleResult(
            recording.audio_file_id,
            (
                _metric("encoder_kl", encoder_kl),
                _metric("f0", f0),
                _metric("n", n),
                _metric("reconstruction", reconstruction),
                _metric("generator_total", total),
            ),
            _cpu(values.waveform[0]),
            _cpu(synthesis.waveform[0]),
            _cpu(synthesis.posterior.latent[0]),
            (_cpu(targets.f0[0]), _cpu(synthesis.acoustic.f0[0])),
            (_cpu(targets.n[0]), _cpu(synthesis.acoustic.n[0])),
            (_cpu(target_mel[0]), _cpu(predicted_mel[0])),
            None,
        )

    def _artifact_mels(
        self,
        target: Tensor,
        prediction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        transform = self.models.reconstruction_loss.transforms[0]
        return transform(target[:, 0]), transform(prediction[:, 0])

    def _generator(
        self,
        step: int,
        audio_file_id: UUID,
        view: str,
    ) -> torch.Generator:
        seed = derive_seed(
            self.runtime_seed,
            self.stage,
            step,
            audio_file_id,
            view,
        )
        return torch.Generator(device=self.device).manual_seed(seed)


def _metric(name: str, value: Tensor) -> TrainingMetric:
    return TrainingMetric(name, float(value.detach().cpu()))


def _cpu(value: Tensor) -> Tensor:
    return value.detach().cpu().clone()
