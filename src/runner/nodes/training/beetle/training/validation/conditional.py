from uuid import UUID

import torch
from torch import Tensor, nn

from ...config.training import TrainingConfig
from ...data.sampling import derive_seed
from ...data.validation_records import ValidationRecording
from ...losses.conditional import compute_conditional_losses
from ...models.model import AcousticModels
from ...models.modules.audio import AcousticFeatures
from ...models.modules.latent_flow import integrate_latent_flow
from ...models.conditional import ConditionalModels
from ..reporting import TrainingMetric
from ..loss_schedules import TrainingSchedules
from ..conditional.features import ConditionalSynthesis
from ..setup import ConditionalInputBuilder
from ..state import LoopState, TrainingPhase
from .batch import merge_validation_recordings
from .types import (
    ConditionalValidationSample,
    ValidationArtifactSet,
    trim_signal_pair,
    trim_waveform_pair,
)


class ConditionalValidationEvaluator:
    def __init__(
        self,
        acoustic: AcousticModels,
        models: ConditionalModels,
        ema_latent_flow: nn.Module,
        input_builder: ConditionalInputBuilder,
        training_config: TrainingConfig,
        runtime_seed: int,
        device: torch.device,
        require_distinct_voices: bool,
    ) -> None:
        self.acoustic = acoustic
        self.models = models
        self.ema_latent_flow = ema_latent_flow
        self.input_builder = input_builder
        self.schedules = TrainingSchedules.from_config(training_config)
        self.runtime_seed = runtime_seed
        self.device = device
        self.require_distinct_voices = require_distinct_voices

    @staticmethod
    def required_model_names() -> tuple[str, ...]:
        return (
            "phoneme_encoder",
            "context_encoders",
            "conditioning",
            "style_encoder",
            "voice_encoder",
            "duration_predictor",
            "latent_flow",
            "latent_flow_ema",
            "aligner",
            "audio_encoder",
            "feature_linear",
            "decoder",
            "generator",
        )

    def modules(self) -> tuple[nn.Module, ...]:
        return (
            *tuple(self.models.children()),
            self.acoustic.feature_linear,
            self.acoustic.decoder,
            self.acoustic.generator,
            self.ema_latent_flow,
        )

    def evaluate_samples(
        self,
        recordings: tuple[ValidationRecording, ...],
        step: int,
    ) -> tuple[ConditionalValidationSample, ...]:
        batch = merge_validation_recordings(recordings).to(self.device)
        loop = LoopState(
            step,
            0,
            TrainingPhase.READY,
            0,
            step,
            0,
            (),
        )
        inputs = self.input_builder.build_validation(self.models, batch, loop)
        losses = compute_conditional_losses(self.models, inputs)
        latent = integrate_latent_flow(
            self.ema_latent_flow,
            inputs.flow_sample.noise,
            inputs.conditions,
            inputs.latent_mask,
            inputs.minimum_flow_steps,
        )
        targets = self.acoustic.acoustic_targets(
            batch.mel,
            batch.jdc_mel,
            batch.frame_mask,
        )
        synthesis = self._synthesize(
            latent,
            inputs.latent_mask,
            targets.f0,
            batch,
            step,
        )
        weights = self.schedules.conditional_weights(step)
        metrics = tuple(
            _metric(name, value)
            for name, value in zip(
                self.schedules.conditional_names,
                losses.values(),
                strict=True,
            )
        )
        metrics = (*metrics, _metric("conditional_total", losses.total(weights)))
        samples = []
        for index, recording in enumerate(recordings):
            frame_count = int(batch.frame_lengths[index])
            sample_count = int(batch.waveform_lengths[index])
            ground_truth, prediction = trim_waveform_pair(
                batch.waveform[index : index + 1],
                synthesis.waveform[index : index + 1],
                sample_count,
            )
            target_mel, predicted_mel = self._artifact_mels(
                batch.waveform[index : index + 1, :, :sample_count],
                synthesis.waveform[index : index + 1, :, :sample_count],
            )
            f0 = trim_signal_pair(
                targets.f0[index],
                synthesis.acoustic.f0[index],
                frame_count,
            )
            n = trim_signal_pair(
                targets.n[index],
                synthesis.acoustic.n[index],
                frame_count,
            )
            samples.append(
                ConditionalValidationSample(
                    recording.audio_file_id,
                    metrics,
                    ValidationArtifactSet(
                        ground_truth,
                        prediction,
                        _cpu(latent[index]),
                        f0,
                        n,
                        (_cpu(target_mel[0]), _cpu(predicted_mel[0])),
                        _cpu(inputs.alignment.soft_alignment[index]),
                    ),
                    derive_seed(
                        self.runtime_seed,
                        step,
                        recording.audio_file_id,
                        "validation",
                    ),
                )
            )
        return tuple(samples)

    def _synthesize(
        self,
        latent: Tensor,
        latent_mask: Tensor,
        target_f0: Tensor,
        batch: object,
        step: int,
    ) -> ConditionalSynthesis:
        predicted = self.acoustic.feature_linear(
            latent,
            latent_mask,
            batch.frame_mask,
        )
        acoustic = AcousticFeatures(target_f0, predicted.n)
        decoded = self.acoustic.decoder(
            latent,
            acoustic.f0,
            acoustic.n,
            latent_mask,
            batch.frame_mask,
        )
        generated = []
        for index, audio_id in enumerate(batch.recording_ids):
            generator = self._generator(step, audio_id, "source")
            generated.append(
                self.acoustic.generator(
                    decoded.features[index : index + 1],
                    decoded.f0[index : index + 1],
                    decoded.mask[index : index + 1],
                    generator,
                )
            )
        waveform = torch.cat(generated, dim=0)
        sample_mask = batch.frame_mask.repeat_interleave(
            self.acoustic.output_hop,
            dim=-1,
        )
        return ConditionalSynthesis(acoustic, decoded, waveform, sample_mask)

    def _artifact_mels(
        self,
        target: Tensor,
        prediction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        transform = self.acoustic.reconstruction_loss.transforms[0]
        return transform(target[:, 0]), transform(prediction[:, 0])

    def _generator(
        self,
        step: int,
        audio_file_id: UUID,
        view: str,
    ) -> torch.Generator:
        seed = derive_seed(
            self.runtime_seed,
            step,
            audio_file_id,
            view,
        )
        return torch.Generator(device=self.device).manual_seed(seed)


def _metric(name: str, value: Tensor) -> TrainingMetric:
    return TrainingMetric(name, float(value.detach().cpu()))


def _cpu(value: Tensor) -> Tensor:
    return value.detach().cpu().clone()
