from uuid import UUID

import torch
from torch import Tensor, nn

from ...config.training import StageConfig
from ...data.sampling import derive_seed
from ...data.validation_types import ValidationRecording
from ...losses.stage2 import compute_stage2_losses
from ...models.model import Stage1Models
from ...models.modules.conditioning import ProjectedConditions
from ...models.modules.latent_flow import integrate_latent_flow
from ...models.stage2 import Stage2Models
from ..reporting import TrainingMetric
from ..loss_schedules import Stage2Schedules
from ..stage2_features import ConditionalSynthesis
from ..stage2_setup import Stage2InputBuilder
from ..state import LoopState, StageKind, TrainingPhase
from .batch import merge_validation_recordings
from .types import ValidationSampleResult, trim_waveform_pair


def one_step_ema_latent(
    ema_latent_flow: nn.Module,
    noise: Tensor,
    conditions: ProjectedConditions,
    mask: Tensor,
) -> Tensor:
    return integrate_latent_flow(ema_latent_flow, noise, conditions, mask, 1)


class Stage2ValidationEvaluator:
    stage = StageKind.STAGE2

    def __init__(
        self,
        stage1: Stage1Models,
        models: Stage2Models,
        ema_latent_flow: nn.Module,
        input_builder: Stage2InputBuilder,
        stage_config: StageConfig,
        runtime_seed: int,
        device: torch.device,
    ) -> None:
        self.stage1 = stage1
        self.models = models
        self.ema_latent_flow = ema_latent_flow
        self.input_builder = input_builder
        self.schedules = Stage2Schedules.from_config(stage_config)
        self.runtime_seed = runtime_seed
        self.device = device

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
            self.stage1.feature_linear,
            self.stage1.decoder,
            self.stage1.generator,
            self.ema_latent_flow,
        )

    def evaluate_samples(
        self,
        recordings: tuple[ValidationRecording, ...],
        step: int,
    ) -> tuple[ValidationSampleResult, ...]:
        _require_contrastive_groups(recordings)
        batch = merge_validation_recordings(recordings).to(self.device)
        loop = LoopState(
            self.stage,
            step,
            0,
            TrainingPhase.READY,
            0,
            step,
            0,
            (),
        )
        inputs = self.input_builder.build_validation(self.models, batch, loop)
        losses = compute_stage2_losses(self.models, self.ema_latent_flow, inputs)
        latent = one_step_ema_latent(
            self.ema_latent_flow,
            inputs.flow_sample.noise,
            inputs.conditions,
            inputs.latent_mask,
        )
        synthesis = self._synthesize(latent, inputs.latent_mask, batch, step)
        targets = self.stage1.acoustic_targets(batch.mel, batch.frame_mask)
        target_mel, predicted_mel = self._artifact_mels(
            batch.waveform,
            synthesis.waveform,
        )
        weights = self.schedules.weights(step)
        metrics = tuple(
            _metric(weight.name, value)
            for weight, value in zip(
                self.schedules.state(step).weights,
                losses.values(),
                strict=True,
            )
        )
        metrics = (*metrics, _metric("stage2_total", losses.total(weights)))
        samples = []
        for index, recording in enumerate(recordings):
            ground_truth, prediction = trim_waveform_pair(
                batch.waveform[index : index + 1],
                synthesis.waveform[index : index + 1],
                int(batch.waveform_lengths[index]),
            )
            samples.append(
                ValidationSampleResult(
                    recording.audio_file_id,
                    metrics,
                    ground_truth,
                    prediction,
                    _cpu(latent[index]),
                    (_cpu(targets.f0[index]), _cpu(synthesis.acoustic.f0[index])),
                    (_cpu(targets.n[index]), _cpu(synthesis.acoustic.n[index])),
                    (_cpu(target_mel[index]), _cpu(predicted_mel[index])),
                    _cpu(inputs.alignment.soft_alignment[index]),
                    derive_seed(
                        self.runtime_seed,
                        self.stage,
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
        batch: object,
        step: int,
    ) -> ConditionalSynthesis:
        acoustic = self.stage1.feature_linear(latent, latent_mask, batch.frame_mask)
        decoded = self.stage1.decoder(
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
                self.stage1.generator(
                    decoded.features[index : index + 1],
                    decoded.f0[index : index + 1],
                    decoded.mask[index : index + 1],
                    generator,
                )
            )
        waveform = torch.cat(generated, dim=0)
        sample_mask = batch.frame_mask.repeat_interleave(
            self.stage1.output_hop,
            dim=-1,
        )
        return ConditionalSynthesis(acoustic, decoded, waveform, sample_mask)

    def _artifact_mels(
        self,
        target: Tensor,
        prediction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        transform = self.stage1.reconstruction_loss.transforms[0]
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


def _require_contrastive_groups(
    recordings: tuple[ValidationRecording, ...],
) -> None:
    voice_ids = tuple(recording.batch.voice_ids[0] for recording in recordings)
    if len(recordings) < 2 or len(set(voice_ids)) < 2:
        raise ValueError(
            "conditional validation requires at least two recordings with distinct voices"
        )


def _metric(name: str, value: Tensor) -> TrainingMetric:
    return TrainingMetric(name, float(value.detach().cpu()))


def _cpu(value: Tensor) -> Tensor:
    return value.detach().cpu().clone()
