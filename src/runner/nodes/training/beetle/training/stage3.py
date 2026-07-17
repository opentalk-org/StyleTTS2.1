from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config.training import StageConfig
from ..data.records import BeetleBatch
from ..data.sampling import derive_seed
from ..losses.acoustic import masked_f0_mse, masked_kl_standard_normal, masked_n_mse
from ..losses.adversarial import discriminator_step_loss, generator_step_loss
from ..losses.stage2 import Stage2LossInput, compute_stage2_losses
from ..models.model import Stage1Models
from ..models.modules.audio import AcousticFeatures
from ..models.modules.decoder import DecoderOutput
from ..models.modules.latent_flow import integrate_latent_flow
from ..models.stage2 import Stage2Models
from .callbacks import TrainingMetric
from .checkpoint import GradientTarget, NamedModuleGradients, StateKind
from .loop import LoopIntervals
from .optimizer import OptimizerSet
from .stage1 import Stage1Trainer
from .stage1_setup import tensor_metric
from .stage2_setup import (
    Stage2InputBuilder,
    Stage3Schedules,
    build_stage3_optimizers,
    named_trainable_stage2_modules,
    trainable_stage2_modules,
    update_latent_flow_ema,
)
from .state import LoopState, StageKind, capture_gradients

__all__ = ["Stage3Trainer", "build_stage3_optimizers"]


@dataclass(frozen=True)
class ConditionalSynthesis:
    acoustic: AcousticFeatures
    decoded: DecoderOutput
    waveform: Tensor
    sample_mask: Tensor


class Stage3Trainer(Stage1Trainer):
    stage = StageKind.STAGE3

    def __init__(
        self,
        stage1: Stage1Models,
        stage2: Stage2Models,
        ema_latent_flow: nn.Module,
        stage_config: StageConfig,
        runtime_seed: int,
        device: torch.device,
        optimizers: OptimizerSet,
        intervals: LoopIntervals,
        config_fingerprint: str,
        data_fingerprint: str,
        initial_loop: LoopState,
        input_builder: Stage2InputBuilder,
    ) -> None:
        if stage2.audio_encoder is not stage1.audio_encoder:
            raise ValueError("Stage 3 requires one shared audio encoder")
        if stage2.f0_extractor is not stage1.f0_extractor:
            raise ValueError("Stage 3 requires one shared F0 extractor")
        super().__init__(
            stage1,
            stage_config,
            runtime_seed,
            device,
            optimizers,
            intervals,
            config_fingerprint,
            data_fingerprint,
            initial_loop,
        )
        self.stage2_models = stage2.to(device).train()
        self.ema_latent_flow = ema_latent_flow.to(device).requires_grad_(False).eval()
        self.input_builder = input_builder
        self.schedules = Stage3Schedules.from_config(stage_config)
        self.models.discriminators.to(device).requires_grad_(True).train()
        for module in self._inference_modules():
            module.requires_grad_(True).train()
        for module in trainable_stage2_modules(self.stage2_models):
            module.requires_grad_(True).train()
        for module in self._frozen_modules():
            module.requires_grad_(False).eval()

    def discriminator_backward(self, batch: BeetleBatch) -> tuple[TrainingMetric, ...]:
        waveform, mel, frame_mask = self._inputs(batch)
        with torch.no_grad(), self._autocast():
            inputs = self.input_builder.build(self.stage2_models, batch, self._loop)
            posterior = self._synthesize(mel, frame_mask, "discriminator-posterior")
            conditional = self._conditional(inputs, frame_mask, "discriminator")
        with self._autocast():
            posterior_loss = discriminator_step_loss(
                self.models.discriminators, waveform, posterior.waveform
            )
            conditional_loss = discriminator_step_loss(
                self.models.discriminators, waveform, conditional.waveform
            )
            loss = (posterior_loss + conditional_loss) * 0.5
            weighted = loss * self._weights().discriminator
        self.optimizers.group("discriminator").backward(
            weighted / self.accumulation_steps
        )
        return (
            tensor_metric("discriminator", loss),
            tensor_metric("discriminator_total", weighted),
        )

    def generator_backward(self, batch: BeetleBatch) -> tuple[TrainingMetric, ...]:
        waveform, mel, frame_mask = self._inputs(batch)
        inputs = self.input_builder.build(self.stage2_models, batch, self._loop)
        targets = self.models.acoustic_targets(mel, frame_mask)
        with self._autocast():
            posterior = self._synthesize(mel, frame_mask, "generator-posterior")
            conditional = self._conditional(inputs, frame_mask, "generator")
            encoder_kl = masked_kl_standard_normal(
                posterior.posterior.mean,
                posterior.posterior.log_scale,
                posterior.posterior.mask,
            )
            f0 = self._mean_acoustic_loss(
                posterior.acoustic,
                conditional.acoustic,
                targets,
                posterior.decoded.mask,
                True,
            )
            n = self._mean_acoustic_loss(
                posterior.acoustic,
                conditional.acoustic,
                targets,
                posterior.decoded.mask,
                False,
            )
            reconstruction = 0.5 * (
                self.models.reconstruction_loss(
                    posterior.waveform, waveform, posterior.sample_mask
                ).total
                + self.models.reconstruction_loss(
                    conditional.waveform, waveform, conditional.sample_mask
                ).total
            )
            posterior_adv = generator_step_loss(
                self.models.discriminators, waveform, posterior.waveform
            )
            conditional_adv = generator_step_loss(
                self.models.discriminators, waveform, conditional.waveform
            )
            adversarial = 0.5 * (
                posterior_adv.adversarial + conditional_adv.adversarial
            )
            feature_matching = 0.5 * (
                posterior_adv.feature_matching + conditional_adv.feature_matching
            )
            stage2_losses = compute_stage2_losses(
                self.stage2_models, self.ema_latent_flow, inputs
            )
            weights = self._weights()
            acoustic_total = (
                encoder_kl * weights.encoder_kl
                + f0 * weights.f0
                + n * weights.n
                + reconstruction * weights.reconstruction
                + adversarial * weights.generator_adversarial
                + feature_matching * weights.feature_matching
            )
            flow_total = stage2_losses.total(
                self.schedules.stage2.weights(self._loop.optimizer_step)
            )
            total = acoustic_total + flow_total
        self.optimizers.group("generator").backward(total / self.accumulation_steps)
        stage2_names = tuple(
            weight.name for weight in self.schedules.stage2.state(0).weights
        )
        return (
            tensor_metric("encoder_kl", encoder_kl),
            tensor_metric("f0", f0),
            tensor_metric("n", n),
            tensor_metric("reconstruction", reconstruction),
            tensor_metric("generator_adversarial", adversarial),
            tensor_metric("feature_matching", feature_matching),
            *tuple(
                tensor_metric(name, value)
                for name, value in zip(
                    stage2_names, stage2_losses.values(), strict=True
                )
            ),
            tensor_metric("generator_total", total),
        )

    def optimizer_step(self, optimizer_step: int) -> tuple[TrainingMetric, ...]:
        metrics = self.optimizers.step(optimizer_step)
        update_latent_flow_ema(
            self.ema_latent_flow,
            self.stage2_models.latent_flow,
            self.stage2_models.latent_flow.config.ema_decay,
        )
        return metrics

    def _conditional(
        self,
        inputs: Stage2LossInput,
        frame_mask: Tensor,
        view: str,
    ) -> ConditionalSynthesis:
        latent = integrate_latent_flow(
            self.stage2_models.latent_flow,
            inputs.flow_sample.noise,
            inputs.conditions,
            inputs.latent_mask,
            1,
        )
        acoustic = self.models.feature_linear(latent, inputs.latent_mask, frame_mask)
        decoded = self.models.decoder(
            latent,
            acoustic.f0,
            acoustic.n,
            inputs.latent_mask,
            frame_mask,
        )
        source_seed = derive_seed(
            self.runtime_seed,
            self.stage,
            self._loop.cycle,
            self._loop.batch_index,
            view,
            "source",
        )
        source = torch.Generator(device=self.device).manual_seed(source_seed)
        waveform = self.models.generator(
            decoded.features, decoded.f0, decoded.mask, source
        )
        sample_mask = frame_mask.repeat_interleave(
            self.models.generator.config.output_hop(), dim=-1
        )
        return ConditionalSynthesis(acoustic, decoded, waveform, sample_mask)

    def _mean_acoustic_loss(
        self,
        posterior: AcousticFeatures,
        conditional: AcousticFeatures,
        target: AcousticFeatures,
        mask: Tensor,
        pitch: bool,
    ) -> Tensor:
        loss = masked_f0_mse if pitch else masked_n_mse
        generated = posterior.f0 if pitch else posterior.n
        conditioned = conditional.f0 if pitch else conditional.n
        expected = target.f0 if pitch else target.n
        return 0.5 * (
            loss(generated, expected, mask) + loss(conditioned, expected, mask)
        )

    def _inference_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.models.audio_encoder,
            self.models.feature_linear,
            self.models.decoder,
            self.models.generator,
        )

    def _frozen_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.models.f0_extractor,
            self.stage2_models.text_encoder,
        )

    def _state_modules(self) -> tuple[tuple[str, StateKind, nn.Module], ...]:
        stage1 = (
            *super()._state_modules(),
            ("discriminators", StateKind.DISCRIMINATOR, self.models.discriminators),
        )
        stage2 = tuple(
            (name, StateKind.MODEL, module)
            for name, module in named_trainable_stage2_modules(self.stage2_models)
        )
        helpers = (
            ("text_encoder", StateKind.FROZEN_MODEL, self.stage2_models.text_encoder),
            ("latent_flow", StateKind.EMA, self.ema_latent_flow),
        )
        return (*stage1, *stage2, *helpers)

    def _gradients(self) -> tuple[NamedModuleGradients, ...]:
        return tuple(
            NamedModuleGradients(name, capture_gradients(module))
            for name, kind, module in self._state_modules()
            if kind in (StateKind.MODEL, StateKind.DISCRIMINATOR)
        )

    def _gradient_targets(self) -> tuple[GradientTarget, ...]:
        return tuple(
            GradientTarget(name, module)
            for name, kind, module in self._state_modules()
            if kind in (StateKind.MODEL, StateKind.DISCRIMINATOR)
        )
