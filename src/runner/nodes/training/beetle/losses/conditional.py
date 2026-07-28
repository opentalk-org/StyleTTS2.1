from dataclasses import dataclass, fields

from torch import Tensor

from ..models.modules.aligner import AlignerOutput
from ..models.modules.audio import AcousticFeatures
from ..models.modules.conditioning import ProjectedConditions
from ..models.modules.embeddings import AcousticStatistics
from ..models.modules.latent_flow import FlowTrainingSample
from ..models.conditional import ConditionalModels
from .alignment import compute_alignment_losses
from .duration import duration_flow_loss
from .embeddings import (
    embedding_consistency_loss,
    speaker_adversarial_loss,
    style_statistics_loss,
    supervised_contrastive_loss,
)
from .flow import base_flow_loss


@dataclass(frozen=True)
class ConditionalLossWeights:
    duration_flow: float
    latent_flow: float
    shortcut: float
    align_s2s: float
    align_mono: float
    align_ctc: float
    voice_contrastive: float
    voice_ge2e: float
    style_contrastive: float
    style_ge2e: float
    style_speaker_adversarial: float
    style_statistics: float
    style_reencoding: float

    @classmethod
    def from_shared(cls, value: float) -> "ConditionalLossWeights":
        return cls(*(value for _ in range(13)))

    def values(self) -> tuple[float, ...]:
        return (
            self.duration_flow,
            self.latent_flow,
            self.shortcut,
            self.align_s2s,
            self.align_mono,
            self.align_ctc,
            self.voice_contrastive,
            self.voice_ge2e,
            self.style_contrastive,
            self.style_ge2e,
            self.style_speaker_adversarial,
            self.style_statistics,
            self.style_reencoding,
        )


@dataclass(frozen=True)
class ConditionalBatchStatistics:
    target_seconds: Tensor
    target_padding_ratio: Tensor
    full_audio_ratio: Tensor
    pre_text_available_ratio: Tensor
    post_text_available_ratio: Tensor
    pre_audio_available_ratio: Tensor
    post_audio_available_ratio: Tensor
    pre_text_random_drop_ratio: Tensor
    post_text_random_drop_ratio: Tensor
    pre_audio_random_drop_ratio: Tensor
    post_audio_random_drop_ratio: Tensor
    pre_text_effective_drop_ratio: Tensor
    post_text_effective_drop_ratio: Tensor
    pre_audio_effective_drop_ratio: Tensor
    post_audio_effective_drop_ratio: Tensor

    def named_values(self) -> tuple[tuple[str, Tensor], ...]:
        return tuple(
            (field.name, getattr(self, field.name))
            for field in fields(self)
        )


@dataclass(frozen=True)
class ConditionalLossInput:
    duration_nll: Tensor
    phoneme_mask: Tensor
    flow_sample: FlowTrainingSample
    conditions: ProjectedConditions
    latent_mask: Tensor
    alignment: AlignerOutput
    phonemes: Tensor
    alignment_mask: Tensor
    target_latent_mask: Tensor
    target_style: Tensor
    style_view_latent: Tensor
    style_view_mask: Tensor
    voice_view_latent: Tensor
    voice_view_mask: Tensor
    voice_group_ids: Tensor
    style_group_ids: Tensor
    style_positive_weights: Tensor
    speaker_ids: Tensor
    statistics_target: AcousticStatistics
    acoustic_target: AcousticFeatures
    contrastive_temperature: float
    reversal_scale: float
    consistency_cosine_weight: float
    consistency_mse_weight: float
    align_blank_id: int
    minimum_flow_steps: int
    batch_statistics: ConditionalBatchStatistics


@dataclass(frozen=True)
class ConditionalLossOutput:
    duration_flow: Tensor
    latent_flow: Tensor
    shortcut: Tensor
    align_s2s: Tensor
    align_mono: Tensor
    align_ctc: Tensor
    voice_contrastive: Tensor
    voice_ge2e: Tensor
    style_contrastive: Tensor
    style_ge2e: Tensor
    style_speaker_adversarial: Tensor
    style_statistics: Tensor
    style_reencoding: Tensor

    def values(self) -> tuple[Tensor, ...]:
        return (
            self.duration_flow,
            self.latent_flow,
            self.shortcut,
            self.align_s2s,
            self.align_mono,
            self.align_ctc,
            self.voice_contrastive,
            self.voice_ge2e,
            self.style_contrastive,
            self.style_ge2e,
            self.style_speaker_adversarial,
            self.style_statistics,
            self.style_reencoding,
        )

    def total(self, weights: ConditionalLossWeights) -> Tensor:
        products = tuple(
            loss * weight
            for loss, weight in zip(self.values(), weights.values(), strict=True)
        )
        return sum(products[1:], products[0])


def compute_conditional_losses(
    models: ConditionalModels,
    inputs: ConditionalLossInput,
) -> ConditionalLossOutput:
    style_views = models.style_encoder(inputs.style_view_latent, inputs.style_view_mask)
    voice_views = models.voice_encoder(inputs.voice_view_latent, inputs.voice_view_mask)
    prediction = models.latent_flow(
        inputs.flow_sample.state,
        inputs.flow_sample.time,
        inputs.flow_sample.step_level,
        inputs.conditions,
        inputs.latent_mask,
    )
    generated_latent = (
        inputs.flow_sample.state + (1 - inputs.flow_sample.time) * prediction
    ) * inputs.latent_mask
    generated_style = models.style_encoder(
        generated_latent,
        inputs.target_latent_mask,
    )
    alignment = compute_alignment_losses(
        inputs.alignment,
        inputs.phonemes,
        inputs.phoneme_mask[:, 0],
        inputs.alignment_mask,
        inputs.align_blank_id,
    )
    speaker_logits = models.style_speaker_classifier(
        inputs.target_style, inputs.reversal_scale
    )
    statistics = models.style_statistics_head(inputs.target_style)
    return ConditionalLossOutput(
        duration_flow=duration_flow_loss(inputs.duration_nll, inputs.phoneme_mask),
        latent_flow=base_flow_loss(
            prediction,
            inputs.flow_sample.velocity,
            inputs.latent_mask,
        ),
        shortcut=prediction.new_zeros(()),
        align_s2s=alignment.s2s,
        align_mono=alignment.mono,
        align_ctc=alignment.ctc,
        voice_contrastive=supervised_contrastive_loss(
            voice_views,
            inputs.voice_group_ids,
            inputs.contrastive_temperature,
        ),
        voice_ge2e=models.voice_ge2e(voice_views, inputs.voice_group_ids),
        style_contrastive=supervised_contrastive_loss(
            style_views,
            inputs.style_group_ids,
            inputs.contrastive_temperature,
            inputs.style_positive_weights,
        ),
        style_ge2e=models.style_ge2e(style_views, inputs.style_group_ids),
        style_speaker_adversarial=speaker_adversarial_loss(
            speaker_logits,
            inputs.speaker_ids,
        ),
        style_statistics=style_statistics_loss(statistics, inputs.statistics_target),
        style_reencoding=embedding_consistency_loss(
            generated_style,
            inputs.target_style.detach(),
            inputs.consistency_cosine_weight,
            inputs.consistency_mse_weight,
        ),
    )
