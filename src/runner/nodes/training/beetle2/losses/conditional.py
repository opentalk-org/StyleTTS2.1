from dataclasses import dataclass

from torch import Tensor

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
class ConditionalLossInput:
    duration_nll: Tensor
    phoneme_mask: Tensor
    flow_velocity: Tensor
    latent_mask: Tensor
    ctc_logits: Tensor
    s2s_logits: Tensor
    soft_alignment: Tensor
    hard_alignment: Tensor
    phonemes: Tensor
    alignment_mask: Tensor
    target_style: Tensor
    voice_group_ids: Tensor
    style_group_ids: Tensor
    style_positive_weights: Tensor
    speaker_ids: Tensor
    statistics_target: Tensor
    contrastive_temperature: float
    consistency_cosine_weight: float
    consistency_mse_weight: float
    align_blank_id: int


@dataclass(frozen=True)
class ConditionalModelOutput:
    flow_prediction: Tensor
    generated_style: Tensor
    style_views: Tensor
    voice_views: Tensor
    speaker_logits: Tensor
    statistics: Tensor
    voice_ge2e: Tensor
    style_ge2e: Tensor


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
    inputs: ConditionalLossInput,
    outputs: ConditionalModelOutput,
) -> ConditionalLossOutput:
    alignment = compute_alignment_losses(
        inputs.ctc_logits,
        inputs.s2s_logits,
        inputs.soft_alignment,
        inputs.hard_alignment,
        inputs.phonemes,
        inputs.phoneme_mask[:, 0],
        inputs.alignment_mask,
        inputs.align_blank_id,
    )
    return ConditionalLossOutput(
        duration_flow=duration_flow_loss(inputs.duration_nll, inputs.phoneme_mask),
        latent_flow=base_flow_loss(
            outputs.flow_prediction,
            inputs.flow_velocity,
            inputs.latent_mask,
        ),
        shortcut=outputs.flow_prediction.new_zeros(()),
        align_s2s=alignment.s2s,
        align_mono=alignment.mono,
        align_ctc=alignment.ctc,
        voice_contrastive=supervised_contrastive_loss(
            outputs.voice_views,
            inputs.voice_group_ids,
            inputs.contrastive_temperature,
        ),
        voice_ge2e=outputs.voice_ge2e,
        style_contrastive=supervised_contrastive_loss(
            outputs.style_views,
            inputs.style_group_ids,
            inputs.contrastive_temperature,
            inputs.style_positive_weights,
        ),
        style_ge2e=outputs.style_ge2e,
        style_speaker_adversarial=speaker_adversarial_loss(
            outputs.speaker_logits,
            inputs.speaker_ids,
        ),
        style_statistics=style_statistics_loss(
            outputs.statistics,
            inputs.statistics_target,
        ),
        style_reencoding=embedding_consistency_loss(
            outputs.generated_style,
            inputs.target_style.detach(),
            inputs.consistency_cosine_weight,
            inputs.consistency_mse_weight,
        ),
    )
