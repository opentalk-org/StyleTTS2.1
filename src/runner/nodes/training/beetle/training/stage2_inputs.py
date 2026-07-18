from typing import Protocol

import torch
from torch import Tensor

from ..config import BeetleConfig
from ..config.training import Stage2ObjectiveConfig
from ..data.records import BeetleBatch
from ..data.sampling import derive_seed
from ..losses.stage2 import Stage2LossInput
from ..models.modules.conditioning import (
    ConditionBank,
    ConditionKeep,
    ConditionVectors,
    ProjectedConditions,
    pairwise_pool_tokens,
)
from ..models.modules.latent_flow import sample_flow_training_case
from ..models.stage2 import Stage2Models
from .distributed import DistributedRuntime
from .stage2_features import (
    WaveformMelExtractor,
    acoustic_statistics,
    boundary_pool,
    group_ids,
    style_weights,
)
from .stage2_setup import Stage2InputBuilder
from .state import LoopState


class SpeakerIndex(Protocol):
    def resolve(
        self, voice_ids: tuple[str | None, ...], device: torch.device
    ) -> Tensor: ...


def build_rate_conditions(
    bank: ConditionBank,
    vectors: ConditionVectors,
    duration_phoneme: Tensor,
    latent_phoneme: Tensor,
    keep: ConditionKeep,
) -> tuple[Tensor, ProjectedConditions]:
    duration = vectors.at_rate(duration_phoneme).dropped_concatenated(keep)
    latent = bank(vectors.at_rate(latent_phoneme), keep)
    return duration, latent


class DefaultStage2InputBuilder(Stage2InputBuilder):
    def __init__(
        self,
        config: BeetleConfig,
        speaker_index: SpeakerIndex,
        runtime: DistributedRuntime,
    ) -> None:
        self.config = config
        self.speaker_index = speaker_index
        self.settings: Stage2ObjectiveConfig = config.stage2_objective
        self.runtime = runtime
        self.device = runtime.device
        self.mel_extractor = WaveformMelExtractor(config).to(runtime.device)

    def build(
        self,
        models: Stage2Models,
        batch: object,
        loop: LoopState,
    ) -> Stage2LossInput:
        return self._build(models, batch, loop, False)

    def build_validation(
        self,
        models: Stage2Models,
        batch: object,
        loop: LoopState,
    ) -> Stage2LossInput:
        return self._build(models, batch, loop, True)

    def _build(
        self,
        models: Stage2Models,
        batch: object,
        loop: LoopState,
        validation: bool,
    ) -> Stage2LossInput:
        if not isinstance(batch, BeetleBatch):
            raise TypeError("Stage 2 input builder requires a BeetleBatch")
        values = batch.to(self.device)
        target_generator = self._generator(loop, "target-posterior")
        with torch.no_grad():
            posterior = models.audio_encoder(
                values.mel, values.frame_mask, target_generator
            )
        alignment = models.aligner(
            values.mel,
            values.frame_mask,
            values.phoneme_ids,
            values.phoneme_mask,
        )
        phoneme = models.phoneme_encoder(values.phoneme_ids, values.phoneme_mask)
        duration_tokens = models.duration_phoneme_encoder(phoneme.tokens, phoneme.mask)
        latent_tokens = models.latent_phoneme_encoder(phoneme.tokens, phoneme.mask)
        full_rate = torch.bmm(latent_tokens, alignment.hard_alignment.detach())
        aligned_tokens, aligned_mask = pairwise_pool_tokens(
            full_rate, values.frame_mask
        )
        if aligned_mask.shape != posterior.mask.shape:
            raise ValueError("aligned phonemes and posterior latents must match")
        target_style = models.style_encoder(posterior.latent, posterior.mask)
        target_voice = models.voice_encoder(posterior.latent, posterior.mask)
        vectors = ConditionVectors(
            style=target_style,
            voice=target_voice,
            pooled_phoneme=phoneme.pooled,
            pre_text=self._text_context(models, values, loop, True),
            post_text=self._text_context(models, values, loop, False),
            pre_audio=self._audio_context(models, values, loop, True),
            post_audio=self._audio_context(models, values, loop, False),
            language=models.language_embedding(values.language_ids),
        )
        keep = (
            keep_all_conditions(values.waveform.shape[0], self.device)
            if validation
            else self.runtime.unwrap(models.condition_bank).sample_keep(
                values.waveform.shape[0],
                self.device,
                self.config.architecture.conditioning.dropout,
                self._generator(loop, "condition-dropout"),
            )
        )
        duration_condition, conditions = build_rate_conditions(
            models.condition_bank,
            vectors,
            duration_tokens,
            aligned_tokens,
            keep,
        )
        duration_nll = models.duration_predictor(
            alignment.durations.detach().clamp_min(1).unsqueeze(1),
            duration_condition,
            phoneme.mask,
            self._generator(loop, "duration"),
        )
        flow_sample = sample_flow_training_case(
            posterior.latent,
            posterior.mask,
            self.config.architecture.latent_flow.minimum_steps,
            self.config.architecture.latent_flow.base_case_probability,
            self._generator(loop, "flow"),
        )
        style_latent, style_mask = self._view_latents(
            models,
            values.style_views,
            values.style_view_lengths,
            loop,
            "style-views",
        )
        voice_latent, voice_mask = self._view_latents(
            models,
            values.voice_views,
            values.voice_view_lengths,
            loop,
            "voice-views",
        )
        style_ids = group_ids(values.style_views, self.device)
        voice_ids = group_ids(values.voice_views, self.device)
        statistics = acoustic_statistics(models, values)
        settings = self.settings
        return Stage2LossInput(
            duration_nll=duration_nll,
            phoneme_mask=phoneme.mask,
            flow_sample=flow_sample,
            conditions=conditions,
            latent_mask=posterior.mask,
            alignment=alignment,
            phonemes=values.phoneme_ids,
            frame_mask=values.frame_mask,
            target_latent_mask=posterior.mask,
            target_style=target_style,
            style_view_latent=style_latent,
            style_view_mask=style_mask,
            voice_view_latent=voice_latent,
            voice_view_mask=voice_mask,
            voice_group_ids=voice_ids,
            style_group_ids=style_ids,
            style_positive_weights=style_weights(values.style_distances),
            speaker_ids=self.speaker_index.resolve(values.voice_ids, self.device),
            statistics_target=statistics,
            contrastive_temperature=settings.contrastive_temperature,
            reversal_scale=settings.reversal_scale,
            consistency_cosine_weight=settings.consistency_cosine_weight,
            consistency_mse_weight=settings.consistency_mse_weight,
            align_blank_id=self.config.architecture.aligner.blank_id,
            align_frame_reduction=self.config.architecture.aligner.frame_reduction,
            minimum_flow_steps=self.config.architecture.latent_flow.minimum_steps,
        )

    def _text_context(
        self,
        models: Stage2Models,
        batch: BeetleBatch,
        loop: LoopState,
        pre: bool,
    ) -> Tensor:
        ids = batch.pre_text_ids if pre else batch.post_text_ids
        lengths = batch.pre_text_lengths if pre else batch.post_text_lengths
        available = batch.pre_text_available if pre else batch.post_text_available
        if ids.shape[1] == 0:
            ids = torch.zeros(ids.shape[0], 1, dtype=torch.long, device=self.device)
        mask = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        mask = mask < lengths.unsqueeze(1)
        encoded = models.phoneme_encoder(ids, mask)
        tokens = models.context_phoneme_encoder(encoded.tokens, encoded.mask)
        conditioning = self.config.architecture.conditioning
        counts = torch.randint(
            conditioning.boundary_k_min,
            conditioning.boundary_k_max + 1,
            (ids.shape[0],),
            device=self.device,
            generator=self._generator(loop, "pre-text" if pre else "post-text"),
        )
        return boundary_pool(tokens, mask, available, counts, pre)

    def _audio_context(
        self,
        models: Stage2Models,
        batch: BeetleBatch,
        loop: LoopState,
        pre: bool,
    ) -> Tensor:
        waveform = batch.pre_audio if pre else batch.post_audio
        lengths = batch.pre_audio_lengths if pre else batch.post_audio_lengths
        available = batch.pre_audio_available if pre else batch.post_audio_available
        mel = self.mel_extractor(waveform, lengths)
        with torch.no_grad():
            posterior = models.audio_encoder(
                mel.values,
                mel.mask,
                self._generator(loop, "pre-audio" if pre else "post-audio"),
            )
        pooled = models.context_audio_encoder(posterior.latent, posterior.mask)
        return pooled * available.unsqueeze(1)

    def _view_latents(
        self,
        models: Stage2Models,
        waveforms: Tensor,
        lengths: Tensor,
        loop: LoopState,
        label: str,
    ) -> tuple[Tensor, Tensor]:
        groups, views, channels, samples = waveforms.shape
        flattened = waveforms.reshape(groups * views, channels, samples)
        flat_lengths = lengths.reshape(groups * views)
        mel = self.mel_extractor(flattened, flat_lengths)
        with torch.no_grad():
            posterior = models.audio_encoder(
                mel.values,
                mel.mask,
                self._generator(loop, label),
            )
        return posterior.latent, posterior.mask

    def _generator(self, loop: LoopState, label: str) -> torch.Generator:
        seed = derive_seed(
            self.config.runtime.seed,
            loop.stage,
            loop.cycle,
            loop.batch_index,
            label,
        )
        return torch.Generator(device=self.device).manual_seed(seed)


def keep_all_conditions(batch_size: int, device: torch.device) -> ConditionKeep:
    if batch_size <= 0:
        raise ValueError("condition batch size must be positive")
    keep = torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device)
    return ConditionKeep(keep, keep, keep, keep, keep, keep, keep, keep, keep)
