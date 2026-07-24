from dataclasses import replace

import torch

from ...config import BeetleConfig
from ...config.training import ConditioningObjectiveConfig
from ...data.records import BeetleBatch
from ...data.sampling import derive_seed
from ...losses.conditional import ConditionalLossInput
from ...models.conditional import ConditionalModels
from ...models.modules.conditioning import ConditionVectors, align_phoneme_tokens
from ...models.modules.latent_flow import sample_flow_training_case
from ..aligned_window import (
    apply_window_ranges,
    sample_window_ranges,
    seconds_to_latent_frames,
)
from ..distributed import DistributedRuntime
from ..setup import ConditionalInputBuilder
from ..state import LoopState
from .context import (
    encode_audio_context,
    encode_text_context,
    encode_view_latents,
)
from .features import (
    ConditionalAcousticTargets,
    WaveformMelExtractor,
    acoustic_statistics,
    group_ids,
    style_weights,
)
from .input_types import (
    CoreConditionalInput,
    SpeakerIndex,
    build_rate_conditions,
    keep_all_conditions,
    require_batch,
)
from .statistics import ContextAvailability, conditional_batch_statistics


class DefaultConditionalInputBuilder(ConditionalInputBuilder):
    def __init__(
        self,
        config: BeetleConfig,
        speaker_index: SpeakerIndex,
        runtime: DistributedRuntime,
    ) -> None:
        self.config = config
        self.speaker_index = speaker_index
        self.settings: ConditioningObjectiveConfig = config.conditioning_objective
        self.runtime = runtime
        self.device = runtime.device
        self.mel_extractor = WaveformMelExtractor(config).to(runtime.device)

    def build(
        self,
        models: ConditionalModels,
        batch: object,
        loop: LoopState,
        acoustic_targets: ConditionalAcousticTargets,
    ) -> ConditionalLossInput:
        values = require_batch(batch).to(self.device)
        return self._build(models, values, loop, acoustic_targets, False)

    def acoustic_targets(
        self,
        models: ConditionalModels,
        batch: object,
    ) -> ConditionalAcousticTargets:
        return acoustic_statistics(models, require_batch(batch).to(self.device))

    def build_validation(
        self,
        models: ConditionalModels,
        batch: object,
        loop: LoopState,
    ) -> ConditionalLossInput:
        values = require_batch(batch).to(self.device)
        targets = acoustic_statistics(models, values)
        return self._build(models, values, loop, targets, True)

    def _build(
        self,
        models: ConditionalModels,
        values: BeetleBatch,
        loop: LoopState,
        acoustic_targets: ConditionalAcousticTargets,
        validation: bool,
    ) -> ConditionalLossInput:
        core = self._core(models, values, loop, acoustic_targets, validation)
        duration_tokens = models.duration_phoneme_encoder(
            core.phoneme.tokens,
            core.phoneme.mask,
        )
        duration_condition, conditions = build_rate_conditions(
            models.condition_bank,
            core.vectors,
            duration_tokens,
            core.aligned_tokens,
            core.keep,
        )
        duration_nll = models.duration_predictor(
            core.alignment.durations.detach().clamp_min(1).unsqueeze(1),
            duration_condition,
            core.phoneme.mask,
            self._generator(loop, "duration"),
        )
        flow_sample = sample_flow_training_case(
            core.window.posterior.latent,
            core.window.posterior.mask,
            self.config.architecture.latent_flow.minimum_steps,
            self.config.architecture.latent_flow.base_case_probability,
            self._generator(loop, "flow"),
        )
        style_latent, style_mask = encode_view_latents(
            models,
            self.mel_extractor,
            values.style_views,
            values.style_view_lengths,
            self._generator(loop, "style-views"),
        )
        voice_latent, voice_mask = encode_view_latents(
            models,
            self.mel_extractor,
            values.voice_views,
            values.voice_view_lengths,
            self._generator(loop, "voice-views"),
        )
        style_ids = group_ids(values.style_views, self.device)
        speaker_ids = group_ids(values.voice_views, self.device)
        settings = self.settings
        return ConditionalLossInput(
            duration_nll=duration_nll,
            phoneme_mask=core.phoneme.mask,
            flow_sample=flow_sample,
            conditions=conditions,
            latent_mask=core.window.posterior.mask,
            alignment=core.alignment,
            phonemes=values.phoneme_ids,
            alignment_mask=core.posterior.mask,
            target_latent_mask=core.window.posterior.mask,
            target_style=core.vectors.style,
            style_view_latent=style_latent,
            style_view_mask=style_mask,
            voice_view_latent=voice_latent,
            voice_view_mask=voice_mask,
            voice_group_ids=speaker_ids,
            style_group_ids=style_ids,
            style_positive_weights=style_weights(values.style_distances),
            speaker_ids=self.speaker_index.resolve(values.speaker_ids, self.device),
            statistics_target=core.acoustic_targets.statistics,
            acoustic_target=core.acoustic_targets.features,
            contrastive_temperature=settings.contrastive_temperature,
            reversal_scale=settings.reversal_scale,
            consistency_cosine_weight=settings.consistency_cosine_weight,
            consistency_mse_weight=settings.consistency_mse_weight,
            align_blank_id=self.config.architecture.aligner.blank_id,
            minimum_flow_steps=self.config.architecture.latent_flow.minimum_steps,
            batch_statistics=core.batch_statistics,
        )

    def _core(
        self,
        models: ConditionalModels,
        values: BeetleBatch,
        loop: LoopState,
        acoustic_targets: ConditionalAcousticTargets,
        validation: bool,
    ) -> CoreConditionalInput:
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
        latent_tokens = models.latent_phoneme_encoder(phoneme.tokens, phoneme.mask)
        aligned_tokens, aligned_mask = align_phoneme_tokens(
            latent_tokens,
            alignment.hard_alignment.detach(),
        )
        if aligned_mask.shape != posterior.mask.shape:
            raise ValueError("aligned phonemes and posterior latents must match")
        torch._assert_async(
            torch.all(aligned_mask == posterior.mask),
            "aligned phonemes and posterior latents must match",
        )
        target_style = models.style_encoder(posterior.latent, posterior.mask)
        target_voice = models.voice_encoder(posterior.latent, posterior.mask)
        ranges = sample_window_ranges(
            posterior.mask[:, 0].sum(dim=1),
            self._latent_frames(0.4),
            None if validation else self._latent_frames(self.config.data.maximum_seconds),
            self._latent_frames(0.4),
            self._latent_frames(5.0),
            self.config.training.full_audio_ratio,
            self._generator(loop, "aligned-window"),
            validation,
        )
        window = apply_window_ranges(
            posterior,
            aligned_tokens,
            alignment.hard_alignment,
            ranges,
        )
        pre_text, pre_text_available = encode_text_context(
            models,
            phoneme.tokens,
            window.pre_phoneme_mask,
        )
        post_text, post_text_available = encode_text_context(
            models,
            phoneme.tokens,
            window.post_phoneme_mask,
        )
        pre_audio, pre_audio_available = encode_audio_context(
            models,
            window.pre_audio,
            window.pre_audio_mask,
        )
        post_audio, post_audio_available = encode_audio_context(
            models,
            window.post_audio,
            window.post_audio_mask,
        )
        vectors = ConditionVectors(
            style=target_style,
            voice=target_voice,
            pooled_phoneme=phoneme.pooled,
            pre_text=pre_text,
            post_text=post_text,
            pre_audio=pre_audio,
            post_audio=post_audio,
            language=models.language_embedding(values.language_ids),
        )
        sampled_keep = (
            keep_all_conditions(values.waveform.shape[0], self.device)
            if validation
            else self.runtime.unwrap(models.condition_bank).sample_keep(
                values.waveform.shape[0],
                self.device,
                self.config.architecture.conditioning.dropout,
                self._generator(loop, "condition-dropout"),
            )
        )
        keep = replace(
            sampled_keep,
            pre_text=sampled_keep.pre_text & pre_text_available,
            post_text=sampled_keep.post_text & post_text_available,
            pre_audio=sampled_keep.pre_audio & pre_audio_available,
            post_audio=sampled_keep.post_audio & post_audio_available,
        )
        statistics = conditional_batch_statistics(
            window,
            ContextAvailability(
                pre_text_available,
                post_text_available,
                pre_audio_available,
                post_audio_available,
            ),
            sampled_keep,
            keep,
            (
                self.config.audio.hop_length
                * self.config.architecture.posterior.downsample_rate
                / self.config.audio.sample_rate
            ),
        )
        return CoreConditionalInput(
            acoustic_targets,
            posterior,
            alignment,
            phoneme,
            window.aligned_phonemes,
            vectors,
            keep,
            window,
            statistics,
        )

    def _latent_frames(self, seconds: float) -> int:
        return seconds_to_latent_frames(
            seconds,
            self.config.audio.sample_rate,
            self.config.audio.hop_length,
            self.config.architecture.posterior.downsample_rate,
        )

    def _generator(self, loop: LoopState, label: str) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(
            derive_seed(self.config.runtime.seed, loop.cycle, loop.batch_index, label)
        )
