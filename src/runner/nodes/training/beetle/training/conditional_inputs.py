import torch
from torch import Tensor

from ..config import BeetleConfig
from ..config.training import ConditioningObjectiveConfig
from ..data.records import BeetleBatch
from ..data.sampling import derive_seed
from ..losses.conditional import ConditionalLossInput
from ..models.modules.conditioning import ConditionVectors, align_phoneme_tokens
from ..models.modules.latent_flow import sample_flow_training_case
from ..models.conditional import ConditionalModels
from .conditional_features import (
    WaveformMelExtractor,
    acoustic_statistics,
    boundary_pool,
    group_ids,
    style_weights,
)
from .conditional_input_types import CoreConditionalInput, SpeakerIndex
from .conditional_input_types import build_rate_conditions, keep_all_conditions, require_batch
from .distributed import DistributedRuntime
from .setup import ConditionalInputBuilder
from .state import LoopState


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
    ) -> ConditionalLossInput:
        return self._build(models, batch, loop, False)

    def build_validation(
        self,
        models: ConditionalModels,
        batch: object,
        loop: LoopState,
    ) -> ConditionalLossInput:
        return self._build(models, batch, loop, True)

    def _build(
        self,
        models: ConditionalModels,
        batch: object,
        loop: LoopState,
        validation: bool,
    ) -> ConditionalLossInput:
        values = require_batch(batch).to(self.device)
        core = self._core(models, values, loop, validation)
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
            core.posterior.latent,
            core.posterior.mask,
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
        speaker_ids = group_ids(values.voice_views, self.device)
        settings = self.settings
        return ConditionalLossInput(
            duration_nll=duration_nll,
            phoneme_mask=core.phoneme.mask,
            flow_sample=flow_sample,
            conditions=conditions,
            latent_mask=core.posterior.mask,
            alignment=core.alignment,
            phonemes=values.phoneme_ids,
            alignment_mask=core.posterior.mask,
            target_latent_mask=core.posterior.mask,
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
        )

    def _core(
        self,
        models: ConditionalModels,
        values: BeetleBatch,
        loop: LoopState,
        validation: bool,
    ) -> CoreConditionalInput:
        acoustic_targets = acoustic_statistics(models, values)
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
        return CoreConditionalInput(
            acoustic_targets,
            posterior,
            alignment,
            phoneme,
            aligned_tokens,
            vectors,
            keep,
        )

    def _text_context(
        self,
        models: ConditionalModels,
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
        models: ConditionalModels,
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
        models: ConditionalModels,
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
            loop.cycle,
            loop.batch_index,
            label,
        )
        return torch.Generator(device=self.device).manual_seed(seed)
