from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Protocol

import torch
from accelerate import Accelerator
from torch import Tensor, nn

from .config import BeetleConfig, TrainingStage
from .config.training import ConditioningObjectiveConfig
from .data.index import DatabaseSegmentIndex
from .data.records import BeetleBatch
from .data.sampling import derive_seed
from .losses.acoustic import LogMelSpectrogram
from .models.acoustic import log_mel_l2_energy
from .models.conditional import ConditionalModels
from .models.modules.aligner import AlignerOutput
from .models.modules.audio import AcousticFeatures, AudioPosterior
from .models.modules.conditioning import (
    ConditionBank,
    ConditionKeep,
    ConditionVectors,
    ProjectedConditions,
    align_phoneme_tokens,
)
from .models.modules.embeddings import AcousticStatistics
from .models.modules.latent_flow import FlowTrainingSample, sample_flow_training_case
from .models.modules.text import PhonemeEncoding
from .aligned_window import (
    AlignedWindow,
    apply_window_ranges,
    safe_context_mask,
    sample_window_ranges,
    seconds_to_latent_frames,
)


class SpeakerIndex(Protocol):
    def resolve(
        self,
        speaker_ids: tuple[str | None, ...],
        device: torch.device,
    ) -> Tensor: ...


class DatabaseSpeakerIndex:
    def __init__(self, index: DatabaseSegmentIndex) -> None:
        voices = {
            item.speaker_id
            for item in index.records.values()
            if item.speaker_id is not None
        }
        voices.update(index.validation.conditional_by_voice)
        self.entries = tuple(sorted(voices))

    def resolve(
        self,
        speaker_ids: tuple[str | None, ...],
        device: torch.device,
    ) -> Tensor:
        indices = tuple(self.entries.index(speaker) for speaker in speaker_ids)
        return torch.tensor(indices, dtype=torch.long, device=device)


@dataclass(frozen=True)
class ConditionalAcousticTargets:
    features: AcousticFeatures
    statistics: AcousticStatistics


@dataclass(frozen=True)
class CoreConditionalInput:
    acoustic_targets: ConditionalAcousticTargets
    posterior: AudioPosterior
    alignment: AlignerOutput
    phoneme: PhonemeEncoding
    aligned_tokens: Tensor
    vectors: ConditionVectors
    keep: ConditionKeep
    window: AlignedWindow
    batch_statistics: ConditionalBatchStatistics


@dataclass(frozen=True)
class ContextAvailability:
    pre_text: Tensor
    post_text: Tensor
    pre_audio: Tensor
    post_audio: Tensor


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
class ConditionalTrainingInput:
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


class WaveformMelExtractor(nn.Module):
    def __init__(self, config: BeetleConfig) -> None:
        super().__init__()
        audio = config.audio
        self.hop_length = audio.hop_length
        self.n_fft = audio.n_fft
        self.transform = LogMelSpectrogram(
            audio.sample_rate,
            audio.n_fft,
            audio.hop_length,
            audio.win_length,
            audio.mel_channels,
            audio.f_min,
            audio.f_max,
        )

    def forward(self, waveform: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        required = max(waveform.shape[2], self.n_fft)
        padded = torch.nn.functional.pad(waveform, (0, required - waveform.shape[2]))
        mel = self.transform(padded[:, 0])
        frame_lengths = torch.div(
            lengths,
            self.hop_length,
            rounding_mode="floor",
        ).clamp_min(1)
        maximum = int(frame_lengths.max().clamp_min(1))
        maximum += maximum % 2
        mel = mel[:, :, :maximum]
        if mel.shape[2] < maximum:
            mel = torch.nn.functional.pad(mel, (0, maximum - mel.shape[2]))
        positions = torch.arange(maximum, device=waveform.device).unsqueeze(0)
        mask = (positions < frame_lengths.unsqueeze(1)).unsqueeze(1)
        return mel, mask


class ConditionalInputBuilder:
    def __init__(
        self,
        config: BeetleConfig,
        speaker_index: SpeakerIndex,
        accelerator: Accelerator,
    ) -> None:
        self.config = config
        self.speaker_index = speaker_index
        self.settings: ConditioningObjectiveConfig = config.conditioning_objective
        self.accelerator = accelerator
        self.device = accelerator.device
        self.mel_extractor = WaveformMelExtractor(config).to(accelerator.device)

    def build(
        self,
        models: ConditionalModels,
        batch: BeetleBatch,
        step: int,
        batch_index: int,
        acoustic_targets: ConditionalAcousticTargets,
    ) -> ConditionalTrainingInput:
        values = batch.to(self.device)
        return self._build(models, values, step, batch_index, acoustic_targets, False)

    def acoustic_targets(
        self,
        models: ConditionalModels,
        batch: BeetleBatch,
    ) -> ConditionalAcousticTargets:
        return acoustic_statistics(models, batch.to(self.device))

    def build_validation(
        self,
        models: ConditionalModels,
        batch: BeetleBatch,
        step: int,
        batch_index: int,
    ) -> ConditionalTrainingInput:
        values = batch.to(self.device)
        targets = acoustic_statistics(models, values)
        return self._build(models, values, step, batch_index, targets, True)

    def _build(
        self,
        models: ConditionalModels,
        values: BeetleBatch,
        step: int,
        batch_index: int,
        acoustic_targets: ConditionalAcousticTargets,
        validation: bool,
    ) -> ConditionalTrainingInput:
        core = self._core(
            models,
            values,
            step,
            batch_index,
            acoustic_targets,
            validation,
        )
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
            self._generator(step, batch_index, "duration"),
        )
        flow_sample = sample_flow_training_case(
            core.window.posterior.latent,
            core.window.posterior.mask,
            self.config.architecture.latent_flow.minimum_steps,
            self._generator(step, batch_index, "flow"),
        )
        style_latent, style_mask = encode_view_latents(
            models,
            self.mel_extractor,
            values.style_views,
            values.style_view_lengths,
            self._generator(step, batch_index, "style-views"),
        )
        voice_latent, voice_mask = encode_view_latents(
            models,
            self.mel_extractor,
            values.voice_views,
            values.voice_view_lengths,
            self._generator(step, batch_index, "voice-views"),
        )
        style_ids = group_ids(values.style_views, self.device)
        speaker_ids = group_ids(values.voice_views, self.device)
        settings = self.settings
        return ConditionalTrainingInput(
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
        step: int,
        batch_index: int,
        acoustic_targets: ConditionalAcousticTargets,
        validation: bool,
    ) -> CoreConditionalInput:
        target_generator = self._generator(step, batch_index, "target-posterior")
        posterior_gradients = (
            self.config.training.stage is TrainingStage.END_TO_END
        )
        with torch.set_grad_enabled(posterior_gradients):
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
            self._generator(step, batch_index, "aligned-window"),
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
            else self.accelerator.unwrap_model(models.condition_bank).sample_keep(
                values.waveform.shape[0],
                self.device,
                self.config.architecture.conditioning.dropout,
                self._generator(step, batch_index, "condition-dropout"),
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

    def _generator(
        self,
        step: int,
        batch_index: int,
        label: str,
    ) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(
            derive_seed(self.config.runtime.seed, step, batch_index, label)
        )


def acoustic_statistics(
    models: ConditionalModels,
    batch: BeetleBatch,
) -> ConditionalAcousticTargets:
    with torch.no_grad():
        f0 = models.f0_extractor(batch.jdc_mel, batch.frame_mask)
    n = log_mel_l2_energy(batch.mel, batch.frame_mask)
    f0_mask = batch.frame_mask[:, 0] & (f0 > 0)
    n_mask = batch.frame_mask[:, 0]
    log_f0 = torch.log(f0.clamp_min(1))
    f0_mean, f0_std = masked_statistics(log_f0, f0_mask)
    n_mean, n_std = masked_statistics(n, n_mask)
    return ConditionalAcousticTargets(
        AcousticFeatures(f0, n),
        AcousticStatistics(f0_mean, f0_std, n_mean, n_std),
    )


def masked_statistics(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    numeric = mask.to(dtype=values.dtype)
    count = numeric.sum(dim=1).clamp_min(1)
    mean = (values * numeric).sum(dim=1) / count
    variance = ((values - mean.unsqueeze(1)).square() * numeric).sum(dim=1) / count
    return mean, torch.sqrt(variance.clamp_min(1e-5))


def encode_text_context(
    models: ConditionalModels,
    tokens: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    safe_mask, available = safe_context_mask(mask)
    encoded = models.context_phoneme_encoder(tokens, safe_mask)
    return encoded * available[:, 0], available


def encode_audio_context(
    models: ConditionalModels,
    latent: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    safe_mask, available = safe_context_mask(mask)
    encoded = models.context_audio_encoder(latent, safe_mask)
    return encoded * available[:, 0], available


def encode_view_latents(
    models: ConditionalModels,
    mel_extractor: WaveformMelExtractor,
    waveforms: Tensor,
    lengths: Tensor,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    groups, views, channels, samples = waveforms.shape
    flattened = waveforms.reshape(groups * views, channels, samples)
    flat_lengths = lengths.reshape(groups * views)
    mel, mask = mel_extractor(flattened, flat_lengths)
    with torch.no_grad():
        posterior = models.audio_encoder(mel, mask, generator)
    return posterior.latent, posterior.mask


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


def keep_all_conditions(batch_size: int, device: torch.device) -> ConditionKeep:
    keep = torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device)
    return ConditionKeep(keep, keep, keep, keep, keep, keep, keep, keep, keep)


def group_ids(views: Tensor, device: torch.device) -> Tensor:
    groups, view_count = views.shape[:2]
    return torch.arange(groups, device=device).repeat_interleave(view_count)


def style_weights(distances: Tensor) -> Tensor:
    flattened = distances.flatten()
    difference = (flattened.unsqueeze(0) - flattened.unsqueeze(1)).abs()
    return 1 / (1 + difference)


def conditional_batch_statistics(
    window: AlignedWindow,
    availability: ContextAvailability,
    sampled_keep: ConditionKeep,
    effective_keep: ConditionKeep,
    seconds_per_frame: float,
) -> ConditionalBatchStatistics:
    ranges = window.ranges
    requested = ranges.target_requested_lengths.float()
    source = ranges.target_source_lengths.float()
    values = {
        "target_seconds": (requested * seconds_per_frame).mean(),
        "target_padding_ratio": 1 - source.sum() / requested.sum(),
        "full_audio_ratio": ranges.full_selected.float().mean(),
        "pre_text_available_ratio": availability.pre_text.float().mean(),
        "post_text_available_ratio": availability.post_text.float().mean(),
        "pre_audio_available_ratio": availability.pre_audio.float().mean(),
        "post_audio_available_ratio": availability.post_audio.float().mean(),
        "pre_text_random_drop_ratio": (~sampled_keep.pre_text).float().mean(),
        "post_text_random_drop_ratio": (~sampled_keep.post_text).float().mean(),
        "pre_audio_random_drop_ratio": (~sampled_keep.pre_audio).float().mean(),
        "post_audio_random_drop_ratio": (~sampled_keep.post_audio).float().mean(),
        "pre_text_effective_drop_ratio": (~effective_keep.pre_text).float().mean(),
        "post_text_effective_drop_ratio": (~effective_keep.post_text).float().mean(),
        "pre_audio_effective_drop_ratio": (~effective_keep.pre_audio).float().mean(),
        "post_audio_effective_drop_ratio": (~effective_keep.post_audio).float().mean(),
    }
    expected = {field.name for field in fields(ConditionalBatchStatistics)}
    if values.keys() != expected:
        raise RuntimeError("conditional batch statistic fields changed")
    return ConditionalBatchStatistics(**values)
