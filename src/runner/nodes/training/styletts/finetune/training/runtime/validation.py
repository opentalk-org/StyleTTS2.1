from pathlib import Path

import torch
import torch.nn.functional as F

from runflow.runtime.cancellation import check_cancel

from ...studio.val_sample_export import (
    ValidationArtifactRenderer,
    ValidationSample,
    ValidationSampleArtifacts,
)
from ...stages import (
    ProsodySource,
    StyleSource,
    TrainingStageSpec,
    ValidationDurationSource,
    stage_for_step,
)
from ..config import TrainingConfig
from ..data import TrainingBatch, ValidationResult
from ..losses import acoustic_losses, reconstruction_loss
from ..setup import TrainingRuntime
from ..utils import (
    length_to_mask,
    log_norm,
    mask_from_lens,
    maximum_path,
)
from .batch_ops import crop_training_batch, prosody_inputs, sample_voice_prompts


def predicted_alignment(
    duration_predictions: torch.Tensor,
    input_lengths: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    durations = torch.round(
        torch.sigmoid(duration_predictions).sum(dim=-1)
    ).clamp_min(1)
    token_positions = torch.arange(
        durations.size(1),
        device=durations.device,
    )
    input_lengths = input_lengths.to(durations.device)
    durations = durations * (
        token_positions.unsqueeze(0) < input_lengths.unsqueeze(1)
    )
    ends = durations.cumsum(dim=1)
    starts = ends - durations
    lengths = [int(value.item()) for value in ends[:, -1]]
    frames = torch.arange(max(lengths), device=durations.device)
    alignment = (
        (frames[None, None, :] >= starts[:, :, None])
        & (frames[None, None, :] < ends[:, :, None])
    )
    return alignment.to(duration_predictions.dtype), lengths


def resize_prosody(
    values: torch.Tensor,
    source_lengths: list[int],
    target_lengths: list[int],
) -> torch.Tensor:
    resized = values.new_zeros((values.size(0), max(target_lengths)))
    for index, (source, target) in enumerate(
        zip(source_lengths, target_lengths, strict=True)
    ):
        resized[index, :target] = F.interpolate(
            values[index, :source].reshape(1, 1, source),
            size=target,
            mode="linear",
            align_corners=False,
        ).reshape(target)
    return resized


def rfsq_usage_metrics(counts: torch.Tensor) -> dict[str, torch.Tensor]:
    metrics = {}
    utilizations = []
    dead_fractions = []
    perplexities = []
    top_shares = []
    level_count = counts.size(1)
    for index, level_counts in enumerate(counts):
        assignments = level_counts.sum()
        probabilities = level_counts.float() / assignments
        active = (level_counts > 0).sum()
        nonzero = probabilities > 0
        perplexity = torch.exp(
            -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        )
        utilization = active.float() / level_count
        top_share = probabilities.max()
        prefix = f"rfsq/stage_{index:02d}"
        metrics[f"{prefix}/active_levels"] = active.float()
        metrics[f"{prefix}/utilization"] = utilization
        metrics[f"{prefix}/dead_level_fraction"] = 1 - utilization
        metrics[f"{prefix}/perplexity"] = perplexity
        metrics[f"{prefix}/normalized_perplexity"] = (
            perplexity / level_count
        )
        metrics[f"{prefix}/top_level_share"] = top_share
        utilizations.append(utilization)
        dead_fractions.append(1 - utilization)
        perplexities.append(perplexity)
        top_shares.append(top_share)
    metrics["rfsq/mean_utilization"] = torch.stack(utilizations).mean()
    metrics["rfsq/mean_dead_level_fraction"] = torch.stack(
        dead_fractions
    ).mean()
    metrics["rfsq/mean_perplexity"] = torch.stack(perplexities).mean()
    metrics["rfsq/mean_top_level_share"] = torch.stack(top_shares).mean()
    metrics["rfsq/assignments_per_stage"] = counts.sum(1).float().mean()
    return metrics


def synthesize_validation(
    runtime: TrainingRuntime,
    batch: TrainingBatch,
    stage: TrainingStageSpec,
    text_encoding: torch.Tensor,
    duration_encoding: torch.Tensor,
    bert: torch.Tensor,
    monotonic: torch.Tensor,
    target_f0: torch.Tensor,
    target_energy: torch.Tensor,
    alpha_flow_noise: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    torch.Tensor | None,
]:
    modules = runtime.models.modules
    device = batch.mels.device
    decode_alignment = monotonic
    decode_lengths = [int(value.item() // 2) for value in batch.mel_lengths]
    duration_predictions = batch.mels.new_zeros(
        (batch.texts.size(0), batch.texts.size(1), 1)
    )
    predicted_f0 = target_f0.new_zeros(target_f0.shape)
    predicted_energy = target_energy.new_zeros(target_energy.shape)
    rfsq_indices = None
    validates_predictions = (
        stage.validation.alpha_flow
        or stage.validation.f0_source is ProsodySource.PREDICTED
        or stage.validation.norm_source is ProsodySource.PREDICTED
        or (
            stage.validation.duration_source
            is ValidationDurationSource.PREDICTED
        )
    )
    if validates_predictions:
        conditioning = prosody_inputs(
            modules.position_embedding,
            monotonic,
            target_f0,
            target_energy,
        )
        with torch.autocast(device_type=device.type, enabled=False):
            encoded_style = modules.prosody_encoder(
                conditioning.float(),
                batch.mel_lengths.to(device) // 2,
            )
            style = encoded_style
            if stage.style_source is StyleSource.QUANTIZED:
                latents = modules.quantizer(encoded_style)
                style = latents.quantized_style
                rfsq_indices = latents.indices
        if stage.validation.alpha_flow:
            continuous_latent = modules.alpha_flow.sample(
                bert,
                conditioning,
                batch.input_lengths,
                noise=alpha_flow_noise,
            )
            style = modules.quantizer.decode_continuous(continuous_latent)
        duration_predictions = modules.duration_predictor(
            duration_encoding,
            style,
            batch.input_lengths,
            duration_encoding.size(-1),
        )
        if (
            stage.validation.duration_source
            is ValidationDurationSource.PREDICTED
        ):
            decode_alignment, decode_lengths = predicted_alignment(
                duration_predictions,
                batch.input_lengths,
            )
    prompt_mels = sample_voice_prompts(batch)
    with torch.autocast(device_type=device.type, enabled=False):
        decoder_voice, decoder_text = modules.voice_encoder(
            prompt_mels.float(),
            text_encoding.float(),
            batch.input_lengths,
            text_encoding.size(-1),
        )
    aligned_text = decoder_text @ decode_alignment
    if validates_predictions:
        aligned_duration = duration_encoding @ decode_alignment
        half_lengths = torch.tensor(decode_lengths, device=device)
        predicted_f0, predicted_energy = modules.prosody_predictor(
            aligned_duration,
            style,
            half_lengths,
            decode_alignment.size(-1),
        )
    source_lengths = [int(value.item()) for value in batch.mel_lengths]
    full_lengths = [value * 2 for value in decode_lengths]
    resized_f0 = resize_prosody(target_f0, source_lengths, full_lengths)
    resized_energy = resize_prosody(target_energy, source_lengths, full_lengths)
    decoder_f0 = (
        predicted_f0
        if stage.validation.f0_source is ProsodySource.PREDICTED
        else resized_f0
    )
    decoder_energy = (
        predicted_energy
        if stage.validation.norm_source is ProsodySource.PREDICTED
        else resized_energy
    )
    positions = torch.arange(max(full_lengths), device=device)
    frame_mask = positions[None, :] < torch.tensor(
        full_lengths,
        device=device,
    )[:, None]
    predicted_f0 = predicted_f0 * frame_mask
    predicted_energy = predicted_energy * frame_mask
    reconstructed = modules.decoder(
        aligned_text,
        decoder_f0,
        decoder_energy,
        decoder_voice,
    )
    return (
        reconstructed,
        duration_predictions,
        predicted_f0,
        predicted_energy,
        resized_f0,
        resized_energy,
        decode_alignment,
        decode_lengths,
        rfsq_indices,
    )


def _validates_predictions(stage: TrainingStageSpec) -> bool:
    validation = stage.validation
    return (
        validation.alpha_flow
        or validation.f0_source is ProsodySource.PREDICTED
        or validation.norm_source is ProsodySource.PREDICTED
        or validation.duration_source is ValidationDurationSource.PREDICTED
    )


class Validator:
    def __init__(
        self,
        config: TrainingConfig,
        runtime: TrainingRuntime,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.artifacts = ValidationArtifactRenderer(
            Path(config.log_dir),
            config.preprocess_params.sr,
        )

    def run(
        self,
        batches,
        step: int,
    ) -> ValidationResult:
        modules = self.runtime.models.modules
        for module in modules.values():
            module.eval()
        stage = stage_for_step(self.config.training_stages, step - 1)
        metric_names = ["mel_loss"]
        if _validates_predictions(stage):
            metric_names.extend(
                (
                    "dur_loss",
                    "F0_loss",
                    "norm_loss",
                    "free_run_duration_ratio",
                    "free_run_duration_ratio_mae",
                )
            )
        zero = torch.zeros((), device=self.runtime.accelerator.device)
        totals = {name: zero.clone() for name in metric_names}
        quantizer = modules.quantizer
        rfsq_counts = torch.zeros(
            (
                quantizer.num_stages,
                quantizer.levels,
            ),
            device=zero.device,
            dtype=torch.long,
        )
        samples: list[ValidationSampleArtifacts] = []
        count = 0
        with torch.no_grad():
            for loaded_batch in batches:
                batch = loaded_batch.batch
                check_cancel()
                batch = batch.to(self.runtime.accelerator.device)
                values, exported, rfsq_indices = self._validate_batch(
                    batch,
                    step,
                    stage,
                    export_samples=not samples,
                )
                for name, value in values.items():
                    totals[name] += value
                samples.extend(exported)
                if rfsq_indices is not None:
                    for index in range(rfsq_counts.size(0)):
                        stage_indices = rfsq_indices[:, index].reshape(-1)
                        stage_indices = stage_indices[stage_indices >= 0]
                        rfsq_counts[index] += torch.bincount(
                            stage_indices,
                            minlength=rfsq_counts.size(1),
                        )
                count += 1
        if count == 0:
            raise ValueError("validation loader produced no batches")
        reduced = {
            name: value / count
            for name, value in totals.items()
        }
        if rfsq_counts.sum() > 0:
            reduced.update(rfsq_usage_metrics(rfsq_counts))
        return ValidationResult(
            reduced,
            samples,
        )

    def _validate_batch(
        self,
        batch: TrainingBatch,
        step: int,
        stage: TrainingStageSpec,
        export_samples: bool,
    ) -> tuple[
        dict[str, torch.Tensor],
        list[ValidationSampleArtifacts],
        torch.Tensor | None,
    ]:
        modules = self.runtime.models.modules
        n_down = self.runtime.models.n_down
        with self.runtime.accelerator.autocast():
            mask = length_to_mask(
                batch.mel_lengths // (2**n_down),
                batch.mels.device,
            )
            text_mask = length_to_mask(
                batch.input_lengths,
                batch.texts.device,
            )
            _, _, soft_alignment = modules.text_aligner(
                batch.mels,
                mask,
                batch.texts,
            )
            soft_alignment = soft_alignment.transpose(-1, -2)
            soft_alignment = soft_alignment[..., 1:].transpose(-1, -2)
            alignment_mask = mask_from_lens(
                soft_alignment,
                batch.input_lengths,
                batch.mel_lengths // (2**n_down),
            )
            soft_alignment = soft_alignment.masked_fill(
                ~alignment_mask.bool(),
                0.0,
            )
            monotonic = maximum_path(soft_alignment, alignment_mask)
            duration_targets = monotonic.sum(axis=-1).detach()
            text_encoding = modules.text_encoder(
                batch.texts,
                batch.input_lengths,
                text_mask,
                batch.language_ids,
            )
            validate_predictions = _validates_predictions(stage)
            bert = batch.mels.new_zeros(
                (batch.texts.size(0), batch.texts.size(1), 1)
            )
            duration_encoding = batch.mels.new_zeros(
                (batch.texts.size(0), 1, batch.texts.size(1))
            )
            if validate_predictions:
                bert = modules.bert(
                    batch.texts,
                    attention_mask=(~text_mask).int(),
                    language_ids=batch.language_ids,
                    modality_ids=batch.modality_ids,
                )
                duration_encoding = modules.bert_encoder(bert).transpose(
                    -1,
                    -2,
                )
            waveform, target_sample_lengths = self._waveform_targets(batch)
            target_f0, _, _ = modules.pitch_extractor(
                batch.mels.unsqueeze(1),
                batch.mel_lengths,
            )
            target_f0 = target_f0.squeeze(-1)
            target_norm = log_norm(batch.mels.unsqueeze(1)).squeeze(1)
            full_mask = length_to_mask(batch.mel_lengths, batch.mels.device)
            target_f0.masked_fill_(full_mask, 0.0)
            target_norm.masked_fill_(full_mask, 0.0)
            source_target_f0 = target_f0
            source_target_norm = target_norm
            alpha_flow_noise = self._alpha_flow_noise(
                batch.texts,
                batch.input_lengths,
                bert,
                modules.alpha_flow.style_dim,
            ) if stage.validation.alpha_flow else None
            (
                reconstructed,
                duration_predictions,
                predicted_f0,
                predicted_norm,
                target_f0,
                target_norm,
                decode_alignment,
                decode_lengths,
                rfsq_indices,
            ) = synthesize_validation(
                self.runtime,
                batch,
                stage,
                text_encoding,
                duration_encoding,
                bert,
                monotonic if validate_predictions else soft_alignment,
                source_target_f0,
                source_target_norm,
                alpha_flow_noise,
            )
            waveform = waveform.unsqueeze(1)
            prediction_sample_lengths = [
                min(length * 600, reconstructed.size(-1))
                for length in decode_lengths
            ]
            if validate_predictions:
                validation_losses = []
                for index, (target_samples, prediction_samples) in enumerate(
                    zip(
                        target_sample_lengths,
                        prediction_sample_lengths,
                        strict=True,
                    )
                ):
                    samples = max(target_samples, prediction_samples)
                    prediction = F.pad(
                        reconstructed[index, 0, :prediction_samples],
                        (0, samples - prediction_samples),
                    ).unsqueeze(0)
                    target = F.pad(
                        waveform[index, 0, :target_samples],
                        (0, samples - target_samples),
                    ).unsqueeze(0)
                    validation_losses.append(
                        self.runtime.losses.stft(prediction, target)
                    )
                mel_loss = acoustic_losses(validation_losses)
            else:
                mel_loss = self._acoustic_reconstruction_loss(
                    batch,
                    text_encoding,
                    soft_alignment,
                    target_f0,
                    target_norm,
                )
            metrics = {"mel_loss": mel_loss}
            if validate_predictions:
                full_lengths = [length * 2 for length in decode_lengths]
                f0_loss = reconstruction_loss(
                    target_f0,
                    predicted_f0,
                    full_lengths,
                    divisor=10,
                )
                norm_loss = reconstruction_loss(
                    target_norm,
                    predicted_norm,
                    full_lengths,
                )
                duration_loss = self._duration_loss(
                    duration_predictions,
                    duration_targets,
                    batch.input_lengths,
                )
                free_run_stage = stage.model_copy(
                    update={
                        "validation": stage.validation.model_copy(
                            update={
                                "duration_source": (
                                    ValidationDurationSource.PREDICTED
                                ),
                            }
                        )
                    }
                )
                (
                    free_reconstructed,
                    _,
                    free_predicted_f0,
                    free_predicted_norm,
                    free_target_f0,
                    free_target_norm,
                    free_alignment,
                    free_lengths,
                    _,
                ) = synthesize_validation(
                    self.runtime,
                    batch,
                    free_run_stage,
                    text_encoding,
                    duration_encoding,
                    bert,
                    monotonic,
                    source_target_f0,
                    source_target_norm,
                    alpha_flow_noise,
                )
                free_prediction_sample_lengths = [
                    min(length * 600, free_reconstructed.size(-1))
                    for length in free_lengths
                ]
                ground_truth_half_lengths = batch.mel_lengths.to(
                    device=free_reconstructed.device,
                    dtype=free_reconstructed.dtype,
                ) / 2
                free_duration_ratios = (
                    torch.as_tensor(
                        free_lengths,
                        device=free_reconstructed.device,
                        dtype=free_reconstructed.dtype,
                    ) / ground_truth_half_lengths
                )
                metrics.update(
                    {
                        "dur_loss": duration_loss,
                        "F0_loss": f0_loss,
                        "norm_loss": norm_loss,
                        "free_run_duration_ratio": free_duration_ratios.mean(),
                        "free_run_duration_ratio_mae": (
                            free_duration_ratios - 1
                        ).abs().mean(),
                    }
                )
        samples = []
        if export_samples:
            sample_count = min(4, waveform.size(0))
            validation_samples = [
                ValidationSample(
                    ground_truth=waveform[
                        index,
                        :,
                        : target_sample_lengths[index],
                    ],
                    prediction=reconstructed[
                        index,
                        :,
                        : prediction_sample_lengths[index],
                    ],
                    target_f0=target_f0[
                        index,
                        : decode_lengths[index] * 2,
                    ],
                    predicted_f0=(
                        predicted_f0[index, : decode_lengths[index] * 2]
                        if validate_predictions
                        else None
                    ),
                    target_n=target_norm[
                        index,
                        : decode_lengths[index] * 2,
                    ],
                    predicted_n=(
                        predicted_norm[index, : decode_lengths[index] * 2]
                        if validate_predictions
                        else None
                    ),
                    soft_attention=soft_alignment[
                        index,
                        : batch.input_lengths[index],
                        : batch.mel_lengths[index] // (2**n_down),
                    ],
                    hard_attention=monotonic[
                        index,
                        : batch.input_lengths[index],
                        : batch.mel_lengths[index] // (2**n_down),
                    ],
                )
                for index in range(sample_count)
            ]
            teacher_forced_samples = self.artifacts.render(
                step,
                validation_samples,
                mode="teacher_forced_timing",
            )
            samples = teacher_forced_samples
            if validate_predictions:
                free_run_samples = [
                    ValidationSample(
                        ground_truth=waveform[
                            index,
                            :,
                            : target_sample_lengths[index],
                        ],
                        prediction=free_reconstructed[
                            index,
                            :,
                            : free_prediction_sample_lengths[index],
                        ],
                        target_f0=free_target_f0[
                            index,
                            : free_lengths[index] * 2,
                        ],
                        predicted_f0=free_predicted_f0[
                            index,
                            : free_lengths[index] * 2,
                        ],
                        target_n=free_target_norm[
                            index,
                            : free_lengths[index] * 2,
                        ],
                        predicted_n=free_predicted_norm[
                            index,
                            : free_lengths[index] * 2,
                        ],
                        soft_attention=soft_alignment[
                            index,
                            : batch.input_lengths[index],
                            : batch.mel_lengths[index] // (2**n_down),
                        ],
                        hard_attention=free_alignment[
                            index,
                            : batch.input_lengths[index],
                            : free_lengths[index],
                        ],
                    )
                    for index in range(sample_count)
                ]
                samples += self.artifacts.render(
                    step,
                    free_run_samples,
                    mode="free_running",
                )
        return metrics, samples, rfsq_indices

    def _acoustic_reconstruction_loss(
        self,
        batch: TrainingBatch,
        text_encoding: torch.Tensor,
        alignment: torch.Tensor,
        target_f0: torch.Tensor,
        target_norm: torch.Tensor,
    ) -> torch.Tensor:
        modules = self.runtime.models.modules
        prompt_mels = sample_voice_prompts(batch)
        voice, encoded_text = modules.voice_encoder(
            prompt_mels,
            text_encoding,
            batch.input_lengths,
            text_encoding.size(-1),
        )
        aligned_text = encoded_text @ alignment
        crops = crop_training_batch(
            batch,
            aligned_text,
            target_f0,
            target_norm,
            target_f0,
            target_norm,
            min(int(batch.mel_lengths.min().item() / 2 - 1), 80),
        )
        aligned_crop, _, _, _, _, cropped_mels, waveform = crops
        lengths = batch.mel_lengths.new_full(
            (cropped_mels.size(0),),
            cropped_mels.size(-1),
        )
        f0, _, _ = modules.pitch_extractor(
            cropped_mels.unsqueeze(1),
            lengths,
        )
        norm = log_norm(cropped_mels.unsqueeze(1)).squeeze(1)
        reconstructed = modules.decoder(
            aligned_crop,
            f0.squeeze(-1),
            norm,
            voice,
        )
        return self.runtime.losses.stft(reconstructed, waveform)

    @staticmethod
    def _alpha_flow_noise(
        texts: torch.Tensor,
        lengths: torch.Tensor,
        like: torch.Tensor,
        latent_dim: int,
    ) -> torch.Tensor:
        """Stable validation noise makes checkpoints directly comparable."""
        rows = []
        for text, length in zip(texts, lengths, strict=True):
            count = int(length.item())
            positions = torch.arange(
                1,
                count + 1,
                device=text.device,
                dtype=torch.long,
            )
            seed = int(
                (((text[:count].long() + 1) * positions).sum().item())
                % 2_147_483_647
            )
            generator = torch.Generator(device=text.device)
            generator.manual_seed(seed)
            rows.append(
                torch.randn(
                    latent_dim,
                    50,
                    generator=generator,
                    device=text.device,
                    dtype=like.dtype,
                )
            )
        return torch.stack(rows)

    @staticmethod
    def _waveform_targets(
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, list[int]]:
        lengths = [int(length.item() // 2) for length in batch.mel_lengths]
        sample_lengths = [
            min(length * 600, batch.waves[index].shape[0])
            for index, length in enumerate(lengths)
        ]
        frames = max(lengths)
        waveform = batch.mels.new_zeros((len(lengths), frames * 600))
        for index, samples in enumerate(sample_lengths):
            waveform[index, :samples] = torch.from_numpy(
                batch.waves[index][:samples]
            ).to(batch.mels.device)
        return waveform.detach(), sample_lengths

    @staticmethod
    def _duration_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.zeros((), device=predictions.device)
        for prediction, target, length in zip(predictions, targets, lengths):
            prediction = prediction[:length, :]
            target = target[:length].long()
            predicted = torch.sigmoid(prediction).sum(axis=1)
            loss += F.l1_loss(
                predicted[1 : length - 1],
                target[1 : length - 1],
            )
        return loss / predictions.size(0)
