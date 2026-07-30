import math
import random
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs, broadcast_object_list
from torch import Tensor, nn
from torch.autograd.functional import jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from .checkpoints import CheckpointManager
from .conditioning import (
    ConditionalInputBuilder,
    ConditionalTrainingInput,
    DatabaseSpeakerIndex,
)
from .config import (
    BeetleConfig,
    OptimizerConfig,
    TrainingStage,
    load_config,
)
from .config.training import ScheduledWeight
from .data import (
    BatchLoadError,
    ContinuousBatchPlanner,
    DataPipelineState,
    DatabaseSegmentIndex,
    DistributedShard,
    RepeatedBatchPipeline,
    ValidationLoader,
    build_data_pipeline,
    repeat_validation_embedding_groups,
    select_validation_audio_ids,
    select_validation_voice_reference_ids,
)
from .data.records import BeetleBatch
from .data.sampling import derive_seed
from .data.validation_records import ValidationRecording
from .losses.acoustic import (
    masked_kl_standard_normal,
    masked_n_smooth_l1,
    masked_pitch_loss,
)
from .losses.adversarial import discriminator_step_loss, generator_step_loss
from .losses.conditional import (
    ConditionalLossInput,
    ConditionalLossOutput,
    ConditionalLossWeights,
    ConditionalModelOutput,
    compute_conditional_losses,
)
from .losses.composition import AcousticLossWeights
from .losses.flow import flow_inner_product, flow_mse
from .logger import logger
from .mlflow_logging import MlflowLogger
from .models.conditional import ConditionalModels
from .models.model import AcousticModels
from .models.modules.audio import AcousticFeatures
from .models.modules.latent_flow import integrate_latent_flow
from .models.modules.segments import AlignedSegments
from .setup import (
    TrainingModels,
    TrainingOptimizers,
    build_models,
    build_optimizers,
    load_text_resources,
    prepare_training,
)
from .validation import ValidationArtifacts, ValidationSample


ACOUSTIC_LOSS_NAMES = (
    "encoder_kl",
    "f0",
    "n",
    "reconstruction",
    "discriminator",
    "generator_adversarial",
    "feature_matching",
)
CONDITIONAL_LOSS_NAMES = (
    "duration_flow",
    "latent_flow",
    "shortcut",
    "align_s2s",
    "align_mono",
    "align_ctc",
    "voice_contrastive",
    "voice_ge2e",
    "style_contrastive",
    "style_ge2e",
    "style_speaker_adversarial",
    "style_statistics",
    "style_reencoding",
)


class TrainingCallbacks(Protocol):
    def check_cancel(self) -> None: ...

    def report_index_progress(self, scanned: int, total: int) -> None: ...

    def report_training_progress(self, step: int, total: int) -> None: ...


class Trainer:
    def __init__(
        self,
        config_path: Path,
        output_path: Path,
        callbacks: TrainingCallbacks,
        resume_path: Path | None = None,
        initialize_path: Path | None = None,
    ) -> None:
        self.config_path = config_path
        self.output_path = output_path
        self.callbacks = callbacks
        self.resume_path = resume_path
        self.initialize_path = initialize_path
        self.config: BeetleConfig
        self.accelerator: Accelerator
        self.model_bundle: TrainingModels
        self.acoustic_models: AcousticModels | None
        self.conditional_models: ConditionalModels | None
        self.optimizers: TrainingOptimizers
        self.checkpoints: CheckpointManager
        self.conditioning: ConditionalInputBuilder | None
        self.pipeline: object
        self.validation_recordings: tuple[ValidationRecording, ...]
        self.logger: MlflowLogger | None
        self.artifacts: ValidationArtifacts | None
        self.step = 0
        self.batch_index = 0
        self.metric_sums: defaultdict[str, float] = defaultdict(float)
        self.sparse_metrics: dict[str, float] = {}
        self.metric_count = 0
        self.optimizer_metric_count = 0
        self.initial_step = 0
        self.items_processed = 0
        self.training_started_at = 0.0
        self.data_wait_seconds = 0.0
        self.compute_seconds = 0.0
        self.validation_seconds = 0.0
        self.checkpoint_seconds = 0.0
        self.reporting_seconds = 0.0
        self.first_validation_succeeded = False
        self.recovery_counts: defaultdict[str, int] = defaultdict(int)

    def prepare(self) -> None:
        self.config = load_config(self.config_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(self.config.data.prefetch.preprocessing_threads)
        random.seed(self.config.runtime.seed)
        np.random.seed(self.config.runtime.seed)
        torch.manual_seed(self.config.runtime.seed)
        precision = {
            "float32": "no",
            "float16": "fp16",
            "bfloat16": "bf16",
        }[self.config.training.precision.value]
        self.accelerator = Accelerator(
            mixed_precision=precision,
            cpu=self.config.runtime.device == "cpu",
            kwargs_handlers=[GradScalerKwargs(init_scale=16.0)],
        )
        if self.accelerator.device.type != self.config.runtime.device:
            raise RuntimeError(
                f"requested {self.config.runtime.device}, "
                f"Accelerate selected {self.accelerator.device}"
            )
        self.callbacks.check_cancel()
        index = DatabaseSegmentIndex.build(
            self.config.data.selection,
            self.config.architecture.language.values,
            self.config.data.maximum_seconds,
            self.config.data.prefetch.page_size,
            self.callbacks,
        )
        index.report.require()
        resources = load_text_resources(self.config)
        validation_loader = ValidationLoader(self.config)
        validation_ids = select_validation_audio_ids(
            index,
            self.config.validation.sample_count,
            self.config.runtime.seed,
            self.config.validation.audio_file_ids,
            self.config.training.stage is not TrainingStage.POSTERIOR,
        )
        validation_voice_ids = (
            validation_ids
            if self.config.training.stage is TrainingStage.POSTERIOR
            else select_validation_voice_reference_ids(
                index,
                validation_ids,
                self.config.runtime.seed,
            )
        )
        validation_source_ids = tuple(
            dict.fromkeys((*validation_ids, *validation_voice_ids))
        )
        validation_source = validation_loader.load_source(validation_source_ids)
        self.validation_recordings = validation_loader.collate(
            validation_source,
            validation_ids,
            validation_voice_ids,
            resources.phoneme_tokenizer,
            resources.text_tokenizer,
        )
        self.model_bundle = build_models(self.config, resources)
        self.optimizers = build_optimizers(self.config, self.model_bundle)
        self.checkpoints = CheckpointManager(
            self.output_path,
            self.config,
            self.accelerator,
        )
        pipeline_state = None
        resume_run_id = None
        if self.initialize_path is not None:
            self.checkpoints.initialize(self.initialize_path, self.model_bundle)
        if self.resume_path is not None:
            resume = self.checkpoints.resume(
                self.resume_path,
                self.model_bundle,
                self.optimizers,
            )
            self.step = resume.step
            self.batch_index = resume.batch_index
            pipeline_state = resume.pipeline
            resume_run_id = resume.mlflow_run_id
        generator_groups = self.optimizers.generator.param_groups
        generator_groups[0]["weight_decay"] = (
            self.config.training.generator_optimizer.weight_decay
        )
        if self.model_bundle.conditional is not None:
            generator_groups[1]["weight_decay"] = (
                self.config.training.latent_flow_weight_decay
            )
            generator_groups[2]["weight_decay"] = (
                self.config.architecture.phoneme.weight_decay
            )
        self.model_bundle, self.optimizers = prepare_training(
            self.accelerator,
            self.model_bundle,
            self.optimizers,
        )
        self.acoustic_models = self.model_bundle.acoustic
        self.conditional_models = self.model_bundle.conditional
        self.conditioning = (
            ConditionalInputBuilder(
                self.config,
                DatabaseSpeakerIndex(index),
                self.accelerator,
            )
            if self.conditional_models is not None
            else None
        )
        shard = DistributedShard(
            self.accelerator.process_index,
            self.accelerator.num_processes,
        )
        initial_state = pipeline_state
        if initial_state is None:
            planner = ContinuousBatchPlanner(
                index,
                self.config.training.batch_size,
                self.config.runtime.seed,
                self.config.data.maximum_seconds,
                self.config.data.grouping,
                shard,
            )
            initial_state = DataPipelineState(
                index.fingerprint,
                planner.state_dict(),
                shard.world_size,
            )
        if self.config.training.overfit_validation_recording:
            recording = repeat_validation_embedding_groups(
                self.validation_recordings[0]
            )
            self.validation_recordings = (recording,)
            self.pipeline = RepeatedBatchPipeline(
                recording.batch,
                index.fingerprint,
                shard.world_size,
                initial_state,
            )
        else:
            self.pipeline = build_data_pipeline(
                self.config,
                self.callbacks,
                index,
                resources.phoneme_tokenizer,
                resources.text_tokenizer,
                initial_state,
                shard,
            )
        run_id = resume_run_id
        if self.accelerator.is_main_process:
            self.logger = MlflowLogger(
                self.config,
                self.accelerator.device.index or 0,
                resume_run_id,
            )
            run_id = self.logger.run_id
        else:
            self.logger = None
        shared_run_id = [run_id]
        broadcast_object_list(shared_run_id)
        self.run_id = str(shared_run_id[0])
        self.artifacts = (
            ValidationArtifacts(
                self.output_path,
                self.config.audio,
                self.logger,
            )
            if self.logger is not None
            else None
        )
        self.initial_step = self.step
        self.training_started_at = time.monotonic()

    def train_step(self, batch: BeetleBatch) -> dict[str, Tensor | float]:
        match self.config.training.stage:
            case TrainingStage.POSTERIOR:
                return self.posterior_step(batch)
            case TrainingStage.LATENT_FLOW:
                return self.latent_flow_step(batch)
            case TrainingStage.END_TO_END:
                return self.end_to_end_step(batch)
            case stage:
                raise RuntimeError(f"unsupported training stage: {stage}")

    def posterior_step(self, batch: BeetleBatch) -> dict[str, Tensor | float]:
        acoustic_models = self.acoustic_models
        if acoustic_models is None or self.optimizers.discriminator is None:
            raise RuntimeError("acoustic stage was not prepared")
        values = batch.to(self.accelerator.device)
        waveform = values.waveform
        frame_count = (
            self.config.adversarial.segment_samples
            // acoustic_models.output_hop
        )
        lengths = values.frame_mask[:, 0].sum(dim=1).clamp_min(frame_count)
        positions = torch.arange(
            values.frame_mask.shape[-1],
            device=self.accelerator.device,
        )
        available = positions.view(1, 1, -1) < lengths.view(-1, 1, 1)
        segment = AlignedSegments.reference_chunks(
            available,
            frame_count,
            acoustic_models.output_hop,
            self.generator("acoustic-segment"),
        )
        real = segment.samples(waveform)
        with self.accelerator.autocast():
            mel = acoustic_models.reconstruction_loss.transforms[0](real[:, 0])
            jdc_mel = acoustic_models.jdc_transform(real[:, 0])
            frame_mask = torch.ones(
                mel.shape[0],
                1,
                mel.shape[-1],
                dtype=torch.bool,
                device=self.accelerator.device,
            )
            target_f0 = acoustic_models.f0_extractor(jdc_mel, frame_mask)
            target_n = acoustic_models.n_target(mel, frame_mask)
            posterior = acoustic_models.audio_encoder(
                jdc_mel,
                frame_mask,
                self.generator("acoustic-latent"),
            )
            predicted = acoustic_models.feature_linear(
                posterior.latent,
                posterior.mask,
                frame_mask,
            )
            f0_ratio = self.loss_weight(self.config.training.f0_prediction)
            n_ratio = self.loss_weight(self.config.training.n_prediction)
            decoder_acoustic = AcousticFeatures(
                target_f0 * (1 - f0_ratio) + predicted.voiced_f0 * f0_ratio,
                target_n * (1 - n_ratio) + predicted.n * n_ratio,
            )
            smoothing_kernels = (0, 3, 7)
            smoothing_index = torch.randint(
                len(smoothing_kernels),
                (1,),
                device=self.accelerator.device,
                generator=self.generator("f0-smoothing"),
            ).item()
            smoothing_kernel = smoothing_kernels[smoothing_index]
            if smoothing_kernel:
                smoothed_f0 = torch.nn.functional.avg_pool1d(
                    decoder_acoustic.f0.unsqueeze(1),
                    smoothing_kernel,
                    stride=1,
                    padding=smoothing_kernel // 2,
                ).squeeze(1)
                decoder_acoustic = AcousticFeatures(
                    smoothed_f0,
                    decoder_acoustic.n,
                )
            decoded = acoustic_models.decoder(
                posterior.latent,
                decoder_acoustic.f0,
                decoder_acoustic.n,
                posterior.mask,
                frame_mask,
            )
            generated = acoustic_models.generator(
                decoded.features,
                decoded.f0,
                decoded.mask,
                self.generator("acoustic-source"),
            )
            with torch.autocast(
                device_type=real.device.type,
                enabled=False,
            ):
                discriminator_real, discriminator_fake = (
                    acoustic_models.phase_augmentation.forward_sync(
                        real.float(),
                        generated.detach().float(),
                    )
                )
            discriminator_output = acoustic_models.discriminators(
                discriminator_real,
                discriminator_fake,
            )
            discriminator = discriminator_step_loss(
                discriminator_output.real.logits,
                discriminator_output.fake.logits,
            )
            weights = self.acoustic_weights()
            discriminator_total = discriminator * weights.discriminator
        self.accelerator.backward(
            discriminator_total / self.config.training.accumulation_steps
        )
        with self.accelerator.autocast():
            encoder_kl = masked_kl_standard_normal(
                posterior.mean,
                posterior.log_scale,
                posterior.mask,
            )
            pitch = masked_pitch_loss(
                predicted.f0,
                predicted.voicing_logits,
                target_f0,
                decoded.mask,
            )
            f0 = pitch.total
            n = masked_n_smooth_l1(
                predicted.n,
                target_n,
                decoded.mask,
            )
            sample_mask = frame_mask.repeat_interleave(
                acoustic_models.output_hop,
                dim=-1,
            )
            reconstruction = acoustic_models.reconstruction_loss(
                generated,
                real,
                sample_mask,
                self.step + 1,
            ).total
            discriminator_parameters = tuple(
                acoustic_models.discriminators.parameters()
            )
            for parameter in discriminator_parameters:
                parameter.requires_grad_(False)
            with torch.autocast(
                device_type=real.device.type,
                enabled=False,
            ):
                generator_real, generator_fake = (
                    acoustic_models.phase_augmentation.forward_sync(
                        real.float(),
                        generated.float(),
                    )
                )
            adversarial_output = acoustic_models.discriminators(
                generator_real,
                generator_fake,
            )
            for parameter in discriminator_parameters:
                parameter.requires_grad_(True)
            adversarial = generator_step_loss(
                adversarial_output.real.logits,
                adversarial_output.fake.logits,
                adversarial_output.real.feature_maps,
                adversarial_output.fake.feature_maps,
                adversarial_output.period_count,
            )
            vocoder_total = (
                reconstruction * weights.reconstruction
                + adversarial.adversarial * weights.generator_adversarial
                + adversarial.feature_matching * weights.feature_matching
            )
            generator_total = (
                encoder_kl * weights.encoder_kl
                + f0 * weights.f0
                + n * weights.n
                + vocoder_total
            )
        self.accelerator.backward(
            generator_total / self.config.training.accumulation_steps
        )
        return {
            "encoder_kl": encoder_kl,
            "f0": f0,
            "f0_regression": pitch.regression,
            "f0_voicing": pitch.voicing,
            "f0_mae_hz": pitch.mae_hz,
            "f0_voicing_accuracy": pitch.voicing_accuracy,
            "n": n,
            "posterior_reconstruction": reconstruction,
            "discriminator": discriminator,
            "generator_adversarial": adversarial.adversarial,
            "feature_period": adversarial.feature_period,
            "feature_resolution": adversarial.feature_resolution,
            "feature_matching": adversarial.feature_matching,
            "vocoder_total": vocoder_total,
            "discriminator_total": discriminator_total,
            "f0_prediction_ratio": f0_ratio,
            "n_prediction_ratio": n_ratio,
            "generator_total": generator_total,
        }

    def latent_flow_step(self, batch: BeetleBatch) -> dict[str, Tensor | float]:
        conditional_models = self.conditional_models
        builder = self.conditioning
        if conditional_models is None or builder is None:
            raise RuntimeError("conditional stage was not prepared")
        inputs = builder.build(
            conditional_models,
            batch,
            self.step,
            self.batch_index,
            validation=False,
        )
        with self.accelerator.autocast():
            style_views = conditional_models.style_encoder(
                inputs.style_view_latent,
                inputs.style_view_mask,
            )
            voice_views = inputs.voice_view_embeddings
            flow_prediction, flow_target = self.alpha_flow_tensors(inputs)
            full_end_time = torch.ones_like(inputs.flow_sample.start_time)
            full_end_time = full_end_time * inputs.latent_mask
            full_prediction = conditional_models.latent_flow(
                inputs.flow_sample.state,
                inputs.flow_sample.start_time,
                full_end_time,
                inputs.conditions,
                inputs.latent_mask,
            )
            generated_latent = (
                inputs.flow_sample.state
                + (full_end_time - inputs.flow_sample.start_time)
                * full_prediction
            ) * inputs.latent_mask
            generated_style = conditional_models.style_encoder(
                generated_latent,
                inputs.target_latent_mask,
            )
            speaker_logits = conditional_models.style_speaker_classifier(
                inputs.target_style,
                inputs.reversal_scale,
            )
            statistics = conditional_models.style_statistics_head(inputs.target_style)
            voice_ge2e = conditional_models.voice_ge2e(
                voice_views,
                inputs.voice_group_ids,
            )
            style_ge2e = conditional_models.style_ge2e(
                style_views,
                inputs.style_group_ids,
            )
            statistics_values = torch.stack(
                (
                    statistics.f0_mean,
                    statistics.f0_std,
                    statistics.n_mean,
                    statistics.n_std,
                ),
                dim=1,
            )
            statistics_target = torch.stack(
                (
                    inputs.statistics_target.f0_mean,
                    inputs.statistics_target.f0_std,
                    inputs.statistics_target.n_mean,
                    inputs.statistics_target.n_std,
                ),
                dim=1,
            )
            outputs = ConditionalModelOutput(
                flow_prediction,
                generated_style,
                style_views,
                voice_views,
                speaker_logits,
                statistics_values,
                voice_ge2e,
                style_ge2e,
            )
            loss_inputs = ConditionalLossInput(
                inputs.duration_nll,
                inputs.phoneme_mask,
                flow_target,
                inputs.flow_sample.alpha,
                inputs.flow_sample.flow_matching_count,
                inputs.latent_mask,
                inputs.alignment.ctc_logits,
                inputs.alignment.s2s_logits,
                inputs.alignment.soft_alignment,
                inputs.alignment.hard_alignment,
                inputs.phonemes,
                inputs.alignment_mask,
                inputs.target_style,
                inputs.voice_group_ids,
                inputs.style_group_ids,
                inputs.style_positive_weights,
                inputs.speaker_ids,
                statistics_target,
                inputs.contrastive_temperature,
                inputs.consistency_cosine_weight,
                inputs.consistency_mse_weight,
                inputs.align_blank_id,
            )
            losses = compute_conditional_losses(loss_inputs, outputs)
            generator_total = losses.total(self.conditional_weights())
        self.accelerator.backward(
            generator_total / self.config.training.accumulation_steps
        )
        metrics: dict[str, Tensor | float] = self.conditional_metrics(losses)
        metrics["latent_flow_mse"] = flow_mse(
            flow_prediction,
            flow_target,
            inputs.latent_mask,
        )
        metrics["trajectory_flow_matching"] = flow_mse(
            flow_prediction,
            inputs.flow_sample.velocity,
            inputs.latent_mask,
        )
        metrics["trajectory_consistency"] = 2 * flow_inner_product(
            inputs.flow_sample.velocity - flow_target,
            flow_prediction,
            inputs.latent_mask,
        )
        metrics["trajectory_sum"] = (
            metrics["trajectory_flow_matching"]
            + metrics["trajectory_consistency"]
        )
        metrics["alpha_flow_ratio"] = inputs.flow_sample.alpha
        for name, value in inputs.batch_statistics.named_values():
            metrics[f"conditioning/{name}"] = value
        metrics["generator_total"] = generator_total
        return metrics

    def alpha_flow_tensors(
        self,
        inputs: ConditionalTrainingInput,
    ) -> tuple[Tensor, Tensor]:
        conditional_models = self.conditional_models
        if conditional_models is None:
            raise RuntimeError("conditional stage was not prepared")
        model = conditional_models.latent_flow
        sample = inputs.flow_sample
        prediction = model(
            sample.state,
            sample.start_time,
            sample.end_time,
            inputs.conditions,
            inputs.latent_mask,
        )
        target = sample.velocity.clone()
        trajectory_start = sample.flow_matching_count
        if trajectory_start == sample.state.shape[0]:
            return prediction, target
        trajectory_mask = inputs.latent_mask[trajectory_start:]
        trajectory_velocity = sample.velocity[trajectory_start:]
        clip = self.config.training.alpha_flow.target_clip
        if sample.alpha == 1.0:
            target[trajectory_start:] = trajectory_velocity.clamp(-clip, clip)
            return prediction, target
        trajectory_conditions = inputs.conditions.slice_from(trajectory_start)
        if sample.alpha == 0.0:
            state = sample.state[trajectory_start:]
            start_time = sample.start_time[trajectory_start:]
            end_time = sample.end_time[trajectory_start:]
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                _, derivative = jvp(
                    partial(
                        model,
                        conditions=trajectory_conditions,
                        mask=trajectory_mask,
                    ),
                    (state, start_time, end_time),
                    (
                        trajectory_velocity,
                        trajectory_mask.to(dtype=start_time.dtype),
                        torch.zeros_like(end_time),
                    ),
                )
            interval = end_time - start_time
            target[trajectory_start:] = trajectory_velocity + interval * derivative
            return prediction, target
        with torch.no_grad():
            future_velocity = model(
                sample.intermediate_state[trajectory_start:],
                sample.intermediate_time[trajectory_start:],
                sample.end_time[trajectory_start:],
                trajectory_conditions,
                trajectory_mask,
            )
        discrete_target = (
            sample.alpha * trajectory_velocity
            + (1 - sample.alpha) * future_velocity
        )
        target[trajectory_start:] = discrete_target.clamp(-clip, clip)
        return prediction, target

    def end_to_end_step(self, batch: BeetleBatch) -> dict[str, Tensor | float]:
        acoustic_metrics = self.posterior_step(batch)
        conditional_metrics = self.latent_flow_step(batch)
        acoustic_total = acoustic_metrics["generator_total"]
        conditional_total = conditional_metrics["generator_total"]
        combined = acoustic_total + conditional_total
        acoustic_metrics.update(conditional_metrics)
        acoustic_metrics["generator_total"] = combined
        return acoustic_metrics

    @torch.no_grad()
    def validation_step(
        self,
        recording: ValidationRecording,
        position: int,
    ) -> dict[str, float]:
        stage = self.config.training.stage
        values = recording.batch.to(self.accelerator.device)
        acoustic_models = self.acoustic_models
        conditional_models = self.conditional_models
        sample_count = int(values.waveform_lengths[0])
        target_waveform = values.waveform[0, :, :sample_count]
        prediction = None
        predicted_latent = None
        predicted_f0 = None
        predicted_n = None
        alignment = None
        soft_alignment = None
        metrics: dict[str, Tensor | float] = {}
        target_f0 = None
        target_n = None
        if stage is TrainingStage.POSTERIOR:
            if acoustic_models is None:
                raise RuntimeError("posterior validation model is unavailable")
            with self.accelerator.autocast():
                target_f0 = acoustic_models.f0_extractor(
                    values.jdc_mel,
                    values.frame_mask,
                )
                target_n = acoustic_models.n_target(values.mel, values.frame_mask)
                posterior = acoustic_models.audio_encoder(
                    values.jdc_mel,
                    values.frame_mask,
                    self.generator(f"validation-{position}-latent"),
                )
                features = acoustic_models.feature_linear(
                    posterior.latent,
                    posterior.mask,
                    values.frame_mask,
                )
                decoded = acoustic_models.decoder(
                    posterior.latent,
                    features.voiced_f0,
                    features.n,
                    posterior.mask,
                    values.frame_mask,
                )
                generated = acoustic_models.generator(
                    decoded.features,
                    decoded.f0,
                    decoded.mask,
                    self.generator(f"validation-{position}-source"),
                )
                prediction = generated[0, :, :sample_count]
                predicted_latent = posterior.latent[0]
                predicted_f0 = features.voiced_f0[0]
                predicted_n = features.n[0]
                metrics["encoder_kl"] = masked_kl_standard_normal(
                    posterior.mean,
                    posterior.log_scale,
                    posterior.mask,
                )
                pitch = masked_pitch_loss(
                    features.f0,
                    features.voicing_logits,
                    target_f0,
                    values.frame_mask,
                )
                metrics["f0"] = pitch.total
                metrics["f0_regression"] = pitch.regression
                metrics["f0_voicing"] = pitch.voicing
                metrics["f0_mae_hz"] = pitch.mae_hz
                metrics["f0_voicing_accuracy"] = pitch.voicing_accuracy
                metrics["n"] = masked_n_smooth_l1(
                    features.n,
                    target_n,
                    values.frame_mask,
                )
                sample_mask = values.frame_mask.repeat_interleave(
                    acoustic_models.output_hop,
                    dim=-1,
                )
                metrics["reconstruction"] = acoustic_models.reconstruction_loss(
                    generated,
                    values.waveform,
                    sample_mask,
                    self.step,
                ).total
                discriminator_output = acoustic_models.discriminators(
                    values.waveform,
                    generated,
                )
                metrics["discriminator"] = discriminator_step_loss(
                    discriminator_output.real.logits,
                    discriminator_output.fake.logits,
                )
                adversarial = generator_step_loss(
                    discriminator_output.real.logits,
                    discriminator_output.fake.logits,
                    discriminator_output.real.feature_maps,
                    discriminator_output.fake.feature_maps,
                    discriminator_output.period_count,
                )
                metrics["generator_adversarial"] = adversarial.adversarial
                metrics["feature_period"] = adversarial.feature_period
                metrics["feature_resolution"] = adversarial.feature_resolution
                metrics["feature_matching"] = adversarial.feature_matching
                weights = self.acoustic_weights()
                vocoder_total = (
                    metrics["reconstruction"] * weights.reconstruction
                    + adversarial.adversarial
                    * weights.generator_adversarial
                    + adversarial.feature_matching
                    * weights.feature_matching
                )
                metrics["discriminator_total"] = (
                    metrics["discriminator"] * weights.discriminator
                )
                metrics["generator_total"] = (
                    metrics["encoder_kl"] * weights.encoder_kl
                    + metrics["f0"] * weights.f0
                    + metrics["n"] * weights.n
                    + vocoder_total
                )
            target_latent = posterior.latent[0]
        else:
            if conditional_models is None or self.conditioning is None:
                raise RuntimeError("conditional validation model is unavailable")
            inputs = self.conditioning.build(
                conditional_models,
                values,
                self.step,
                position,
                validation=True,
            )
            with self.accelerator.autocast():
                predicted_duration = conditional_models.duration_predictor.sample(
                    inputs.duration_condition,
                    inputs.phoneme_mask,
                    self.generator(f"validation-{position}-duration"),
                )
                duration_mask = inputs.phoneme_mask.to(
                    dtype=predicted_duration.dtype
                )
                metrics["duration_mae_frames"] = (
                    (predicted_duration - inputs.duration_target).abs()
                    * duration_mask
                ).sum() / duration_mask.sum().clamp_min(1)
                style_views = conditional_models.style_encoder(
                    inputs.style_view_latent,
                    inputs.style_view_mask,
                )
                voice_views = inputs.voice_view_embeddings
                flow_prediction, flow_target = self.alpha_flow_tensors(inputs)
                full_end_time = torch.ones_like(inputs.flow_sample.start_time)
                full_end_time = full_end_time * inputs.latent_mask
                full_prediction = conditional_models.latent_flow(
                    inputs.flow_sample.state,
                    inputs.flow_sample.start_time,
                    full_end_time,
                    inputs.conditions,
                    inputs.latent_mask,
                )
                generated_training_latent = (
                    inputs.flow_sample.state
                    + (full_end_time - inputs.flow_sample.start_time)
                    * full_prediction
                ) * inputs.latent_mask
                generated_style = conditional_models.style_encoder(
                    generated_training_latent,
                    inputs.target_latent_mask,
                )
                speaker_logits = conditional_models.style_speaker_classifier(
                    inputs.target_style,
                    inputs.reversal_scale,
                )
                statistics = conditional_models.style_statistics_head(
                    inputs.target_style
                )
                voice_ge2e = conditional_models.voice_ge2e(
                    voice_views,
                    inputs.voice_group_ids,
                )
                style_ge2e = conditional_models.style_ge2e(
                    style_views,
                    inputs.style_group_ids,
                )
                statistics_values = torch.stack(
                    (
                        statistics.f0_mean,
                        statistics.f0_std,
                        statistics.n_mean,
                        statistics.n_std,
                    ),
                    dim=1,
                )
                statistics_target = torch.stack(
                    (
                        inputs.statistics_target.f0_mean,
                        inputs.statistics_target.f0_std,
                        inputs.statistics_target.n_mean,
                        inputs.statistics_target.n_std,
                    ),
                    dim=1,
                )
                outputs = ConditionalModelOutput(
                    flow_prediction,
                    generated_style,
                    style_views,
                    voice_views,
                    speaker_logits,
                    statistics_values,
                    voice_ge2e,
                    style_ge2e,
                )
                loss_inputs = ConditionalLossInput(
                    inputs.duration_nll,
                    inputs.phoneme_mask,
                    flow_target,
                    inputs.flow_sample.alpha,
                    inputs.flow_sample.flow_matching_count,
                    inputs.latent_mask,
                    inputs.alignment.ctc_logits,
                    inputs.alignment.s2s_logits,
                    inputs.alignment.soft_alignment,
                    inputs.alignment.hard_alignment,
                    inputs.phonemes,
                    inputs.alignment_mask,
                    inputs.target_style,
                    inputs.voice_group_ids,
                    inputs.style_group_ids,
                    inputs.style_positive_weights,
                    inputs.speaker_ids,
                    statistics_target,
                    inputs.contrastive_temperature,
                    inputs.consistency_cosine_weight,
                    inputs.consistency_mse_weight,
                    inputs.align_blank_id,
                )
                losses = compute_conditional_losses(loss_inputs, outputs)
                metrics.update(self.conditional_metrics(losses))
                metrics["latent_flow_mse"] = flow_mse(
                    flow_prediction,
                    flow_target,
                    inputs.latent_mask,
                )
                metrics["trajectory_flow_matching"] = flow_mse(
                    flow_prediction,
                    inputs.flow_sample.velocity,
                    inputs.latent_mask,
                )
                metrics["trajectory_consistency"] = 2 * flow_inner_product(
                    inputs.flow_sample.velocity - flow_target,
                    flow_prediction,
                    inputs.latent_mask,
                )
                metrics["trajectory_sum"] = (
                    metrics["trajectory_flow_matching"]
                    + metrics["trajectory_consistency"]
                )
                metrics["alpha_flow_ratio"] = inputs.flow_sample.alpha
                conditional_total = losses.total(self.conditional_weights())
                generation_noise = torch.randn(
                    inputs.flow_sample.noise.shape,
                    dtype=inputs.flow_sample.noise.dtype,
                    device=inputs.flow_sample.noise.device,
                    generator=self.generator(
                        f"validation-{position}-generation-noise"
                    ),
                ) * inputs.latent_mask
                generated_latent = integrate_latent_flow(
                    conditional_models.latent_flow,
                    generation_noise,
                    inputs.conditions,
                    inputs.latent_mask,
                    self.config.training.alpha_flow.sampling_steps,
                )
            target_latent = (
                inputs.flow_sample.velocity + inputs.flow_sample.noise
            )[0]
            predicted_latent = generated_latent[0]
            alignment = inputs.alignment.hard_alignment[0]
            soft_alignment = inputs.alignment.soft_alignment[0]
            target_f0 = inputs.acoustic_target.f0
            target_n = inputs.acoustic_target.n
            if stage is TrainingStage.LATENT_FLOW:
                with self.accelerator.autocast():
                    predicted_acoustic = conditional_models.feature_linear(
                        generated_latent,
                        inputs.latent_mask,
                        values.frame_mask,
                    )
                    decoded = conditional_models.decoder(
                        generated_latent,
                        predicted_acoustic.voiced_f0,
                        predicted_acoustic.n,
                        inputs.latent_mask,
                        values.frame_mask,
                    )
                    generated = conditional_models.generator(
                        decoded.features,
                        decoded.f0,
                        decoded.mask,
                        self.generator(f"validation-{position}-flow-source"),
                    )
                prediction = generated[0, :, :sample_count]
                predicted_f0 = predicted_acoustic.voiced_f0[0]
                predicted_n = predicted_acoustic.n[0]
            if stage is TrainingStage.END_TO_END:
                if acoustic_models is None:
                    raise RuntimeError("end-to-end acoustic model is unavailable")
                with self.accelerator.autocast():
                    predicted_acoustic = acoustic_models.feature_linear(
                        generated_latent,
                        inputs.latent_mask,
                        values.frame_mask,
                    )
                    decoded = acoustic_models.decoder(
                        generated_latent,
                        predicted_acoustic.voiced_f0,
                        predicted_acoustic.n,
                        inputs.latent_mask,
                        values.frame_mask,
                    )
                    generated = acoustic_models.generator(
                        decoded.features,
                        decoded.f0,
                        decoded.mask,
                        self.generator(f"validation-{position}-flow-source"),
                    )
                    posterior = acoustic_models.audio_encoder(
                        values.jdc_mel,
                        values.frame_mask,
                        self.generator(f"validation-{position}-posterior-latent"),
                    )
                    posterior_acoustic = acoustic_models.feature_linear(
                        posterior.latent,
                        posterior.mask,
                        values.frame_mask,
                    )
                    posterior_decoded = acoustic_models.decoder(
                        posterior.latent,
                        posterior_acoustic.voiced_f0,
                        posterior_acoustic.n,
                        posterior.mask,
                        values.frame_mask,
                    )
                    posterior_waveform = acoustic_models.generator(
                        posterior_decoded.features,
                        posterior_decoded.f0,
                        posterior_decoded.mask,
                        self.generator(
                            f"validation-{position}-posterior-source"
                        ),
                    )
                    metrics["encoder_kl"] = masked_kl_standard_normal(
                        posterior.mean,
                        posterior.log_scale,
                        posterior.mask,
                    )
                    pitch = masked_pitch_loss(
                        posterior_acoustic.f0,
                        posterior_acoustic.voicing_logits,
                        inputs.acoustic_target.f0,
                        values.frame_mask,
                    )
                    metrics["f0"] = pitch.total
                    metrics["f0_regression"] = pitch.regression
                    metrics["f0_voicing"] = pitch.voicing
                    metrics["f0_mae_hz"] = pitch.mae_hz
                    metrics["f0_voicing_accuracy"] = pitch.voicing_accuracy
                    metrics["n"] = masked_n_smooth_l1(
                        posterior_acoustic.n,
                        inputs.acoustic_target.n,
                        values.frame_mask,
                    )
                    sample_mask = values.frame_mask.repeat_interleave(
                        acoustic_models.output_hop,
                        dim=-1,
                    )
                    metrics["reconstruction"] = (
                        acoustic_models.reconstruction_loss(
                            posterior_waveform,
                            values.waveform,
                            sample_mask,
                            self.step,
                        ).total
                    )
                    discriminator_output = acoustic_models.discriminators(
                        values.waveform,
                        posterior_waveform,
                    )
                    metrics["discriminator"] = discriminator_step_loss(
                        discriminator_output.real.logits,
                        discriminator_output.fake.logits,
                    )
                    adversarial = generator_step_loss(
                        discriminator_output.real.logits,
                        discriminator_output.fake.logits,
                        discriminator_output.real.feature_maps,
                        discriminator_output.fake.feature_maps,
                        discriminator_output.period_count,
                    )
                    metrics["generator_adversarial"] = adversarial.adversarial
                    metrics["feature_period"] = adversarial.feature_period
                    metrics["feature_resolution"] = adversarial.feature_resolution
                    metrics["feature_matching"] = adversarial.feature_matching
                    weights = self.acoustic_weights()
                    vocoder_total = (
                        metrics["reconstruction"] * weights.reconstruction
                        + adversarial.adversarial
                        * weights.generator_adversarial
                        + adversarial.feature_matching
                        * weights.feature_matching
                    )
                    metrics["discriminator_total"] = (
                        metrics["discriminator"] * weights.discriminator
                    )
                    acoustic_total = (
                        metrics["encoder_kl"] * weights.encoder_kl
                        + metrics["f0"] * weights.f0
                        + metrics["n"] * weights.n
                        + vocoder_total
                    )
                    metrics["generator_total"] = (
                        acoustic_total + conditional_total
                    )
                prediction = generated[0, :, :sample_count]
                predicted_f0 = predicted_acoustic.voiced_f0[0]
                predicted_n = predicted_acoustic.n[0]
        scalars = {
            f"validation/{name}": (
                float(value.detach().float().cpu())
                if isinstance(value, Tensor)
                else float(value)
            )
            for name, value in metrics.items()
        }
        if self.artifacts is not None:
            self.artifacts.publish(
                self.step,
                position,
                ValidationSample(
                    target_waveform,
                    prediction,
                    target_latent,
                    predicted_latent,
                    target_f0[0] if target_f0 is not None else None,
                    predicted_f0,
                    target_n[0] if target_n is not None else None,
                    predicted_n,
                    alignment,
                    soft_alignment,
                ),
                scalars,
            )
        return scalars

    def train(self) -> int:
        self.prepare()
        try:
            while self.step < self.config.training.total_steps:
                self.callbacks.check_cancel()
                step_started_at = time.monotonic()
                step_data_wait = 0.0
                generator_lr = self.learning_rate(
                    self.config.training.generator_optimizer
                )
                self.set_learning_rate(self.optimizers.generator, generator_lr)
                self.optimizers.generator.zero_grad(set_to_none=True)
                if self.optimizers.discriminator is not None:
                    discriminator_lr = self.learning_rate(
                        self.config.training.discriminator_optimizer
                    )
                    self.set_learning_rate(
                        self.optimizers.discriminator,
                        discriminator_lr,
                    )
                    self.optimizers.discriminator.zero_grad(set_to_none=True)
                else:
                    discriminator_lr = None
                update_metrics: defaultdict[str, float] = defaultdict(float)
                update_metric_count = 0
                update_failed = False
                for _ in range(self.config.training.accumulation_steps):
                    data_wait_started_at = time.monotonic()
                    try:
                        batch = self.pipeline.next_batch()
                    except BatchLoadError as error:
                        if not self.recovery_allowed():
                            raise
                        self.pipeline.mark_consumed()
                        self.batch_index += 1
                        self.skip_update("data", error)
                        update_failed = True
                        break
                    except Exception as error:
                        if not self.recovery_allowed():
                            raise
                        state = self.pipeline.state_dict()
                        try:
                            self.pipeline.load_state_dict(state)
                        except Exception as restart_error:
                            logger.warning(
                                "data pipeline restart deferred: %s",
                                restart_error,
                            )
                        self.skip_update("prefetch", error)
                        update_failed = True
                        break
                    waited = time.monotonic() - data_wait_started_at
                    step_data_wait += waited
                    self.data_wait_seconds += waited
                    try:
                        metrics = self.train_step(batch)
                    except Exception as error:
                        if not self.recovery_allowed():
                            raise
                        self.pipeline.mark_consumed()
                        self.batch_index += 1
                        reason = (
                            "oom"
                            if isinstance(error, torch.cuda.OutOfMemoryError)
                            else "batch"
                        )
                        self.skip_update(reason, error)
                        update_failed = True
                        break
                    self.pipeline.mark_consumed()
                    self.batch_index += 1
                    self.items_processed += (
                        batch.waveform.shape[0] * self.accelerator.num_processes
                    )
                    for name, value in metrics.items():
                        scalar = (
                            float(value.detach().float().cpu())
                            if isinstance(value, Tensor)
                            else float(value)
                        )
                        update_metrics[name] += scalar
                    update_metric_count += 1
                if update_failed:
                    continue
                for name, value in update_metrics.items():
                    self.metric_sums[name] += value
                self.metric_count += update_metric_count
                active_optimizers = [self.optimizers.generator]
                if self.optimizers.discriminator is not None:
                    active_optimizers.append(self.optimizers.discriminator)
                self.accelerator.unscale_gradients(active_optimizers)
                generator_parameters = tuple(
                    parameter
                    for group in self.optimizers.generator.param_groups
                    for parameter in group["params"]
                )
                generator_norm = self.parameter_gradient_norm(
                    generator_parameters
                )
                discriminator_norm = None
                if self.optimizers.discriminator is not None:
                    discriminator_parameters = tuple(
                        parameter
                        for group in self.optimizers.discriminator.param_groups
                        for parameter in group["params"]
                    )
                    discriminator_norm = self.parameter_gradient_norm(
                        discriminator_parameters
                    )
                gradients_are_finite = torch.isfinite(generator_norm)
                if discriminator_norm is not None:
                    gradients_are_finite &= torch.isfinite(discriminator_norm)
                if not gradients_are_finite:
                    scaler = self.accelerator.scaler
                    if scaler is not None:
                        scaler.update()
                    self.skip_update(
                        "nonfinite",
                        FloatingPointError(
                            "optimizer gradient norm is non-finite"
                        ),
                    )
                    continue
                gradient_metrics, generator_norm, discriminator_norm = (
                    self.gradient_metrics()
                )
                scaler = self.accelerator.scaler
                if scaler is None:
                    if self.optimizers.discriminator is not None:
                        self.optimizers.discriminator.step()
                    self.optimizers.generator.step()
                else:
                    if self.optimizers.discriminator is not None:
                        scaler.step(self.optimizers.discriminator.optimizer)
                    scaler.step(self.optimizers.generator.optimizer)
                    scaler.update()
                self.step += 1
                for name, value in gradient_metrics.items():
                    if name.endswith(("_clip_coefficient", "_was_clipped")):
                        self.sparse_metrics[name] = value
                    else:
                        self.metric_sums[name] += value
                self.metric_sums["optimizer/generator_learning_rate"] += generator_lr
                self.metric_sums["optimizer/plbert_learning_rate"] += generator_lr
                self.metric_sums["optimizer/generator_gradient_norm"] += float(
                    generator_norm
                )
                self.metric_sums["optimizer/generator_amp_scale"] += (
                    scaler.get_scale() if scaler is not None else 1.0
                )
                self.metric_sums["skipped_steps"] += 0.0
                self.optimizer_metric_count += 1
                if discriminator_lr is not None and discriminator_norm is not None:
                    self.metric_sums[
                        "optimizer/discriminator_learning_rate"
                    ] += discriminator_lr
                    self.metric_sums[
                        "optimizer/discriminator_gradient_norm"
                    ] += float(discriminator_norm)
                    self.metric_sums["optimizer/discriminator_amp_scale"] += (
                        scaler.get_scale() if scaler is not None else 1.0
                    )
                self.compute_seconds += max(
                    time.monotonic() - step_started_at - step_data_wait,
                    0.0,
                )
                try:
                    self.callbacks.report_training_progress(
                        self.step,
                        self.config.training.total_steps,
                    )
                except Exception as error:
                    if not self.recovery_allowed():
                        raise
                    self.record_recovery("progress", error)
                if self.step % self.config.runtime.log_every_steps == 0:
                    reporting_started_at = time.monotonic()
                    try:
                        self.log_training_metrics()
                    except Exception as error:
                        if not self.recovery_allowed():
                            raise
                        self.record_recovery("reporting", error)
                    self.reporting_seconds += (
                        time.monotonic() - reporting_started_at
                    )
                if self.step % self.config.training.validation_every_steps == 0:
                    validation_started_at = time.monotonic()
                    try:
                        self.run_validation()
                    except Exception as error:
                        if not self.first_validation_succeeded:
                            raise
                        self.record_recovery("validation", error)
                    self.validation_seconds += (
                        time.monotonic() - validation_started_at
                    )
                if self.step % self.config.checkpoint.every_steps == 0:
                    checkpoint_started_at = time.monotonic()
                    self.save_checkpoint()
                    self.checkpoint_seconds += (
                        time.monotonic() - checkpoint_started_at
                    )
            if self.step % self.config.training.validation_every_steps:
                try:
                    self.run_validation()
                except Exception as error:
                    if not self.first_validation_succeeded:
                        raise
                    self.record_recovery("validation", error)
            if self.step % self.config.checkpoint.every_steps:
                self.save_checkpoint()
            if self.logger is not None:
                self.logger.finish()
            return self.step
        except BaseException:
            if self.logger is not None:
                self.logger.fail()
            raise
        finally:
            try:
                self.pipeline.close()
            except Exception as error:
                logger.warning("data pipeline close failed: %s", error)

    def recovery_allowed(self) -> bool:
        elapsed = time.monotonic() - self.training_started_at
        return self.first_validation_succeeded or elapsed >= 300

    def skip_update(self, reason: str, error: Exception) -> None:
        self.optimizers.generator.zero_grad(set_to_none=True)
        if self.optimizers.discriminator is not None:
            self.optimizers.discriminator.zero_grad(set_to_none=True)
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        self.metric_sums["skipped_steps"] += 1
        self.record_recovery(reason, error)

    def record_recovery(self, reason: str, error: Exception) -> None:
        self.recovery_counts[reason] += 1
        logger.warning(
            "training recovered from %s failure at step %d: %s: %s",
            reason,
            self.step,
            type(error).__name__,
            error,
        )

    def save_checkpoint(self) -> None:
        try:
            self.checkpoints.save(
                self.step,
                self.batch_index,
                self.pipeline.state_dict(),
                self.run_id,
                self.model_bundle,
                self.optimizers,
            )
        except Exception as error:
            self.record_recovery("checkpoint", error)

    def run_validation(self) -> None:
        modules = self.training_modules()
        modes = tuple(module.training for module in modules)
        for module in modules:
            module.eval()
        totals: defaultdict[str, float] = defaultdict(float)
        successful = 0
        try:
            for position, recording in enumerate(
                self.validation_recordings,
                start=1,
            ):
                try:
                    metrics = self.validation_step(recording, position)
                except Exception as error:
                    if not self.first_validation_succeeded:
                        raise
                    self.record_recovery("validation", error)
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    continue
                successful += 1
                for name, value in metrics.items():
                    totals[name] += value
            if successful:
                averages = {
                    name: value / successful for name, value in totals.items()
                }
                averages = self.reduce_metrics(averages)
                if self.logger is not None:
                    self.logger.log_metrics(averages, self.step)
                    self.logger.flush()
            self.first_validation_succeeded = True
        finally:
            for module, mode in zip(modules, modes, strict=True):
                module.train(mode)

    def log_training_metrics(self) -> None:
        loss_divisor = max(1, self.metric_count)
        optimizer_divisor = max(1, self.optimizer_metric_count)
        metrics = {}
        for name, value in self.metric_sums.items():
            optimizer_metric = name.startswith(("optimizer/", "gradient/"))
            optimizer_metric = optimizer_metric or name == "skipped_steps"
            output_name = name if optimizer_metric else f"train/{name}"
            if name == "skipped_steps":
                divisor = 1
            else:
                divisor = optimizer_divisor if optimizer_metric else loss_divisor
            metrics[output_name] = value / divisor
        metrics.update(self.sparse_metrics)
        metrics = self.reduce_metrics(metrics)
        if self.logger is not None:
            elapsed = time.monotonic() - self.training_started_at
            measured_steps = self.step - self.initial_step
            steps_per_second = measured_steps / elapsed
            items_per_second = self.items_processed / elapsed
            remaining_steps = self.config.training.total_steps - self.step
            eta_seconds = remaining_steps / steps_per_second
            metrics.update(
                {
                    "performance/items_per_second": items_per_second,
                    "performance/steps_per_second": steps_per_second,
                    "performance/elapsed_seconds": elapsed,
                    "performance/eta_seconds": eta_seconds,
                    "performance/eta_hours": eta_seconds / 3600,
                }
            )
            measured = (
                self.data_wait_seconds
                + self.compute_seconds
                + self.validation_seconds
                + self.checkpoint_seconds
                + self.reporting_seconds
            )
            overhead = {
                "data_wait": self.data_wait_seconds,
                "compute": self.compute_seconds,
                "validation": self.validation_seconds,
                "checkpoint": self.checkpoint_seconds,
                "reporting": self.reporting_seconds,
                "residual": max(elapsed - measured, 0.0),
            }
            metrics.update(
                {
                    f"overhead/{name}_percent": 100 * value / elapsed
                    for name, value in overhead.items()
                }
            )
            metrics.update(self.logger.health_metrics())
            pending_metrics = metrics["overhead/pending_metric_operations"]
            pending_artifacts = metrics["overhead/pending_artifact_jobs"]
            metrics["overhead/metric_queue_utilization_percent"] = (
                100 * pending_metrics / 256
            )
            metrics["overhead/artifact_queue_utilization_percent"] = (
                100 * pending_artifacts / 4096
            )
            metrics.update(
                {
                    f"recovery/{name}_failures": float(value)
                    for name, value in self.recovery_counts.items()
                }
            )
            metrics.update(self.logger.sample_system_metrics())
            self.logger.log_metrics(metrics, self.step)
        self.metric_sums.clear()
        self.sparse_metrics.clear()
        self.metric_count = 0
        self.optimizer_metric_count = 0

    def reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        names = tuple(metrics)
        values = torch.tensor(
            [metrics[name] for name in names],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        reduced = self.accelerator.reduce(values, reduction="mean")
        return {
            name: float(value)
            for name, value in zip(names, reduced.cpu(), strict=True)
        }

    def gradient_metrics(
        self,
    ) -> tuple[dict[str, float], Tensor, Tensor | None]:
        generator_parameters = tuple(
            parameter
            for group in self.optimizers.generator.param_groups
            for parameter in group["params"]
        )
        generator_norm = self.parameter_gradient_norm(generator_parameters)
        owned_parameters = {id(parameter) for parameter in generator_parameters}
        groups: list[tuple[str, tuple[nn.Module, ...], bool]] = []
        acoustic_models = self.acoustic_models
        if acoustic_models is not None:
            groups.extend(
                (
                    ("audio_encoder", (acoustic_models.audio_encoder,), False),
                    ("feature_linear", (acoustic_models.feature_linear,), False),
                    ("decoder", (acoustic_models.decoder,), False),
                    ("generator", (acoustic_models.generator,), True),
                )
            )
        conditional_models = self.conditional_models
        if conditional_models is not None:
            groups.extend(
                (
                    ("plbert", (conditional_models.plbert,), False),
                    (
                        "phoneme_encoders",
                        (
                            conditional_models.phoneme_encoder,
                            conditional_models.latent_phoneme_encoder,
                            conditional_models.duration_phoneme_encoder,
                        ),
                        False,
                    ),
                    (
                        "context_encoders",
                        (
                            conditional_models.context_phoneme_encoder,
                            conditional_models.context_audio_encoder,
                        ),
                        False,
                    ),
                    (
                        "conditioning",
                        (
                            conditional_models.language_embedding,
                            conditional_models.condition_bank,
                        ),
                        False,
                    ),
                    ("style_encoder", (conditional_models.style_encoder,), False),
                    ("voice_encoder", (conditional_models.voice_encoder,), False),
                    (
                        "duration_predictor",
                        (conditional_models.duration_predictor,),
                        False,
                    ),
                    ("latent_flow", (conditional_models.latent_flow,), False),
                    ("aligner", (conditional_models.aligner,), False),
                    (
                        "style_auxiliaries",
                        (
                            conditional_models.style_speaker_classifier,
                            conditional_models.style_statistics_head,
                            conditional_models.style_ge2e,
                        ),
                        False,
                    ),
                    (
                        "voice_auxiliaries",
                        (conditional_models.voice_ge2e,),
                        False,
                    ),
                )
            )
        diagnostics = (self.step + 1) % 250 == 0
        metrics: dict[str, float] = {}
        coefficients = []
        maximum = self.config.training.generator_optimizer.maximum_gradient_norm
        for name, modules, clip in groups:
            parameters = tuple(
                parameter
                for parameter in self.module_parameters(modules)
                if id(parameter) in owned_parameters
            )
            if clip:
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, maximum))
                coefficient = min(1.0, maximum / (norm + 1e-6))
            else:
                norm = float(self.parameter_gradient_norm(parameters))
                coefficient = 1.0
            metrics[f"gradient/{name}"] = norm
            coefficients.append(coefficient)
            if diagnostics:
                metrics[f"gradient/{name}_clip_coefficient"] = coefficient
                metrics[f"gradient/{name}_was_clipped"] = float(
                    coefficient < 1.0
                )
        generator_coefficient = min(coefficients)
        if diagnostics:
            metrics["optimizer/generator_clip_coefficient"] = (
                generator_coefficient
            )
            metrics["optimizer/generator_was_clipped"] = float(
                generator_coefficient < 1.0
            )
        discriminator_norm = None
        discriminator_optimizer = self.optimizers.discriminator
        if discriminator_optimizer is not None and acoustic_models is not None:
            discriminator_parameters = tuple(
                parameter
                for group in discriminator_optimizer.param_groups
                for parameter in group["params"]
            )
            maximum = (
                self.config.training.discriminator_optimizer.maximum_gradient_norm
            )
            discriminator_norm = torch.nn.utils.clip_grad_norm_(
                discriminator_parameters,
                maximum,
            )
            coefficient = min(
                1.0,
                maximum / (float(discriminator_norm) + 1e-6),
            )
            metrics["gradient/discriminators"] = float(discriminator_norm)
            if diagnostics:
                metrics["gradient/discriminators_clip_coefficient"] = coefficient
                metrics["gradient/discriminators_was_clipped"] = float(
                    coefficient < 1.0
                )
                metrics["optimizer/discriminator_clip_coefficient"] = coefficient
                metrics["optimizer/discriminator_was_clipped"] = float(
                    coefficient < 1.0
                )
        return metrics, generator_norm, discriminator_norm

    def module_parameters(
        self,
        modules: tuple[nn.Module, ...],
    ) -> tuple[nn.Parameter, ...]:
        parameters = []
        identities = set()
        for module in modules:
            for parameter in module.parameters():
                if id(parameter) not in identities:
                    parameters.append(parameter)
                    identities.add(id(parameter))
        return tuple(parameters)

    def parameter_gradient_norm(
        self,
        parameters: tuple[nn.Parameter, ...],
    ) -> Tensor:
        gradients = tuple(
            parameter.grad.detach().float().norm(2)
            for parameter in parameters
            if parameter.grad is not None
        )
        if not gradients:
            return torch.zeros((), device=self.accelerator.device)
        return torch.stack(gradients).norm(2)

    def acoustic_weights(self) -> AcousticLossWeights:
        values = tuple(
            self.loss_weight(getattr(self.config.training.losses, name))
            for name in ACOUSTIC_LOSS_NAMES
        )
        return AcousticLossWeights(*values)

    def conditional_weights(self) -> ConditionalLossWeights:
        values = tuple(
            self.loss_weight(getattr(self.config.training.losses, name))
            for name in CONDITIONAL_LOSS_NAMES
        )
        return ConditionalLossWeights(*values)

    def conditional_metrics(
        self,
        losses: ConditionalLossOutput,
    ) -> dict[str, Tensor]:
        return {
            name: value
            for name, value in zip(
                CONDITIONAL_LOSS_NAMES,
                losses.values(),
                strict=True,
            )
        }

    def loss_weight(self, schedule: ScheduledWeight) -> float:
        relative = self.step - schedule.start_step
        if relative < 0:
            return 0.0
        if schedule.warmup_steps and relative < schedule.warmup_steps:
            return schedule.value * relative / schedule.warmup_steps
        return schedule.value

    def learning_rate(self, settings: OptimizerConfig) -> float:
        if settings.warmup_steps and self.step < settings.warmup_steps:
            return settings.learning_rate * self.step / settings.warmup_steps
        decay_position = self.step - settings.warmup_steps
        if decay_position >= settings.decay_steps:
            return settings.learning_rate * settings.minimum_learning_rate_ratio
        ratio = max(0, decay_position) / settings.decay_steps
        cosine = 0.5 * (1 + math.cos(math.pi * ratio))
        minimum = settings.learning_rate * settings.minimum_learning_rate_ratio
        return minimum + cosine * (settings.learning_rate - minimum)

    def set_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
    ) -> None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    def training_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = []
        if self.acoustic_models is not None:
            modules.append(self.acoustic_models)
        if self.conditional_models is not None:
            modules.append(self.conditional_models)
        return tuple(modules)

    def generator(self, label: str) -> torch.Generator:
        seed = derive_seed(
            self.config.runtime.seed,
            self.step,
            self.batch_index,
            label,
        )
        return torch.Generator(device=self.accelerator.device).manual_seed(seed)
