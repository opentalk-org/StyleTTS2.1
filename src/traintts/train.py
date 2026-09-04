import logging
import random
import shutil
import time
from pathlib import Path

import givemedata_client as gmd
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from .tracking import TrackerRun

from .checkpoints import CheckpointPublisher
from .config import load_training_config
from .profiling import configure_profiling, profiling_fn
from .parameter_monitor import write_model_graph
from .reporting import TrainingReporter, start_run
from .runtime.trainer import Trainer
from .runtime.validation import Validator
from .setup import build_accelerator, build_training_runtime
from .stages import TrainableModule, stage_for_step
from .telemetry_metrics import TrainingTelemetry

logger = logging.getLogger(__name__)


def train(
    config_path: str,
    *,
    run: TrackerRun | None,
    data_client: gmd.GiveMeDataClient,
) -> None:
    config = load_training_config(config_path)
    logger.info(
        "config loaded dataset=%s total_steps=%s stages=%s device=%s precision=%s seed=%s",
        config.data_params.dataset_id,
        config.total_steps,
        len(config.training_stages),
        config.device,
        config.precision,
        config.seed,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    configure_profiling(config.profiling_enabled)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, log_dir / Path(config_path).name)
    accelerator = build_accelerator(config)
    owns_run = run is None and accelerator.is_main_process
    if owns_run:
        run = start_run(config, data_client)
    telemetry = (
        TrainingReporter(run, config.total_steps)
        if accelerator.is_main_process and run is not None
        else None
    )
    logger.info(
        "building models and loading weights pretrained=%s",
        config.pretrained_model or "from scratch",
    )
    runtime = build_training_runtime(config, accelerator)
    if accelerator.is_main_process and run is not None:
        graph_path = log_dir / "model_graph.json"
        write_model_graph(runtime.models.modules, graph_path)
        run.log_artifact(graph_path, "monitor", step=runtime.initial_step)
    logger.info(
        "runtime ready device=%s processes=%s initial_step=%s",
        accelerator.device,
        accelerator.num_processes,
        runtime.initial_step,
    )
    # one session serves the whole run: the service drives the batch schedule
    # (sequence of sequences), the loop just asks for data
    trainer = Trainer(config, runtime)
    validator = Validator(config, runtime)
    checkpoints = CheckpointPublisher(config, runtime, data_client=data_client)
    timing = TrainingTelemetry.start(config.total_steps, trainer.step)
    logged_total_loss = 0.0
    logged_steps = 0
    validation_loss = None
    active_stage = None

    modality_id = config.PLBERT_config.get("modality_id", 0)
    train_batches = gmd.dataloader(
        data_client,
        device=config.device,
        modality_id=modality_id,
    )
    validation_batches = gmd.dataloader(
        data_client,
        validation=True,
        device=config.device,
        samples_per_epoch=config.data_params.validation_samples,
        modality_id=modality_id,
    )
    logger.info(
        "givemedata session ready session=%s dataset=%s",
        data_client.run_id,
        config.data_params.dataset_id,
    )
    batch_iterator = iter(train_batches)

    while trainer.step < config.total_steps:
        stage = stage_for_step(config.training_stages, trainer.step)
        if stage is not active_stage:
            logger.info(
                "stage starting name=%r step=%s/%s max_audio_seconds=%s trainable=%s",
                stage.name,
                trainer.step,
                config.total_steps,
                stage.max_audio_seconds,
                [module.value for module in stage.trainable_modules],
            )
            active_stage = stage
        trainer.set_training_mode()
        while (
            trainer.step < config.total_steps
            and stage_for_step(config.training_stages, trainer.step)
            is active_stage
        ):
            data_wait_started = time.monotonic()
            try:
                batch = next(batch_iterator)
            except StopIteration:
                break
            timing.data_wait_seconds += time.monotonic() - data_wait_started
            compute_started = time.monotonic()
            if (
                config.profiling_enabled
                and accelerator.is_main_process
                and trainer.step == 20
            ):
                configure_profiling(False)
                try:
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    ) as torch_profile:
                        step_metrics = trainer.train_step(batch)
                finally:
                    configure_profiling(True)
                trace_path = log_dir / "training_step_000000021_trace.json"
                torch_profile.export_chrome_trace(str(trace_path))
                if run is not None:
                    run.log_artifact(trace_path, "profiling", step=trainer.step + 1)
            else:
                with profiling_fn("train_step"):
                    step_metrics = trainer.train_step(batch)
            timing.compute_seconds += time.monotonic() - compute_started
            trainer.step += 1
            step = trainer.step
            if TrainableModule.DECODER in stage.trainable_modules:
                crop_frames = min(
                    int(batch.mel_lengths.min().item() / 2 - 1),
                    int(
                        stage.max_decoder_seconds
                        * config.preprocess_params.sr
                        / config.preprocess_params.spect_params.hop_length
                        / 2
                    ),
                )
                audio_seconds = (
                    len(batch.audio_durations)
                    * crop_frames
                    * config.preprocess_params.spect_params.hop_length
                    * 2
                    / config.preprocess_params.sr
                )
            else:
                audio_seconds = sum(batch.audio_durations)
            batch_totals = torch.tensor(
                (len(batch.audio_durations), audio_seconds),
                device=accelerator.device,
                dtype=torch.float64,
            )
            reduced_totals = accelerator.reduce(
                batch_totals,
                reduction="sum",
            )
            timing.items_processed += reduced_totals[0].item()
            timing.audio_seconds_trained += reduced_totals[1].item()
            metrics = dict(step_metrics)
            metrics.update(timing.metrics(step))
            if accelerator.is_main_process:
                reporting_started = time.monotonic()
                telemetry.log_train(step, metrics)
                timing.reporting_seconds += (
                    time.monotonic() - reporting_started
                )
            if not metrics["step_skipped"]:
                logged_total_loss += _scalar(metrics["total"])
                logged_steps += 1
            if (
                accelerator.is_main_process
                and step % config.log_every_steps == 0
            ):
                logger.info(
                    "step=%s/%s total_loss=%.5f elapsed_seconds=%.3f",
                    step,
                    config.total_steps,
                    logged_total_loss / max(1, logged_steps),
                    time.monotonic() - timing.started_at,
                )
                logged_total_loss = 0.0
                logged_steps = 0
            validate = (
                step % config.validation_every_steps == 0
                or step == config.total_steps
            )
            if validate:
                validation_started = time.monotonic()
                if accelerator.is_main_process:
                    logger.info("validation starting step=%s", step)
                    with profiling_fn("validation"):
                        result = validator.run(validation_batches, step)
                    validation_loss = result.metrics["mel_loss"]
                    telemetry.log_validation(
                        step,
                        result.metrics,
                        result.samples,
                        config.log_dir,
                    )
                    logger.info(
                        "validation done step=%s mel_loss=%.5f samples=%s seconds=%.1f",
                        step,
                        _scalar(validation_loss),
                        len(result.samples),
                        time.monotonic() - validation_started,
                    )
                    if trainer.step < config.total_steps:
                        trainer.set_training_mode()
                timing.validation_seconds += (
                    time.monotonic() - validation_started
                )

            checkpoint = (
                step % config.checkpoint_every_steps == 0
                or step == config.total_steps
            )
            if checkpoint and accelerator.is_main_process:
                checkpoint_started = time.monotonic()
                checkpoints.publish(
                    step,
                    validation_loss,
                )
                timing.checkpoint_seconds += (
                    time.monotonic() - checkpoint_started
                )
            if validate or checkpoint:
                accelerator.wait_for_everyone()
    logger.info(
        "training finished step=%s/%s elapsed_seconds=%.1f",
        trainer.step,
        config.total_steps,
        time.monotonic() - timing.started_at,
    )
    if owns_run:
        run.close()


def _scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().item())
    return float(value)
