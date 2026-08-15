import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from runflow.runtime.cancellation import check_cancel
from runner.nodes.training.common.mlflow_run import TrackerRun

from .checkpoints import CheckpointPublisher
from .config import load_training_config
from .data import build_dataloader
from .mlflow_logging import MlflowLogger, start_run
from .profiling import configure_profiling, profiling_fn
from .runtime import Trainer, Validator
from .setup import build_accelerator, build_training_runtime
from ..stages import TrainableModule, stage_for_step
from .telemetry_metrics import TrainingTelemetry

logger = logging.getLogger(__name__)


def train(config_path: str, *, run: TrackerRun | None) -> None:
    config = load_training_config(config_path)
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
        run = start_run(config)
    telemetry = (
        MlflowLogger(run, config.total_steps)
        if accelerator.is_main_process and run is not None
        else None
    )
    runtime = build_training_runtime(config, accelerator)
    trainer = Trainer(config, runtime)
    validator = Validator(config, runtime)
    checkpoints = CheckpointPublisher(config, runtime)
    timing = TrainingTelemetry.start(config.total_steps, trainer.step)
    logged_total_loss = 0.0
    logged_steps = 0
    validation_loss = None
    active_stage = None
    train_batches = None
    validation_batches = None

    while trainer.step < config.total_steps:
        stage = stage_for_step(config.training_stages, trainer.step)
        if stage is not active_stage:
            dataset_config = {
                "symbols": config.symbols,
                "max_text_tokens": config.PLBERT_config["model_params"][
                    "max_position_embeddings"
                ],
                "plbert_languages": config.PLBERT_config.get("languages"),
                "plbert_modality_id": config.PLBERT_config.get("modality_id", 0),
            }
            train_batches = build_dataloader(
                config.data_params.dataset_id,
                config.data_params.validation_samples,
                max_seconds=stage.max_audio_seconds,
                num_workers=0,
                dataset_config=dataset_config,
                device=config.device,
                seed=config.seed,
            ).prepare(accelerator)
            validation_batches = build_dataloader(
                config.data_params.dataset_id,
                config.data_params.validation_samples,
                max_seconds=stage.max_audio_seconds,
                validation=True,
                num_workers=0,
                device=config.device,
                seed=config.seed,
                dataset_config=dataset_config,
            )
            active_stage = stage
        assert train_batches is not None
        assert validation_batches is not None
        trainer.set_training_mode()
        batch_iterator = iter(train_batches)
        while (
            trainer.step < config.total_steps
            and stage_for_step(config.training_stages, trainer.step)
            is active_stage
        ):
            data_wait_started = time.monotonic()
            try:
                loaded_batch = next(batch_iterator)
            except StopIteration:
                break
            batch = loaded_batch.batch
            loader_telemetry = loaded_batch.telemetry
            timing.data_wait_seconds += time.monotonic() - data_wait_started
            check_cancel()
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
                    run.log_artifact(trace_path, "profiling")
            else:
                with profiling_fn("train_step"):
                    step_metrics = trainer.train_step(batch)
            timing.compute_seconds += time.monotonic() - compute_started
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
            if not trainer.update_completed:
                continue
            trainer.step += 1
            step = trainer.step
            metrics = dict(step_metrics)
            metrics.update(timing.metrics(step))
            fetch_mib = loader_telemetry.bucket_fetch_bytes / (1024 * 1024)
            metrics.update({
                "performance/bucket_fetch_seconds": (
                    loader_telemetry.bucket_fetch_seconds
                ),
                "performance/bucket_fetch_mib": fetch_mib,
                "performance/bucket_fetch_mib_per_second": (
                    fetch_mib / max(loader_telemetry.bucket_fetch_seconds, 1e-9)
                ),
                "performance/bucket_request_seconds": (
                    loader_telemetry.bucket_request_seconds
                ),
                "performance/bucket_request_count": loader_telemetry.bucket_request_count,
                "performance/bucket_error_count": loader_telemetry.bucket_error_count,
                "performance/failed_queries": loader_telemetry.failed_queries,
            })
            metrics.update(
                {
                    f"performance/failed_queries/{code}": count
                    for code, count in loader_telemetry.failed_query_codes.items()
                }
            )
            if accelerator.is_main_process:
                assert telemetry is not None
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
                    assert telemetry is not None
                    with profiling_fn("validation"):
                        result = validator.run(validation_batches, step)
                    validation_loss = result.metrics["mel_loss"]
                    telemetry.log_validation(
                        step,
                        result.metrics,
                        result.samples,
                        config.log_dir,
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
    if owns_run:
        assert run is not None
        run.close()


def _scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().item())
    return float(value)
