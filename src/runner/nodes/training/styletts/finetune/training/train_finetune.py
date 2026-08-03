import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from runflow.runtime.cancellation import check_cancel
from runner.nodes.training.common.mlflow_run import TrackerRun

from .checkpoints import CheckpointPublisher
from .config import load_training_config
from .data import build_dataloader
from .mlflow_logging import MlflowLogger, start_run
from .profiling import configure_profiling, profiling_fn
from .runtime import Trainer, Validator
from .setup import build_accelerator, build_training_runtime
from ..stages import stage_for_step
from .telemetry_metrics import TrainingTelemetry
from .utils import get_data_path_list

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
    train_list, validation_list = get_data_path_list(
        config.data_params.train_data,
        config.data_params.val_data,
    )
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
                "max_audio_seconds": stage.max_audio_seconds,
                "max_text_tokens": config.PLBERT_config["model_params"][
                    "max_position_embeddings"
                ],
                "plbert_languages": config.PLBERT_config.get("languages"),
                "plbert_modality_id": config.PLBERT_config.get("modality_id", 0),
            }
            train_batches = build_dataloader(
                train_list,
                config.data_params.root_path,
                OOD_data=config.data_params.OOD_data,
                min_length=config.data_params.min_length,
                batch_size=stage.batch_size,
                num_workers=0,
                dataset_config=dataset_config,
                device=config.device,
                seed=config.seed,
                stream_cache=None,
            ).prepare(accelerator)
            validation_batches = build_dataloader(
                validation_list,
                config.data_params.root_path,
                OOD_data=config.data_params.OOD_data,
                min_length=config.data_params.min_length,
                batch_size=stage.batch_size,
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
                batch = next(batch_iterator)
            except StopIteration:
                break
            timing.data_wait_seconds += time.monotonic() - data_wait_started
            check_cancel()
            compute_started = time.monotonic()
            with profiling_fn("train_step"):
                step_metrics = trainer.train_step(batch)
            timing.compute_seconds += time.monotonic() - compute_started
            trainer.step += 1
            step = trainer.step
            timing.items_processed += (
                batch.texts.shape[0] * accelerator.num_processes
            )
            metrics = dict(step_metrics)
            metrics.update(timing.metrics(step))
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
                    trainer.running_std,
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
