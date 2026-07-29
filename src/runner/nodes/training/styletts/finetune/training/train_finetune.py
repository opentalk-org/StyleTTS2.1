import logging
import shutil
import time
from pathlib import Path

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
from .utils import get_data_path_list

logger = logging.getLogger(__name__)


def train(config_path: str, *, run: TrackerRun | None) -> None:
    config = load_training_config(config_path)
    configure_profiling(config.profiling_enabled)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, log_dir / Path(config_path).name)
    train_list, validation_list = get_data_path_list(
        config.data_params.train_data,
        config.data_params.val_data,
    )
    train_batches = build_dataloader(
        train_list,
        config.data_params.root_path,
        OOD_data=config.data_params.OOD_data,
        min_length=config.data_params.min_length,
        batch_size=config.batch_size,
        num_workers=0,
        dataset_config={"symbols": config.symbols},
        device=config.device,
        stream_cache=None,
    )
    validation_batches = build_dataloader(
        validation_list,
        config.data_params.root_path,
        OOD_data=config.data_params.OOD_data,
        min_length=config.data_params.min_length,
        batch_size=config.batch_size,
        validation=True,
        num_workers=0,
        device=config.device,
        dataset_config={"symbols": config.symbols},
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
    train_batches = train_batches.prepare(accelerator)
    validation_batches = validation_batches.prepare(accelerator)
    runtime = build_training_runtime(config, accelerator)
    trainer = Trainer(config, runtime)
    validator = Validator(config, runtime)
    checkpoints = CheckpointPublisher(config, runtime)
    training_started = time.monotonic()
    logged_mel_loss = 0.0
    logged_steps = 0
    validation_loss = None

    while trainer.step < config.total_steps:
        trainer.set_training_mode()
        for batch in train_batches:
            check_cancel()
            if trainer.step == config.total_steps:
                break
            with profiling_fn("train_step"):
                step_metrics = trainer.train_step(batch)
            trainer.step += 1
            step = trainer.step
            metrics = dict(step_metrics)
            metrics["elapsed_seconds"] = time.monotonic() - training_started
            if accelerator.is_main_process:
                assert telemetry is not None
                telemetry.log_train(step, metrics)
            if not metrics["step_skipped"]:
                logged_mel_loss += _scalar(metrics["mel_loss"])
                logged_steps += 1
            if (
                accelerator.is_main_process
                and step % config.log_every_steps == 0
            ):
                logger.info(
                    "step=%s/%s mel_loss=%.5f elapsed_seconds=%.3f",
                    step,
                    config.total_steps,
                    logged_mel_loss / max(1, logged_steps),
                    time.monotonic() - training_started,
                )
                logged_mel_loss = 0.0
                logged_steps = 0

            validate = (
                step % config.validation_every_steps == 0
                or step == config.total_steps
            )
            if validate:
                with profiling_fn("validation"):
                    result = validator.run(validation_batches, step)
                validation_loss = result.metrics["mel_loss"]
                if accelerator.is_main_process:
                    assert telemetry is not None
                    telemetry.log_validation(
                        step,
                        result.metrics,
                        result.samples,
                        config.log_dir,
                    )
                trainer.set_training_mode()

            checkpoint = (
                step % config.checkpoint_every_steps == 0
                or step == config.total_steps
            )
            if checkpoint and accelerator.is_main_process:
                checkpoints.publish(
                    step,
                    validation_loss,
                    trainer.running_std,
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
