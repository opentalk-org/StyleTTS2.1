from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
import os
from pathlib import Path
from uuid import UUID

import torch

from runner.nodes.training.common.mlflow_run import start_mlflow_run

from runner.nodes.training.styletts3.testing.discriminator_backend import (
    DiscriminatorBackend,
    build_discriminator,
)
from runner.nodes.training.styletts3.testing.istftnet2_mb import ISTFTNet2MB
from runner.nodes.training.styletts3.testing.vocoder_training.audio_data import (
    build_train_loader,
    prepare_backend_audio,
)
from runner.nodes.training.styletts3.testing.vocoder_training.geometry import (
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
)
from runner.nodes.training.styletts3.testing.vocoder_training.reporting import EpochReporter
from runner.nodes.training.styletts3.testing.vocoder_training.trainer import (
    TrainingConfig,
    train_vocoder,
)

EXPERIMENT = "istftnet2_mb"
DEFAULT_EPOCHS = 5
DEFAULT_VALIDATION_INTERVAL_EPOCHS = 1
DEFAULT_VALIDATION_SAMPLES = 16
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 8
LEARNING_RATE = 8e-4
ADAM_BETAS = (0.5, 0.9)
DATABASE_URL_ENV = "RUNFLOW_PGBOUNCER_DATABASE_URL"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train plain iSTFTNet2-MB on backend WAVs with selectable GAN discriminators."
    )
    parser.add_argument("--dataset-id", required=True, type=UUID)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--epochs", type=_positive_int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--validation-interval-epochs",
        type=_positive_int,
        default=DEFAULT_VALIDATION_INTERVAL_EPOCHS,
    )
    parser.add_argument("--validation-samples", type=_positive_int, default=DEFAULT_VALIDATION_SAMPLES)
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=_nonnegative_int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--discriminator",
        choices=[backend.value for backend in DiscriminatorBackend],
        default=DiscriminatorBackend.WAVE_UNET.value,
    )
    parser.add_argument("--max-train-items", type=_positive_int)
    parser.add_argument("--max-steps-per-epoch", type=_positive_int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("iSTFTNet2-MB training requires CUDA")
    _configure_database(arguments.database_url)

    output_dir: Path = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = prepare_backend_audio(
        arguments.dataset_id,
        output_dir / "audio_cache",
        arguments.validation_samples,
        arguments.max_train_items,
    )
    loader = build_train_loader(splits.train, arguments.batch_size, arguments.workers)
    _configure_cuda()

    run = start_mlflow_run(
        experiment=EXPERIMENT,
        name=output_dir.name,
        config={
            "dataset_id": str(arguments.dataset_id),
            "epochs": arguments.epochs,
            "validation_interval_epochs": arguments.validation_interval_epochs,
            "validation_samples": arguments.validation_samples,
            "batch_size": arguments.batch_size,
            "workers": arguments.workers,
            "max_train_items": arguments.max_train_items,
            "max_steps_per_epoch": arguments.max_steps_per_epoch,
            "sample_rate": SAMPLE_RATE,
            "segment_samples": SEGMENT_SAMPLES,
            "learning_rate": LEARNING_RATE,
            "adam_betas": ADAM_BETAS,
        },
    )
    reporter = EpochReporter(output_dir, run, SAMPLE_RATE)
    try:
        train_vocoder(
            ISTFTNet2MB(),
            build_discriminator(arguments.discriminator),
            loader,
            splits.validation,
            TrainingConfig(
                epochs=arguments.epochs,
                learning_rate=LEARNING_RATE,
                betas=ADAM_BETAS,
                max_steps_per_epoch=arguments.max_steps_per_epoch,
                validation_interval_epochs=arguments.validation_interval_epochs,
            ),
            torch.device("cuda"),
            reporter,
        )
    finally:
        reporter.close()
    return 0


def _configure_cuda() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def _configure_database(database_url: str | None) -> None:
    if database_url is not None:
        os.environ[DATABASE_URL_ENV] = database_url
    if DATABASE_URL_ENV not in os.environ:
        raise RuntimeError(
            f"set {DATABASE_URL_ENV} or pass --database-url to connect to backend audio"
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
