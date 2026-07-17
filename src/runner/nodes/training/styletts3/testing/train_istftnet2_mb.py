from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
import os
from pathlib import Path
from uuid import UUID

import torch
import torch.nn as nn

from runner.nodes.training.common.mlflow_run import start_mlflow_run

from runner.nodes.training.styletts3.testing.paper_istftnet2_mb import PaperISTFTNet2MB
from runner.nodes.training.styletts3.testing.discriminator_backend import (
    DiscriminatorBackend,
    build_discriminator,
)
from runner.nodes.training.styletts3.testing.istftnet2_mb import ISTFTNet2MB
from runner.nodes.training.styletts3.testing.vocoder_training.audio_data import (
    build_train_loader,
    prepare_backend_audio,
)
from runner.nodes.training.styletts3.testing.vocoder_training.profiles import (
    VocoderProfile,
    profile_geometry,
)
from runner.nodes.training.styletts3.testing.vocoder_training.reporting import EpochReporter
from runner.nodes.training.styletts3.testing.vocoder_training.trainer import train_vocoder
from runner.nodes.training.styletts3.testing.vocoder_training.training_config import TrainingConfig

EXPERIMENT = "istftnet2_mb"
DEFAULT_EPOCHS = 5
DEFAULT_VALIDATION_INTERVAL_EPOCHS = 1
DEFAULT_VALIDATION_SAMPLES = 16
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 8
GENERATOR_LEARNING_RATE = 2e-4
WAVE_UNET_DISCRIMINATOR_LEARNING_RATE = 2e-4
STYLETTS_DISCRIMINATOR_LEARNING_RATE = 2e-4
ADAM_BETAS = (0.5, 0.9)
DATABASE_URL_ENV = "RUNFLOW_PGBOUNCER_DATABASE_URL"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train native or paper-reference iSTFTNet2-MB on backend WAVs."
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
        "--profile",
        choices=[profile.value for profile in VocoderProfile],
        default=VocoderProfile.NATIVE_300.value,
    )
    parser.add_argument(
        "--discriminator",
        choices=[backend.value for backend in DiscriminatorBackend],
    )
    parser.add_argument("--max-train-items", type=_positive_int)
    parser.add_argument("--max-steps-per-epoch", type=_positive_int)
    parser.add_argument("--max-steps", type=_positive_int)
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
    selected_profile = VocoderProfile(arguments.profile)
    signal = profile_geometry(selected_profile)
    discriminator_name = resolve_discriminator(selected_profile, arguments.discriminator)
    max_steps = arguments.max_steps if arguments.max_steps is not None else signal.target_steps

    output_dir: Path = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = prepare_backend_audio(
        arguments.dataset_id,
        output_dir / "audio_cache",
        arguments.validation_samples,
        arguments.max_train_items,
        signal,
    )
    loader = build_train_loader(
        splits.train,
        arguments.batch_size,
        arguments.workers,
        signal,
    )
    _configure_cuda()
    discriminator_learning_rate = (
        WAVE_UNET_DISCRIMINATOR_LEARNING_RATE
        if discriminator_name == DiscriminatorBackend.WAVE_UNET.value
        else STYLETTS_DISCRIMINATOR_LEARNING_RATE
    )

    run = start_mlflow_run(
        experiment=f"{EXPERIMENT}_{selected_profile.value}",
        name=output_dir.name,
        config={
            "dataset_id": str(arguments.dataset_id),
            "profile": selected_profile.value,
            "epochs": arguments.epochs,
            "validation_interval_epochs": arguments.validation_interval_epochs,
            "validation_samples": arguments.validation_samples,
            "batch_size": arguments.batch_size,
            "workers": arguments.workers,
            "discriminator": discriminator_name,
            "max_train_items": arguments.max_train_items,
            "max_steps_per_epoch": arguments.max_steps_per_epoch,
            "max_steps": max_steps,
            "sample_rate": signal.sample_rate,
            "segment_samples": signal.segment_samples,
            "synthesis_hop": signal.synthesis_hop,
            "conditioning_n_fft": signal.conditioning.n_fft,
            "conditioning_win_length": signal.conditioning.win_length,
            "conditioning_hop_length": signal.conditioning.hop_length,
            "conditioning_fmin": signal.conditioning.fmin,
            "conditioning_fmax": signal.conditioning.fmax,
            "generator_learning_rate": GENERATOR_LEARNING_RATE,
            "discriminator_learning_rate": discriminator_learning_rate,
            "adam_betas": ADAM_BETAS,
            "paper_deviation": (
                "styletts_gan_with_relative_loss"
                if selected_profile is VocoderProfile.PAPER_256
                else "none"
            ),
        },
    )
    reporter = EpochReporter(output_dir, run, signal.sample_rate)
    try:
        train_vocoder(
            build_generator(selected_profile),
            build_discriminator(discriminator_name),
            loader,
            splits.validation,
            TrainingConfig(
                epochs=arguments.epochs,
                generator_learning_rate=GENERATOR_LEARNING_RATE,
                discriminator_learning_rate=discriminator_learning_rate,
                betas=ADAM_BETAS,
                max_steps_per_epoch=arguments.max_steps_per_epoch,
                validation_interval_epochs=arguments.validation_interval_epochs,
                max_steps=max_steps,
            ),
            torch.device("cuda"),
            reporter,
            signal,
        )
    finally:
        reporter.close()
    return 0


def resolve_discriminator(profile: VocoderProfile, requested: str | None) -> str:
    if requested is None:
        return (
            DiscriminatorBackend.STYLETTS.value
            if profile is VocoderProfile.PAPER_256
            else DiscriminatorBackend.WAVE_UNET.value
        )
    if profile is VocoderProfile.PAPER_256 and requested != DiscriminatorBackend.STYLETTS.value:
        raise ValueError("paper_256 requires the styletts discriminator")
    return requested


def build_generator(profile: VocoderProfile) -> nn.Module:
    if profile is VocoderProfile.PAPER_256:
        return PaperISTFTNet2MB()
    assert profile is VocoderProfile.NATIVE_300, f"unsupported vocoder profile: {profile}"
    return ISTFTNet2MB()


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
