from __future__ import annotations

from dataclasses import asdict, replace
import logging
from pathlib import Path

import torch

from runner.nodes.training.common.mlflow_run import start_mlflow_run

from .cli import parse_args
from .config import RunConfig, SIGNAL, TRAINING
from .data import prepare_audio, training_loader
from .discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionSpectralDiscriminator,
)
from .model import Generator, JDCNet
from .reporting import Reporter
from .trainer import train

EXPERIMENT = "istftnet2_mb_training"
JDC_CHECKPOINT = (
    Path(__file__).parents[1] / "hiftnet" / "Utils" / "JDC" / "bst.t7"
)


def main() -> int:
    arguments = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("HiFTNet-PQMF training requires CUDA")
    torch.manual_seed(TRAINING.seed)
    torch.cuda.manual_seed(TRAINING.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = prepare_audio(
        arguments.dataset_id,
        arguments.cache_dir.resolve(),
        arguments.validation_samples,
        arguments.max_train_items,
        SIGNAL,
    )
    training_config = replace(
        TRAINING,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
    )
    loader = training_loader(
        audio.training,
        training_config.batch_size,
        training_config.workers,
        SIGNAL,
    )
    run_config = RunConfig(
        epochs=arguments.epochs,
        checkpoint_interval=arguments.checkpoint_interval,
        validation_interval=arguments.validation_interval,
        max_steps=arguments.max_steps,
    )
    f0_model = JDCNet()
    f0_model.load_state_dict(
        torch.load(
            JDC_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )["model"]
    )
    generator = Generator(f0_model)
    run = start_mlflow_run(
        experiment=EXPERIMENT,
        name=arguments.run_name or output_dir.name,
        config={
            "dataset_id": str(arguments.dataset_id),
            "architecture": "hiftnet_pqmf_4band",
            "parameters": sum(
                parameter.numel()
                for parameter in generator.parameters()
                if parameter.requires_grad
            ),
            "validation_samples": arguments.validation_samples,
            "max_train_items": arguments.max_train_items,
            **asdict(SIGNAL),
            **asdict(training_config),
            **asdict(run_config),
        },
    )
    reporter = Reporter(output_dir / "samples", run, SIGNAL)
    try:
        train(
            generator,
            MultiPeriodDiscriminator(),
            MultiResolutionSpectralDiscriminator(),
            loader,
            audio.validation,
            SIGNAL,
            training_config,
            run_config,
            output_dir,
            reporter,
            torch.device("cuda"),
        )
    finally:
        reporter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
