from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from experiments.bert_g2p_asr_ppo.data import download_parquets

from .config import DataConfig, ExperimentConfig, TrainConfig
from .train import train
from .validate import validate_checkpoint


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="PL-BERT plus BiLSTM RNN-T G2P experiment")
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--train-files", type=int, default=DataConfig().train_files)
    download.add_argument("--validation-files", type=int, default=DataConfig().validation_files)
    training = commands.add_parser("train")
    training.add_argument("--steps", type=int, default=TrainConfig().steps)
    training.add_argument("--batch-size", type=int, default=TrainConfig().batch_size)
    training.add_argument("--validation-interval", type=int, default=TrainConfig().validation_interval)
    training.add_argument("--validation-batches", type=int, default=TrainConfig().validation_batches)
    training.add_argument("--checkpoint", type=Path)
    training.add_argument("--resume-optimizer", action="store_true")
    validation = commands.add_parser("validate")
    validation.add_argument("checkpoint", type=Path)
    validation.add_argument("--beam-width", type=int, default=10)
    validation.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    validation.add_argument("--validation-batches", type=int, default=TrainConfig().validation_batches)
    return root


def main() -> None:
    args = parser().parse_args()
    config = ExperimentConfig()
    if args.command == "download":
        data = replace(config.data, train_files=args.train_files, validation_files=args.validation_files)
        print(*download_parquets(data), sep="\n")
        print(*download_parquets(data, validation=True), sep="\n")
        return
    if args.command == "validate":
        metrics = validate_checkpoint(
            args.checkpoint,
            config,
            args.beam_width,
            args.batch_size,
            args.validation_batches,
        )
        print(metrics)
        return
    training = replace(
        config.train,
        steps=args.steps,
        batch_size=args.batch_size,
        validation_interval=args.validation_interval,
        validation_batches=args.validation_batches,
    )
    checkpoint, run_id = train(
        replace(config, train=training),
        args.checkpoint,
        args.resume_optimizer,
    )
    print(checkpoint, run_id)


if __name__ == "__main__":
    main()
