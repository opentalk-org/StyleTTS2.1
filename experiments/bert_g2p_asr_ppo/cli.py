from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from uuid import UUID

from .config import DataConfig, ExperimentConfig, PpoConfig, SftConfig
from .data import download_parquets
from .ppo import train_ppo
from .sft import train_sft


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Two-copy BERT G2P with frozen-ASR PPO")
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--train-files", type=int, default=DataConfig().train_files)
    download.add_argument("--validation-files", type=int, default=DataConfig().validation_files)
    sft = commands.add_parser("sft")
    sft.add_argument("--steps", type=int, default=SftConfig().steps)
    sft.add_argument("--batch-size", type=int, default=SftConfig().batch_size)
    sft.add_argument("--validation-interval", type=int, default=SftConfig().validation_interval)
    sft.add_argument("--validation-batches", type=int, default=SftConfig().validation_batches)
    pipeline = commands.add_parser("pipeline")
    pipeline.add_argument("--sft-steps", type=int, default=SftConfig().steps)
    pipeline.add_argument("--ppo-steps", type=int, default=500)
    pipeline.add_argument("--batch-size", type=int, default=4)
    pipeline.add_argument("--dataset-id", type=UUID, default=PpoConfig().dataset_id)
    ppo = commands.add_parser("ppo")
    ppo.add_argument("checkpoint", type=Path)
    ppo.add_argument("mlflow_run_id")
    ppo.add_argument("--steps", type=int, default=500)
    ppo.add_argument("--batch-size", type=int, default=4)
    ppo.add_argument("--dataset-id", type=UUID, default=PpoConfig().dataset_id)
    ppo.add_argument("--validation-interval", type=int, default=25)
    ppo.add_argument("--validation-batches", type=int, default=4)
    return root


def main() -> None:
    args = parser().parse_args()
    config = ExperimentConfig()
    if args.command == "download":
        data = replace(config.data, train_files=args.train_files, validation_files=args.validation_files)
        print(*download_parquets(data), sep="\n")
        print(*download_parquets(data, validation=True), sep="\n")
        return
    if args.command == "sft":
        sft = replace(
            config.sft,
            steps=args.steps,
            batch_size=args.batch_size,
            validation_interval=args.validation_interval,
            validation_batches=args.validation_batches,
        )
        config = replace(config, sft=sft)
        checkpoint, run_id = train_sft(config)
        print(checkpoint, run_id)
        return
    if args.command == "ppo":
        ppo = replace(
            config.ppo,
            steps=args.steps,
            batch_size=args.batch_size,
            dataset_id=args.dataset_id,
            validation_interval=args.validation_interval,
            validation_batches=args.validation_batches,
        )
        print(train_ppo(replace(config, ppo=ppo), args.checkpoint, args.mlflow_run_id))
        return
    sft = replace(config.sft, steps=args.sft_steps, batch_size=args.batch_size)
    ppo = replace(config.ppo, steps=args.ppo_steps, batch_size=args.batch_size, dataset_id=args.dataset_id)
    config = replace(config, sft=sft, ppo=ppo)
    checkpoint, run_id = train_sft(config)
    print(train_ppo(config, checkpoint, run_id))


if __name__ == "__main__":
    main()
