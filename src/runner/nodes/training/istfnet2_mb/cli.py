from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from .config import TRAINING


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the four-band PQMF vocoder."
    )
    parser.add_argument("--dataset-id", required=True, type=UUID)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=positive_int, default=TRAINING.epochs)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=TRAINING.batch_size,
    )
    parser.add_argument("--workers", type=nonnegative_int, default=TRAINING.workers)
    parser.add_argument("--validation-samples", type=positive_int, default=16)
    parser.add_argument("--validation-interval", type=positive_int, default=1_000)
    parser.add_argument("--checkpoint-interval", type=positive_int, default=5_000)
    parser.add_argument("--max-train-items", type=positive_int)
    parser.add_argument("--max-steps", type=positive_int)
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed
