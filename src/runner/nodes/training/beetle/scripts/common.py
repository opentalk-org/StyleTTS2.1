import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..training.callbacks import ArtifactEvent, ProgressEvent, StandaloneCallbacks
from ..training.execution import run_training

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Arguments:
    config: Path
    output: Path
    resume: Path | None


class CliCallbacks(StandaloneCallbacks):
    def __init__(self) -> None:
        super().__init__(_log_progress, _log_artifact)

    def report_index_progress(self, scanned: int, total: int) -> None:
        logger.info("indexed %d/%d database segments", scanned, total)


def parse_arguments(argv: Sequence[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(prog="beetle-train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    values = parser.parse_args(argv)
    return Arguments(values.config, values.output, values.resume)


def run_cli(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    callbacks = CliCallbacks()
    callbacks.install_signal_handlers()
    final = run_training(
        arguments.config,
        arguments.output,
        arguments.resume,
        callbacks,
    )
    logger.info(
        "training stopped at optimizer_step=%d microstep=%d phase=%s",
        final.optimizer_step,
        final.microstep,
        final.phase,
    )


def _log_progress(event: ProgressEvent) -> None:
    metrics = " ".join(f"{metric.name}={metric.value:.6g}" for metric in event.metrics)
    logger.info(
        "training step=%d microstep=%d phase=%s%s",
        event.optimizer_step,
        event.microstep,
        event.phase.value,
        f" {metrics}" if metrics else "",
    )


def _log_artifact(event: ArtifactEvent) -> None:
    logger.info("artifact %s (%s)", event.path, event.media_type)
