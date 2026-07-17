import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..training.callbacks import ArtifactEvent, ProgressEvent, StandaloneCallbacks
from ..training.execution import run_stage
from ..training.state import StageKind

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageArguments:
    config: Path
    output: Path
    resume: Path | None
    stage1_checkpoint: Path | None
    stage2_checkpoint: Path | None


class CliCallbacks(StandaloneCallbacks):
    def __init__(self) -> None:
        super().__init__(_log_progress, _log_artifact)

    def report_index_progress(self, scanned: int, total: int) -> None:
        logger.info("indexed %d/%d database segments", scanned, total)


def parse_stage_arguments(
    stage: StageKind,
    argv: Sequence[str] | None = None,
) -> StageArguments:
    parser = argparse.ArgumentParser(prog=f"beetle-{stage.value}")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    if stage in (StageKind.STAGE2, StageKind.STAGE3):
        parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    if stage is StageKind.STAGE3:
        parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    values = parser.parse_args(argv)
    return StageArguments(
        config=values.config,
        output=values.output,
        resume=values.resume,
        stage1_checkpoint=(
            values.stage1_checkpoint
            if stage in (StageKind.STAGE2, StageKind.STAGE3)
            else None
        ),
        stage2_checkpoint=(
            values.stage2_checkpoint if stage is StageKind.STAGE3 else None
        ),
    )


def run_cli(stage: StageKind, argv: Sequence[str] | None = None) -> None:
    arguments = parse_stage_arguments(stage, argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    callbacks = CliCallbacks()
    callbacks.install_signal_handlers()
    final = run_stage(
        stage,
        arguments.config,
        arguments.output,
        arguments.resume,
        callbacks,
        arguments.stage1_checkpoint,
        arguments.stage2_checkpoint,
    )
    logger.info(
        "%s stopped at optimizer_step=%d microstep=%d phase=%s",
        stage.value,
        final.optimizer_step,
        final.microstep,
        final.phase,
    )


def _log_progress(event: ProgressEvent) -> None:
    metrics = " ".join(
        f"{metric.name}={metric.value:.6g}" for metric in event.metrics
    )
    logger.info(
        "%s step=%d microstep=%d phase=%s%s",
        event.stage.value,
        event.optimizer_step,
        event.microstep,
        event.phase.value,
        f" {metrics}" if metrics else "",
    )


def _log_artifact(event: ArtifactEvent) -> None:
    logger.info("artifact %s (%s)", event.path, event.media_type)
