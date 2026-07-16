from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class TrackerRun(Protocol):
    """Metric-sink surface the finetune/vocoder loggers depend on. Kept minimal so
    a real trackio run and the no-op fallback are interchangeable."""

    def track(self, value: object, name: str, step: int, epoch: int | None = None) -> None: ...

    def close(self) -> None: ...


class NoopWandbRun:
    """Used when trackio is unavailable or a run fails to start, so training never
    fails just because metric logging could not be initialized."""

    def track(self, value: object, name: str, step: int, epoch: int | None = None) -> None:
        return None

    def close(self) -> None:
        return None


class WandbRun:
    """Adapter over a trackio run that maps a single ``(name, value, step)`` point
    onto ``run.log({name: value}, step=step)``. Media values (``trackio.Audio``,
    ``trackio.Image``) log through the same path. ``epoch`` is accepted for call-site
    parity but trackio keys everything by step."""

    def __init__(self, run: Any) -> None:
        self._run = run

    def track(self, value: object, name: str, step: int, epoch: int | None = None) -> None:
        del epoch
        self._run.log({name: value}, step=step)

    def close(self) -> None:
        self._run.finish()


def start_wandb_run(*, project: str, name: str, config: dict[str, Any]) -> TrackerRun:
    """Start a trackio run so finetune metrics and samples land in the wandb-style
    dashboard on the Runs page. Trackio writes to the shared ``TRACKIO_DIR`` that the
    ``runflow-trackio`` server reads, so the run appears live. Returns a no-op run if
    trackio is missing or init fails, so training never breaks over logging."""
    try:
        import trackio
    except ImportError:
        logger.warning("trackio not installed; training metrics will not be logged")
        return NoopWandbRun()
    try:
        run = trackio.init(project=project, name=name, config=config)
        logger.info("trackio run started project=%s name=%s", project, name)
        return WandbRun(run)
    except Exception:
        logger.warning("failed to start trackio run; training metrics will not be logged", exc_info=True)
        return NoopWandbRun()
