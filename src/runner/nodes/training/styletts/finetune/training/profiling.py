import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch


logger = logging.getLogger(__name__)
_enabled = False
_stack: ContextVar[tuple[str, ...]] = ContextVar("profiling_stack", default=())
_step: ContextVar[int | None] = ContextVar("profiling_step", default=None)


def configure_profiling(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def set_profiling_step(step: int | None) -> None:
    _step.set(step)


@contextmanager
def profiling_fn(name: str) -> Iterator[None]:
    if not _enabled:
        yield
        return

    parent = _stack.get()
    path = (*parent, name)
    token = _stack.set(path)
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    started_ns = time.time_ns()
    started_allocated = torch.cuda.memory_allocated(device)
    started_reserved = torch.cuda.memory_reserved(device)
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        ended_ns = time.time_ns()
        ended_allocated = torch.cuda.memory_allocated(device)
        ended_reserved = torch.cuda.memory_reserved(device)
        event = {
            "name": name,
            "path": list(path),
            "depth": len(parent),
            "step": _step.get(),
            "rank": int(os.environ.get("RANK", "0")),
            "device": device,
            "start_ns": started_ns,
            "end_ns": ended_ns,
            "duration_ms": (ended_ns - started_ns) / 1_000_000,
            "start_allocated_bytes": started_allocated,
            "end_allocated_bytes": ended_allocated,
            "start_reserved_bytes": started_reserved,
            "end_reserved_bytes": ended_reserved,
        }
        _log_event(event)
        _stack.reset(token)


def _log_event(event: dict[str, object]) -> None:
    logger.info("PROFILE_EVENT %s", json.dumps(event, separators=(",", ":")))
