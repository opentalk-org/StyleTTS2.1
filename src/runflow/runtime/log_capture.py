"""Route stdout/stderr writes into a logger so library progress bars (tqdm) and prints land in
per-node logs instead of only the process console.

A single tee is installed process-wide over sys.stdout/sys.stderr on first use. The routing target
is held in a context variable, so it is inherited by every thread spawned through
``asyncio.to_thread`` (which copies the current context). While a context has a target set, writes
made from that context — or any thread it spawns — become log records; every other context passes
straight through to the original stream. The tee is only consulted by code that reads
``sys.stdout``/``sys.stderr`` dynamically (tqdm captures ``file=sys.stderr`` when the bar is
created), so logging handlers bound to the original streams at startup are unaffected — no
recursion, console output preserved.

Plain ``threading.Thread`` does not copy the parent context the way ``asyncio.to_thread`` does, so
library-spawned threads (prefetchers, thread pools) would escape capture. Installing the tee also
patches ``threading.Thread`` to snapshot the starting thread's context and run the new thread inside
it, extending capture to those threads. Note the snapshot is taken at ``start()``, so a pooled
thread keeps the target of whichever node first started it; capture does not reach child processes
(e.g. DataLoader workers with ``num_workers>0``), which share none of this process's state.

Line-assembly buffers are per-thread: a single ``print`` completes on one thread, so partial-line
state never needs to cross threads even though the target does.
"""
from __future__ import annotations

import contextvars
import logging
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar


_PROGRESS_MIN_INTERVAL = 0.5

_target: ContextVar[tuple[logging.Logger, int] | None] = ContextVar("runflow_output_target", default=None)


def current_output_logger() -> logging.Logger | None:
    """The logger stdout/stderr is routed to in the current context, or None outside capture."""
    target = _target.get()
    return target[0] if target is not None else None


def _next_break(text: str) -> tuple[int, str] | None:
    candidates = [(pos, char) for char in ("\n", "\r") if (pos := text.find(char)) != -1]
    return min(candidates) if candidates else None


class _RoutedStream:
    def __init__(self, original) -> None:
        self._original = original
        self._local = threading.local()

    def _buffer(self) -> dict:
        state = getattr(self._local, "state", None)
        if state is None:
            state = {"buffer": "", "emitting": False, "last": 0.0, "pending": None}
            self._local.state = state
        return state

    def write(self, text: str):
        target = _target.get()
        state = self._buffer()
        if target is None or state["emitting"] or not text:
            return self._original.write(text)
        logger, level = target
        state["buffer"] += text
        while (found := _next_break(state["buffer"])) is not None:
            index, terminator = found
            segment = state["buffer"][:index]
            state["buffer"] = state["buffer"][index + 1:]
            self._emit(state, logger, level, segment, throttle=terminator == "\r")
        return len(text)

    def _emit(self, state: dict, logger: logging.Logger, level: int, segment: str, throttle: bool) -> None:
        line = segment.strip()
        if not line:
            return
        if throttle:
            now = time.monotonic()
            if now - state["last"] < _PROGRESS_MIN_INTERVAL:
                state["pending"] = line
                return
        self._log(state, logger, level, line)

    def _log(self, state: dict, logger: logging.Logger, level: int, line: str) -> None:
        state["emitting"] = True
        try:
            logger.log(level, "%s", line)
        finally:
            state["emitting"] = False
        state["last"] = time.monotonic()
        state["pending"] = None

    def flush_partial(self) -> None:
        target = _target.get()
        state = self._buffer()
        if target is None:
            return
        logger, level = target
        remainder = state["buffer"].strip()
        state["buffer"] = ""
        if remainder:
            self._log(state, logger, level, remainder)
        elif state["pending"] is not None:
            self._log(state, logger, level, state["pending"])
        state["pending"] = None

    def flush(self) -> None:
        self._original.flush()

    def isatty(self) -> bool:
        if _target.get() is not None:
            return False
        return self._original.isatty()

    def __getattr__(self, name):
        return getattr(self._original, name)


_install_lock = threading.Lock()
_stdout: _RoutedStream | None = None
_stderr: _RoutedStream | None = None
_threads_patched = False


def _patch_thread_context_inheritance() -> None:
    global _threads_patched
    if _threads_patched:
        return
    original_start = threading.Thread.start
    original_bootstrap = threading.Thread._bootstrap_inner

    def start(self) -> None:
        self._runflow_context = contextvars.copy_context()
        original_start(self)

    def bootstrap_inner(self) -> None:
        context = getattr(self, "_runflow_context", None)
        if context is None:
            original_bootstrap(self)
        else:
            context.run(original_bootstrap, self)

    # _bootstrap_inner is the single entry every thread runs, so this covers both target= threads
    # and Thread subclasses that override run().
    threading.Thread.start = start
    threading.Thread._bootstrap_inner = bootstrap_inner
    _threads_patched = True


def _install() -> tuple[_RoutedStream, _RoutedStream]:
    global _stdout, _stderr
    with _install_lock:
        _patch_thread_context_inheritance()
        if _stdout is None:
            _stdout = _RoutedStream(sys.stdout)
            sys.stdout = _stdout
        if _stderr is None:
            _stderr = _RoutedStream(sys.stderr)
            sys.stderr = _stderr
        return _stdout, _stderr


@contextmanager
def route_output_to_logger(logger: logging.Logger | None, level: int = logging.INFO):
    """Capture stdout/stderr writes made in the current context into ``logger``.

    No-op when ``logger`` is None. The target propagates into threads spawned through
    ``asyncio.to_thread`` for the duration of the block; other contexts keep writing to the real
    streams. The scheduler wraps every node's execution in this, so nodes never wire it themselves.
    """
    if logger is None:
        yield
        return
    stdout, stderr = _install()
    token = _target.set((logger, level))
    try:
        yield
    finally:
        stdout.flush_partial()
        stderr.flush_partial()
        _target.reset(token)
