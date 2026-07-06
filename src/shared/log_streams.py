"""Route stdout/stderr writes into a logger so library progress bars (tqdm) and prints
land in per-node logs instead of only the process console.

A single tee is installed process-wide over sys.stdout/sys.stderr on first use. Writes are
routed per calling thread: while a thread is inside ``route_output_to_logger`` its writes are
turned into log records; every other thread passes straight through to the original stream. The
tee is only consulted by code that reads ``sys.stdout``/``sys.stderr`` dynamically (tqdm captures
``file=sys.stderr`` when the bar is created), so logging handlers bound to the original streams at
startup are unaffected — no recursion, console output preserved.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import contextmanager


_PROGRESS_MIN_INTERVAL = 0.5  # seconds between throttled (\r) progress updates


def _next_break(text: str) -> tuple[int, str] | None:
    candidates = [(pos, char) for char in ("\n", "\r") if (pos := text.find(char)) != -1]
    return min(candidates) if candidates else None


class _ThreadRoutedStream:
    def __init__(self, original) -> None:
        self._original = original
        self._local = threading.local()

    def _state(self) -> dict:
        state = getattr(self._local, "state", None)
        if state is None:
            state = {"logger": None, "level": logging.INFO, "buffer": "", "emitting": False, "last": 0.0, "pending": None}
            self._local.state = state
        return state

    def set_target(self, logger: logging.Logger, level: int) -> None:
        state = self._state()
        state.update(logger=logger, level=level, buffer="", pending=None, last=0.0)

    def clear_target(self) -> None:
        state = self._state()
        remainder = state["buffer"].strip()
        if remainder:
            self._log(state, remainder)
        elif state["pending"] is not None:
            self._log(state, state["pending"])
        state.update(logger=None, buffer="", pending=None)

    def write(self, text: str):
        state = self._state()
        if state["logger"] is None or state["emitting"] or not text:
            return self._original.write(text)
        state["buffer"] += text
        while (found := _next_break(state["buffer"])) is not None:
            index, terminator = found
            segment = state["buffer"][:index]
            state["buffer"] = state["buffer"][index + 1:]
            self._emit(state, segment, throttle=terminator == "\r")
        return len(text)

    def _emit(self, state: dict, segment: str, throttle: bool) -> None:
        line = segment.strip()
        if not line:
            return
        if throttle:
            now = time.monotonic()
            if now - state["last"] < _PROGRESS_MIN_INTERVAL:
                state["pending"] = line
                return
        self._log(state, line)

    def _log(self, state: dict, line: str) -> None:
        logger = state["logger"]
        if logger is None:
            return
        state["emitting"] = True
        try:
            logger.log(state["level"], "%s", line)
        finally:
            state["emitting"] = False
        state["last"] = time.monotonic()
        state["pending"] = None

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        if self._state()["logger"] is not None:
            return False
        return bool(getattr(self._original, "isatty", lambda: False)())

    def __getattr__(self, name):
        return getattr(self._original, name)


_install_lock = threading.Lock()
_stdout: _ThreadRoutedStream | None = None
_stderr: _ThreadRoutedStream | None = None


def _install() -> tuple[_ThreadRoutedStream, _ThreadRoutedStream]:
    global _stdout, _stderr
    with _install_lock:
        if _stdout is None:
            _stdout = _ThreadRoutedStream(sys.stdout)
            sys.stdout = _stdout
        if _stderr is None:
            _stderr = _ThreadRoutedStream(sys.stderr)
            sys.stderr = _stderr
        return _stdout, _stderr


@contextmanager
def route_output_to_logger(logger: logging.Logger | None, level: int = logging.INFO):
    """Capture stdout/stderr writes made on the current thread into ``logger``.

    No-op when ``logger`` is None. Other threads keep writing to the real streams.
    """
    if logger is None:
        yield
        return
    stdout, stderr = _install()
    stdout.set_target(logger, level)
    stderr.set_target(logger, level)
    try:
        yield
    finally:
        stdout.clear_target()
        stderr.clear_target()
