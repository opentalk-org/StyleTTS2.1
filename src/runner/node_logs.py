import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path

from runflow.core.node import Node
from runflow.runtime.log_capture import current_output_logger


MAX_NODE_LOG_BYTES = 1_000_000
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
SAFE_NODE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def node_log_path(work_dir: Path, run_id: str, node_id: str) -> Path:
    safe_node_id = SAFE_NODE_ID.sub("_", node_id)
    return (work_dir / run_id / "logs" / f"{safe_node_id}.log").resolve()


class CappedNodeLogHandler(logging.Handler):
    def __init__(self, path: Path, on_dirty: Callable[[], None]) -> None:
        super().__init__()
        self.path = path.resolve()
        self.on_dirty = on_dirty
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.setFormatter(logging.Formatter(LOG_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        line = f"{self.format(record)}\n".encode("utf-8", errors="replace")
        with self.path.open("ab") as file:
            file.write(line)
        self._trim()
        self.on_dirty()

    def _trim(self) -> None:
        size = self.path.stat().st_size
        if size <= MAX_NODE_LOG_BYTES:
            return
        with self.path.open("rb") as file:
            file.seek(-MAX_NODE_LOG_BYTES, 2)
            data = file.read()
        self.path.write_bytes(data)


class ContextForwardHandler(logging.Handler):
    """Forward records from arbitrary module loggers (``logging.getLogger(__name__)``) into the log
    of whichever node is currently executing. Node execution runs inside ``route_output_to_logger``,
    so ``current_output_logger`` names that node; a helper module needs no special wiring to have its
    logs land in the right node log. Records already owned by the node logger are left to the node
    logger's own handler to avoid duplicates."""

    def emit(self, record: logging.LogRecord) -> None:
        target = current_output_logger()
        if target is None or record.name == target.name:
            return
        for handler in target.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)


_forwarding_lock = threading.Lock()
_forwarding_installed = False


def ensure_context_forwarding() -> None:
    """Install the single process-wide forwarder on the root logger. Idempotent: a second call is a
    no-op, so concurrent runs cannot forward the same record twice."""
    global _forwarding_installed
    with _forwarding_lock:
        if _forwarding_installed:
            return
        logging.getLogger().addHandler(ContextForwardHandler())
        _forwarding_installed = True


class NodeLogManager:
    def __init__(self, work_dir: Path, run_id: str, on_dirty: Callable[[str, Path], None]) -> None:
        self.work_dir = work_dir
        self.run_id = run_id
        self.on_dirty = on_dirty
        self._handlers: list[tuple[logging.Logger, CappedNodeLogHandler]] = []

    def attach(self, nodes: list[Node]) -> None:
        ensure_context_forwarding()
        for node in nodes:
            path = node_log_path(self.work_dir, self.run_id, node.id)
            handler = CappedNodeLogHandler(path, lambda node_id=node.id, path=path: self.on_dirty(node_id, path))
            node.logger.addHandler(handler)
            self._handlers.append((node.logger, handler))

    def detach(self) -> None:
        for logger, handler in self._handlers:
            logger.removeHandler(handler)
            handler.close()
        self._handlers = []


def read_node_log(work_dir: Path, run_id: str, node_id: str) -> tuple[str, bool]:
    path = node_log_path(work_dir, run_id, node_id)
    if not path.exists():
        return "", False
    data = path.read_bytes()
    truncated = len(data) >= MAX_NODE_LOG_BYTES
    return data.decode("utf-8", errors="replace"), truncated
