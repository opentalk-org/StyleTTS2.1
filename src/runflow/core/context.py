from __future__ import annotations

from dataclasses import dataclass, field
from inspect import isawaitable
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Iterator
from uuid import uuid4

from runflow.core.config import RuntimeConfig
from runflow.core.events import RunEvent
from runflow.runtime.cancellation import CancelToken, cancellable


@dataclass
class ExecutionContext:
    run_id: str = field(default_factory=lambda: f"run_{uuid4().hex[:8]}")
    work_dir: Path = Path("work")
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("outputs")
    device: str = "cuda"
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    input_items: list[Any] = field(default_factory=list)
    window_index: int = 0
    cancel_token: CancelToken = field(default_factory=CancelToken)
    event_sink: Callable[[RunEvent], Awaitable[None] | None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir).resolve()
        self.cache_dir = Path(self.cache_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        if isinstance(self.config, dict):
            self.config = RuntimeConfig.model_validate(self.config)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def window_dir(self) -> Path:
        path = self.work_dir / self.run_id / f"window_{self.window_index:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def node_dir(self, node_id: str) -> Path:
        path = self.window_dir / node_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cancel(self) -> None:
        self.cancel_token.cancel()

    def check_cancel(self) -> None:
        self.cancel_token.check()

    def cancellable[T](self, items: Iterable[T]) -> Iterator[T]:
        return cancellable(items)

    async def emit_event(
        self,
        kind: str,
        message: str = "",
        node_id: str | None = None,
        port: str | None = None,
        target_node_id: str | None = None,
        target_port: str | None = None,
        worker_index: int | None = None,
        batch_index: int | None = None,
        batch_size: int | None = None,
        lineage_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        if self.event_sink is None:
            return
        event = RunEvent(
            kind=kind,
            run_id=self.run_id,
            message=message,
            node_id=node_id,
            port=port,
            target_node_id=target_node_id,
            target_port=target_port,
            window_index=self.window_index,
            worker_index=worker_index,
            batch_index=batch_index,
            batch_size=batch_size,
            lineage_id=lineage_id,
            detail=detail or {},
        )
        result = self.event_sink(event)
        if isawaitable(result):
            await result

    async def report_progress(
        self,
        node_id: str,
        current: float | None = None,
        total: float | None = None,
        message: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        progress_detail = dict(detail or {})
        progress_detail["current"] = current
        progress_detail["total"] = total
        await self.emit_event("node_progress", message=message, node_id=node_id, detail=progress_detail)
