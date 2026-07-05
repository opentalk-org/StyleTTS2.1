from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from runflow.core.context import ExecutionContext
from runflow.core.events import RunEvent
from runflow.core.graph import Graph
from runflow.registry.node_registry import NodeRegistry
from runflow.runtime.scheduler import WindowedScheduler
from runflow.tmp_nodes.register import register_builtin_nodes
from runflow.ui.graph_import import load_graph_json

from runner.graphs import InlineGraphRunRequest, build_inline_graph
from runner.schemas import RunEventResponse, RunStartRequest, RunState, RunStatus


class DuplicateRunError(ValueError):
    pass


@dataclass
class RunRecord:
    run_id: str
    workflow_path: Path
    task: asyncio.Task[None]
    state: RunState
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    events: list[RunEventResponse] = field(default_factory=list)
    next_event_sequence: int = 1

    def to_status(self) -> RunStatus:
        return RunStatus(
            run_id=self.run_id,
            state=self.state,
            workflow_path=self.workflow_path,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            error=self.error,
            event_count=len(self.events),
        )


class RunnerManager:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def start(self, request: RunStartRequest) -> RunStatus:
        run_id = self._run_id(request)
        async with self._lock:
            if run_id in self._runs:
                raise DuplicateRunError(f"Run already exists: {run_id}")

        graph = self._load_graph(request.workflow_path)
        context = self._context(request.context, run_id)

        async with self._lock:
            if run_id in self._runs:
                raise DuplicateRunError(f"Run already exists: {run_id}")

            task = asyncio.create_task(self._execute(run_id, graph, context), name=f"runner:{run_id}")
            record = RunRecord(
                run_id=run_id,
                workflow_path=request.workflow_path,
                task=task,
                state=RunState.QUEUED,
                created_at=self._now(),
            )
            self._runs[run_id] = record
            return record.to_status()

    async def start_inline_graph(self, request: InlineGraphRunRequest) -> RunStatus:
        run_id = self._inline_run_id(request)
        async with self._lock:
            if run_id in self._runs:
                raise DuplicateRunError(f"Run already exists: {run_id}")

        graph = build_inline_graph(request)
        context = self._context(request.context, run_id)

        async with self._lock:
            if run_id in self._runs:
                raise DuplicateRunError(f"Run already exists: {run_id}")

            task = asyncio.create_task(self._execute(run_id, graph, context), name=f"runner:{run_id}")
            record = RunRecord(
                run_id=run_id,
                workflow_path=Path("inline_graph"),
                task=task,
                state=RunState.QUEUED,
                created_at=self._now(),
            )
            self._runs[run_id] = record
            return record.to_status()

    async def stop(self, run_id: str) -> RunStatus:
        async with self._lock:
            record = self._record(run_id)
            if record.state is RunState.QUEUED:
                record.state = RunState.STOPPED
                record.finished_at = self._now()
                record.task.cancel()
            elif record.state is RunState.RUNNING:
                record.state = RunState.STOPPING
                record.task.cancel()
            return record.to_status()

    async def status(self, run_id: str) -> RunStatus:
        async with self._lock:
            return self._record(run_id).to_status()

    async def events(self, run_id: str, after: int = 0) -> list[RunEventResponse]:
        async with self._lock:
            record = self._record(run_id)
            return [event for event in record.events if event.sequence > after]

    async def list_statuses(self) -> list[RunStatus]:
        async with self._lock:
            return [record.to_status() for record in self._runs.values()]

    async def _execute(self, run_id: str, graph: Graph, context: ExecutionContext) -> None:
        context.event_sink = lambda event: self._record_event(run_id, event)
        scheduler = WindowedScheduler(graph, context)
        await self._mark_started(run_id)
        try:
            await scheduler.arun()
        except asyncio.CancelledError:
            await self._mark_finished(run_id, RunState.STOPPED, None)
            raise
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            await self._record_event(
                run_id,
                RunEvent(
                    kind="run_failed",
                    run_id=run_id,
                    message=message,
                    detail={"traceback": "".join(traceback.format_exception(error))},
                ),
            )
            await self._mark_finished(run_id, RunState.FAILED, message)
            return
        await self._mark_finished(run_id, RunState.SUCCEEDED, None)

    async def _record_event(self, run_id: str, event: RunEvent) -> None:
        async with self._lock:
            record = self._record(run_id)
            response = RunEventResponse(
                sequence=record.next_event_sequence,
                kind=event.kind,
                run_id=event.run_id,
                created_at=event.created_at,
                message=event.message,
                node_id=event.node_id,
                port=event.port,
                target_node_id=event.target_node_id,
                target_port=event.target_port,
                window_index=event.window_index,
                worker_index=event.worker_index,
                batch_index=event.batch_index,
                batch_size=event.batch_size,
                lineage_id=event.lineage_id,
                detail=event.detail,
            )
            record.events.append(response)
            record.next_event_sequence += 1

    async def _mark_started(self, run_id: str) -> None:
        async with self._lock:
            record = self._record(run_id)
            record.state = RunState.RUNNING
            record.started_at = self._now()

    async def _mark_finished(self, run_id: str, state: RunState, error: str | None) -> None:
        async with self._lock:
            record = self._record(run_id)
            record.state = state
            record.finished_at = self._now()
            record.error = error

    def _record(self, run_id: str) -> RunRecord:
        if run_id not in self._runs:
            raise KeyError(f"Unknown run: {run_id}")
        return self._runs[run_id]

    def _run_id(self, request: RunStartRequest) -> str:
        if request.run_id is not None:
            return request.run_id
        return f"run_{uuid4().hex[:8]}"

    def _inline_run_id(self, request: InlineGraphRunRequest) -> str:
        if request.run_id is not None:
            return request.run_id
        return f"graph_{uuid4().hex[:8]}"

    def _load_graph(self, workflow_path: Path) -> Graph:
        registry = register_builtin_nodes(NodeRegistry())
        return load_graph_json(workflow_path, registry)

    def _context(self, context_request, run_id: str) -> ExecutionContext:
        return ExecutionContext(
            run_id=run_id,
            work_dir=context_request.work_dir,
            cache_dir=context_request.cache_dir,
            output_dir=context_request.output_dir,
            device=context_request.device,
            config=context_request.config,
            input_items=context_request.input_items,
        )

    def _now(self) -> datetime:
        return datetime.now(UTC)
