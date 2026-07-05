from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from fastapi import WebSocket

from shared.event_store import RunEventStore
from shared.schemas import (
    InlineGraphRunRequest,
    RunEventResponse,
    RunnerStatus,
    RunSnapshot,
    RunState,
    RunStatus,
)
from backend.websocket_hub import WebSocketHub


class DuplicateRunError(ValueError):
    pass


class CommandBus(Protocol):
    async def publish_start_graph(self, request: InlineGraphRunRequest) -> None:
        ...

    async def publish_stop(self, run_id: str, runner_id: str | None) -> None:
        ...


@dataclass
class BackendRunRecord:
    run_id: str
    workflow_path: Path
    state: RunState
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    runner_id: str | None = None
    event_store: RunEventStore = field(default_factory=RunEventStore)

    def to_status(self) -> RunStatus:
        return RunStatus(
            run_id=self.run_id,
            state=self.state,
            workflow_path=self.workflow_path,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            error=self.error,
            event_count=self.event_store.total_event_count,
        )


class BackendManager:
    def __init__(self) -> None:
        self._runs: dict[str, BackendRunRecord] = {}
        self._hub = WebSocketHub()
        self._command_bus: CommandBus | None = None

    def set_command_bus(self, command_bus: CommandBus) -> None:
        self._command_bus = command_bus

    async def start_inline_graph(self, request: InlineGraphRunRequest) -> RunStatus:
        run_id = self._inline_run_id(request)
        if run_id in self._runs:
            raise DuplicateRunError(f"Run already exists: {run_id}")

        command_request = request.model_copy(update={"run_id": run_id})
        record = BackendRunRecord(
            run_id=run_id,
            workflow_path=Path("inline_graph"),
            state=RunState.QUEUED,
            created_at=self._now(),
        )
        self._runs[run_id] = record
        await self._command_bus_checked().publish_start_graph(command_request)
        await self._broadcast_run_status(record)
        return record.to_status()

    async def stop(self, run_id: str) -> RunStatus:
        record = self._record(run_id)
        if record.state in {RunState.QUEUED, RunState.RUNNING}:
            if record.runner_id is None:
                return record.to_status()
            record.state = RunState.STOPPING
            await self._command_bus_checked().publish_stop(run_id, record.runner_id)
            await self._broadcast_run_status(record)
        return record.to_status()

    async def status(self, run_id: str) -> RunStatus:
        return self._record(run_id).to_status()

    async def list_statuses(self) -> RunnerStatus:
        runs = [record.to_status() for record in self._runs.values()]
        active_states = {RunState.QUEUED, RunState.RUNNING, RunState.STOPPING}
        active_runs = [run for run in runs if run.state in active_states]
        return RunnerStatus(total_runs=len(runs), active_runs=len(active_runs), runs=runs)

    async def events(self, run_id: str, after: int = 0) -> list[RunEventResponse]:
        return self._record(run_id).event_store.recent_after(after)

    async def errors(self, run_id: str) -> list[RunEventResponse]:
        return list(self._record(run_id).event_store.errors)

    async def snapshot(self, run_id: str) -> RunSnapshot:
        return self._record(run_id).event_store.snapshot(run_id)

    async def record_event(self, event: RunEventResponse) -> None:
        record = self._ensure_record(event.run_id)
        record.event_store.record(event)
        self._apply_lifecycle_event(record, event)
        await self._hub.broadcast_global(
            {
                "type": "run_event",
                "status": record.to_status().model_dump(mode="json"),
                "event": event.model_dump(mode="json"),
                "snapshot": record.event_store.snapshot(event.run_id).model_dump(mode="json"),
            },
        )

    async def connect_socket(self, websocket: WebSocket) -> None:
        await self._hub.connect_global(websocket)
        try:
            status = await self.list_statuses()
            await websocket.send_json({"type": "runner_status", "status": status.model_dump(mode="json")})
            while True:
                await websocket.receive_text()
        finally:
            self._hub.disconnect_global(websocket)

    async def _broadcast_run_status(self, record: BackendRunRecord) -> None:
        await self._hub.broadcast_global(
            {
                "type": "run_status",
                "status": record.to_status().model_dump(mode="json"),
            },
        )

    def _apply_lifecycle_event(self, record: BackendRunRecord, event: RunEventResponse) -> None:
        if event.kind == "run_claimed":
            record.runner_id = str(event.detail["runner_id"])
        elif event.kind == "run_started":
            record.state = RunState.RUNNING
            record.started_at = event.created_at
        elif event.kind == "run_completed":
            record.state = RunState.SUCCEEDED
            record.finished_at = event.created_at
        elif event.kind == "run_stopped":
            record.state = RunState.STOPPED
            record.finished_at = event.created_at
        elif event.kind == "run_failed":
            record.state = RunState.FAILED
            record.finished_at = event.created_at
            record.error = event.message

    def _ensure_record(self, run_id: str) -> BackendRunRecord:
        if run_id not in self._runs:
            self._runs[run_id] = BackendRunRecord(
                run_id=run_id,
                workflow_path=Path("unknown"),
                state=RunState.RUNNING,
                created_at=self._now(),
            )
        return self._runs[run_id]

    def _record(self, run_id: str) -> BackendRunRecord:
        if run_id not in self._runs:
            raise KeyError(f"Unknown run: {run_id}")
        return self._runs[run_id]

    def _command_bus_checked(self) -> CommandBus:
        if self._command_bus is None:
            raise RuntimeError("NATS command bus is not connected")
        return self._command_bus

    def _inline_run_id(self, request: InlineGraphRunRequest) -> str:
        if request.run_id is not None:
            return request.run_id
        return f"graph_{uuid4().hex[:8]}"

    def _now(self) -> datetime:
        return datetime.now(UTC)
