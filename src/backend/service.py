import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from fastapi import WebSocket

from shared.logging_setup import get_logger
from shared.event_store import RunEventStore
from shared.schemas import (
    InlineGraphRunRequest,
    RunEventResponse,
    NodeLogResponseMessage,
    RunnerStatus,
    RunSnapshot,
    RunState,
    RunStatus,
)
from backend.jobs.persistence import persist_job, persist_node_log
from backend.websocket_hub import WebSocketHub
from shared.db import database_session
from shared.db.jobs import crud as jobs_crud

IMMEDIATE_EVENT_KINDS = {"run_claimed", "run_started", "run_completed", "run_stopped", "run_failed", "node_failed", "node_lifecycle_failed", "node_loaded", "node_unloaded"}


class DuplicateRunError(ValueError):
    pass


class CommandBus(Protocol):
    async def publish_start_graph(self, request: InlineGraphRunRequest) -> None:
        ...

    async def publish_stop(self, run_id: str, runner_id: str | None) -> None:
        ...

    async def publish_node_lifecycle(self, run_id: str, node_id: str, command: str, runner_id: str | None) -> None:
        ...

    async def request_node_log(self, run_id: str, node_id: str, runner_id: str | None, work_dir: Path | None = None) -> NodeLogResponseMessage:
        ...


@dataclass
class BackendRunRecord:
    run_id: str
    name: str
    workflow_path: Path
    state: RunState
    created_at: datetime
    graph_request: InlineGraphRunRequest | None = None
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
        self.logger = get_logger("backend.manager")
        self._runs: dict[str, BackendRunRecord] = {}
        self._hub = WebSocketHub()
        self._command_bus: CommandBus | None = None
        self._snapshot_tasks: dict[str, asyncio.Task[None]] = {}
        # Runs removed while active: late runner events must not resurrect them.
        self._deleted: set[str] = set()

    def set_command_bus(self, command_bus: CommandBus) -> None:
        self._command_bus = command_bus

    async def start_inline_graph(self, request: InlineGraphRunRequest, name: str | None = None) -> RunStatus:
        run_id = self._inline_run_id(request)
        if run_id in self._runs:
            raise DuplicateRunError(f"Run already exists: {run_id}")
        self._deleted.discard(run_id)

        command_request = request.model_copy(update={"run_id": run_id})
        record = BackendRunRecord(
            run_id=run_id,
            name=name or run_id,
            workflow_path=Path("inline_graph"),
            state=RunState.QUEUED,
            created_at=datetime.now(UTC),
            graph_request=command_request,
            runner_id=command_request.runner_id,
        )
        self._runs[run_id] = record
        persist_job(record)
        self.logger.info("queue run run_id=%s", run_id)
        await self._command_bus_checked().publish_start_graph(command_request)
        await self._broadcast_run_status(record)
        return record.to_status()

    async def stop(self, run_id: str) -> RunStatus:
        record = self._record(run_id)
        if record.state in {RunState.QUEUED, RunState.RUNNING}:
            if record.runner_id is None:
                return record.to_status()
            record.state = RunState.STOPPING
            self.logger.info("stop run run_id=%s runner_id=%s", run_id, record.runner_id)
            await self._command_bus_checked().publish_stop(run_id, record.runner_id)
            await self._broadcast_run_status(record)
        return record.to_status()

    async def remove(self, run_id: str) -> None:
        """Remove a job, stopping it first if it is still active.

        Unlike a plain delete, this works on running jobs: it asks the runner to
        cancel, tombstones the run so trailing events do not re-create it, then
        force-deletes the persisted record.
        """
        record = self._runs.get(run_id)
        existed = record is not None
        if record is not None and record.state in {RunState.QUEUED, RunState.RUNNING, RunState.STOPPING} and record.runner_id is not None:
            self.logger.info("stop before remove run_id=%s runner_id=%s", run_id, record.runner_id)
            try:
                await self._command_bus_checked().publish_stop(run_id, record.runner_id)
            except Exception:
                self.logger.exception("stop before remove failed run_id=%s", run_id)
        self._deleted.add(run_id)
        self._runs.pop(run_id, None)
        task = self._snapshot_tasks.pop(run_id, None)
        if task is not None:
            task.cancel()
        self.logger.info("remove run run_id=%s", run_id)
        with database_session() as session:
            try:
                jobs_crud.delete_job(session, run_id, force=True)
            except KeyError:
                if not existed:
                    raise

    async def remove_many(self, run_ids: list[str]) -> None:
        for run_id in run_ids:
            try:
                await self.remove(run_id)
            except KeyError:
                self.logger.info("skip remove of missing job run_id=%s", run_id)

    async def remove_all(self) -> None:
        with database_session() as session:
            persisted = jobs_crud.list_all_job_ids(session)
        await self.remove_many(list({*persisted, *self._runs.keys()}))

    async def rename(self, run_id: str, name: str) -> None:
        record = self._runs.get(run_id)
        if record is not None:
            record.name = name
            persist_job(record)
            return
        with database_session() as session:
            jobs_crud.rename_job(session, run_id, name)

    async def load_node(self, run_id: str, node_id: str) -> RunStatus:
        return await self._publish_node_lifecycle(run_id, node_id, "load_node")

    async def unload_node(self, run_id: str, node_id: str) -> RunStatus:
        return await self._publish_node_lifecycle(run_id, node_id, "unload_node")

    async def node_log(self, run_id: str, node_id: str) -> NodeLogResponseMessage:
        record = self._runs.get(run_id)
        if record is not None and record.runner_id is not None and record.state in {RunState.QUEUED, RunState.RUNNING, RunState.STOPPING}:
            self.logger.info("read live node log run_id=%s node_id=%s runner_id=%s", run_id, node_id, record.runner_id)
            work_dir = record.graph_request.context.work_dir if record.graph_request is not None else None
            response = await self._command_bus_checked().request_node_log(run_id, node_id, record.runner_id, work_dir)
            persist_node_log(response)
            return response
        try:
            with database_session() as session:
                item = jobs_crud.get_node_log(session, run_id, node_id)
                return NodeLogResponseMessage(request_id="db", run_id=run_id, node_id=node_id, content=item.content, truncated=item.truncated, error=item.error)
        except KeyError as error:
            raise RuntimeError(f"Node log is not available: {run_id}/{node_id}") from error

    async def graph(self, run_id: str) -> InlineGraphRunRequest:
        record = self._runs.get(run_id)
        if record is not None and record.graph_request is not None:
            return record.graph_request
        try:
            with database_session() as session:
                return InlineGraphRunRequest.model_validate(jobs_crud.get_job(session, run_id).graph_request)
        except KeyError as error:
            raise RuntimeError(f"Run graph is not available: {run_id}") from error

    async def status(self, run_id: str) -> RunStatus:
        return self._record(run_id).to_status()

    async def list_statuses(self) -> RunnerStatus:
        runs = [record.to_status() for record in self._runs.values()]
        active_states = {RunState.QUEUED, RunState.RUNNING, RunState.STOPPING}
        active_runs = [run for run in runs if run.state in active_states]
        return RunnerStatus(total_runs=len(runs), active_runs=len(active_runs), runs=runs)

    async def errors(self, run_id: str) -> list[RunEventResponse]:
        return list(self._record(run_id).event_store.errors)

    async def snapshot(self, run_id: str) -> RunSnapshot:
        record = self._runs.get(run_id)
        if record is not None:
            return record.event_store.snapshot(run_id)
        # Not live in memory (e.g. reopened from history or after a backend restart):
        # serve the per-node snapshot persisted on the job so the failed node still shows.
        with database_session() as session:
            job = jobs_crud.get_job(session, run_id)
        if job.snapshot is not None:
            return RunSnapshot.model_validate(job.snapshot)
        return RunEventStore().snapshot(run_id)

    async def record_event(self, event: RunEventResponse) -> None:
        if event.run_id in self._deleted:
            return
        record = self._ensure_record(event.run_id)
        record.event_store.record(event)
        self._apply_lifecycle_event(record, event)
        persist_job(record)
        if event.kind in IMMEDIATE_EVENT_KINDS:
            await self._broadcast_run_snapshot(record)
            return
        self._schedule_run_snapshot(record.run_id)

    async def connect_socket(self, websocket: WebSocket) -> None:
        await self._hub.connect_global(websocket)
        try:
            status = await self.list_statuses()
            await websocket.send_json({"type": "runner_status", "status": status.model_dump(mode="json")})
            while True:
                self._handle_socket_message(websocket, await websocket.receive_text())
        finally:
            self._hub.disconnect_global(websocket)

    async def _broadcast_run_status(self, record: BackendRunRecord) -> None:
        await self._hub.broadcast_global(
            {
                "type": "run_status",
                "status": record.to_status().model_dump(mode="json"),
            },
        )

    async def _broadcast_run_snapshot(self, record: BackendRunRecord) -> None:
        task = self._snapshot_tasks.pop(record.run_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        await self._hub.broadcast_run(
            record.run_id,
            {
                "type": "run_snapshot",
                "run_id": record.run_id,
                "status": record.to_status().model_dump(mode="json"),
                "snapshot": record.event_store.snapshot(record.run_id).model_dump(mode="json"),
            },
        )

    def _schedule_run_snapshot(self, run_id: str) -> None:
        task = self._snapshot_tasks.get(run_id)
        if task is None or task.done():
            self._snapshot_tasks[run_id] = asyncio.create_task(self._broadcast_run_snapshot_later(run_id))

    async def _broadcast_run_snapshot_later(self, run_id: str) -> None:
        await asyncio.sleep(0.1)
        if run_id in self._runs:
            await self._broadcast_run_snapshot(self._runs[run_id])

    def _handle_socket_message(self, websocket: WebSocket, raw_message: str) -> None:
        message = json.loads(raw_message)
        if message["type"] == "watch_run":
            self._hub.watch_run(websocket, message["run_id"])

    def _apply_lifecycle_event(self, record: BackendRunRecord, event: RunEventResponse) -> None:
        if event.kind == "run_claimed":
            record.runner_id = str(event.detail["runner_id"])
            record.graph_request = InlineGraphRunRequest.model_validate(event.detail["request"])
            self.logger.info("run claimed run_id=%s runner_id=%s", record.run_id, record.runner_id)
        elif event.kind == "run_started":
            record.state = RunState.RUNNING
            record.started_at = event.created_at
            self.logger.info("run started run_id=%s", record.run_id)
        elif event.kind == "run_completed":
            record.state = RunState.SUCCEEDED
            record.finished_at = event.created_at
            self.logger.info("run completed run_id=%s", record.run_id)
        elif event.kind == "run_stopped":
            record.state = RunState.STOPPED
            record.finished_at = event.created_at
            self.logger.info("run stopped run_id=%s", record.run_id)
        elif event.kind == "run_failed":
            record.state = RunState.FAILED
            record.finished_at = event.created_at
            record.error = event.message
            self.logger.error("run failed run_id=%s error=%s", record.run_id, event.message)
        if event.kind in {"run_completed", "run_stopped", "run_failed"}:
            self._schedule_node_log_sync(record)

    def _ensure_record(self, run_id: str) -> BackendRunRecord:
        if run_id not in self._runs:
            self._runs[run_id] = BackendRunRecord(
                run_id=run_id,
                name=run_id,
                workflow_path=Path("unknown"),
                state=RunState.RUNNING,
                created_at=datetime.now(UTC),
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

    async def _publish_node_lifecycle(self, run_id: str, node_id: str, command: str) -> RunStatus:
        record = self._record(run_id)
        if record.state is not RunState.RUNNING:
            raise RuntimeError(f"Run is not active: {run_id}")
        if record.runner_id is None:
            raise RuntimeError(f"Run has no claimed runner: {run_id}")
        self.logger.info(
            "node lifecycle command run_id=%s node_id=%s command=%s runner_id=%s",
            run_id,
            node_id,
            command,
            record.runner_id,
        )
        await self._command_bus_checked().publish_node_lifecycle(run_id, node_id, command, record.runner_id)
        return record.to_status()

    def _inline_run_id(self, request: InlineGraphRunRequest) -> str:
        if request.run_id is not None:
            return request.run_id
        return f"graph_{uuid4().hex[:8]}"

    def _schedule_node_log_sync(self, record: BackendRunRecord) -> None:
        if record.runner_id is None or record.graph_request is None:
            return
        work_dir = record.graph_request.context.work_dir
        asyncio.create_task(self._sync_node_logs(record.run_id, record.runner_id, [node.id for node in record.graph_request.nodes], work_dir))

    async def _sync_node_logs(self, run_id: str, runner_id: str, node_ids: list[str], work_dir: Path | None = None) -> None:
        for node_id in node_ids:
            try:
                response = await self._command_bus_checked().request_node_log(run_id, node_id, runner_id, work_dir)
                persist_node_log(response)
            except Exception:
                self.logger.exception("sync node log failed run_id=%s node_id=%s", run_id, node_id)
