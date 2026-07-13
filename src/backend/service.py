import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import WebSocket

from backend.db_watcher import BackendDatabaseWatcher
from backend.run_records import job_status
from backend.websocket_hub import WebSocketHub
from shared.db import database_session
from shared.db.jobs import coordination_crud
from shared.db.jobs import crud as jobs_crud
from shared.db.jobs.schemas import JobUpsert
from shared.schemas import (
    InlineGraphRunRequest,
    NodeLogResponseMessage,
    RunEventResponse,
    RunnerStatus,
    RunSnapshot,
    RunState,
    RunStatus,
)


class DuplicateRunError(ValueError):
    pass


class BackendManager:
    def __init__(self) -> None:
        self._hub = WebSocketHub()
        self._watcher = BackendDatabaseWatcher(self._broadcast_changed_runs)
        self._watcher_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._watcher_task = asyncio.create_task(self._watcher.run(), name="backend:database-watcher")

    async def close(self) -> None:
        if self._watcher_task is None:
            return
        self._watcher_task.cancel()
        await asyncio.gather(self._watcher_task, return_exceptions=True)
        self._watcher_task = None

    async def start_inline_graph(self, request: InlineGraphRunRequest, name: str | None = None) -> RunStatus:
        run_id = request.run_id if request.run_id is not None else f"graph_{uuid4().hex[:8]}"
        command_request = request.model_copy(update={"run_id": run_id})
        now = datetime.now(UTC)
        payload = JobUpsert(
            run_id=run_id,
            name=name or run_id,
            state=RunState.QUEUED.value,
            desired_state="running",
            target_runner_id=command_request.runner_id,
            graph_request=command_request.model_dump(mode="json"),
            created_at=now,
            started_at=None,
            finished_at=None,
            error=None,
        )
        with database_session() as session:
            try:
                jobs_crud.get_job(session, run_id)
            except KeyError:
                job = jobs_crud.upsert_job(session, payload)
            else:
                raise DuplicateRunError(f"Run already exists: {run_id}")
        return job_status(job)

    async def stop(self, run_id: str) -> RunStatus:
        with database_session() as session:
            job = coordination_crud.set_desired_job_state(session, run_id, "stopped")
        return job_status(job)

    async def remove(self, run_id: str) -> None:
        with database_session() as session:
            job = jobs_crud.get_job(session, run_id)
            if job.state in {"queued", "running", "stopping"}:
                coordination_crud.set_desired_job_state(session, run_id, "stopped")
        for _ in range(20):
            with database_session() as session:
                job = jobs_crud.get_job(session, run_id)
                if job.state not in {"queued", "running", "stopping"}:
                    break
            await asyncio.sleep(0.25)
        with database_session() as session:
            jobs_crud.delete_job(session, run_id, force=True)

    async def remove_many(self, run_ids: list[str]) -> None:
        for run_id in run_ids:
            try:
                await self.remove(run_id)
            except KeyError:
                continue

    async def remove_all(self) -> None:
        with database_session() as session:
            run_ids = jobs_crud.list_all_job_ids(session)
        await self.remove_many(run_ids)

    async def rename(self, run_id: str, name: str) -> None:
        with database_session() as session:
            jobs_crud.rename_job(session, run_id, name)

    async def load_node(self, run_id: str, node_id: str) -> RunStatus:
        return await self._set_node_state(run_id, node_id, True)

    async def unload_node(self, run_id: str, node_id: str) -> RunStatus:
        return await self._set_node_state(run_id, node_id, False)

    async def node_log(self, run_id: str, node_id: str) -> NodeLogResponseMessage:
        with database_session() as session:
            item = jobs_crud.get_node_log(session, run_id, node_id)
        return NodeLogResponseMessage(
            request_id="db",
            run_id=run_id,
            node_id=node_id,
            content=item.content,
            truncated=item.truncated,
            error=item.error,
        )

    async def graph(self, run_id: str) -> InlineGraphRunRequest:
        with database_session() as session:
            return InlineGraphRunRequest.model_validate(jobs_crud.get_job(session, run_id).graph_request)

    async def status(self, run_id: str) -> RunStatus:
        with database_session() as session:
            return job_status(jobs_crud.get_job(session, run_id))

    async def list_statuses(self) -> RunnerStatus:
        with database_session() as session:
            runs = [job_status(job) for job in jobs_crud.list_run_jobs(session)]
        active = sum(run.state in {RunState.QUEUED, RunState.RUNNING, RunState.STOPPING} for run in runs)
        return RunnerStatus(total_runs=len(runs), active_runs=active, runs=runs)

    async def errors(self, run_id: str) -> list[RunEventResponse]:
        snapshot = await self.snapshot(run_id)
        now = datetime.now(UTC)
        return [
            RunEventResponse(
                sequence=index,
                kind="node_failed",
                run_id=run_id,
                created_at=now,
                message=node.error,
                node_id=node.node_id,
                port=None,
                target_node_id=None,
                target_port=None,
                window_index=None,
                worker_index=None,
                batch_index=None,
                batch_size=None,
                lineage_id=None,
                detail={},
            )
            for index, node in enumerate(snapshot.nodes, 1)
            if node.error is not None
        ]

    async def snapshot(self, run_id: str) -> RunSnapshot:
        with database_session() as session:
            job = jobs_crud.get_job(session, run_id)
        if job.snapshot is not None:
            return RunSnapshot.model_validate(job.snapshot)
        return RunSnapshot(run_id=run_id, total_event_count=0, error_count=0, event_counts={}, nodes=[])

    async def connect_socket(self, websocket: WebSocket) -> None:
        await self._hub.connect_global(websocket)
        try:
            await websocket.send_json({"type": "runner_status", "status": (await self.list_statuses()).model_dump(mode="json")})
            while True:
                message = json.loads(await websocket.receive_text())
                if message["type"] == "watch_run":
                    self._hub.watch_run(websocket, message["run_id"])
        finally:
            self._hub.disconnect_global(websocket)

    async def _set_node_state(self, run_id: str, node_id: str, loaded: bool) -> RunStatus:
        with database_session() as session:
            job = jobs_crud.get_job(session, run_id)
            if job.state != RunState.RUNNING.value:
                raise RuntimeError(f"Run is not active: {run_id}")
            coordination_crud.set_desired_node_state(session, run_id, node_id, loaded)
        return job_status(job)

    async def _broadcast_changed_runs(self, run_ids: list[str]) -> None:
        await self._hub.broadcast_global(
            {"type": "runner_status", "status": (await self.list_statuses()).model_dump(mode="json")}
        )
        for run_id in run_ids:
            try:
                status = await self.status(run_id)
                snapshot = await self.snapshot(run_id)
            except KeyError:
                continue
            await self._hub.broadcast_run(
                run_id,
                {
                    "type": "run_snapshot",
                    "run_id": run_id,
                    "status": status.model_dump(mode="json"),
                    "snapshot": snapshot.model_dump(mode="json"),
                },
            )
