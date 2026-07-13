import asyncio
import os
import socket
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from runflow.core.events import RunEvent
from shared.db import database_session
from shared.db.jobs import coordination_crud
from shared.db.jobs.schemas import JobStateReplacement, NodeLogUpsert, NodeStateReplacement, RunnerStateFlush
from shared.event_store import RunEventStore
from shared.schemas import RunEventResponse


SNAPSHOT_FLUSH_SECONDS = 0.5
LOG_FLUSH_SECONDS = 1.0
HEARTBEAT_FLUSH_SECONDS = 5.0


@dataclass
class BufferedRun:
    store: RunEventStore = field(default_factory=RunEventStore)
    state: str = "running"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    release_claim: bool = False
    version: int = 1
    last_flush: float = 0


@dataclass
class DirtyLog:
    path: Path
    version: int = 1
    last_flush: float = 0


class RunnerStateBuffer:
    def __init__(self, runner_id: str, port: int, gpu_index: int | None) -> None:
        self.runner_id = runner_id
        self.hostname = socket.gethostname()
        self.port = port
        self.gpu_index = gpu_index
        self.process_id = os.getpid()
        self.runs: dict[str, BufferedRun] = {}
        self._logs: dict[tuple[str, str], DirtyLog] = {}
        self._node_states: dict[tuple[str, str], NodeStateReplacement] = {}
        self._sequences: dict[str, int] = {}
        self._log_lock = threading.Lock()
        self._flush_lock = asyncio.Lock()
        self._last_heartbeat = 0.0

    def register_run(self, run_id: str) -> None:
        self.runs[run_id] = BufferedRun(started_at=datetime.now(UTC))

    def event_sink(self, event: RunEvent) -> None:
        buffered = self.runs[event.run_id]
        self._sequences[event.run_id] = self._sequences.get(event.run_id, 0) + 1
        buffered.store.record(self._response(event, self._sequences[event.run_id]))
        buffered.version += 1

    def mark_log_dirty(self, run_id: str, node_id: str, path: Path) -> None:
        key = (run_id, node_id)
        with self._log_lock:
            if key not in self._logs:
                self._logs[key] = DirtyLog(path=path)
                return
            self._logs[key].version += 1

    def mark_terminal(self, run_id: str, state: str, error: str | None = None) -> None:
        if state == "stopped":
            self.event_sink(RunEvent(kind="run_stopped", run_id=run_id, message="run stopped"))
        elif state == "failed":
            assert error is not None, "failed runs require an error"
            self.event_sink(RunEvent(kind="run_failed", run_id=run_id, message=error))
        elif state != "succeeded":
            raise ValueError(f"Unsupported terminal run state: {state}")
        buffered = self.runs[run_id]
        buffered.state = state
        buffered.finished_at = datetime.now(UTC)
        buffered.error = error
        buffered.release_claim = True
        buffered.version += 1

    def record_node_state(self, replacement: NodeStateReplacement) -> None:
        self._node_states[(replacement.run_id, replacement.node_id)] = replacement

    async def flush_due(self, force_run_id: str | None = None) -> None:
        async with self._flush_lock:
            now = monotonic()
            job_versions, jobs = self._due_jobs(now, force_run_id)
            log_versions, logs = self._due_logs(now, force_run_id)
            node_states = [
                state for key, state in self._node_states.items()
                if force_run_id is None or key[0] == force_run_id
            ]
            heartbeat_due = now - self._last_heartbeat >= HEARTBEAT_FLUSH_SECONDS
            if not jobs and not logs and not node_states and not heartbeat_due:
                return
            payload = RunnerStateFlush(
                runner_id=self.runner_id,
                hostname=self.hostname,
                port=self.port,
                gpu_index=self.gpu_index,
                process_id=self.process_id,
                active_run_ids=[run_id for run_id, run in self.runs.items() if not run.release_claim],
                capabilities={},
                jobs=jobs,
                node_states=node_states,
                logs=logs,
            )
            await asyncio.to_thread(self._flush, payload)
            self._accept_flush(now, job_versions, log_versions)
            for state in node_states:
                self._node_states.pop((state.run_id, state.node_id), None)
            self._last_heartbeat = now

    def discard_run(self, run_id: str) -> None:
        self.runs.pop(run_id)
        self._sequences.pop(run_id, None)
        with self._log_lock:
            for key in [key for key in self._logs if key[0] == run_id]:
                del self._logs[key]
        for key in [key for key in self._node_states if key[0] == run_id]:
            del self._node_states[key]

    def terminal_flush_complete(self, run_id: str) -> bool:
        buffered = self.runs[run_id]
        return buffered.release_claim and buffered.version == 0

    def _due_jobs(self, now: float, force_run_id: str | None) -> tuple[dict[str, int], list[JobStateReplacement]]:
        versions: dict[str, int] = {}
        replacements: list[JobStateReplacement] = []
        for run_id, buffered in self.runs.items():
            if buffered.version == 0:
                continue
            if run_id != force_run_id and now - buffered.last_flush < SNAPSHOT_FLUSH_SECONDS:
                continue
            versions[run_id] = buffered.version
            replacements.append(
                JobStateReplacement(
                    run_id=run_id,
                    state=buffered.state,
                    snapshot=buffered.store.snapshot(run_id).model_dump(mode="json"),
                    started_at=buffered.started_at,
                    finished_at=buffered.finished_at,
                    error=buffered.error,
                    release_claim=buffered.release_claim,
                )
            )
        return versions, replacements

    def _due_logs(self, now: float, force_run_id: str | None) -> tuple[dict[tuple[str, str], int], list[NodeLogUpsert]]:
        versions: dict[tuple[str, str], int] = {}
        replacements: list[NodeLogUpsert] = []
        with self._log_lock:
            items = list(self._logs.items())
        for key, dirty in items:
            if dirty.version == 0:
                continue
            if key[0] != force_run_id and now - dirty.last_flush < LOG_FLUSH_SECONDS:
                continue
            data = dirty.path.read_bytes() if dirty.path.exists() else b""
            versions[key] = dirty.version
            replacements.append(
                NodeLogUpsert(
                    run_id=key[0],
                    node_id=key[1],
                    content=data.decode("utf-8", errors="replace"),
                    truncated=len(data) >= 1_000_000,
                )
            )
        return versions, replacements

    def _accept_flush(self, now: float, jobs: dict[str, int], logs: dict[tuple[str, str], int]) -> None:
        for run_id, version in jobs.items():
            buffered = self.runs[run_id]
            buffered.last_flush = now
            if buffered.version == version:
                buffered.version = 0
        with self._log_lock:
            for key, version in logs.items():
                dirty = self._logs[key]
                dirty.last_flush = now
                if dirty.version == version:
                    dirty.version = 0

    def _flush(self, payload: RunnerStateFlush) -> None:
        with database_session() as session:
            coordination_crud.flush_runner_state(session, payload)

    def _response(self, event: RunEvent, sequence: int) -> RunEventResponse:
        return RunEventResponse(
            sequence=sequence,
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
