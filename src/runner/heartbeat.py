import asyncio
import os
import socket
from datetime import UTC, datetime
from typing import Callable

from nats.js.client import JetStreamContext

from shared.db import database_session
from shared.db.runners import crud as runner_crud
from shared.db.runners.schemas import RunnerCreate
from shared.jetstream import EVENT_STREAM, RUNNER_HEARTBEAT_SUBJECT, encode_model
from shared.schemas import RunnerHeartbeatMessage


class RunnerHeartbeatPublisher:
    def __init__(self, runner_id: str, js: JetStreamContext) -> None:
        self.runner_id = runner_id
        self.js = js
        self.hostname = socket.gethostname()
        self.port = int(os.environ["RUNFLOW_RUNNER_PORT"]) if "RUNFLOW_RUNNER_PORT" in os.environ else 0
        gpu_index = os.environ["RUNFLOW_RUNNER_GPU_INDEX"] if "RUNFLOW_RUNNER_GPU_INDEX" in os.environ else None
        self.gpu_index = int(gpu_index) if gpu_index is not None else None

    def register_runner(self) -> None:
        payload = RunnerCreate(
            name=self.runner_id,
            hostname=self.hostname,
            port=self.port,
            gpu_index=self.gpu_index,
        )
        with database_session() as session:
            runner_crud.upsert_runner(session, payload)

    async def run(self, active_run_ids: Callable[[], list[str]]) -> None:
        while True:
            await self.publish(active_run_ids())
            await asyncio.sleep(5)

    async def publish(self, active_run_ids: list[str]) -> None:
        message = RunnerHeartbeatMessage(
            runner_id=self.runner_id,
            hostname=self.hostname,
            process_id=os.getpid(),
            port=self.port,
            gpu_index=self.gpu_index,
            active_run_ids=active_run_ids,
            created_at=datetime.now(UTC),
        )
        await self.js.publish(RUNNER_HEARTBEAT_SUBJECT, encode_model(message), stream=EVENT_STREAM)
