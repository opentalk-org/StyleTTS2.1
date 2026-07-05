from __future__ import annotations

import asyncio

from nats.aio.client import Client as NatsClient
from nats.js.client import JetStreamContext
from nats.js.errors import FetchTimeoutError

from backend.service import BackendManager
from shared.jetstream import (
    BACKEND_EVENT_DURABLE,
    COMMAND_STREAM,
    DEFAULT_NATS_URL,
    EVENT_STREAM,
    EVENT_SUBJECTS,
    START_COMMAND_SUBJECT,
    connect,
    decode_json,
    encode_model,
    ensure_streams,
    stop_command_subject,
)
from shared.schemas import InlineGraphRunRequest, RunnerEventMessage, StartGraphRunCommand, StopRunCommand


class BackendNatsBus:
    def __init__(self, manager: BackendManager, url: str = DEFAULT_NATS_URL) -> None:
        self.manager = manager
        self.url = url
        self.nc: NatsClient | None = None
        self.js: JetStreamContext | None = None
        self._event_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.nc = await connect(self.url)
        self.js = self.nc.jetstream()
        await ensure_streams(self.js)
        self._event_task = asyncio.create_task(self._consume_events(), name="backend:nats-events")

    async def stop(self) -> None:
        if self._event_task is not None:
            self._event_task.cancel()
            await asyncio.gather(self._event_task, return_exceptions=True)
            self._event_task = None
        if self.nc is not None:
            await self.nc.drain()
            self.nc = None
            self.js = None

    async def publish_start_graph(self, request: InlineGraphRunRequest) -> None:
        command = StartGraphRunCommand(request=request)
        await self._js().publish(
            START_COMMAND_SUBJECT,
            encode_model(command),
            stream=COMMAND_STREAM,
        )

    async def publish_stop(self, run_id: str, runner_id: str | None) -> None:
        if runner_id is None:
            return
        command = StopRunCommand(run_id=run_id)
        await self._js().publish(
            stop_command_subject(runner_id),
            encode_model(command),
            stream=COMMAND_STREAM,
        )

    async def _consume_events(self) -> None:
        subscription = await self._js().pull_subscribe(
            EVENT_SUBJECTS[0],
            durable=BACKEND_EVENT_DURABLE,
            stream=EVENT_STREAM,
        )
        while True:
            try:
                messages = await subscription.fetch(50, timeout=1)
            except FetchTimeoutError:
                continue
            for message in messages:
                payload = decode_json(message.data)
                event_message = RunnerEventMessage.model_validate(payload)
                await self.manager.record_event(event_message.event)
                await message.ack()

    def _js(self) -> JetStreamContext:
        if self.js is None:
            raise RuntimeError("JetStream is not connected")
        return self.js
