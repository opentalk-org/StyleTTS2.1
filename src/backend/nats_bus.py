import asyncio
import os
from pathlib import Path
from uuid import uuid4

from nats.aio.client import Client as NatsClient
from nats.errors import ConnectionClosedError, ConnectionReconnectingError, TimeoutError as NatsTimeoutError
from nats.js.client import JetStreamContext
from nats.js.errors import FetchTimeoutError

from backend.service import BackendManager
from shared.jetstream import (
    BACKEND_HEARTBEAT_DURABLE,
    BACKEND_EVENT_DURABLE,
    COMMAND_STREAM,
    DEFAULT_NATS_URL,
    EVENT_STREAM,
    EVENT_SUBJECTS,
    RUNNER_HEARTBEAT_SUBJECT,
    START_COMMAND_SUBJECT,
    connect,
    decode_json,
    encode_model,
    ensure_streams,
    node_command_subject,
    node_log_command_subject,
    node_log_response_subject,
    start_command_subject,
    stop_command_subject,
)
from backend.runners.service import runner_live_registry
from shared.logging_setup import get_logger
from shared.schemas import (
    InlineGraphRunRequest,
    NodeLifecycleCommand,
    NodeLogRequestCommand,
    NodeLogResponseMessage,
    RunnerEventMessage,
    RunnerHeartbeatMessage,
    StartGraphRunCommand,
    StopRunCommand,
)


class BackendNatsBus:
    def __init__(self, manager: BackendManager, url: str | None = None) -> None:
        self.logger = get_logger("backend.nats")
        self.manager = manager
        self.url = url if url is not None else os.environ.get("NATS_URL", DEFAULT_NATS_URL)
        self.nc: NatsClient | None = None
        self.js: JetStreamContext | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.logger.info("connecting to nats url=%s", self.url)
        self.nc = await connect(self.url)
        self.js = self.nc.jetstream()
        await ensure_streams(self.js)
        self._event_task = asyncio.create_task(self._consume_events(), name="backend:nats-events")
        self._heartbeat_task = asyncio.create_task(self._consume_heartbeats(), name="backend:nats-heartbeats")
        self.logger.info("nats event consumer started")

    async def stop(self) -> None:
        if self._event_task is not None:
            self.logger.info("stopping nats event consumer")
            self._event_task.cancel()
            await asyncio.gather(self._event_task, return_exceptions=True)
            self._event_task = None
        if self._heartbeat_task is not None:
            self.logger.info("stopping nats heartbeat consumer")
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            self._heartbeat_task = None
        if self.nc is not None:
            self.logger.info("draining nats connection")
            try:
                await self.nc.drain()
            except (ConnectionClosedError, ConnectionReconnectingError):
                self.logger.info("nats connection closed before drain completed")
            self.nc = None
            self.js = None

    async def publish_start_graph(self, request: InlineGraphRunRequest) -> None:
        command = StartGraphRunCommand(request=request)
        subject = start_command_subject(request.runner_id) if request.runner_id is not None else START_COMMAND_SUBJECT
        self.logger.info("publish start command run_id=%s runner_id=%s", request.run_id, request.runner_id)
        await self._js().publish(
            subject,
            encode_model(command),
            stream=COMMAND_STREAM,
        )

    async def publish_stop(self, run_id: str, runner_id: str | None) -> None:
        if runner_id is None:
            return
        command = StopRunCommand(run_id=run_id)
        self.logger.info("publish stop command run_id=%s runner_id=%s", run_id, runner_id)
        await self._js().publish(
            stop_command_subject(runner_id),
            encode_model(command),
            stream=COMMAND_STREAM,
        )

    async def publish_node_lifecycle(self, run_id: str, node_id: str, command: str, runner_id: str | None) -> None:
        if runner_id is None:
            return
        payload = NodeLifecycleCommand(command=command, run_id=run_id, node_id=node_id)
        self.logger.info(
            "publish node lifecycle command run_id=%s node_id=%s command=%s runner_id=%s",
            run_id,
            node_id,
            command,
            runner_id,
        )
        await self._js().publish(
            node_command_subject(runner_id),
            encode_model(payload),
            stream=COMMAND_STREAM,
        )

    async def request_node_log(self, run_id: str, node_id: str, runner_id: str | None, work_dir: Path | None = None) -> NodeLogResponseMessage:
        if runner_id is None:
            raise RuntimeError(f"Run has no claimed runner: {run_id}")
        request_id = uuid4().hex
        response_subject = node_log_response_subject(request_id)
        subscription = await self._js().pull_subscribe(response_subject, stream=EVENT_STREAM)
        payload = NodeLogRequestCommand(request_id=request_id, run_id=run_id, node_id=node_id, work_dir=work_dir)
        self.logger.info("publish node log request run_id=%s node_id=%s runner_id=%s", run_id, node_id, runner_id)
        await self._js().publish(node_log_command_subject(runner_id), encode_model(payload), stream=COMMAND_STREAM)
        try:
            messages = await subscription.fetch(1, timeout=5)
        except (FetchTimeoutError, NatsTimeoutError) as error:
            raise RuntimeError(f"Timed out reading node log from runner: {runner_id}") from error
        message = messages[0]
        response = NodeLogResponseMessage.model_validate(decode_json(message.data))
        await message.ack()
        return response

    async def _consume_events(self) -> None:
        subscription = await self._js().pull_subscribe(
            EVENT_SUBJECTS[0],
            durable=BACKEND_EVENT_DURABLE,
            stream=EVENT_STREAM,
        )
        while True:
            try:
                messages = await subscription.fetch(50, timeout=1)
            except (FetchTimeoutError, NatsTimeoutError):
                continue
            for message in messages:
                try:
                    payload = decode_json(message.data)
                    event_message = RunnerEventMessage.model_validate(payload)
                    await self.manager.record_event(event_message.event)
                except Exception:
                    self.logger.exception("failed to record runner event")
                    raise
                finally:
                    await message.ack()

    async def _consume_heartbeats(self) -> None:
        subscription = await self._js().pull_subscribe(
            RUNNER_HEARTBEAT_SUBJECT,
            durable=BACKEND_HEARTBEAT_DURABLE,
            stream=EVENT_STREAM,
        )
        while True:
            try:
                messages = await subscription.fetch(50, timeout=1)
            except (FetchTimeoutError, NatsTimeoutError):
                continue
            for message in messages:
                try:
                    payload = decode_json(message.data)
                    runner_live_registry.record(RunnerHeartbeatMessage.model_validate(payload))
                except Exception:
                    self.logger.exception("failed to record runner heartbeat")
                    raise
                finally:
                    await message.ack()

    def _js(self) -> JetStreamContext:
        if self.js is None:
            raise RuntimeError("JetStream is not connected")
        return self.js
