from __future__ import annotations

import argparse
import asyncio
import traceback
from datetime import UTC, datetime
from uuid import uuid4

from nats.aio.msg import Msg
from nats.js.api import ConsumerConfig
from nats.js.client import JetStreamContext
from nats.js.errors import FetchTimeoutError

from runflow.core.context import ExecutionContext
from runflow.core.events import RunEvent
from runflow.runtime.scheduler import WindowedScheduler
from runner.graphs import build_inline_graph
from shared.jetstream import (
    COMMAND_STREAM,
    DEFAULT_NATS_URL,
    EVENT_STREAM,
    START_COMMAND_SUBJECT,
    RUNNER_COMMAND_DURABLE,
    connect,
    decode_json,
    encode_model,
    ensure_streams,
    event_subject,
    stop_command_subject,
)
from shared.schemas import InlineGraphRunRequest, RunEventResponse, RunnerEventMessage, StartGraphRunCommand, StopRunCommand


class RunnerWorker:
    def __init__(self, runner_id: str, nats_url: str = DEFAULT_NATS_URL) -> None:
        self.runner_id = runner_id
        self.nats_url = nats_url
        self.js: JetStreamContext | None = None
        self._active_runs: dict[str, asyncio.Task[None]] = {}
        self._sequences: dict[str, int] = {}
        self._sequence_lock = asyncio.Lock()

    async def run(self) -> None:
        nc = await connect(self.nats_url)
        self.js = nc.jetstream()
        await ensure_streams(self._js())
        await asyncio.gather(
            self._consume_starts(),
            self._consume_stops(stop_command_subject(self.runner_id), f"{self.runner_id}-targeted-stops"),
        )

    async def _consume_starts(self) -> None:
        subscription = await self._js().pull_subscribe(
            START_COMMAND_SUBJECT,
            durable=RUNNER_COMMAND_DURABLE,
            stream=COMMAND_STREAM,
            config=ConsumerConfig(ack_wait=30),
        )
        while True:
            try:
                messages = await subscription.fetch(1, timeout=1)
            except FetchTimeoutError:
                continue
            for message in messages:
                asyncio.create_task(self._handle_start(message), name=f"runner:{self.runner_id}:start")

    async def _consume_stops(self, subject: str, durable: str) -> None:
        subscription = await self._js().pull_subscribe(
            subject,
            durable=durable,
            stream=COMMAND_STREAM,
            config=ConsumerConfig(ack_wait=30),
        )
        while True:
            try:
                messages = await subscription.fetch(10, timeout=1)
            except FetchTimeoutError:
                continue
            for message in messages:
                await self._handle_stop(message)

    async def _handle_start(self, message: Msg) -> None:
        payload = decode_json(message.data)
        command = StartGraphRunCommand.model_validate(payload)
        request = command.request
        run_id = self._run_id(request)
        if run_id in self._active_runs:
            await message.ack()
            return

        await self._publish_custom_event(run_id, "run_claimed", "run claimed", {"runner_id": self.runner_id})
        task = asyncio.create_task(self._execute_run(run_id, request), name=f"runflow:{run_id}")
        self._active_runs[run_id] = task
        heartbeat = asyncio.create_task(self._heartbeat(message, task), name=f"runflow:{run_id}:ack-heartbeat")
        try:
            await task
            await message.ack()
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            del self._active_runs[run_id]

    async def _handle_stop(self, message: Msg) -> None:
        payload = decode_json(message.data)
        command = StopRunCommand.model_validate(payload)
        task = self._active_runs.get(command.run_id)
        if task is not None:
            task.cancel()
        await message.ack()

    async def _execute_run(self, run_id: str, request: InlineGraphRunRequest) -> None:
        graph = build_inline_graph(request)
        context = self._context(request, run_id)
        context.event_sink = self._publish_run_event
        scheduler = WindowedScheduler(graph, context)
        try:
            await scheduler.arun()
        except asyncio.CancelledError:
            await self._publish_custom_event(run_id, "run_stopped", "run stopped", {"runner_id": self.runner_id})
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            await self._publish_custom_event(
                run_id,
                "run_failed",
                message,
                {"runner_id": self.runner_id, "traceback": "".join(traceback.format_exception(error))},
            )

    async def _heartbeat(self, message: Msg, task: asyncio.Task[None]) -> None:
        while not task.done():
            await message.in_progress()
            await asyncio.sleep(10)

    async def _publish_run_event(self, event: RunEvent) -> None:
        response = await self._response(event)
        await self._publish_response(response)

    async def _publish_custom_event(
        self,
        run_id: str,
        kind: str,
        message: str,
        detail: dict[str, object],
    ) -> None:
        response = RunEventResponse(
            sequence=await self._next_sequence(run_id),
            kind=kind,
            run_id=run_id,
            created_at=datetime.now(UTC),
            message=message,
            node_id=None,
            port=None,
            target_node_id=None,
            target_port=None,
            window_index=None,
            worker_index=None,
            batch_index=None,
            batch_size=None,
            lineage_id=None,
            detail=detail,
        )
        await self._publish_response(response)

    async def _publish_response(self, response: RunEventResponse) -> None:
        message = RunnerEventMessage(event=response)
        await self._js().publish(
            event_subject(response.run_id),
            encode_model(message),
            stream=EVENT_STREAM,
        )

    async def _response(self, event: RunEvent) -> RunEventResponse:
        return RunEventResponse(
            sequence=await self._next_sequence(event.run_id),
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

    async def _next_sequence(self, run_id: str) -> int:
        async with self._sequence_lock:
            if run_id not in self._sequences:
                self._sequences[run_id] = 0
            self._sequences[run_id] += 1
            return self._sequences[run_id]

    def _context(self, request: InlineGraphRunRequest, run_id: str) -> ExecutionContext:
        return ExecutionContext(
            run_id=run_id,
            work_dir=request.context.work_dir,
            cache_dir=request.context.cache_dir,
            output_dir=request.context.output_dir,
            device=request.context.device,
            config=request.context.config,
            input_items=request.context.input_items,
        )

    def _run_id(self, request: InlineGraphRunRequest) -> str:
        if request.run_id is not None:
            return request.run_id
        return f"graph_{uuid4().hex[:8]}"

    def _js(self) -> JetStreamContext:
        if self.js is None:
            raise RuntimeError("JetStream is not connected")
        return self.js


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-id", default=f"runner_{uuid4().hex[:8]}")
    parser.add_argument("--nats-url", default=DEFAULT_NATS_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker = RunnerWorker(runner_id=args.runner_id, nats_url=args.nats_url)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
