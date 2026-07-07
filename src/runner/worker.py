import asyncio
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from nats.errors import TimeoutError as NatsTimeoutError
from nats.aio.msg import Msg
from nats.js.api import ConsumerConfig
from nats.js.client import JetStreamContext
from nats.js.errors import FetchTimeoutError

from runflow.core.context import ExecutionContext
from runflow.core.events import RunEvent
from runflow.runtime.scheduler import WindowedScheduler
from runner.graphs import build_inline_graph
from runner.hardware import apply_detected_resources
from runner.heartbeat import RunnerHeartbeatPublisher
from runner.node_logs import NodeLogManager, publish_node_log_response
from shared.jetstream import (
    COMMAND_STREAM, DEFAULT_NATS_URL, EVENT_STREAM, RUNNER_COMMAND_DURABLE, START_COMMAND_SUBJECT,
    connect, decode_json, encode_model, ensure_streams, event_subject,
    node_command_subject, node_log_command_subject, start_command_subject, stop_command_subject,
)
from shared.logging_setup import get_logger
from shared.schemas import InlineGraphRunRequest, NodeLifecycleCommand, RunEventResponse, RunnerEventMessage, StartGraphRunCommand, StopRunCommand

# A start command is attempted once and never redelivered. A run that fails is
# reported as run_failed and acked, so redelivery only ever happens when a start
# is left un-acked (runner crash / ack timeout); capping delivery to 1 keeps such
# a command from being re-dispatched forever and re-running the graph in a loop.
START_COMMAND_MAX_DELIVER = 1

class RunnerWorker:
    def __init__(self, runner_id: str, nats_url: str = DEFAULT_NATS_URL) -> None:
        self.logger = get_logger(f"runner.{runner_id}")
        self.runner_id = runner_id
        self.nats_url = nats_url
        self.js: JetStreamContext | None = None
        self._active_runs: dict[str, asyncio.Task[None]] = {}
        self._active_schedulers: dict[str, WindowedScheduler] = {}
        self._run_work_dirs: dict[str, Path] = {}
        self._pending_stops: set[str] = set()
        self._sequences: dict[str, int] = {}
        self._sequence_lock = asyncio.Lock()

    async def run(self) -> None:
        self.logger.info("connecting to nats url=%s", self.nats_url)
        nc = await connect(self.nats_url)
        self.js = nc.jetstream()
        await ensure_streams(self._js())
        heartbeat = RunnerHeartbeatPublisher(self.runner_id, self._js())
        heartbeat.register_runner()
        self.logger.info("runner subscriptions starting")
        await asyncio.gather(
            heartbeat.run(lambda: list(self._active_runs.keys())),
            self._consume_starts(START_COMMAND_SUBJECT, RUNNER_COMMAND_DURABLE),
            self._consume_starts(start_command_subject(self.runner_id), f"{self.runner_id}-targeted-starts"),
            self._consume_commands(stop_command_subject(self.runner_id), f"{self.runner_id}-targeted-stops", self._handle_stop),
            self._consume_commands(node_command_subject(self.runner_id), f"{self.runner_id}-node-lifecycle", self._handle_node_lifecycle),
            self._consume_commands(node_log_command_subject(self.runner_id), f"{self.runner_id}-node-logs", self._handle_node_log),
        )

    async def _consume_starts(self, subject: str, durable: str) -> None:
        self.logger.info("consume starts subject=%s durable=%s", subject, durable)
        subscription = await self._js().pull_subscribe(
            subject,
            durable=durable,
            stream=COMMAND_STREAM,
            config=ConsumerConfig(ack_wait=30, max_deliver=START_COMMAND_MAX_DELIVER),
        )
        while True:
            try:
                messages = await subscription.fetch(1, timeout=1)
            except (FetchTimeoutError, NatsTimeoutError, asyncio.TimeoutError):
                continue
            for message in messages:
                asyncio.create_task(self._handle_start(message), name=f"runner:{self.runner_id}:start")

    async def _consume_commands(self, subject: str, durable: str, handler) -> None:
        self.logger.info("consume commands subject=%s durable=%s", subject, durable)
        subscription = await self._js().pull_subscribe(
            subject,
            durable=durable,
            stream=COMMAND_STREAM,
            config=ConsumerConfig(ack_wait=30),
        )
        while True:
            try:
                messages = await subscription.fetch(10, timeout=1)
            except (FetchTimeoutError, NatsTimeoutError, asyncio.TimeoutError):
                continue
            for message in messages:
                await handler(message)

    async def _handle_start(self, message: Msg) -> None:
        payload = decode_json(message.data)
        command = StartGraphRunCommand.model_validate(payload)
        request = command.request
        run_id = self._run_id(request)
        if run_id in self._active_runs:
            await message.ack()
            return

        self.logger.info("run claimed run_id=%s", run_id)
        await self._publish_custom_event(
            run_id,
            "run_claimed",
            "run claimed",
            {"runner_id": self.runner_id, "request": request.model_dump(mode="json")},
        )
        task = asyncio.create_task(self._execute_run(run_id, request), name=f"runflow:{run_id}")
        self._active_runs[run_id] = task
        if run_id in self._pending_stops:
            self.logger.info("applying pending stop run_id=%s", run_id)
            self._pending_stops.discard(run_id)
            task.cancel()
        heartbeat = asyncio.create_task(self._heartbeat(message, task), name=f"runflow:{run_id}:ack-heartbeat")
        try:
            await task
            await message.ack()
        except asyncio.CancelledError:
            self.logger.info("run stopped before execution started run_id=%s", run_id)
            await self._publish_custom_event(run_id, "run_stopped", "run stopped", {"runner_id": self.runner_id})
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
            self.logger.info("stop command run_id=%s", command.run_id)
            scheduler = self._active_schedulers.get(command.run_id)
            if scheduler is not None:
                scheduler.cancel()
            else:
                task.cancel()
        else:
            self.logger.info("stop command pending run_id=%s", command.run_id)
            self._pending_stops.add(command.run_id)
        await message.ack()

    async def _handle_node_lifecycle(self, message: Msg) -> None:
        payload = decode_json(message.data)
        command = NodeLifecycleCommand.model_validate(payload)
        scheduler = self._active_schedulers.get(command.run_id)
        try:
            if scheduler is None:
                raise RuntimeError(f"Run is not active on runner {self.runner_id}: {command.run_id}")
            if command.command == "load_node":
                self.logger.info("load node command run_id=%s node_id=%s", command.run_id, command.node_id)
                await scheduler.load_node(command.node_id)
            elif command.command == "unload_node":
                self.logger.info("unload node command run_id=%s node_id=%s", command.run_id, command.node_id)
                await scheduler.unload_node(command.node_id)
            else:
                raise RuntimeError(f"Unsupported node lifecycle command: {command.command}")
        except Exception as error:
            self.logger.exception("node lifecycle command failed")
            await self._publish_custom_event(
                command.run_id,
                "node_lifecycle_failed",
                f"{command.node_id} lifecycle failed: {type(error).__name__}: {error}",
                {"runner_id": self.runner_id, "traceback": "".join(traceback.format_exception(error))},
                command.node_id,
            )
        await message.ack()

    async def _handle_node_log(self, message: Msg) -> None:
        await publish_node_log_response(self._js(), message, self._run_work_dir, self.logger)

    async def _execute_run(self, run_id: str, request: InlineGraphRunRequest) -> None:
        log_manager: NodeLogManager | None = None
        try:
            self.logger.info("run starting run_id=%s", run_id)
            graph = build_inline_graph(request)
            context = self._context(request, run_id)
            self._run_work_dirs[run_id] = context.work_dir
            context.event_sink = self._threadsafe_event_sink()
            log_manager = NodeLogManager(context.work_dir, run_id)
            log_manager.attach(list(graph.nodes.values()))
            scheduler = WindowedScheduler(graph, context)
            self._active_schedulers[run_id] = scheduler
            await scheduler.arun()
            self.logger.info("run completed run_id=%s", run_id)
        except asyncio.CancelledError:
            self.logger.info("run stopped run_id=%s", run_id)
            await self._publish_custom_event(run_id, "run_stopped", "run stopped", {"runner_id": self.runner_id})
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.logger.exception("run failed run_id=%s", run_id)
            await self._publish_custom_event(
                run_id,
                "run_failed",
                message,
                {"runner_id": self.runner_id, "traceback": "".join(traceback.format_exception(error))},
            )
        finally:
            if log_manager is not None:
                log_manager.detach()
            self._active_schedulers.pop(run_id, None)

    def _threadsafe_event_sink(self):
        loop = asyncio.get_running_loop()

        def sink(event: RunEvent):
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop is loop:
                return self._publish_run_event(event)
            return asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._publish_run_event(event), loop))

        return sink

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
        node_id: str | None = None,
    ) -> None:
        response = RunEventResponse(
            sequence=await self._next_sequence(run_id),
            kind=kind,
            run_id=run_id,
            created_at=datetime.now(UTC),
            message=message,
            node_id=node_id,
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
        config = request.context.config.model_copy(
            update={"resources": apply_detected_resources(request.context.config.resources)}
        )
        return ExecutionContext(
            run_id=run_id,
            work_dir=request.context.work_dir,
            cache_dir=request.context.cache_dir,
            output_dir=request.context.output_dir,
            device=request.context.device,
            config=config,
            input_items=request.context.input_items,
        )

    def _run_id(self, request: InlineGraphRunRequest) -> str:
        return request.run_id if request.run_id is not None else f"graph_{uuid4().hex[:8]}"

    def _run_work_dir(self, run_id: str) -> Path:
        scheduler = self._active_schedulers.get(run_id)
        if scheduler is not None:
            return scheduler.context.work_dir
        if run_id not in self._run_work_dirs:
            raise RuntimeError(f"Unknown run log directory: {run_id}")
        return self._run_work_dirs[run_id]

    def _js(self) -> JetStreamContext:
        if self.js is None:
            raise RuntimeError("JetStream is not connected")
        return self.js
