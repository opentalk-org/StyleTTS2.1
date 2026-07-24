import asyncio
import shutil
import traceback
from pathlib import Path

from runflow.core.context import ExecutionContext
from runflow.core.events import RunEvent
from runflow.runtime.scheduler import WindowedScheduler
from runner.graphs import GraphNodeBuildError, build_inline_graph
from runner.hardware import apply_detected_resources, default_memory_budget_mb
from runner.node_logs import NodeLogManager
from runner.state_buffer import RunnerStateBuffer
from shared.logging_setup import get_logger
from shared.schemas import InlineGraphRunRequest


class RunExecution:
    def __init__(self, run_id: str, request: InlineGraphRunRequest, state: RunnerStateBuffer) -> None:
        self.run_id = run_id
        self.request = request
        self.state = state
        self.scheduler: WindowedScheduler | None = None
        self.logger = get_logger(f"runner.run.{run_id}")

    async def execute(self) -> None:
        log_manager: NodeLogManager | None = None
        self.state.register_run(self.run_id)
        try:
            graph = await asyncio.to_thread(build_inline_graph, self.request)
            context = self._context()
            await asyncio.to_thread(self._cleanup_stale_run_dir, context.work_dir)
            context.event_sink = self._threadsafe_event_sink()
            log_manager = NodeLogManager(
                context.work_dir,
                self.run_id,
                lambda node_id, path: self.state.mark_log_dirty(self.run_id, node_id, path),
            )
            log_manager.attach(list(graph.nodes.values()))
            self.scheduler = WindowedScheduler(graph, context)
            await self.scheduler.arun()
            self.state.mark_terminal(self.run_id, "succeeded")
        except asyncio.CancelledError:
            if self.scheduler is not None:
                self.scheduler.cancel()
            self.state.mark_terminal(self.run_id, "stopped")
        except GraphNodeBuildError as error:
            cause = error.__cause__ if error.__cause__ is not None else error
            message = f"{type(cause).__name__}: {cause}"
            self.logger.exception("run failed during node construction")
            self.state.event_sink(
                RunEvent(
                    kind="node_failed",
                    run_id=self.run_id,
                    message=message,
                    node_id=error.node_id,
                    detail={
                        "node_type": error.node_type,
                        "stage": "graph_build",
                        "traceback": "".join(traceback.format_exception(error)),
                    },
                )
            )
            self.state.mark_terminal(self.run_id, "failed", message)
        except Exception as error:
            self.logger.error("run failed: %s", "".join(traceback.format_exception(error)))
            self.state.mark_terminal(self.run_id, "failed", f"{type(error).__name__}: {error}")
        finally:
            if log_manager is not None:
                log_manager.detach()
            await self.state.flush_due(force_run_id=self.run_id)

    def cancel(self) -> None:
        if self.scheduler is not None:
            self.scheduler.cancel()

    def _threadsafe_event_sink(self):
        loop = asyncio.get_running_loop()

        async def record(event: RunEvent) -> None:
            self.state.event_sink(event)

        def sink(event: RunEvent):
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop is loop:
                return record(event)
            return asyncio.wrap_future(asyncio.run_coroutine_threadsafe(record(event), loop))

        return sink

    def _context(self) -> ExecutionContext:
        request_config = self.request.context.config
        memory_budget_mb = request_config.memory_budget_mb
        if memory_budget_mb is None:
            memory_budget_mb = default_memory_budget_mb()
        config = request_config.model_copy(
            update={
                "resources": apply_detected_resources(request_config.resources),
                "memory_budget_mb": memory_budget_mb,
            }
        )
        return ExecutionContext(
            run_id=self.run_id,
            work_dir=self.request.context.work_dir,
            cache_dir=self.request.context.cache_dir,
            output_dir=self.request.context.output_dir,
            device=self.request.context.device,
            config=config,
            input_items=self.request.context.input_items,
        )

    def _cleanup_stale_run_dir(self, work_dir: Path) -> None:
        run_dir = work_dir / self.run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
