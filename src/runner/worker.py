import asyncio
import os

from runner.job_poller import JobPoller
from runner.run_execution import RunExecution
from runner.state_buffer import RunnerStateBuffer
from shared.db.jobs.schemas import NodeStateReplacement
from shared.db.notifications import PostgresNotifier
from shared.logging_setup import get_logger
from shared.schemas import InlineGraphRunRequest


POLL_SECONDS = 0.5


class RunnerWorker:
    def __init__(self, runner_id: str) -> None:
        self.runner_id = runner_id
        self.logger = get_logger(f"runner.{runner_id}")
        port = int(os.environ["RUNFLOW_RUNNER_PORT"]) if "RUNFLOW_RUNNER_PORT" in os.environ else 0
        gpu = os.environ["RUNFLOW_RUNNER_GPU_INDEX"] if "RUNFLOW_RUNNER_GPU_INDEX" in os.environ else None
        self.state = RunnerStateBuffer(runner_id, port, int(gpu) if gpu is not None else None)
        self.poller = JobPoller(runner_id)
        self.notifier = PostgresNotifier(["runflow_runs"])
        self.executions: dict[str, RunExecution] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}

    async def run(self) -> None:
        self.logger.info("PostgreSQL runner starting")
        notify_task = asyncio.create_task(self.notifier.run(), name=f"runner:{self.runner_id}:notify")
        try:
            await self.state.flush_due()
            while True:
                try:
                    await self._claim_jobs()
                    await self._reconcile()
                    await self.state.flush_due()
                    self._discard_flushed_terminals()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.logger.exception("PostgreSQL coordination iteration failed")
                    await asyncio.sleep(1)
                await self.notifier.wait(POLL_SECONDS)
        finally:
            notify_task.cancel()
            await asyncio.gather(notify_task, return_exceptions=True)

    async def _claim_jobs(self) -> None:
        available = max(0, 1 - len(self.tasks))
        for claimed in await self.poller.claim(available) if available else []:
            request = InlineGraphRunRequest.model_validate(claimed.graph_request)
            execution = RunExecution(claimed.run_id, request, self.state)
            self.executions[claimed.run_id] = execution
            task = asyncio.create_task(execution.execute(), name=f"runflow:{claimed.run_id}")
            self.tasks[claimed.run_id] = task
            task.add_done_callback(lambda _task, run_id=claimed.run_id: self._finished(run_id))
            self.logger.info("run claimed run_id=%s", claimed.run_id)

    async def _reconcile(self) -> None:
        run_ids = list(self.tasks)
        desired = await self.poller.desired_states(run_ids)
        for run_id, state in desired.items():
            if state == "stopped":
                self.executions[run_id].cancel()
                self.tasks[run_id].cancel()

        for node_state in await self.poller.pending_node_states(run_ids):
            execution = self.executions[node_state.run_id]
            try:
                if execution.scheduler is None:
                    continue
                if node_state.desired_loaded:
                    await execution.scheduler.load_node(node_state.node_id)
                else:
                    await execution.scheduler.unload_node(node_state.node_id)
                error = None
            except Exception as exc:
                self.logger.exception("node state reconciliation failed")
                error = f"{type(exc).__name__}: {exc}"
            self.state.record_node_state(
                NodeStateReplacement(
                    run_id=node_state.run_id,
                    node_id=node_state.node_id,
                    desired_loaded=node_state.desired_loaded,
                    observed_loaded=node_state.desired_loaded if error is None else node_state.observed_loaded,
                    error=error,
                )
            )

    def _finished(self, run_id: str) -> None:
        self.tasks.pop(run_id)
        self.executions.pop(run_id)
        if self.state.terminal_flush_complete(run_id):
            self.state.discard_run(run_id)

    def _discard_flushed_terminals(self) -> None:
        for run_id in list(self.state.runs):
            if run_id not in self.tasks and self.state.terminal_flush_complete(run_id):
                self.state.discard_run(run_id)
