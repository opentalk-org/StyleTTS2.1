import asyncio
from collections.abc import Sequence

from shared.db import database_session
from shared.db.jobs import coordination_crud
from shared.db.jobs.models import RunNodeState
from shared.db.jobs.schemas import ClaimedJob


class JobPoller:
    def __init__(self, runner_id: str) -> None:
        self.runner_id = runner_id

    async def claim(self, limit: int) -> list[ClaimedJob]:
        return await asyncio.to_thread(self._claim, limit)

    async def desired_states(self, run_ids: Sequence[str]) -> dict[str, str]:
        return await asyncio.to_thread(self._desired_states, list(run_ids))

    async def pending_node_states(self, run_ids: Sequence[str]) -> list[RunNodeState]:
        return await asyncio.to_thread(self._pending_node_states, list(run_ids))

    def _claim(self, limit: int) -> list[ClaimedJob]:
        with database_session() as session:
            return coordination_crud.claim_jobs(session, self.runner_id, limit)

    def _desired_states(self, run_ids: list[str]) -> dict[str, str]:
        if not run_ids:
            return {}
        with database_session() as session:
            return coordination_crud.desired_job_states(session, self.runner_id, run_ids)

    def _pending_node_states(self, run_ids: list[str]) -> list[RunNodeState]:
        if not run_ids:
            return []
        with database_session() as session:
            rows = coordination_crud.pending_node_states(session, run_ids)
            session.expunge_all()
            return rows
