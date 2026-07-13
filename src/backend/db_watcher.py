import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from shared.db import database_session
from shared.db.jobs import crud as jobs_crud
from shared.db.notifications import PostgresNotifier


class BackendDatabaseWatcher:
    def __init__(self, on_runs_changed: Callable[[list[str]], Awaitable[None]]) -> None:
        self.on_runs_changed = on_runs_changed
        self.notifier = PostgresNotifier(["runflow_runs", "runflow_runners"])
        self.updated_after = datetime.now(UTC)

    async def run(self) -> None:
        notify_task = asyncio.create_task(self.notifier.run(), name="backend:postgres-notify")
        try:
            while True:
                rows = await asyncio.to_thread(self._changed_jobs)
                if rows:
                    self.updated_after = max(row.updated_at for row in rows)
                    await self.on_runs_changed([row.run_id for row in rows])
                await self.notifier.wait(0.5)
        finally:
            notify_task.cancel()
            await asyncio.gather(notify_task, return_exceptions=True)

    def _changed_jobs(self):
        with database_session() as session:
            rows = list(jobs_crud.list_jobs_updated_after(session, self.updated_after))
            session.expunge_all()
            return rows
