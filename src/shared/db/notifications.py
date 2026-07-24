import asyncio
import os
from collections.abc import Sequence

import psycopg


NOTIFY_DATABASE_URL_ENV = "RUNFLOW_NOTIFY_DATABASE_URL"


class PostgresNotifier:
    def __init__(self, channels: Sequence[str]) -> None:
        self.channels = tuple(channels)
        self.changed = asyncio.Event()

    async def run(self) -> None:
        await self._listen()

    async def wait(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self.changed.wait(), timeout)
        except TimeoutError:
            return
        self.changed.clear()

    async def _listen(self) -> None:
        connection = await psycopg.AsyncConnection.connect(self._url(), autocommit=True)
        try:
            for channel in self.channels:
                await connection.execute(f'LISTEN "{channel}"')
            async for _notification in connection.notifies():
                self.changed.set()
        finally:
            await connection.close()

    def _url(self) -> str:
        return os.environ[NOTIFY_DATABASE_URL_ENV].replace("postgresql+psycopg://", "postgresql://", 1)
