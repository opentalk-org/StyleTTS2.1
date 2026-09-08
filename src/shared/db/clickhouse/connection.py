import os
from threading import local

import clickhouse_connect
from clickhouse_connect.driver.client import Client

CLICKHOUSE_URL_ENV = "RUNFLOW_CLICKHOUSE_URL"


class _ThreadClient(local):
    client: Client | None = None
    process_id: int | None = None


_thread_client = _ThreadClient()


def clickhouse_client() -> Client:
    """Return the ClickHouse client owned by the current thread and process."""
    process_id = os.getpid()
    if _thread_client.client is None or _thread_client.process_id != process_id:
        _thread_client.client = clickhouse_connect.get_client(
            dsn=os.environ[CLICKHOUSE_URL_ENV]
        )
        _thread_client.process_id = process_id
    return _thread_client.client


def clickhouse_healthy() -> bool:
    """Check connectivity without changing server state."""
    return clickhouse_client().ping()
