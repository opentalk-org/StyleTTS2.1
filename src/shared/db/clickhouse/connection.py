import os
from functools import lru_cache

import clickhouse_connect
from clickhouse_connect.driver.client import Client

CLICKHOUSE_URL_ENV = "RUNFLOW_CLICKHOUSE_URL"


@lru_cache(maxsize=1)
def clickhouse_client() -> Client:
    """Return the process-wide ClickHouse client."""
    return clickhouse_connect.get_client(dsn=os.environ[CLICKHOUSE_URL_ENV])


def clickhouse_healthy() -> bool:
    """Check connectivity without changing server state."""
    return clickhouse_client().ping()
