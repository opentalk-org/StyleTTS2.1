from collections.abc import Mapping
from typing import Any

from clickhouse_connect.driver.client import Client

MUTATION_SETTINGS = {"mutations_sync": 2}


def delete_rows(
    client: Client,
    table: str,
    predicate: str,
    parameters: Mapping[str, Any],
) -> None:
    """Delete rows synchronously so subsequent reads observe completion."""
    client.command(
        f"ALTER TABLE {table} DELETE WHERE {predicate}",
        parameters=dict(parameters),
        settings=MUTATION_SETTINGS,
    )
