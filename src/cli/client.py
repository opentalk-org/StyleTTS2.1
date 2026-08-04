"""Thin HTTP client for the runflow backend.

Every call maps to an existing backend endpoint (see ``src/backend/api.py``);
nothing here talks to a runner directly. Because the backend reads and
persists each node's log from whichever runner ran it — local or remote — this
client works identically regardless of where the runner lives.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from cli.log_merge import NodeLog

DEFAULT_TIMEOUT = 30.0


def default_backend_url() -> str:
    """Resolve the backend base URL from the same env the dev stack uses."""
    explicit = os.environ.get("RUNFLOW_BACKEND_URL") or os.environ.get("VITE_BACKEND_URL")
    if explicit:
        return explicit.rstrip("/")
    host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    port = os.environ.get("BACKEND_PORT", "8001")
    return f"http://{host}:{port}"


class BackendClient:
    def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = (base_url or default_backend_url()).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def __enter__(self) -> "BackendClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str) -> Any:
        response = self._client.get(path)
        response.raise_for_status()
        return response.json()

    def list_runs(self) -> dict[str, Any]:
        """GET /runs -> RunnerStatus."""
        return self._get("/runs")

    def start_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /graphs/runs -> RunStatus."""
        response = self._client.post("/graphs/runs", json=payload)
        response.raise_for_status()
        return response.json()

    def run_status(self, run_id: str) -> dict[str, Any]:
        """GET /runs/{run_id} -> RunStatus."""
        return self._get(f"/runs/{run_id}")

    def run_graph(self, run_id: str) -> dict[str, Any]:
        """GET /runs/{run_id}/graph -> InlineGraphRunRequest."""
        return self._get(f"/runs/{run_id}/graph")

    def run_snapshot(self, run_id: str) -> dict[str, Any]:
        """GET /runs/{run_id}/snapshot -> RunSnapshot (per-node status/error)."""
        return self._get(f"/runs/{run_id}/snapshot")

    def run_errors(self, run_id: str) -> list[dict[str, Any]]:
        """GET /runs/{run_id}/errors -> list[RunEventResponse]."""
        return self._get(f"/runs/{run_id}/errors")

    def node_log(self, run_id: str, node_id: str) -> NodeLog:
        """GET /runs/{run_id}/nodes/{node_id}/logs -> NodeLogResponseMessage."""
        data = self._get(f"/runs/{run_id}/nodes/{node_id}/logs")
        return NodeLog(
            node_id=node_id,
            content=data.get("content", ""),
            truncated=bool(data.get("truncated", False)),
            error=data.get("error"),
        )

    def node_ids(self, run_id: str) -> list[str]:
        """Node ids for a run, ordered like the UI (by x, then y, then id)."""
        nodes = self.run_graph(run_id).get("nodes", [])
        ordered = sorted(nodes, key=lambda node: (node.get("x", 0), node.get("y", 0), node["id"]))
        return [node["id"] for node in ordered]

    def aggregated_logs(self, run_id: str) -> list[NodeLog]:
        """Fetch every node's log for a run (the inputs to the merge)."""
        return [self.node_log(run_id, node_id) for node_id in self.node_ids(run_id)]

    def failed_nodes(self, run_id: str) -> list[dict[str, Any]]:
        """Nodes reported as failed, from the run snapshot (node_id, status, error)."""
        nodes = self.run_snapshot(run_id).get("nodes", [])
        return [node for node in nodes if node.get("status") == "failed" or node.get("error")]
