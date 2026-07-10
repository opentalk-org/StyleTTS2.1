"""``runflow-cli`` — inspect workflow runs from the terminal.

Run it against a running backend (the same one the UI uses):

    python -m cli runs                      # list runs
    python -m cli run <run_id>              # one run's status
    python -m cli logs <run_id>             # aggregated logs (merged by time)
    python -m cli node-log <run_id> <node>  # one node's log
    python -m cli failed <run_id>           # failed node(s): error + traceback + log
    python -m cli perf <run_id>             # per-node latency & throughput metrics

The backend URL defaults to $RUNFLOW_BACKEND_URL / $VITE_BACKEND_URL, else
http://$BACKEND_HOST:$BACKEND_PORT (127.0.0.1:8001 in the dev stack). Override
with ``--backend URL``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import httpx

from cli.client import BackendClient, default_backend_url
from cli.log_merge import format_merged


def _print_run_row(run: dict) -> None:
    started = run.get("started_at") or "-"
    finished = run.get("finished_at") or "-"
    error = f"  error={run['error']}" if run.get("error") else ""
    print(f"{run['run_id']:<24} {run['state']:<10} events={run.get('event_count', 0):<5} "
          f"start={started} end={finished}{error}")


def cmd_runs(client: BackendClient, _: argparse.Namespace) -> int:
    status = client.list_runs()
    runs = status.get("runs", [])
    if not runs:
        print("No runs.")
        return 0
    print(f"{status.get('total_runs', len(runs))} run(s), {status.get('active_runs', 0)} active:\n")
    for run in runs:
        _print_run_row(run)
    return 0


def cmd_run(client: BackendClient, args: argparse.Namespace) -> int:
    _print_run_row(client.run_status(args.run_id))
    return 0


def cmd_logs(client: BackendClient, args: argparse.Namespace) -> int:
    logs = client.aggregated_logs(args.run_id)
    truncated = [log.node_id for log in logs if log.truncated]
    if truncated:
        print(f"# showing latest 1 MB for: {', '.join(truncated)}", file=sys.stderr)
    merged = format_merged(logs)
    if merged:
        print(merged)
    else:
        print("No log lines for any node yet.", file=sys.stderr)
    return 0


def cmd_node_log(client: BackendClient, args: argparse.Namespace) -> int:
    log = client.node_log(args.run_id, args.node_id)
    if log.error:
        print(f"[error] {log.error}", file=sys.stderr)
    if log.truncated:
        print("# showing latest 1 MB", file=sys.stderr)
    print(log.content, end="" if log.content.endswith("\n") else "\n")
    return 0


def cmd_failed(client: BackendClient, args: argparse.Namespace) -> int:
    failed = client.failed_nodes(args.run_id)
    if not failed:
        print("No failed nodes.")
        return 0

    # Tracebacks arrive on the error events, keyed by node id.
    tracebacks: dict[str, str] = {}
    for event in client.run_errors(args.run_id):
        node_id = event.get("node_id")
        traceback = (event.get("detail") or {}).get("traceback")
        if node_id and traceback:
            tracebacks[node_id] = traceback

    for node in failed:
        node_id = node["node_id"]
        print(f"===== {node_id}  (status={node.get('status')}) =====")
        if node.get("error"):
            print(f"error: {node['error']}")
        if node_id in tracebacks:
            print(tracebacks[node_id].rstrip())
        log = client.node_log(args.run_id, node_id)
        if log.content.strip():
            print("--- node log ---")
            print(log.content.rstrip())
        print()
    return 1


def _duration(milliseconds: float) -> str:
    """Format a millisecond duration like the UI (ms / s / m)."""
    if milliseconds < 1000:
        return f"{milliseconds:.0f}ms"
    if milliseconds < 60_000:
        return f"{milliseconds / 1000:.1f}s"
    return f"{milliseconds / 60_000:.1f}m"


def _rate(value: float) -> str:
    """Format an items/s rate like the UI (one decimal below 10, else whole)."""
    return f"{value:.1f}" if value < 10 else f"{value:.0f}"


def _current_elapsed_ms(started_at: str | None) -> float:
    """Milliseconds since the in-flight batch started, or 0 if none is running."""
    if not started_at:
        return 0.0
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return 0.0
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds() * 1000)


def cmd_perf(client: BackendClient, args: argparse.Namespace) -> int:
    """Per-node latency & throughput, matching the UI's node-performance table."""
    nodes = client.run_snapshot(args.run_id).get("nodes", [])
    if not nodes:
        print("No nodes in snapshot.")
        return 0

    header = (f"{'node':<28} {'active total':>12} {'queue avg':>10} {'resource avg':>12} "
              f"{'p95 batch':>10} {'items/s':>8} {'max q':>6}")
    print(header)
    print("-" * len(header))
    for node in nodes:
        metrics = node.get("performance") or {}
        batches = metrics.get("batches", 0) or 1
        current_started = metrics.get("current_batch_started_at")
        wait_batches = (metrics.get("batches", 0) + (1 if current_started else 0)) or 1
        current_elapsed = _current_elapsed_ms(current_started)
        completed_total = (metrics.get("total_resource_wait_ms", 0.0) + metrics.get("total_load_ms", 0.0)
                           + metrics.get("total_execute_ms", 0.0) + metrics.get("total_unload_ms", 0.0)
                           + metrics.get("total_route_ms", 0.0))
        active_total = _duration(completed_total + current_elapsed)
        queue_avg = _duration((metrics.get("total_queue_wait_ms", 0.0)
                               + metrics.get("current_queue_wait_ms", 0.0)) / wait_batches)
        resource_avg = _duration(metrics.get("total_resource_wait_ms", 0.0) / batches)
        p95 = _duration(metrics.get("p95_batch_ms", 0.0))
        throughput = _rate(metrics.get("items_per_second", 0.0))
        max_q = str(metrics.get("max_queue_size", 0))
        print(f"{node['node_id']:<28} {active_total:>12} {queue_avg:>10} {resource_avg:>12} "
              f"{p95:>10} {throughput:>8} {max_q:>6}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runflow-cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default=None,
                        help=f"backend base URL (default: {default_backend_url()})")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("runs", help="list all runs").set_defaults(func=cmd_runs)

    run = sub.add_parser("run", help="show one run's status")
    run.add_argument("run_id")
    run.set_defaults(func=cmd_run)

    logs = sub.add_parser("logs", help="aggregated logs, merged by time")
    logs.add_argument("run_id")
    logs.set_defaults(func=cmd_logs)

    node_log = sub.add_parser("node-log", help="one node's raw log")
    node_log.add_argument("run_id")
    node_log.add_argument("node_id")
    node_log.set_defaults(func=cmd_node_log)

    failed = sub.add_parser("failed", help="failed node(s): error, traceback, and log")
    failed.add_argument("run_id")
    failed.set_defaults(func=cmd_failed)

    perf = sub.add_parser("perf", help="per-node latency & throughput metrics")
    perf.add_argument("run_id")
    perf.set_defaults(func=cmd_perf)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with BackendClient(args.backend) as client:
            return args.func(client, args)
    except httpx.HTTPStatusError as error:
        detail = _error_detail(error)
        print(f"backend error {error.response.status_code}: {detail}", file=sys.stderr)
        return 2
    except httpx.HTTPError as error:
        print(f"cannot reach backend at {args.backend or default_backend_url()}: {error}", file=sys.stderr)
        return 2


def _error_detail(error: httpx.HTTPStatusError) -> str:
    try:
        return error.response.json().get("detail", error.response.text)
    except ValueError:
        return error.response.text


if __name__ == "__main__":
    raise SystemExit(main())
