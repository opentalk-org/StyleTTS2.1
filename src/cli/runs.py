"""``runflow-cli`` — inspect workflow runs from the terminal.

Run it against a running backend (the same one the UI uses):

    python -m cli runs                      # list runs
    python -m cli run <run_id>              # one run's status
    python -m cli logs <run_id>             # aggregated logs (merged by time)
    python -m cli node-log <run_id> <node>  # one node's log
    python -m cli failed <run_id>           # failed node(s): error + traceback + log
    python -m cli perf <run_id>             # graph flow and per-node performance

The backend URL defaults to $RUNFLOW_BACKEND_URL / $VITE_BACKEND_URL, else
http://$BACKEND_HOST:$BACKEND_PORT (127.0.0.1:8001 in the dev stack). Override
with ``--backend URL``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def cmd_import_ds_v2_metadata(client: BackendClient, args: argparse.Namespace) -> int:
    workflow_path = Path(args.workflow)
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    request = workflow["data"]
    source = next(node for node in request["nodes"] if node["type"] == "HetznerDsV2Source")
    source["params"].update(
        host=args.host,
        row_offset=args.row_offset,
        row_limit=args.row_limit,
        text_column=args.text_column,
        name_prefix=args.name_prefix,
        download_retries=args.download_retries,
    )
    status = client.start_graph(request)
    print(f"started ds_v2 metadata import: {status['run_id']} ({status['state']})")
    print(f"monitor: python -m cli run {status['run_id']}")
    print(f"logs:   python -m cli logs {status['run_id']}")
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


def cmd_perf(client: BackendClient, args: argparse.Namespace) -> int:
    """Show graph throughput and factual per-node flow measurements."""
    snapshot = client.run_snapshot(args.run_id)
    graph = snapshot["performance"]
    nodes = snapshot["nodes"]
    if not nodes:
        print("No nodes in snapshot.")
        return 0

    print(
        f"graph: {graph['rolling_throughput']:.2f}/s rolling · {graph['average_throughput']:.2f}/s average · "
        f"{graph['completed_items']}/{graph['started_items']} complete · {graph['inflight_items']} in flight · "
        f"p95 {_duration(graph['latency_p95_ms'])}"
    )
    header = (f"{'node':<28} {'in/s':>8} {'out/s':>8} {'queue':>11} {'delta/s':>9} "
              f"{'busy':>7} {'resource':>9} {'blocked':>10} {'p95':>9}")
    print(header)
    print("-" * len(header))
    for node in nodes:
        metrics = node["performance"]
        queue = f"{metrics['queue_size']}/{metrics['queue_capacity']}"
        busy = f"{metrics['busy_ratio'] * 100:.0f}%"
        resource = f"{metrics['resource_wait_ratio'] * 100:.0f}%"
        print(
            f"{node['node_id']:<28} {_rate(metrics['arrival_rate']):>8} {_rate(metrics['departure_rate']):>8} "
            f"{queue:>11} {metrics['queue_growth_rate']:>+9.1f} {busy:>7} {resource:>9} "
            f"{_duration(metrics['downstream_blocked_ms']):>10} {_duration(metrics['batch_p95_ms']):>9}"
        )
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

    metadata_import = sub.add_parser(
        "import-ds-v2-metadata",
        help="start an idempotent Hetzner ds_v2 metadata import",
    )
    metadata_import.add_argument(
        "--workflow",
        default="workflows/ds_v2_metadata_import.json",
        help="metadata import workflow (default: %(default)s)",
    )
    metadata_import.add_argument("--host", default="hetzner-storagebox")
    metadata_import.add_argument("--row-offset", type=int, default=0)
    metadata_import.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="maximum rows; omitted imports every metadata CSV",
    )
    metadata_import.add_argument(
        "--text-column",
        choices=("text_src", "text_parakeet", "text_whisper", "text_canary"),
        default="text_src",
    )
    metadata_import.add_argument("--name-prefix", default="ds_v2")
    metadata_import.add_argument("--download-retries", type=int, choices=range(1, 11), default=3)
    metadata_import.set_defaults(func=cmd_import_ds_v2_metadata)

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

    perf = sub.add_parser("perf", help="graph flow and per-node performance")
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
