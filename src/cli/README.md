# runflow-cli

Inspect workflow runs from the terminal, over the same backend HTTP API the UI
uses. Nothing here talks to NATS or a runner directly — the backend fetches and
persists each node's log from whichever runner executed it, so the CLI works the
same whether the runner is **local or remote**.

Use it to check on a run you launched from the UI (or via `POST /graphs/runs`):
a node that passes here will behave the same in the UI, because it ran through
the identical dispatch path.

## Usage

```bash
python -m cli runs                       # list all runs (id, state, timings)
python -m cli run <run_id>               # one run's status
python -m cli logs <run_id>              # aggregated logs, merged by time, node-tagged
python -m cli node-log <run_id> <node>   # one node's raw log
python -m cli failed <run_id>            # failed node(s): error + traceback + log
```

`logs` prints exactly what the UI's "All node logs (merged by time)" popover
shows — one `node_id<TAB>text` line per record, chronologically merged, with
multi-line entries (tracebacks) kept together. `failed` is the shortcut for the
common case: find which node failed and dump its error, traceback, and log.

## Backend URL

Resolved in this order:

1. `--backend URL`
2. `$RUNFLOW_BACKEND_URL` / `$VITE_BACKEND_URL`
3. `http://$BACKEND_HOST:$BACKEND_PORT` (defaults to `127.0.0.1:8001`, the dev stack)

## Exit codes

- `0` — success
- `1` — `failed` found failed node(s)
- `2` — backend unreachable or returned an error
