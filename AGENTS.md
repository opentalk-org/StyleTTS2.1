# AGENTS.md

This file is the onboarding note for coding agents working on this repository. Keep it short and update it only for rules that should apply to nearly every task.

## Project purpose

This project is a ComfyUI-style workflow system for typed, batched, concurrent pipelines. The core package is `runflow`, a domain-agnostic runtime for node graphs with custom datatypes, typed ports, batching, generic resource policies, artifact storage, cancellation, and event snapshots. Audio processing is one target workflow, not a runtime assumption.

## Repository layout

- `src/runflow/` — core graph/runtime library. This must stay domain-agnostic.
- `src/backend/` — FastAPI backend that serves the UI/API, loads workflows, persists status, publishes runner commands over NATS, and exposes runtime results.
- `src/runner/` — NATS-backed worker/CLI that runs workflows using `runflow` and owns project-specific node definitions.
- `src/shared/` — shared database, storage, event, logging, and schema code used by both backend and runner.
- `src/frontend/` — visual editor/client UI for nodes, typed sockets, workflow editing, and job monitoring.
- `workflows/` — example graph JSON definitions used as smoke workflows.

## Technology Stack

- backend: fastapi, sqlalchemy, pydantic, postgresql, nats jetstream, s3-compatible object storage
- frontend: react, tanstack query/form, tailwind css, zustand for local state
- runflow: pydantic
- runner: nats worker/cli, sqlalchemy through shared db facades, pydantic

## Architecture rules

- Do not add audio-specific assumptions to `src/runflow`. Use generic names such as item, artifact, datatype, port, node, resource, queue, batch, window, and task.
- Nodes declare capabilities through metadata: input/output ports, datatypes, batch policy, resource policy, and cache policy. Avoid hardcoding behavior in the scheduler based on node names.
- Resource scheduling is generic. Prefer `resources={"io": 1}`, `resources={"accelerator": 1, "vram_gb": 8}`, or project-defined resource keys over CPU/GPU-specific node classes.
- Windowing is generic. A window is a bounded set of source items based on count, cost, or caller-provided metadata; it must not depend on audio files.
- Keep node lifecycle separate from node logic. Loading/unloading belongs behind the node manager, not scattered through the scheduler.
- Prefer artifact references over passing large payloads in memory. Branches should reuse stored artifacts instead of duplicating data.

## Node quality

- A good node is typed, batch-aware, cancellable, and honest about resources. Declare precise ports, settings, `BATCH_POLICY`, `RESOURCE_POLICY`, and `QUEUE_MAX_SIZE`; do not rely on scheduler special cases.
- Never use union datatypes for node ports. Model alternatives as separate explicit ports, separate nodes, or a concrete wrapper datatype with a clear discriminator.
- Design for high throughput. Process the whole incoming `batch` when the underlying library supports it, avoid per-item model reloads, use bulk DB/storage calls, and stream or chunk large inputs instead of materializing full datasets in memory.
- Keep expensive resources behind `setup`/`teardown` and `keep_loaded` policies. Load models, remote clients, and caches once per node lifecycle; release file handles, subprocesses, GPU memory, and temporary directories reliably.
- Make cancellation work in long loops and multi-step jobs. Call `context.check_cancel()` between chunks, batches, downloads, subprocess waits, and training epochs; propagate cancellation into child processes or library callbacks when possible.
- Report useful progress for long-running nodes with `context.report_progress(...)` or `__progress__` outputs. Prefer item counts, bytes, epochs, or rows over vague messages.
- Preserve lineage and output shape. For normal transforms, return one output per input item; for fan-out nodes, only expand from a single input item unless the scheduler contract is explicitly handled.
- Fail with actionable errors. Validate settings and input metadata early, include the missing field/resource in the exception, and avoid swallowing partial failures that would make runs look successful.
- Register new node types and datatypes in the runner registry/schema export path so the frontend can discover ports, settings, runtime defaults, and categories. 
- Reuse Ports, Instead creating variants prefer single one generalized because it allows to connecting various nodes.

## Database and stored files

- Backend and runner code must access PostgreSQL through the shared CRUD facades in `src/shared/db/<feature>/crud.py`. Do not bypass them with ad hoc SQLAlchemy queries unless the feature CRUD does not exist yet.
- Use `shared.db.connection.database_session` for SQLAlchemy sessions and keep transaction boundaries inside CRUD/service functions.
- Audio file bytes are managed by `src/shared/db/audio/crud.py`; waveform packs are managed by `src/shared/db/waveforms/crud.py`. Call the public CRUD functions; callers should not manage pack files, byte offsets, stale bytes, or pruning directly. Use bulk options when audio files can be counted in hundreds.
- Checkpoints, configs, and extra files are managed by `src/shared/db/assets/crud.py`. Checkpoints are folder artifacts: create/update them from a folder path and use `get_checkpoint_path` to get the cached local folder. Extra files are single-object artifacts; use `get_extra_file_path` when a local cached file path is needed.
- PostgreSQL is the source of truth for metadata and bucket object keys. S3/RustFS bucket objects and local caches are implementation details behind shared CRUD/storage helpers.

## Project structure

- Use feature-based structure inside `src/frontend`, `src/backend`, `src/runner`, and `src/shared/db` where feature ownership is clear. `src/runflow` is organized around runtime capabilities such as `core`, `planning`, `registry`, `runtime`, and `ui`; keep it domain-agnostic.
- Keep each feature split into clear layers. A feature may contain local files such as `schemas.py`, `models.py`, `crud.py`, `actions.py`, `service.py`, `api.py`, `query.ts`, `logic.ts`, and `components.tsx` as appropriate for that feature.
- Frontend features should separate API calls, query/cache hooks, state or domain logic, and rendering components. Prefer names such as `api`, `query`, `logic`, and `components` within the feature.
- Backend features should separate request/response schemas, database models, persistence operations, actions or services, and route/API wiring. Prefer names such as `schemas`, `models`, `crud`, `actions`, `service`, and `api` within the feature.
- Runner node families live under `src/runner/nodes/<family>/`; keep their datatypes, models, runtime code, and registration close to the node family.
- Node families must not import implementation modules owned by another node family. Copy an external reference into the owning family when reuse through a domain-agnostic shared module is not appropriate.
- Name modules, classes, functions, and variables after stable responsibilities and domain concepts, not implementation discussions or decision history.

## Frontend implementation

- Stack is already installed under `src/frontend`: Vite + React + TypeScript, Tailwind v4, TanStack Query, TanStack Form, Zustand. Do not add a new toolchain.
- Tailwind is v4 and **config-less** (`@import "tailwindcss"` in `src/frontend/src/index.css`). There is no `tailwind.config.js`. Define the color palette and design tokens in an `@theme` block in `index.css`, then use them as normal Tailwind utility classes.
- Reusable, domain-agnostic UI and helpers live in `src/frontend/src/shared/` (`ui`, `data`, `feedback`, `media`, `schema-form`, icons, formatting). Reuse these instead of re-styling per feature.
- Each feature folder under `src/frontend/src/features/<feature>/` keeps its own `api.ts`, `query.ts`, `logic.ts`, and `components.tsx` (split further into a local `components/` folder when a feature grows past the file-size limit).
- Large lists (datasets, audio files, jobs — target is ~5 million rows) **must** be virtualized with `@tanstack/react-virtual`. Never render a full row set into the DOM; window it. Assume any list view can hit millions of rows.
- Zustand holds local UI/client state; TanStack Query owns server state and caching. Do not duplicate server data into Zustand.
- Frontend features generally call the backend through `src/frontend/src/app/backend.ts` from their feature `api.ts`, and cache server data through `query.ts`. Keep any temporary mock data behind the same API/query seam.

## Development workflow

- Run all project commands through `./nix/run-venv.sh <command> [arguments...]`. This is the single supported entry point for the Nix development shell, project `.venv`, local service environment, and repository user; do not assemble equivalent `nix develop`, `runuser`, `.venv`, database, MLflow, or S3 command lines in individual sessions.
- Before changing behavior, find the relevant example or smoke workflow and run it before and after the change when practical.
- Do not run backend, runner, CLI, Python, pytest, uv, npm, node, or equivalent `nix develop`/`.venv` commands directly from the host shell. The wrapper provides the correct shared libraries and environment.
- Manage the local NATS JetStream, backend, and runner through the single shared Zellij session: `./nix/run-venv.sh runflow-dev-session`. This attaches to the existing `runflow-dev` session or creates it and starts the stack once. Check it with `./nix/run-venv.sh runflow-dev-status`; stop it with `./nix/run-venv.sh runflow-dev-stop`. Never start a second stack or run `runflow-dev` directly when the shared session exists. Do not use `sudo`.
- The project virtual environment is at `/workspace/styletts_studio_v2/.venv`, but access it only through `./nix/run-venv.sh`; do not invoke `.venv/bin/python`, `python`, or other venv binaries directly from the host shell.
- Torch and other project Python libraries are available through the project interpreter: use `./nix/run-venv.sh python -m <module> ...` or `./nix/run-venv.sh python <script.py>`. Verify the environment with `./nix/run-venv.sh python -c 'import torch; print(torch.__version__)'`.
- Add or change Python dependencies in `pyproject.toml`, then update the lock with `./nix/run-venv.sh uv ...`; do not install packages with `pip install` into the environment.
- For backend or frontend changes, use the package manager and commands already present in that subproject, but invoke them through `./nix/run-venv.sh`. Do not introduce a new toolchain unless requested.
- Always test nodes by running them through a real graph, never by importing the node class and calling `execute()` by hand. Direct calls skip registration, port typing, `setup`/`teardown`, batching, resource leasing, and routing, so "works in my script" does not mean "works from the UI". Build a small graph (reuse the `Testing*` input feeders in `src/runner/nodes/testing/`), submit it via `POST /graphs/runs` (the exact path the UI uses — runs on the same local or remote runner), then inspect results with the `src/cli` tool through the wrapper: `./nix/run-venv.sh python -m cli runs`, `./nix/run-venv.sh python -m cli logs <run_id>` for aggregated logs, `./nix/run-venv.sh python -m cli node-log <run_id> <node>`, and `./nix/run-venv.sh python -m cli failed <run_id>` to pull a failed node's error and traceback. See `src/cli/README.md`. A node that runs clean here runs clean in the UI.
- Do not keep committed tests in the repo unless explicitly requested. Temporary throwaway tests/scripts are allowed and preferable for validating behavior, but remove them before finishing.

## Deployment (hub + remote runners)

The flake builds two images. The **hub** (`nix build .#image`, entrypoint `nix/entrypoint.sh`) runs the full stack (Postgres, PgBouncer, NATS, RustFS, backend, frontend, MLflow, a runner) and is stateful — run it on a persistent GPU box (Vast.ai). The **runner** (`nix build .#runner-image`, entrypoint `nix/runner-entrypoint.sh`) is a slim, self-contained GPU worker (baked ML venv + tailscale + proxychains); run it on any GPU host and it reaches the hub's DB/NATS/S3/MLflow services as a client over a self-hosted Headscale tailnet.

- **Shared config lives in `flake.nix`**: `commonEnv` (creds/ports, baked into both images — single source of truth), `hubEnv`, `runnerEnv`. Secrets/topology are injected at runtime, never baked. The one fill-in file is `.env` (gitignored); `deploy/hub.env.example` documents the hub's runtime env.
- **Baked runner venv**: `runnerVenv` in `flake.nix` builds the full Python/ML env (torch/vllm/flashinfer/… ~13GB) with `uv sync --frozen` so the runner image is self-contained (no runtime `uv sync`). It is a `__noChroot` (impure, network-enabled) derivation — a fixed-output hash is not viable because source-built extensions embed build paths. Builds need `sandbox = relaxed` (set in CI via the installer `extra-conf` and locally via `--option sandbox relaxed`). Because the venv is prebaked, the runner image drops `uv`/`rustc`/`cargo` and keeps only `gcc`+`nvcc` for triton/inductor JIT.
- **Tailscale**: `nix/tailscale-up.sh` (the `tailscale-up` command) joins Headscale for both images via `TAILSCALE_LOGIN_SERVER` + `TAILSCALE_AUTHKEY`. Both hub and runner use **userspace** networking (`TAILSCALE_USERSPACE=1`) since managed containers (Vast.ai) have no `/dev/net/tun`. The hub publishes its services to the tailnet with `tailscale serve --tcp` (raw passthrough) for the ports in `TAILSCALE_SERVE_PORTS` (6432/4222/9000); the runner is launched under `proxychains4` so its raw-TCP Postgres/NATS/boto3 route through the tailnet SOCKS proxy. It no-ops without `TAILSCALE_AUTHKEY`, preserving the single-box path.
- **Runner identity**: `nix/runner-entrypoint.sh` derives a unique `RUNNER_ID` per replica (from `$SALAD_MACHINE_ID` if present, else the container hostname) — it keys the heartbeat DB row and NATS routing, so it must be unique. `HF_TOKEN` is not passed to runners; they read it from the hub's `integration_settings` table over the tailnet.
- **Hub run**: `docker run --gpus all -v runflow-data:/data --env-file .env <hub-image>`. No `--device`/`--cap-add` needed (userspace tailscale). Ports are not published — only tailnet peers reach DB/NATS/S3 (via `tailscale serve`).
- **Runner run**: `docker run --gpus all -v runflow-data:/data -e TAILSCALE_LOGIN_SERVER=… -e TAILSCALE_AUTHKEY=… -e RUNFLOW_HUB_HOST=<hub tailnet name/IP> ghcr.io/<owner>/runflow-studio-runner:latest` on any GPU host. It joins the tailnet, proxychains to the hub, and registers its heartbeat.
- **Runner image CI**: `git push` to `main` triggers `.github/workflows/deploy.yml` → `nix build .#runner-image` → push `ghcr.io/<owner>/runflow-studio-runner:<sha>` and `:latest`. Needs `sandbox = relaxed` (installer `extra-conf`) for the `__noChroot` venv build, and secret `GHCR_OWNER`. CI builds from the **committed** tree — untracked/uncommitted files (e.g. new `src/**`) will be missing from the image, so commit before pushing.
- **Headscale**: `deploy/headscale/prompt.md` is a ready system prompt for deploying the self-hosted Headscale control server on an IPv6-only VPS (dual-stack front via Cloudflare Tunnel, public DERP map, reusable pre-auth key). Use an **ephemeral** reusable pre-auth key for runners so churned nodes auto-remove.

## Change discipline

- Keep changes small and localized.
- Work in the current local checkout and local state by default. Do not create git worktrees or new branches unless explicitly requested.
- Do not rename concepts casually; `runflow` vocabulary should remain stable across runtime, backend, and frontend.
- Do not commit generated artifacts, caches, model weights, temporary audio, transcripts, or local run outputs unless explicitly requested.
- If a task exposes a domain leak in `src/runflow`, fix the abstraction instead of patching around it.

# Coding Rules

## File and Folder Structure

* Keep each file under 300 lines.
* Keep each folder under 16 files.
* If a file or folder exceeds these limits, split the code into smaller modules.
* Prefer the single responsibility principle. When code does too many things, split it into focused units.

## Imports

* Always place imports at the top of the module.
* Avoid inline imports inside function bodies, type annotations, or interface fields.
* Inline imports are allowed only when there is a strict circular-dependency reason, and that reason must be documented.

## Avoid

* Avoid 1–3 line functions unless they meaningfully improve readability or express intent.
* Avoid nested class or function definitions.
* Avoid `.get()` when direct indexing with `[]` is appropriate.
* Do not use default values or fallback behavior unless explicitly requested or clearly justified.
* Avoid raw dictionaries and strings for structured data. Prefer dataclasses, Pydantic models, and enums.
* Avoid excessive checking. Prefer clear failures over silent defaults, silent skips, or hidden fallback behavior.
* Do not overuse `if` statements.
* Do not repeatedly check values that already have defaults or guaranteed invariants.

## If Statements

Use `if` statements only when both branches are valid, expected paths. avoid defensive coding. let if fail.

### Wrong

```rust
if condition {
    // happy path
} else {
    // "shouldn't happen" - silently ignored
}
```

### Right

```rust
assert!(condition, "invariant violated: ...");
```

Or:

```rust
return Err(LimboError::InternalError("unexpected state".into()));
```

Or:

```rust
unreachable!("impossible state: ...");
```

If only one branch should ever be reached, use an assertion, explicit error, or `unreachable!` instead of silently ignoring the impossible branch.

## Comments and Documentation

### Do

* Document why something exists, not what the code does.
* Document functions, structs, enums, and enum variants when useful.
* Explain why a decision, constraint, or workaround is necessary.

### Don’t

* Do not write comments that simply repeat the code.
* Do not reference AI conversations or prompts.
* Do not use temporal markers such as “added,” “existing code,” “new,” or “Phase 1.”
* Do not add comments or docstrings to unchanged code unless requested.

## Avoid Over-Engineering

* Make only the changes directly requested or clearly necessary.
* Do not add features beyond the request.
* Do not add error handling for impossible scenarios.
* Do not create abstractions for one-time operations.
* Prefer three similar lines over premature abstraction.


## Project Assumptions

* This is a greenfield project.
* Do not add legacy-code support, compatibility fallbacks, or migration paths unless explicitly requested.
