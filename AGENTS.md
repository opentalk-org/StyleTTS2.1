# AGENTS.md

This file is the onboarding note for coding agents working on this repository. Keep it short and update it only for rules that should apply to nearly every task.

## Project purpose

This project is a ComfyUI-style workflow system for typed, batched, concurrent pipelines. The core package is `runflow`, a domain-agnostic runtime for node graphs with custom datatypes, typed ports, batching, generic resource policies, bounded queues, artifact storage, and resumable execution. Audio processing is one target workflow, not a runtime assumption.

## Repository layout

- `src/runflow/` — core graph/runtime library. This must stay domain-agnostic.
- `src/backend/` — backend/API layer that loads workflows, persists status, and exposes runtime results. it manage runners and serve data for frontend.
- `src/runner` - runner that runs workflows using runflow, it expose api for it. it contain custom nodes
- `src/frontend/` — visual editor/client UI for nodes, typed sockets, workflow editing, and job monitoring.

## Technology Stack

- backend: fastapi, sqlalchemy, alembic, pydantic, postgresql
- frontend: react, tanstack query/form, tailwind css, zustand for local state
- runflow: pydantic
- runner: fastapi, sqlalchemy (the same db as backend, it use backend files like models/services), pydantic 

## Architecture rules

- Do not add audio-specific assumptions to `src/runflow`. Use generic names such as item, artifact, datatype, port, node, resource, queue, batch, window, and task.
- Nodes declare capabilities through metadata: input/output ports, datatypes, batch policy, resource policy, and cache policy. Avoid hardcoding behavior in the scheduler based on node names.
- Resource scheduling is generic. Prefer `resources={"io": 1}`, `resources={"accelerator": 1, "vram_gb": 8}`, or project-defined resource keys over CPU/GPU-specific node classes.
- Windowing is generic. A window is a bounded set of source items based on count, cost, or caller-provided metadata; it must not depend on audio files.
- Keep node lifecycle separate from node logic. Loading/unloading belongs behind the node manager, not scattered through the scheduler.
- Prefer artifact references over passing large payloads in memory. Branches should reuse stored artifacts instead of duplicating data.

## Database and stored files

- Backend and runner code must access PostgreSQL through the shared CRUD facades in `src/shared/db/<feature>/crud.py`. Do not bypass them with ad hoc SQLAlchemy queries unless the feature CRUD does not exist yet.
- Use `shared.db.connection.database_session` for SQLAlchemy sessions and keep transaction boundaries inside CRUD/service functions.
- Audio file bytes are managed by `src/shared/db/audio/crud.py`. Call the public single or bulk audio CRUD functions; callers should not manage pack files, byte offsets, stale bytes, or pruning directly. use bulk options if audio files can be counted in hundrends.
- Checkpoints and extra files are managed by `src/shared/db/assets/crud.py`. Checkpoints are folder artifacts: create/update them from a folder path and use `get_checkpoint_path` to get the cached local folder. Extra files are single-object artifacts; use `get_extra_file_path` when a local cached file path is needed.
- PostgreSQL is the source of truth for metadata and bucket object keys. Bucket objects and local caches are implementation details behind shared CRUD.

## Project structure

- Use feature-based structure inside `src/frontend`, `src/backend`, and `src/runflow`. Avoid broad technical folders such as `models/`, `schemas/`, `components/`, or `utils/` at the package root when the code belongs to one feature.
- Keep each feature split into clear layers. A feature may contain local files such as `schemas.py`, `models.py`, `crud.py`, `actions.py`, `service.py`, `api.py`, `query.ts`, `logic.ts`, and `components.tsx` as appropriate for that feature.
- Frontend features should separate API calls, query/cache hooks, state or domain logic, and rendering components. Prefer names such as `api`, `query`, `logic`, and `components` within the feature.
- Backend features should separate request/response schemas, database models, persistence operations, actions or services, and route/API wiring. Prefer names such as `schemas`, `models`, `crud`, `actions`, `service`, and `api` within the feature.
- `runflow` features should keep domain concepts local to the capability they implement, with schemas, datatypes, ports, policies, execution logic, and tests grouped by feature instead of by technical file type.

## Frontend implementation

- Stack is already installed under `src/frontend`: Vite + React + TypeScript, Tailwind v4, TanStack Query, TanStack Form, Zustand. Do not add a new toolchain.
- Tailwind is v4 and **config-less** (`@import "tailwindcss"` in `src/frontend/src/index.css`). There is no `tailwind.config.js`. Define the color palette and design tokens in an `@theme` block in `index.css`, then use them as normal Tailwind utility classes.
- Reusable, domain-agnostic UI (buttons, inputs, badges, table/virtual-list primitives, layout shell) lives in a single `src/frontend/src/shared/` folder. Reuse these instead of re-styling per feature. Do not scatter shared primitives across features.
- Each feature folder under `src/frontend/src/features/<feature>/` keeps its own `api.ts`, `query.ts`, `logic.ts`, and `components.tsx` (split further into a local `components/` folder when a feature grows past the file-size limit).
- Large lists (datasets, audio files, jobs — target is ~5 million rows) **must** be virtualized with `@tanstack/react-virtual`. Never render a full row set into the DOM; window it. Assume any list view can hit millions of rows.
- Zustand holds local UI/client state; TanStack Query owns server state and caching. Do not duplicate server data into Zustand.
- Current milestone is a UI scaffold: features use mockup data and mockup actions (no real backend calls yet). Keep mock data behind the same `api.ts`/`query.ts` seam a real backend would use, so wiring the backend later is a drop-in swap.

## Development workflow

- Before changing behavior, find the relevant test, example, or smoke workflow and run it before and after the change.
- Start the local NATS JetStream, backend, and one runner with `nix develop --command runflow-dev`.
- Open the UI at `http://127.0.0.1:8000/ui`; stop the local stack with Ctrl-C.
- For backend or frontend changes, use the package manager and commands already present in that subproject. Do not introduce a new toolchain unless requested.
- When adding a new node type, update the registry/schema export path so the frontend can discover its ports and parameters.
- THERE IS NO TESTS AND DONT ADD THEM.

## Change discipline

- Keep changes small and localized.
- Do not rename concepts casually; `runflow` vocabulary should remain stable across runtime, backend, and frontend.
- Do not commit generated artifacts, caches, model weights, temporary audio, transcripts, or local run outputs unless explicitly requested.
- If a task exposes a domain leak in `src/runflow`, fix the abstraction instead of patching around it.
