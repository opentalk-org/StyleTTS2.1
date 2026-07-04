# AGENTS.md

This file is the onboarding note for coding agents working on this repository. Keep it short and update it only for rules that should apply to nearly every task.

## Project purpose

This project is a ComfyUI-style workflow system for typed, batched, concurrent pipelines. The core package is `runflow`, a domain-agnostic runtime for node graphs with custom datatypes, typed ports, batching, generic resource policies, bounded queues, artifact storage, and resumable execution. Audio processing is one target workflow, not a runtime assumption.

## Repository layout

- `src/runflow/` — core graph/runtime library. This must stay domain-agnostic.
- `src/backend/` — backend/API layer that loads workflows, runs jobs, persists status, and exposes runtime results, it also contain project specific nodes like whisper.
- `src/frontend/` — visual editor/client UI for nodes, typed sockets, workflow editing, and job monitoring.

## Technology Stack

- backend: fastapi, sqlalchemy, alembic, pydantic, postgresql
- frontend: react, tanstack query/form, tailwind css, zustand for local state
- runflow: pydantic

## Architecture rules

- Do not add audio-specific assumptions to `src/runflow`. Use generic names such as item, artifact, datatype, port, node, resource, queue, batch, window, and task.
- Nodes declare capabilities through metadata: input/output ports, datatypes, batch policy, resource policy, and cache policy. Avoid hardcoding behavior in the scheduler based on node names.
- Resource scheduling is generic. Prefer `resources={"io": 1}`, `resources={"accelerator": 1, "vram_gb": 8}`, or project-defined resource keys over CPU/GPU-specific node classes.
- Windowing is generic. A window is a bounded set of source items based on count, cost, or caller-provided metadata; it must not depend on audio files.
- Keep node lifecycle separate from node logic. Loading/unloading belongs behind the node manager, not scattered through the scheduler.
- Prefer artifact references over passing large payloads in memory. Branches should reuse stored artifacts instead of duplicating data.

## Project structure

- Use feature-based structure inside `src/frontend`, `src/backend`, and `src/runflow`. Avoid broad technical folders such as `models/`, `schemas/`, `components/`, or `utils/` at the package root when the code belongs to one feature.
- Keep each feature split into clear layers. A feature may contain local files such as `schemas.py`, `models.py`, `crud.py`, `actions.py`, `service.py`, `api.py`, `query.ts`, `logic.ts`, and `components.tsx` as appropriate for that feature.
- Frontend features should separate API calls, query/cache hooks, state or domain logic, and rendering components. Prefer names such as `api`, `query`, `logic`, and `components` within the feature.
- Backend features should separate request/response schemas, database models, persistence operations, actions or services, and route/API wiring. Prefer names such as `schemas`, `models`, `crud`, `actions`, `service`, and `api` within the feature.
- `runflow` features should keep domain concepts local to the capability they implement, with schemas, datatypes, ports, policies, execution logic, and tests grouped by feature instead of by technical file type.

## Development workflow

- Before changing behavior, find the relevant test, example, or smoke workflow and run it before and after the change.
- For backend or frontend changes, use the package manager and commands already present in that subproject. Do not introduce a new toolchain unless requested.
- When adding a new node type, update the registry/schema export path so the frontend can discover its ports and parameters.
- THERE IS NO TESTS AND DONT ADD THEM.

## Change discipline

- Keep changes small and localized.
- Do not rename concepts casually; `runflow` vocabulary should remain stable across runtime, backend, and frontend.
- Do not commit generated artifacts, caches, model weights, temporary audio, transcripts, or local run outputs unless explicitly requested.
- If a task exposes a domain leak in `src/runflow`, fix the abstraction instead of patching around it.
