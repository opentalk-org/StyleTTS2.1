# Next Stage Workflow And Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the legacy workflow editor into the current React UI, generate runnable workflow JSON dynamically from node settings and audio selections, and support multiple backend-connected runners with RustFS-backed S3 settings.

**Architecture:** The frontend owns workflow composition and selection UX. The backend validates, persists, compiles, and starts typed graph requests. Node definitions and node schema registration live under `src/runner/`; `src/runflow/` stays a domain-agnostic runtime with no action vocabulary and no `tmp_nodes`.

**Tech Stack:** React, TypeScript, Tailwind v4, TanStack Query, Zustand, FastAPI, Pydantic, SQLAlchemy, NATS JetStream, PostgreSQL, RustFS/S3, Nix.

---

## Context Read

- `next_stage.md` requires dynamic workflow JSON from selected audio, dataset audio, or all audio into bucket load, processing, and save nodes.
- `src/backend/ui/static/` contains the legacy workflow editor to port; `src/frontend/src/features/workflows/WorkflowsScreen.tsx` is a placeholder.
- `src/runflow/ui/schema_export.py` exports registries, but current registration imports `src/runflow/tmp_nodes`; all node registration moves to `src/runner/nodes`.
- `src/backend/service.py` and `src/runner/worker.py` already start inline graphs over NATS and target commands to the claiming runner.
- `src/shared/db/workflows`, `src/shared/db/runners`, RustFS, and S3 env setup exist but need typed API wiring and env alignment.
- Repo rule: there are no tests; do not add tests. Use build, typecheck, import, compile, and smoke commands for verification.
- Follow-up requirement: processing nodes may stub model internals, but source/load/save nodes must use real shared DB and bucket data, and node settings must be production-facing.

## Execution Rule

After every checked step below, re-read the design with:

```bash
sed -n '1,220p' next_stage.md
```

Then compare the implementation against the remaining design gaps list in this plan. If a gap changes, update the checklist in the same commit as the implementation step.

## Design Gaps To Close

- [ ] React workflow editor is ported from `src/backend/ui/static` into current React.
- [ ] Workflow editor files are split into reusable schema form, graph logic, canvas, node, edge, inspector, picker, settings, and run components.
- [ ] Workflow editor uses a bottom canvas bar with one node picker button and one global settings button.
- [ ] Frontend can aggregate node settings into graph JSON and reuse workflow editor controls for workflow launch forms.
- [ ] Backend can persist and compile typed workflow definitions into `InlineGraphRunRequest`.
- [ ] Selected audio, dataset audio, and all audio are represented as typed launch sources.
- [ ] Bucket-backed audio loading and transcript saving are runner-owned graph nodes or compiled graph inputs, not runner actions.
- [ ] `src/runflow/tmp_nodes` is removed; all project/audio node definitions, datatypes, and schema registration live under `src/runner/nodes`.
- [ ] No `action` or `actions` vocabulary is introduced under `src/runflow` or `src/runner`.
- [ ] Backend exposes multiple runner status and manual extra runner registration.
- [ ] Cluster tab is renamed/reworked from Ray to runners.
- [ ] Nix dev and image entrypoints start one runner per detected GPU and keep local non-GPU dev usable.
- [ ] Storage settings include RustFS defaults and align env vars with `shared.storage.ObjectStoreConfig`.

## File Structure Plan

- Frontend: add shared schema-form controls, workflow feature files/components, runner cluster API/query, and storage settings API/query.
- Backend/shared: add workflow, runner, and settings routers/services/CRUD while keeping `src/backend/api.py` as root composition.
- Runner/nix: add `src/runner/nodes/`, split `worker.py`, remove `src/runflow/tmp_nodes`, and split `flake.nix` before runner-launch edits.

## Runner Custom Node Inventory

- Sources/IO: `SelectedAudioSource`, `DatasetAudioSource`, `AllAudioSource`, `LoadBucketAudio`, `SaveTranscript`, `SaveAudioArtifact`.
- ASR/text/segmentation: `WhisperTranscribe`, `CanaryTranscribe`, `ParakeetTranscribe`, `PhonemizeTranscript`, `VadDetect`, `CutAudioBySegments`, `SortformerDiarization`, `CutAudioBySpeakers`.
- Processing/writeback: `DeepFilterNetDenoise`, `NormalizeLoudness`, `CalculateAudioStats`, `AddAudioToDataset`, `RemoveAudioFromDataset`, `AssignVoice`, `DeleteAudioRecords`.
- Model-backed nodes may return deterministic stub outputs until real model integrations exist, but every node consumes or writes real IDs, metadata, and bucket-backed payloads.

---

### Task 1: Baseline And Guardrails

**Files:**
- Read: `next_stage.md`, `src/backend/ui/static/*.js`, `src/frontend/src/features/workflows/WorkflowsScreen.tsx`
- Read: `src/backend/service.py`, `src/runner/worker.py`, `src/runflow/tmp_nodes/register.py`, `src/runflow/tmp_nodes/audio/datatypes.py`

- [ ] Run `git status --short` and record unrelated modified files before editing.
- [ ] Run `find src/runflow/tmp_nodes -type f | sort` and record every node/datatype module that must move to `src/runner/nodes`.
- [ ] Run `npm run build` from `src/frontend`; expected result is either PASS or a current baseline failure to preserve in notes.
- [ ] Run `python -m compileall src/backend src/runner src/runflow src/shared`; expected result is either PASS or a current baseline failure to preserve in notes.
- [ ] Run the design re-read command from the Execution Rule and leave the Design Gaps list unchanged.

### Task 2: Move Nodes Out Of Runflow

**Files:**
- Create: `src/runner/nodes/__init__.py`
- Create: `src/runner/nodes/datatypes.py`
- Create: `src/runner/nodes/registry.py`
- Create: `src/runner/nodes/audio_sources.py`
- Create: `src/runner/nodes/audio_io.py`
- Create: `src/runner/nodes/asr.py`
- Create: `src/runner/nodes/audio_processing.py`
- Create: `src/runner/nodes/text_processing.py`
- Create: `src/runner/nodes/dataset_writeback.py`
- Modify: `src/runner/graphs.py`
- Modify: `src/backend/api.py`
- Delete: `src/runflow/tmp_nodes/`

- [ ] Move audio/project datatypes from `src/runflow/tmp_nodes/audio/datatypes.py` into `src/runner/nodes/datatypes.py`.
- [ ] Move node registration from `src/runflow/tmp_nodes/register.py` into `src/runner/nodes/registry.py`, and have backend schema export plus runner graph construction use that registry.
- [ ] Replace file-demo input settings like `directory`, `patterns`, `repeat_count`, and `sleep_sec` with real source/load/save settings: selected audio ids, dataset id, include virtual files, sample rate, channels, transcript format, overwrite mode, and target dataset/output settings.
- [ ] Create the custom node inventory from this plan with production-facing settings; model-backed nodes can stub outputs, but must operate on real audio IDs, metadata, or bucket-loaded payload metadata.
- [ ] Delete `src/runflow/tmp_nodes/` after imports are migrated; `rg -n "tmp_nodes" src` must return no matches.
- [ ] Run `rg -n "action|actions|Action" src/runflow src/runner`; expected result is no matches.
- [ ] Run `python -m compileall src/backend src/shared src/runner src/runflow`.
- [ ] Run the design re-read command and update Design Gaps for runner-owned nodes and tmp-node removal.

### Task 3: Shared React Schema Form

**Files:**
- Create: `src/frontend/src/shared/schema-form/types.ts`
- Create: `src/frontend/src/shared/schema-form/logic.ts`
- Create: `src/frontend/src/shared/schema-form/SchemaField.tsx`
- Create: `src/frontend/src/shared/schema-form/SchemaForm.tsx`
- Create: `src/frontend/src/shared/schema-form/SchemaObjectField.tsx`
- Create: `src/frontend/src/shared/schema-form/SchemaMapField.tsx`
- Modify: `src/frontend/src/shared/icons.tsx`

- [ ] Define typed TypeScript models for the JSON schema shapes exported by `/schema`: primitive fields, enums, arrays, nullable values, object fields, maps, defaults, `$defs`, and `$ref`.
- [ ] Port the legacy `schemaType`, `resolveSchemaRef`, `valueToText`, `textToValue`, and map-entry behavior from `forms.js` into pure functions in `logic.ts`.
- [ ] Build form components using existing `Input`, `Select`, `NumberInput`, `Toggle`, `Button`, and `Field`; keep every created file under 300 lines.
- [ ] Verify object fields and `resources: dict[str, float]` render as editable map rows for runtime config.
- [ ] Run `npm run build` from `src/frontend`.
- [ ] Run the design re-read command and update Design Gaps if schema-form reuse is complete.

### Task 4: Workflow Domain Types And Graph Logic

**Files:**
- Create: `src/frontend/src/features/workflows/types.ts`
- Create: `src/frontend/src/features/workflows/logic.ts`
- Create: `src/frontend/src/features/workflows/store.ts`
- Create: `src/frontend/src/features/workflows/templates.ts`

- [ ] Define `WorkflowSchema`, `WorkflowTypeSchema`, `WorkflowNodeSchema`, `WorkflowPortSchema`, `WorkflowGraph`, `WorkflowNode`, `WorkflowEdge`, `WorkflowRunContext`, and `WorkflowLaunchSource` TypeScript types.
- [ ] Port `typeAccepts`, `nodeAccent`, `addNode`, `deleteNode`, `renameNode`, `connect`, and `graphPayload` into pure functions that do not call React or the DOM.
- [ ] Add typed launch source variants for `selected_audio`, `dataset_audio`, and `all_audio`.
- [ ] Add a Whisper transcript template matching the design using runner-owned node types: audio source node, bucket load node, `Whisper`, and save transcript node.
- [ ] Store graph, selected nodes, wire draft, viewport, schema, runtime config, and active run id in Zustand.
- [ ] Run `npm run build` from `src/frontend`.
- [ ] Run the design re-read command and update Design Gaps for workflow JSON generation progress.

### Task 5: Workflow Editor Canvas Port

**Files:**
- Modify: `src/frontend/src/features/workflows/WorkflowsScreen.tsx`
- Create: `src/frontend/src/features/workflows/components/WorkflowCanvas.tsx`
- Create: `src/frontend/src/features/workflows/components/WorkflowNodeCard.tsx`
- Create: `src/frontend/src/features/workflows/components/WorkflowEdges.tsx`
- Create: `src/frontend/src/features/workflows/components/WorkflowInspector.tsx`
- Create: `src/frontend/src/features/workflows/components/WorkflowBottomBar.tsx`
- Create: `src/frontend/src/features/workflows/components/NodePickerPopover.tsx`
- Create: `src/frontend/src/features/workflows/components/RuntimeSettingsPopover.tsx`
- Create: `src/frontend/src/features/workflows/components/WorkflowRunPanel.tsx`

- [ ] Replace the placeholder screen with a full-height graph editor surface.
- [ ] Port node rendering, type-colored sockets, Bezier edges, pan, zoom, drag, wire creation, selection, delete, and marquee selection from the legacy UI.
- [ ] Implement the requested bottom bar: first button opens node picker popup, second button opens global runtime settings popup.
- [ ] Render node settings and node runtime settings with the shared schema form.
- [ ] Keep node lifecycle load/unload controls visually present but disabled until backend run wiring is in place.
- [ ] Run `npm run build` from `src/frontend`.
- [ ] Run the design re-read command and update Design Gaps for the React UI port.

### Task 6: Workflow API Query And Run Wiring

**Files:**
- Create: `src/frontend/src/features/workflows/api.ts`
- Create: `src/frontend/src/features/workflows/query.ts`
- Modify: `src/frontend/src/features/workflows/store.ts`
- Modify: `src/frontend/src/features/workflows/components/WorkflowRunPanel.tsx`
- Modify: `src/frontend/src/features/workflows/components/WorkflowNodeCard.tsx`

- [ ] Add API functions for `/schema`, `/graphs/runs`, `/runs`, `/runs/{run_id}`, `/runs/{run_id}/snapshot`, `/runs/{run_id}/graph`, and node lifecycle endpoints.
- [ ] Use TanStack Query for schema and run status; keep live UI state in Zustand only.
- [ ] Add WebSocket handling for `runner_status`, `run_status`, and `run_snapshot`, preserving the old UI behavior.
- [ ] Convert the current graph and runtime settings into `InlineGraphRunRequest` and start/stop runs from React.
- [ ] Show queued/completed/remaining/loaded node metrics from snapshots on node cards.
- [ ] Run `npm run build` from `src/frontend`.
- [ ] Run the design re-read command and update Design Gaps for frontend graph execution.

### Task 7: Typed Workflow Persistence And Compilation

**Files:**
- Modify: `src/shared/db/workflows/models.py`
- Modify: `src/shared/db/workflows/schemas.py`
- Modify: `src/shared/db/workflows/crud.py`
- Create: `src/backend/workflows/schemas.py`
- Create: `src/backend/workflows/service.py`
- Create: `src/backend/workflows/api.py`
- Modify: `src/backend/api.py`
- Modify: `src/shared/schemas.py`

- [ ] Replace raw workflow payload handling with Pydantic models for workflow definition, graph nodes, graph edges, runtime context, launch source, compile request, and compile response.
- [ ] Keep SQLAlchemy JSONB storage as a persistence detail by serializing/deserializing the typed workflow model at the CRUD boundary.
- [ ] Add backend routes for listing, saving, reading, compiling, and starting workflow definitions.
- [ ] Compile `selected_audio`, `dataset_audio`, and `all_audio` sources into typed graph input data or source-node parameters.
- [ ] Validate compiled graph requests by building the graph before publishing to NATS.
- [ ] Run `python -m compileall src/backend src/shared src/runner src/runflow`.
- [ ] Run the design re-read command and update Design Gaps for dynamic workflow JSON generation.

### Task 8: Real Data Workflow Node Wiring

**Files:**
- Modify: `src/runner/nodes/audio_sources.py`
- Modify: `src/runner/nodes/audio_io.py`
- Modify: `src/runner/nodes/asr.py`
- Modify: `src/runner/graphs.py`
- Modify: `src/backend/api.py`

- [ ] Add source nodes for selected audio ids, dataset audio ids, and all audio ids using shared DB CRUD facades.
- [ ] Add a bucket audio load node that reads audio bytes only through `shared.db.audio.crud` or shared asset CRUD.
- [ ] Add a transcript save node that writes output through a backend/shared facade rather than inlining storage details.
- [ ] Keep ASR and processing execution stubbed where needed, but emit deterministic transcript/audio records derived from real input ids and metadata.
- [ ] Confirm `src/runflow` scheduler, graph, ports, types, and policies are domain-agnostic after tmp-node removal.
- [ ] Run `rg -n "action|actions|Action" src/runflow src/runner`; expected result is no matches.
- [ ] Run `python -m compileall src/backend src/shared src/runner src/runflow`.
- [ ] Run the design re-read command and update Design Gaps for bucket-backed source/load/save nodes.

### Task 9: Multiple Runner Status And Runners UI

**Files:**
- Modify: `src/shared/db/runners/models.py`
- Modify: `src/shared/db/runners/schemas.py`
- Modify: `src/shared/db/runners/crud.py`
- Modify: `src/shared/schemas.py`
- Modify: `src/shared/jetstream.py`
- Create: `src/backend/runners/schemas.py`
- Create: `src/backend/runners/service.py`
- Create: `src/backend/runners/api.py`
- Modify: `src/backend/nats_bus.py`
- Modify: `src/backend/service.py`
- Create: `src/runner/heartbeat.py`
- Create: `src/runner/cli.py`
- Modify: `src/runner/worker.py`
- Modify: `src/frontend/src/features/cluster/ClusterScreen.tsx`
- Create: `src/frontend/src/features/cluster/api.ts`
- Create: `src/frontend/src/features/cluster/query.ts`

- [ ] Split runner CLI parsing and `main()` from `worker.py` into `runner/cli.py`, then update module entrypoint behavior.
- [ ] Add runner heartbeat messages over NATS with runner id, hostname, process id, GPU index, advertised resources, active run ids, and timestamp.
- [ ] Persist manual runner records and merge them with live heartbeat state in backend runner service.
- [ ] Add `GET /runners` and `POST /runners` for runner status and extra runner registration.
- [ ] Rename the Cluster screen content from Ray to runners and render runner cards/table with online, stale, busy, active runs, resources, and endpoint details.
- [ ] Run `python -m compileall src/backend src/shared src/runner`.
- [ ] Run `npm run build` from `src/frontend`.
- [ ] Run the design re-read command and update Design Gaps for multiple runners.

### Task 10: Nix Multi-GPU Runner Launch And RustFS Env Alignment

**Files:**
- Create: `nix/rustfs-package.nix`
- Create: `nix/frontend-static.nix`
- Create: `nix/runner-launch.sh`
- Modify: `flake.nix`
- Modify: `nix/runflow-dev.sh`
- Modify: `nix/entrypoint.sh`

- [ ] Move RustFS package construction out of `flake.nix` into `nix/rustfs-package.nix`.
- [ ] Move frontend static package construction out of `flake.nix` into `nix/frontend-static.nix`.
- [ ] Add `nix/runner-launch.sh` with GPU count detection using `RUNFLOW_RUNNER_COUNT` first, then `nvidia-smi -L`, then one local runner when no GPU is present.
- [ ] Launch runners as `runner-gpu-0`, `runner-gpu-1`, etc. with `CUDA_VISIBLE_DEVICES` set per process and `RUNFLOW_RUNNER_GPU_INDEX` exported.
- [ ] Export `RUNFLOW_S3_BUCKET`, `RUNFLOW_S3_ENDPOINT_URL`, `RUNFLOW_S3_REGION`, `RUNFLOW_S3_ACCESS_KEY_ID`, and `RUNFLOW_S3_SECRET_ACCESS_KEY` from dev and image entrypoints with RustFS defaults.
- [ ] Keep `flake.nix`, `nix/runflow-dev.sh`, and `nix/entrypoint.sh` under 300 lines after edits.
- [ ] Run `nix flake check` if available in the local environment; otherwise run `nix develop --command bash -lc 'command -v runflow-dev'`.
- [ ] Run the design re-read command and update Design Gaps for GPU runner startup and S3 defaults.

### Task 11: Persisted Storage Settings UI

**Files:**
- Create: `src/shared/db/settings/models.py`
- Create: `src/shared/db/settings/schemas.py`
- Create: `src/shared/db/settings/crud.py`
- Create: `src/shared/db/settings/__init__.py`
- Modify: `src/shared/db/connection.py`
- Modify: `src/shared/storage/object_store.py`
- Create: `src/backend/settings/api.py`
- Modify: `src/backend/api.py`
- Create: `src/frontend/src/features/settings/api.ts`
- Create: `src/frontend/src/features/settings/query.ts`
- Modify: `src/frontend/src/features/settings/store.ts`
- Modify: `src/frontend/src/features/settings/SettingsScreen.tsx`

- [ ] Add a typed storage settings model with RustFS defaults: bucket `runflow`, endpoint `http://127.0.0.1:9000`, region `us-east-1`, access key `runflow`, secret key `runflow-secret`.
- [ ] Load object store config from persisted settings when backend code has a database session and from `RUNFLOW_S3_*` env for runner/low-level storage code.
- [ ] Add backend `GET /settings/storage` and `PUT /settings/storage`.
- [ ] Add a Settings screen section for S3/RustFS with endpoint, bucket, region, access key, and secret key controls.
- [ ] Use TanStack Query for persisted settings and keep local Zustand for UI-only preferences.
- [ ] Run `python -m compileall src/backend src/shared`.
- [ ] Run `npm run build` from `src/frontend`.
- [ ] Run the design re-read command and update Design Gaps for storage settings.

### Task 12: End-To-End Smoke Verification

**Files:**
- Read: all files modified by Tasks 2-11
- Read: `next_stage.md`

- [ ] Run `npm run build` from `src/frontend`; expected result: PASS.
- [ ] Run `python -m compileall src/backend src/shared src/runner src/runflow`; expected result: PASS.
- [ ] Run `test ! -d src/runflow/tmp_nodes`; expected result: PASS.
- [ ] Run `rg -n "tmp_nodes|action|actions|Action" src/runflow src/runner`; expected result: no matches.
- [ ] Start the local stack with `nix develop --command runflow-dev`.
- [ ] Open `http://127.0.0.1:8000/ui` and confirm the Workflow screen renders the React editor, node picker popup, global settings popup, and template graph.
- [ ] Start a Whisper transcript workflow from selected mock audio or a dataset source and confirm a run appears, is claimed by one runner, and streams node metrics.
- [ ] Open the Runners screen and confirm all detected runner processes appear with heartbeat state.
- [ ] Open Settings and confirm RustFS/S3 defaults are visible and editable.
- [ ] Stop the local stack with Ctrl-C.
- [ ] Run the design re-read command and verify every Design Gap is checked or explicitly noted as a remaining out-of-scope decision.
