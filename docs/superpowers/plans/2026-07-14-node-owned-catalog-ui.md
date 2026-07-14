# Node-Owned Catalog UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the frontend pretrained-download catalog exclusively from runner-owned `CatalogDownload` node schema metadata and verify the Smart Turn download through a connected real graph.

**Architecture:** A typed runner registry owns catalog display and request metadata. `CatalogDownloadSettings` exports that registry through a JSON Schema extension; the existing backend schema route transports it unchanged, and the frontend parses and renders it dynamically. Runtime catalog tasks remain runner-owned.

**Tech Stack:** Python 3.12, Pydantic JSON Schema, FastAPI schema transport, React, TypeScript, TanStack Query, Nix development shell, Hugging Face snapshot download.

## Global Constraints

- Dependency direction is `runner nodes -> generated workflow schema -> backend transport -> frontend UI`.
- Backend and frontend must not define catalog entries.
- Run Python, frontend, runner, and CLI commands through `nix develop --command ...`.
- Keep files below 300 lines and folders below 16 files.
- Do not retain temporary tests, graph requests, or downloaded verification output in the repository.
- Verify node behavior through a real graph.

---

### Task 1: Runner-owned catalog registry and schema export

**Files:**
- Create: `src/runner/nodes/assets/catalog_runtime/entries.py`
- Modify: `src/runner/nodes/assets/catalog.py`
- Modify: `src/runner/nodes/assets/catalog_runtime/tasks.py`
- Test temporarily: `/tmp/test_catalog_schema.py`

**Interfaces:**
- Produces: `CatalogKey(StrEnum)`.
- Produces: `CatalogEntry(BaseModel)` with `name`, `file`, `group`, `catalog_key`, and `item`.
- Produces: `CATALOG_ENTRIES: tuple[CatalogEntry, ...]`.
- Produces: `catalog_entries_schema() -> list[dict[str, str]]`.
- Exports: `CatalogDownload.settings["x-catalog-items"]` in `/schema`.

- [ ] **Step 1: Write the failing schema regression**

Create `/tmp/test_catalog_schema.py`:

```python
from runflow.registry.node_registry import NodeRegistry
from runner.nodes.registry import register_runner_nodes


schema = register_runner_nodes(NodeRegistry()).to_schema()["CatalogDownload"]
items = schema["settings"]["x-catalog-items"]
assert len(items) >= 32
assert len({(item["catalog_key"], item["item"]) for item in items}) == len(items)
smart_turn = next(item for item in items if item["item"] == "pipecat-ai/smart-turn-v3")
assert smart_turn == {
    "name": "Smart Turn v3.2 · CPU ONNX",
    "file": "smart-turn-v3.2-cpu.onnx",
    "group": "Turn detection",
    "catalog_key": "turn_models",
    "item": "pipecat-ai/smart-turn-v3",
}
```

- [ ] **Step 2: Verify RED**

Run `nix develop --command python /tmp/test_catalog_schema.py`.
Expected: `KeyError: 'x-catalog-items'`.

- [ ] **Step 3: Create the typed registry**

Create `entries.py` with `CatalogKey`, immutable `CatalogEntry`, and one tuple containing every item currently split across `catalogDefaults.ts` and `logic.ts`. Preserve their exact name/file/group/catalog-key/item values. Add runner-supported Raon OpenTTS and Smart Turn entries:

```python
CatalogEntry(
    name="Raon OpenTTS · 1B",
    file="KRAFTON/Raon-OpenTTS-1B",
    group="TTS",
    catalog_key=CatalogKey.TTS_MODELS,
    item="raon_opentts",
),
CatalogEntry(
    name="Smart Turn v3.2 · CPU ONNX",
    file="smart-turn-v3.2-cpu.onnx",
    group="Turn detection",
    catalog_key=CatalogKey.TURN_MODELS,
    item="pipecat-ai/smart-turn-v3",
),
```

Serialize with `entry.model_dump(mode="json")`. Assert registry key/item pairs are unique at module load because duplicates make frontend node generation ambiguous.

- [ ] **Step 4: Export registry from the node schema**

Move `CatalogKey` usage in `catalog.py` to the registry and configure settings:

```python
class CatalogDownloadSettings(StrictSettings):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"x-catalog-items": catalog_entries_schema()},
    )
```

Derive `_TTS_DEFAULT_REPOS` and Smart Turn model/file constants in `tasks.py` from typed registry entries, so those runner defaults are not separately repeated.

- [ ] **Step 5: Verify GREEN and source quality**

Run:

```bash
nix develop --command python /tmp/test_catalog_schema.py
nix develop --command python -m compileall -q src/runner/nodes/assets
git diff --check -- src/runner/nodes/assets
```

Expected: all commands exit zero and every runner file remains below 300 lines.

---

### Task 2: Schema-generated frontend catalog

**Files:**
- Create: `src/frontend/src/features/checkpoints/catalog.ts`
- Delete: `src/frontend/src/features/checkpoints/catalogDefaults.ts`
- Modify: `src/frontend/src/features/checkpoints/logic.ts`
- Modify: `src/frontend/src/features/checkpoints/CheckpointsScreen.tsx`
- Modify: `src/frontend/src/shared/schema-form/types.ts`
- Test temporarily: `src/frontend/src/features/checkpoints/catalog.tmp.ts`

**Interfaces:**
- Consumes: `WorkflowSchema.nodes.CatalogDownload.settings["x-catalog-items"]`.
- Produces: `CatalogItem` with string `group`.
- Produces: `catalogItemsFromSchema(schema: WorkflowSchema) -> CatalogItem[]`.
- Produces: `groupCatalogItems(items: CatalogItem[]) -> Record<string, CatalogItem[]>`.

- [ ] **Step 1: Add a temporary parser regression**

Create a temporary TypeScript module that imports `catalogItemsFromSchema` and `groupCatalogItems`, constructs a minimal `WorkflowSchema` containing two catalog entries in distinct groups, and throws unless both entries and groups are returned. Run it with `nix develop --command node --experimental-strip-types src/frontend/src/features/checkpoints/catalog.tmp.ts`.

Expected RED: module resolution fails because `catalog.ts` does not exist.

- [ ] **Step 2: Implement schema parsing**

Extend `JsonSchema` with `"x-catalog-items"?: unknown`. In `catalog.ts`, validate that the extension is an array and that each entry contains string `name`, `file`, `group`, `catalog_key`, and `item` values. Map snake-case `catalog_key` to frontend `catalogKey`. Throw an actionable error containing `CatalogDownload x-catalog-items` for malformed schema; do not use a local fallback.

Implement grouping without a fixed group list:

```typescript
export function groupCatalogItems(items: CatalogItem[]): Record<string, CatalogItem[]> {
  const groups: Record<string, CatalogItem[]> = {};
  for (const item of items) (groups[item.group] ??= []).push(item);
  return groups;
}
```

- [ ] **Step 3: Remove hardcoded catalog ownership**

Delete `catalogDefaults.ts`. Remove `CatalogItem`, `CATALOG`, and catalog grouping from `logic.ts`. Update `CheckpointsScreen` to use `useWorkflowSchemaQuery`, parse catalog entries from its returned schema, and render the resulting dynamic groups. Preserve the existing card and download mutation behavior.

- [ ] **Step 4: Verify frontend and hardcoding removal**

Run:

```bash
nix develop --command npm --prefix src/frontend run build
nix develop --command node --experimental-strip-types src/frontend/src/features/checkpoints/catalog.tmp.ts
rg -n "CORE_CATALOG_ITEMS|export const CATALOG|catalogDefaults" src/frontend/src
```

Expected: build exits zero and `rg` returns no matches. Remove the temporary TypeScript regression after the build.

---

### Task 3: Smart Turn real download and connection verification

**Files:**
- Test temporarily: `/tmp/smart_turn_catalog_verification.json`

**Interfaces:**
- Verifies: `CatalogDownload.checkpoint -> SmartTurnPredict.checkpoint`.
- Verifies: `LoadAudio.audio -> SmartTurnPredict.audio`.

- [ ] **Step 1: Confirm upstream artifact metadata**

Confirm `pipecat-ai/smart-turn-v3` exposes `smart-turn-v3.2-cpu.onnx`, then compare the downloaded checkpoint file size and SHA-256 when available. The authoritative upstream SHA-256 is `2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f`.

- [ ] **Step 2: Submit the registered connected graph**

Build a temporary graph from `workflows/smart_turn_predict.json`, replacing its selected audio ID only if that record is unavailable. Submit through `POST /graphs/runs`, then inspect the run and node logs using the CLI through Nix.

- [ ] **Step 3: Verify success and clean up**

Confirm the run succeeds, the catalog node resolves a single `smart_turn` checkpoint, and `SmartTurnPredict` loads the ONNX file and emits a probability. Remove all temporary files.

Run final checks:

```bash
nix develop --command python -m compileall -q src/runner/nodes/assets src/runner/nodes/smart_turn
nix develop --command npm --prefix src/frontend run build
git diff --check
git status --short
```
