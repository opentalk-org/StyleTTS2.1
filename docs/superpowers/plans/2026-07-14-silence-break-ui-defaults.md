# Silence Break UI Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export useful `InsertSilenceBreaks` defaults and make JSON Schema number fields accept arbitrary decimal precision without losing intermediate input.

**Architecture:** Add defaults at the Pydantic settings source so existing schema export and workflow node creation pick them up automatically. Split numeric draft parsing from the React control, allowing temporary tests to verify parsing while the component owns focus-aware draft synchronization.

**Tech Stack:** Python, Pydantic, React, TypeScript, native HTML number inputs, Node 22 type stripping, Vite.

## Global Constraints

- Work in the current checkout and preserve unrelated dirty changes.
- Run Python, Node, and npm only through `nix develop --command ...`.
- Keep files below 300 lines and folders below 16 files.
- Do not add dependencies or a new test framework.
- Temporary tests must be removed before completion.

---

### Task 1: InsertSilenceBreaks defaults
**Files:**
- Modify: `src/runner/nodes/audio_segments/silence_breaks.py`
- Test temporarily: `.tmp_tests/test_silence_break_defaults.py`

**Interfaces:**
- Produces: Pydantic defaults and `settings_defaults` for all six node settings.

- [ ] **Step 1: Write a failing schema-default test**

```python
expected = {
    "silence_threshold": 0.01,
    "window_size": 20,
    "min_break_time": 100,
    "insert_at_start": False,
    "insert_at_end": False,
    "drop_prob": 0.0,
}
assert create_node_registry().to_schema()["InsertSilenceBreaks"]["settings_defaults"] == expected
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `nix develop --command python .tmp_tests/test_silence_break_defaults.py`

Expected: failure because every field is currently required and defaults export as an empty object.

- [ ] **Step 3: Add the approved defaults**

```python
class InsertSilenceBreaksSettings(StrictSettings):
    silence_threshold: float = Field(default=0.01, ge=0.0, le=1.0, title="Silence RMS threshold")
    window_size: int = Field(default=20, gt=0, title="RMS window size (ms)")
    min_break_time: int = Field(default=100, gt=0, title="Minimum break time (ms)")
    insert_at_start: bool = Field(default=False, title="Insert at start")
    insert_at_end: bool = Field(default=False, title="Insert at end")
    drop_prob: float = Field(default=0.0, ge=0.0, le=1.0, title="Break drop probability")
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `nix develop --command python .tmp_tests/test_silence_break_defaults.py`

Expected: one passing check.

### Task 2: Numeric draft parsing
**Files:**
- Create: `src/frontend/src/shared/schema-form/number.ts`
- Test temporarily: `.tmp_tests/test_schema_number.ts`

**Interfaces:**
- Produces: `numericDraftValue(draft: string, type: "integer" | "number") -> number | undefined`.
- Produces: `numericStep(type: "integer" | "number") -> 1 | "any"`.

- [ ] **Step 1: Write a failing Node test**

```typescript
import assert from "node:assert/strict";
import { numericDraftValue, numericStep } from "../src/frontend/src/shared/schema-form/number.ts";

assert.equal(numericDraftValue("0.", "number"), undefined);
assert.equal(numericDraftValue("0.001", "number"), 0.001);
assert.equal(numericDraftValue("1.2", "integer"), undefined);
assert.equal(numericDraftValue("12", "integer"), 12);
assert.equal(numericStep("number"), "any");
assert.equal(numericStep("integer"), 1);
```

- [ ] **Step 2: Run the test and confirm the missing module failure**

Run: `nix develop --command node --experimental-strip-types .tmp_tests/test_schema_number.ts`

Expected: failure because `number.ts` does not exist.

- [ ] **Step 3: Implement the pure helpers**

```typescript
export type NumericSchemaType = "integer" | "number";

export function numericDraftValue(draft: string, type: NumericSchemaType): number | undefined {
  if (!draft || draft.endsWith(".") || draft === "-" || draft === "+") return undefined;
  if (type === "integer" && !/^[+-]?\d+$/.test(draft)) return undefined;
  const value = Number(draft);
  return Number.isFinite(value) ? value : undefined;
}

export function numericStep(type: NumericSchemaType): 1 | "any" {
  return type === "integer" ? 1 : "any";
}
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `nix develop --command node --experimental-strip-types .tmp_tests/test_schema_number.ts`

Expected: the script exits zero.

### Task 3: Focus-aware schema number control
**Files:**
- Create: `src/frontend/src/shared/schema-form/SchemaNumberField.tsx`
- Modify: `src/frontend/src/shared/schema-form/SchemaField.tsx`

**Interfaces:**
- Consumes: `numericDraftValue`, `numericStep`, schema minimum/maximum, external numeric value.
- Produces: native numeric control that retains incomplete drafts and emits only valid values.

- [ ] **Step 1: Add the component**

Use `useEffect`, `useRef`, and `useState`. Initialize the draft with
`valueToText(value, schema)`. While focused, retain the exact input string.
Emit only values returned by `numericDraftValue`. On blur, stop editing and
normalize the draft to the current external value. When not editing, synchronize
external value changes through an effect. Pass `min`, `max`, and `step` to the
native input.

```tsx
<Input
  filled
  className="h-9"
  type="number"
  value={draft}
  min={schema.minimum}
  max={schema.maximum}
  step={numericStep(type)}
  onFocus={() => { editing.current = true; }}
  onBlur={() => { editing.current = false; setDraft(valueToText(value, schema)); }}
  onChange={(event) => updateDraft(event.target.value)}
/>
```

- [ ] **Step 2: Route numeric schema fields to the component**

In `SchemaField`, return `SchemaNumberField` for `integer` and `number` before
the generic text rendering path. Keep booleans, enums, objects, and strings
unchanged.

- [ ] **Step 3: Run frontend typecheck and build**

Run: `nix develop --command npm --prefix src/frontend run build`

Expected: TypeScript and Vite build exit zero.

### Task 4: Final verification and cleanup
**Files:**
- Remove: `.tmp_tests/test_silence_break_defaults.py`
- Remove: `.tmp_tests/test_schema_number.ts`

- [ ] **Step 1: Verify backend defaults and frontend behavior checks**

Run both temporary test commands, then run the frontend build and
`nix develop --command python -m compileall -q src/runner/nodes/audio_segments/silence_breaks.py`.

- [ ] **Step 2: Verify the live schema**

Request `/schema` from the shared stack and assert that
`nodes.InsertSilenceBreaks.settings_defaults` equals the six approved values.

- [ ] **Step 3: Remove temporary checks and inspect scope**

Remove `.tmp_tests` and confirm `git diff --check`, frontend build, file limits,
and absence of generated temporary artifacts. Leave all unrelated changes
untouched and do not commit implementation files unless requested.
