# Statistics Manual Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the statistics page from automatically selecting and loading its first saved entry while preserving explicit session selections.

**Architecture:** A pure selection reconciler keeps valid explicit IDs and clears stale IDs without inventing a selection. `StatisticsScreen` uses it when the entry list changes, renders an explicit unselected state, and disables deletion until a valid summary exists.

**Tech Stack:** React, TypeScript, Zustand, TanStack Query, Node type stripping, Vite.

## Global Constraints

- Initial `entryId` remains `null` when entries load.
- Preserve an explicitly selected valid ID for the current app session.
- Clear selected IDs absent from the refreshed entry list.
- Keep the detail query disabled for `null` IDs.
- Keep picker and compute controls visible while unselected.
- Run all project commands through `nix develop --command`.
- Keep regression scripts temporary and remove them before completion.

---

### Task 1: Selection Reconciliation

**Files:**
- Create: `src/frontend/src/features/statistics/selection.ts`
- Test: `/tmp/test_statistics_selection.ts`

**Interfaces:**
- Consumes: `currentId: string | null` and `entries: StatisticsSummary[] | undefined`.
- Produces: `reconcileStatisticsEntryId(currentId, entries): string | null`.

- [ ] **Step 1: Write the failing temporary regression**

Create a TypeScript script importing `reconcileStatisticsEntryId`. Assert `null` stays `null` with entries present, a valid ID survives, an unloaded list preserves the current ID, a stale ID becomes `null`, and an empty loaded list clears a selected ID.

- [ ] **Step 2: Verify RED**

Run:

```bash
nix develop --command node --experimental-strip-types /tmp/test_statistics_selection.ts
```

Expected: module-not-found failure because `selection.ts` does not exist.

- [ ] **Step 3: Implement the reconciler**

Create `selection.ts`, import `StatisticsSummary` as a type, preserve the current ID while entries are undefined, and otherwise return it only when it is non-null and present in the loaded list.

- [ ] **Step 4: Verify GREEN**

Run the temporary script again and expect exit 0.

---

### Task 2: Screen Behavior

**Files:**
- Modify: `src/frontend/src/features/statistics/StatisticsScreen.tsx`

**Interfaces:**
- Consumes: `reconcileStatisticsEntryId` from Task 1.
- Produces: manual report selection, a prompt while unselected, and disabled deletion without a valid selection.

- [ ] **Step 1: Replace automatic selection**

In the entry-list effect, reconcile the current ID and call `setEntryId` only when the reconciled value differs. Never read `entries[0]`.

- [ ] **Step 2: Add the unselected report state**

Below the sticky toolbar, render an `EmptyState` with title `Select a statistics entry` and a short picker instruction when `entryId` is `null`. Only show `Loading entry…` when a non-null entry is fetching.

- [ ] **Step 3: Disable deletion while unselected**

Pass `disabled={!summary}` to the delete `IconButton` and add disabled cursor/opacity classes locally so the control is visibly inactive.

- [ ] **Step 4: Run regression and frontend build**

Run:

```bash
nix develop --command node --experimental-strip-types /tmp/test_statistics_selection.ts
nix develop --command npm --prefix src/frontend run build
```

Expected: both commands exit 0.

---

### Task 3: Final Verification and Cleanup

**Files:**
- Test: `/tmp/test_statistics_selection.ts`

- [ ] **Step 1: Inspect the final diff and structural limits**

Run `git diff --check`, verify touched files remain below 300 lines, and verify the statistics feature folder remains below 16 files.

- [ ] **Step 2: Run fresh verification**

Run the regression and frontend production build from the repository root.

- [ ] **Step 3: Remove the temporary regression**

Delete `/tmp/test_statistics_selection.ts` with `apply_patch` and confirm it no longer exists.
