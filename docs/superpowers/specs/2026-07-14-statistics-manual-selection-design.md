# Statistics Manual Selection Design

## Goal

Opening the statistics page must not automatically load the first saved statistics entry. A report loads only after the user explicitly selects it.

## Selection behavior

The Zustand statistics UI store remains the owner of the selected entry ID. Its initial value stays `null`, and the statistics screen must not replace `null` with the first fetched entry.

An explicitly selected entry remains selected for the current application session, including navigation away from and back to the statistics page. If the selected entry is deleted or disappears from the refreshed entry list, the screen clears the stale ID to `null`.

## Empty state

When saved entries exist but `entryId` is `null`, the toolbar remains visible with the entry picker and compute controls. The report area displays an empty state prompting the user to select a statistics entry. It must not display `Loading entry…`, and the entry-detail query remains disabled.

The delete control is disabled while no valid entry is selected. Existing behavior for an empty statistics library remains unchanged.

## Validation

Extract the entry-ID reconciliation rule into a small pure function so a temporary regression can verify:

- `null` remains `null` even when entries exist;
- a valid explicit selection remains selected;
- a stale selection becomes `null`;
- an empty entry list clears a selection.

Run the frontend production build after wiring the screen to the reconciler and adding the unselected empty state.

## Scope

This change does not persist the selection across browser sessions, add URL routing, alter entry ordering, change statistics queries, or change report computation.
