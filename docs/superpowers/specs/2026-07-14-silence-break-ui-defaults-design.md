# Silence Break UI Defaults Design

## Goal

Make `InsertSilenceBreaks` immediately usable when added in the workflow editor
and allow schema-generated decimal settings such as `0.01` to be entered
without the controlled input collapsing intermediate text.

## Node defaults

`InsertSilenceBreaksSettings` will export these defaults through the existing
runner schema path:

- `silence_threshold = 0.01`
- `window_size = 20` milliseconds
- `min_break_time = 100` milliseconds
- `insert_at_start = false`
- `insert_at_end = false`
- `drop_prob = 0.0`

New workflow nodes already clone `settings_defaults`, so no node-specific
frontend behavior is needed.

## Numeric schema input

Add a focused schema-number control under `src/frontend/src/shared/schema-form/`.
It receives the JSON Schema type, value, minimum, maximum, and change callback.
The control keeps a local string draft so valid intermediate input such as
`0.`, `0.0`, or `-` is not replaced by a parsed number during the same
keystroke.

For JSON Schema `number`, the native input uses `step="any"` and accepts any
decimal precision. For `integer`, it uses `step="1"` and only emits parsed
integers. Minimum and maximum constraints are forwarded to the native input.
When external graph state changes, the displayed draft synchronizes to it.

An empty or incomplete draft remains local and does not write `null` or an
invalid number into graph state. A complete finite value is emitted immediately.
The existing schema conversion helpers remain responsible for non-numeric
types.

## Scope

The fix applies to all fields rendered from JSON Schema as `number` or
`integer`. It does not change the separate reusable `NumberInput` component or
feature-specific numeric controls.

## Verification

Temporary frontend checks will cover multi-decimal entry, intermediate draft
preservation, integer-only emission, external-value synchronization, and
minimum/maximum attributes. Backend schema verification will assert the six
`InsertSilenceBreaks` defaults. The frontend typecheck/build and runner schema
export must pass. Temporary tests will be removed before completion.
