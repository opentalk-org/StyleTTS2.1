# Node Build Failure Highlighting Design

## Goal

Show a graph-construction validation failure on the exact current-format workflow node that failed, while keeping failures without node identity as general run errors.

## Design

`build_inline_graph` already wraps node construction errors in `GraphNodeBuildError`, which carries `node_id` and `node_type`. `RunExecution` will convert that exception into the existing current `RunEvent(kind="node_failed")` representation before marking the job terminal. `RunnerStateBuffer` and `RunEventStore` will then persist the node snapshot as `failed`, which is the state already consumed by the frontend card.

No frontend inference, old graph parsing, compatibility fallback, schema migration, or legacy NATS behavior will be introduced. Non-node exceptions continue to populate only the job-level error.

## Verification

A throwaway asynchronous `unittest` regression test will force `build_inline_graph` to raise `GraphNodeBuildError` and assert that the state buffer receives a node-scoped failure event followed by the terminal run failure. The test will be run through `nix develop` before and after implementation, then removed as required by the repository test policy.
