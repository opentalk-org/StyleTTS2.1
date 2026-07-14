# Node-Owned Catalog UI Design

## Goal

Make runner catalog node definitions the only source of truth for downloadable
catalog entries. The backend transports generated node schema without owning or
augmenting catalog definitions, and the frontend renders that schema without a
hardcoded catalog copy.

The dependency direction is strictly:

`runner nodes -> generated workflow schema -> backend transport -> frontend UI`

Neither the backend nor frontend supplies catalog definitions to runner nodes.

## Runner ownership

Add a typed catalog-entry model under
`src/runner/nodes/assets/catalog_runtime/`. Each entry declares its display name,
group, displayed file or repository, `CatalogDownload` catalog key, and requested
item. The registry includes every item currently shown by the checkpoint screen,
plus `pipecat-ai/smart-turn-v3` in a turn-detection group.

`CatalogDownloadSettings` publishes serialized entries as JSON Schema extension
metadata. Catalog task execution remains runner-owned and continues to validate
requested catalog keys and items. Adding or removing a catalog entry requires no
frontend edit.

## Backend transport

The existing workflow-schema route continues to call the registered node schema
export. It does not import catalog definitions, add a catalog endpoint, or merge
backend-owned values into the schema.

## Frontend generation

The checkpoint feature reads catalog entries from the `CatalogDownload` node's
settings schema. It validates the required primitive fields, groups entries by
the supplied group string, and generates the existing download cards.

Remove `catalogDefaults.ts` and the hardcoded `CATALOG` array. Catalog groups are
dynamic strings so adding a runner-owned group does not require a frontend type
or grouping update. A missing or malformed schema extension is a clear error;
the frontend does not silently restore a stale local fallback.

Download actions continue to construct a one-node `CatalogDownload` graph using
the selected entry and runtime defaults from the same workflow schema.

## Smart Turn verification

The runner entry points to `pipecat-ai/smart-turn-v3` and downloads only
`smart-turn-v3.2-cpu.onnx`. The upstream repository currently exposes that exact
file. Verification will submit a real graph connecting
`CatalogDownload.checkpoint` to `SmartTurnPredict.checkpoint`, provide real audio,
and confirm model loading and inference complete through the registered nodes.

## Tests

Temporary regression checks will verify that:

- the exported `CatalogDownload` schema contains all runner-owned entries;
- Smart Turn has `turn_models` and `pipecat-ai/smart-turn-v3` values;
- the frontend parser and dynamic grouping consume schema entries;
- no frontend hardcoded catalog remains;
- the frontend typecheck/build succeeds;
- the real Smart Turn graph downloads, connects, loads, and runs successfully.

Temporary test and graph files will be removed before completion.
