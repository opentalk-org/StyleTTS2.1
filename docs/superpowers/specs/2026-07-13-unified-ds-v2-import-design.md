# Unified ds_v2 Import Design

## Goal

Use one ds_v2 source node for metadata-only and byte-import workflows, and one
audio-record save node for external references and stored audio bytes. Both
nodes retain a single typed `Audio` input or output contract and batch
high-volume persistence work.

## Unified ds_v2 source

The registered node type is `HetznerDsV2Source`. It discovers sorted metadata
CSVs under `/home/ds_v2_metadata`, applies `row_offset` and `row_limit` globally
across their rows, and emits one `Audio` item per selected row. The settings no
longer expose a remote Parquet path.

The `import_audio` setting controls only whether emitted items contain bytes:

- `false` emits metadata, transcript segments, and the external storage
  reference without downloading Parquet files;
- `true` groups selected rows by the processed Parquet path derived from each
  discovered metadata CSV filename,
  downloads every required Parquet file once through the existing cache,
  validates row identity, and attaches the selected audio bytes.

Selection order and absolute source indices are identical in both modes.
Missing metadata fields, missing inferred Parquet files, invalid row indices,
and identity mismatches fail explicitly. There is no metadata fallback from
Parquet.

## Unified audio record save

`SaveAudioRecord` gains `storage_mode`, with values `stored` and `external`.
Both modes consume and emit the existing `Audio` and `SaveResult` ports.

`stored` requires audio bytes and uses the existing bulk packed-audio CRUD.
`bulk_import_packs` continues to select larger import packs. `external`
requires no audio bytes, constructs the external Parquet location from source
metadata, and uses the existing bulk external-record CRUD. The separate
`SaveExternalAudioRecord` node is removed from registration.

## Workflows and validation

Both ds_v2 smoke workflows use `HetznerDsV2Source` and `SaveAudioRecord` with
matching source/import and save/storage modes. Validation uses temporary checks
for selection and conversion helpers, then submits small real graphs through
`POST /graphs/runs` and inspects them with the project CLI. Temporary tests are
removed before completion.
