# ClickHouse CRUD cutover plan

## Scope

Replace PostgreSQL access under `src/` for the tables now owned by ClickHouse:

- `audio_files`
- `audio_segments`, including PostgreSQL `alignments`
- `datasets`
- `dataset_audio_files`
- `bucket_files`, including PostgreSQL `waveform_packs`
- `assets`, replacing PostgreSQL `checkpoints` and `extra_files`
- `audio_waveforms`
- `statistics_entries`
- `configs`
- `mos_comparisons`

Keep PostgreSQL CRUD for jobs, node state, runners, workflows, initialization,
settings, notifications, and Alembic state. Do not route these control-plane
tables through ClickHouse.

## Target boundaries

Keep feature facades under `src/shared/db/<feature>/crud.py`. Backend and runner
code must call those facades and must not import ClickHouse clients directly.

Add minimal shared ClickHouse infrastructure under `src/shared/db/clickhouse/`:

| Module | Responsibility |
| --- | --- |
| `client.py` | Client lifecycle, configuration, health checks, and common execution settings |
| `mutations.py` | Execute typed update/delete mutations with `mutations_sync = 2` |

Each feature CRUD owns its SQL, inserts, row decoding, and DTOs. Do not create a
generic repository, query-builder abstraction, or generic command model. Return
Pydantic models or frozen dataclasses, never driver rows or raw dictionaries.
Remove SQLAlchemy entities for retired tables after cutover.

PostgreSQL `Session` arguments disappear from ClickHouse-only facade functions.
Mixed operations are split into an application service that explicitly owns the
PostgreSQL transaction and ClickHouse operation. A ClickHouse client must never
be hidden inside `database_session`. CRUD obtains it from the ClickHouse client
provider; callers do not pass database-driver objects between features.

## Read and write rules

### ReplacingMergeTree tables

For datasets, memberships, assets, waveforms, statistics, configs, and MOS
comparisons:

- Insert replacements with a strictly increasing `updated_at` for the logical
  key. A retry reuses the original timestamp and payload.
- Point and small-ID reads may use `FINAL` after applying the key filter.
- Broad reads collapse rows with canonical `argMax` queries or a measured
  projection.
- Rare deletions use explicit ClickHouse mutations. Serialize them behind all
  earlier writes for the same key and wait for completion before acknowledging
  success, preventing a delayed insert from resurrecting a row.

### Segments

`audio_segments` is plain `MergeTree` with one logical row per
`(audio_file_id, id)`. Its physical order is
`(updated_at, audio_file_id, id)`. Reads never use `FINAL`.

- Insert new segments in blocks.
- Update text, phonemes, annotations, alignment, speaker, or position with
  explicit update mutations filtered by both identity columns and
  `mutations_sync = 2`.
- Delete by both identity columns.
- Replace a transcript by deleting all rows for its audio ID, waiting for the
  mutation, and inserting the replacement block.
- Serialize segment writes per audio ID. A replacement may briefly expose an
  empty transcript; callers publish the audio ID only after completion.
- Strip PostgreSQL `metadata._source` during backfill. Store typed segment data
  once and retain only extra metadata in `metadata`.

### Audio files

`audio_files` is plain `MergeTree` with one physical row per `id`. Reads never
use `FINAL`. Insert new files in blocks and use explicit synchronous mutations
for rename, model score, language, prompts, timestamps, and storage locations.
Each mutation updates the changed columns and `updated_at` together. Serialize
writes per audio ID and return only after every replica has applied the mutation.
Preserve `updated_at DateTime64(9)` as catalog data, not a replacement version.
The physical audio order is `(updated_at, id, duration)`.

## Existing module disposition

| Current module | Target design |
| --- | --- |
| `audio/catalog.py` | ClickHouse catalog queries returning `AudioFileRecord` pages |
| `audio/files.py` | Audio metadata insert, synchronous mutation, deletion, and object reads |
| `audio/segments.py` | Plain-MergeTree segment insert and synchronous mutations |
| `audio/annotations/crud.py` | Bulk audio/segment mutation operations |
| `audio/segment_previews.py` | One ClickHouse query per audio page |
| `audio/segment_catalog.py` | Dataset-membership-to-segment ClickHouse scan |
| `audio/speaker_annotations.py` | ClickHouse scans and bounded mutations |
| `audio/storage_locations.py` | ClickHouse joins returning typed object ranges |
| `audio/packed.py`, `pack_store.py` | Complete immutable pack builder and publisher |
| `audio/maintenance.py` | Reference scan, compaction publication, and object GC |
| `datasets/crud.py` | Dataset and membership ClickHouse facade |
| `assets/crud.py`, `file_store.py` | Unified asset/config metadata plus object cache facade |
| `waveforms/crud.py`, `pack_store.py` | Waveform metadata and immutable pack facade |
| `statistics/crud.py` | Statistics ClickHouse facade |
| `mos/crud.py` | MOS ClickHouse facade; absorb and remove `mos/mutations.py` |

Keep `audio/crud.py` and feature `__init__.py` files as explicit public export
surfaces. Do not retain PostgreSQL and ClickHouse implementations behind runtime
switches after cutover.

### Packed objects

Audio and waveform packs are immutable `bucket_files` rows. Replace the current
open-pack database allocator with this sequence:

1. Build a complete pack in a local temporary file and assign byte offsets.
2. Upload the completed object.
3. Insert its `bucket_files` row.
4. Publish audio or waveform location rows.

Compaction creates a new pack and publishes new locations. It never updates a
pack row. Garbage collection deletes object bytes only after no current location
references the pack and the reader grace period has elapsed.

## Facade-by-facade conversion

### Audio catalog and bytes

Convert `src/shared/db/audio/catalog.py`, `files.py`, `external.py`,
`storage_locations.py`, and the exports in `crud.py`.

- Replace ORM `AudioFile` returns with `AudioFileRecord`.
- Implement ID lookup and bulk lookup with direct key-filtered queries.
- Implement catalog paging by `updated_at DESC, id` and `duration DESC, id`.
- Remove the `name` and `segments` sort modes from backend schemas, frontend
  state, toolbar options, cursor encoding, and `_audio_sort`.
- Preserve name and metadata text filtering initially; benchmark representative
  data before adding a text index.
- Read bucket paths and byte ranges by joining `audio_files` to `bucket_files`
  inside ClickHouse, then read bytes through object storage.
- Insert audio metadata only after its immutable pack exists. Compaction updates
  the existing audio location with a mutation after the replacement pack exists.
- Bulk deletion removes segments, waveforms, memberships, and policy-selected
  MOS rows before deleting audio rows and eventually collecting packs.

Convert `pack_store.py`, `packed.py`, `maintenance.py`, and both maintenance
CLIs to immutable pack construction. Remove all `used_bytes`, `sealed`, active
pack, row-lock, and decrement-counter logic.

### Segments and annotations

Convert `segments.py`, `segment_previews.py`, `segment_catalog.py`,
`speaker_annotations.py`, and `annotations/crud.py`.

- Map PostgreSQL `source_id` to ClickHouse `id`.
- Flatten `alignments.data` into `audio_segments.alignment`.
- Read transcript rows by audio ID and order by `position`.
- Use one bulk query for previews across an audio page.
- Express dataset training and speaker scans as ClickHouse joins among
  memberships, audio, and segments.
- Replace bulk speaker assignment and annotation writes with bounded mutation
  batches. Wait for each batch and report progress.
- Keep model-inferred `audio_files.score` independent from MOS labels.
- Replace segment-count fields with query aggregates only where the UI still
  needs a displayed count.

### Datasets and memberships

Convert every function in `src/shared/db/datasets/crud.py`.

- Return `DatasetRecord`, never a SQLAlchemy relationship graph.
- Enforce unique active dataset names in the service before insert.
- Read counts, duration bins, minimum duration, metadata values, TTS reference
  candidates, and training streams with dataset-keyed joins across monthly
  partitions.
- Add memberships with deduplicated insert blocks, monotonic `updated_at`, and
  immutable `created_at`; replacements preserve their creation month.
- Remove memberships using explicit mutations.
- Delete a dataset by deleting its membership partition contents, applying MOS
  and statistics policy, and then deleting the dataset row.
- Replace per-dataset loops with one query using `IN` or a temporary external
  table when many dataset IDs are supplied.

### Assets and configs

Convert `src/shared/db/assets/crud.py` and `file_store.py`.

- Map checkpoints to `assets.kind = 'checkpoint'` and extra files to
  `assets.kind = 'file'`.
- Return one `AssetRecord` with an enum discriminator; expose typed checkpoint
  and file DTOs at the existing facade boundary where callers need them.
- Store `run_id` as `UUID`; remove assumptions that it is a job string.
- Upload checkpoint folders or individual objects before publishing metadata.
- Preserve `get_checkpoint_path` and `get_extra_file_path` cache behavior.
- Replace updates with newer asset rows and rare deletes with mutations followed
  by delayed object garbage collection.
- Move config list/create/update operations to ClickHouse and return
  `ConfigRecord`. Keep `type` extensible rather than a schema enum.

### Waveforms

Convert `src/shared/db/waveforms/crud.py`, `pack_store.py`, and the backend
waveform service.

- Replace `session.get(AudioWaveform, ...)` with the waveform facade.
- Use the immutable pack publication sequence.
- Replace a waveform by inserting a newer location row after the new pack is
  durable.
- Delete waveform rows with explicit mutations; remove `used_bytes` accounting.
- Keep `format_version` out of ClickHouse and update codecs through an explicit
  compatible payload contract instead.

### Statistics

Convert every function in `src/shared/db/statistics/crud.py`.

- Insert statistics in blocks and return typed `StatisticsEntryRecord` values.
- List by optional dataset and `created_at DESC`.
- Decode `payload` and `metadata` at the facade boundary.
- Delete with a mutation.
- When a dataset is deleted, replace affected statistics rows with
  `dataset_id = NULL` before removing the dataset.

### MOS comparisons

Replace `src/shared/db/mos/crud.py` and remove the PostgreSQL-specific mutation
module.

- Sample pairs and validate membership entirely in ClickHouse.
- Store human `score_a`, `score_b`, preference, and previous-score state only in
  `mos_comparisons`; never write these values to model-inferred
  `audio_files.score`.
- Serialize create, correction, and undo commands per dataset.
- Corrections insert a replacement comparison with a later `updated_at` and
  update any affected next-comparison previous-score fields.
- Undo deletes the comparison, repairs the next comparison's previous-score
  fields, and waits for mutations before returning.
- Page history by `(created_at, id)` for a dataset across monthly partitions and
  bulk-fetch all referenced audio records in one query.

## Remove ORM leakage from callers

Update these known callers before deleting PostgreSQL models:

| Caller | Required change |
| --- | --- |
| `backend/audio/responses.py` | Accept audio DTOs instead of `AudioFile` |
| `backend/audio/waveform_service.py` | Use waveform facade instead of `session.get` |
| `backend/datasets/api.py` | Accept `DatasetRecord` |
| `backend/mos/api.py` | Accept MOS and audio DTOs |
| `shared/db/mos/schemas.py` | Remove `AudioFile` from `MosPair` |
| Runner audio-segment nodes | Consume facade DTOs and bulk methods |
| Runner MOS dataset | Stream ClickHouse MOS DTOs |
| Runner TTS references | Consume dataset/audio DTOs |
| Asset catalog runtime | Consume asset DTOs and UUID run IDs |

Search again for imports of the retired model modules and direct `Session`
queries before removing them. The final result must have no backend or runner
imports of ClickHouse-owned SQLAlchemy models.

## Delivery order

1. Add the ClickHouse client, typed command/query helpers, DTOs, and health check.
2. Implement read-only facades for configs, assets, statistics, waveforms,
   datasets, audio, segments, and MOS.
3. Backfill current PostgreSQL rows, flatten alignments, strip duplicated segment
   source metadata, and record counts and hashes.
4. Shadow reads and compare ordered IDs, payload hashes, counts, duration sums,
   dataset membership, segment counts, and MOS history.
5. Convert immutable pack writers and garbage collection.
6. Convert writes feature by feature, using idempotent command IDs and bulk
   inserts. Keep PostgreSQL writes temporarily for rollback.
7. Cut over readers in the same feature order and monitor query latency, mutation
   backlog, replacement-row amplification, and object-reference integrity.
8. Stop PostgreSQL writes, replay the final delta, verify parity, and take a
   recoverable PostgreSQL snapshot.
9. Remove retired SQLAlchemy models, relationships, CRUD SQL, and PostgreSQL
   tables. Keep shared facade names stable where useful.

## Completion criteria

- No SQLAlchemy query references a ClickHouse-owned table.
- No backend or runner module imports a retired ORM model.
- Every bulk path uses blocks rather than per-row ClickHouse requests.
- Audio catalog, transcript, dataset training, speaker, waveform, asset,
  statistics, config, and MOS API responses match parity fixtures.
- Audio and segment reads perform no `FINAL`.
- Delete operations wait for their mutations and cannot be followed by an older
  replacement insert.
- Object garbage collection proves a pack or asset is unreferenced before
  deleting bytes.
- PostgreSQL retains only the documented control-plane tables.
