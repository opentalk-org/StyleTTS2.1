# Shared high-volume data migration to ClickHouse

## Decision

Use the repository's existing ClickHouse `default` schema for bulk data and
object metadata. Small mutable catalogs use `ReplacingMergeTree`; audio files,
segments, and completed packs use plain `MergeTree`. PostgreSQL remains the
control plane for jobs and other transactional coordination. Object bytes remain
in S3-compatible storage.

The Atlas schema is declared in [`schema/default.ch.hcl`](schema/default.ch.hcl).

## Ownership after migration

### ClickHouse tables

| Table | Contains PostgreSQL data from |
| --- | --- |
| `audio_files` | `audio_files` |
| `audio_segments` | `segments` with `alignments.data` flattened into each row |
| `datasets` | `datasets` |
| `dataset_audio_files` | `dataset_audio_files` |
| `bucket_files` | `bucket_files` and `waveform_packs`, distinguished by `kind` |
| `assets` | `checkpoints` and `extra_files`, distinguished by `kind` |
| `audio_waveforms` | `audio_waveforms` |
| `statistics_entries` | `statistics_entries` |
| `configs` | `configs` |
| `mos_comparisons` | `mos_comparisons` |

### PostgreSQL tables retained

| Table | Responsibility |
| --- | --- |
| `alembic_version` | PostgreSQL migration state |
| `initialization` | Application initialization state |
| `jobs` | Run queue, claims, leases, and snapshots |
| `node_logs` | Current per-node logs |
| `run_node_states` | Node lifecycle coordination |
| `runners` | Runner registration and heartbeats |
| `settings` | Singleton object-store and external-integration configuration |
| `workflows` | Workflow definitions |

### Source-table mapping

| PostgreSQL table | Destination | Reason |
| --- | --- | --- |
| `alembic_version` | PostgreSQL | PostgreSQL migration state |
| `alignments` | ClickHouse, flattened into `audio_segments.alignment` | Always consumed with its segment; avoid a 22M-row one-to-one join |
| `audio_files` | ClickHouse `audio_files` | 10.2M catalog rows and bulk search/scan workload |
| `audio_waveforms` | ClickHouse `audio_waveforms` | Per-audio waveform descriptor and pack byte range |
| `bucket_files` | ClickHouse `bucket_files` (`kind = 'audio'`) | Immutable completed audio-pack metadata |
| `checkpoints` | ClickHouse `assets` (`kind = 'checkpoint'`) | Folder archives share one object-metadata catalog with files |
| `configs` | ClickHouse `configs` | Phoneme alphabets and saved training preset documents |
| `dataset_audio_files` | ClickHouse `dataset_audio_files` | 10.9M memberships and dataset-scoped scans |
| `datasets` | ClickHouse `datasets` | Keep dataset definitions with their high-volume membership rows |
| `extra_files` | ClickHouse `assets` (`kind = 'file'`) | Single objects share one object-metadata catalog with checkpoints |
| `initialization` | PostgreSQL | Singleton application control state |
| `integration_settings` | PostgreSQL `settings` | Merge external-service fields into the unified singleton |
| `jobs` | PostgreSQL | Claims, leases, desired state, and notifications require transactions |
| `mos_comparisons` | ClickHouse `mos_comparisons` | Human preference labels are append-friendly, dataset-scoped history |
| `node_logs` | PostgreSQL | Current per-node log snapshot updated with job state |
| `run_node_states` | PostgreSQL | Mutable node lifecycle coordination |
| `runners` | PostgreSQL | Heartbeats and active-run coordination |
| `segments` | ClickHouse `audio_segments` | 22.2M transcript rows and analytical speaker/text scans |
| `speaker_cluster_audits` | Remove | Retire persisted speaker-cluster audit state |
| `speaker_cluster_summaries` | Remove | Retire persisted cluster reconciliation state |
| `speaker_clustering_artifacts` | Remove | Retire persisted clustering artifact manifests |
| `speaker_clustering_runs` | Remove | Retire persisted clustering run state |
| `speaker_embedding_runs` | Remove | Retire persisted embedding run state |
| `speaker_embedding_shards` | Remove | Retire persisted embedding shard manifests |
| `statistics_entries` | ClickHouse `statistics_entries` | Dataset-derived analytical report documents |
| `storage_settings` | PostgreSQL `settings` | Merge object-store fields into the unified singleton |
| `waveform_packs` | ClickHouse `bucket_files` (`kind = 'waveform'`) | Shares the same immutable pack representation |
| `workflow_reviews` | Remove | Retire workflow review and continuation persistence |
| `workflows` | PostgreSQL | Small mutable workflow definitions |

Audio and waveform pack metadata share `bucket_files`, distinguished by an enum.
Audio bucket metadata and placement move together to ClickHouse. The audio row
contains `bucket_file_id`, byte offset/length, and the external-object reference;
a null bucket identifies external storage. Waveform descriptors and waveform
pack placement also move together to ClickHouse. Callers continue using shared
CRUD facades.

## PostgreSQL retirement dependencies

No ClickHouse-owned table can be dropped before its shared CRUD implementation
has moved. After that code cutover, the database constraints require this order:

| ClickHouse-owned PostgreSQL table | PostgreSQL dependency before drop |
| --- | --- |
| `checkpoints` | Backfill into `assets` as `kind = 'checkpoint'`; move asset CRUD first |
| `extra_files` | Backfill into `assets` as `kind = 'file'`; its speaker-table foreign keys disappear when those tables are removed |
| `alignments` | Drop with `segments` after transcript CRUD moves |
| `segments` | Drop after `alignments`; its audio FK disappears with both tables |
| `dataset_audio_files` | Drop after dataset/audio CRUD moves |
| `audio_waveforms` | Move waveform CRUD, then drop with PostgreSQL `waveform_packs` |
| `waveform_packs` | Backfill into `bucket_files` as `kind = 'waveform'`, then drop after `audio_waveforms` |
| `bucket_files` | Drop with `audio_files`; `audio_files.bucket_file_id` references it |
| `mos_comparisons` | Move MOS CRUD and stop writing human labels into `audio_files.score`, then drop its PostgreSQL table and foreign keys |
| `audio_files` | Drop with its bucket-file dependency after audio and MOS CRUD move |
| `datasets` | Drop after dataset and MOS CRUD move |
| `statistics_entries` | Move statistics CRUD, then drop its PostgreSQL table and dataset FK |
| `configs` | Move config CRUD, backfill the nine documents, then drop its PostgreSQL table |

Removing the PostgreSQL foreign keys also removes its `CASCADE`, `RESTRICT`, and
`SET NULL` behavior. Dataset/audio deletion services must replace it explicitly:
delete MOS rows according to product policy, and replace statistics rows with a
null dataset reference. Jobs,
workflows, and other JSON payloads also contain audio/dataset/checkpoint UUIDs without database
foreign keys; their IDs remain valid references but require service-level
existence checks.

## Unified PostgreSQL settings

Replace `storage_settings` and `integration_settings` with one singleton
`settings` table containing explicit columns:

| Group | Columns |
| --- | --- |
| Object storage | `bucket`, `folder`, `endpoint_url`, `region_name`, `access_key_id`, `secret_access_key` |
| Integrations | `hf_token`, `openrouter_token`, `mlflow_url` |

Keep the existing `StorageSettingsPayload` and `IntegrationSettingsPayload` as
feature DTOs, but have both shared CRUD facades read and update their respective
columns on the same row. Backfill the current two singleton rows, switch the
settings API and object-store factory, then drop both old tables. Do not use a
generic key/value table or an untyped JSON settings document.

## PostgreSQL references to ClickHouse rows

PostgreSQL stores ClickHouse UUIDs as ordinary typed columns. Shared services,
not SQLAlchemy relationships, resolve them in bulk:

1. Read or validate referenced IDs in one ClickHouse query.
2. Perform the PostgreSQL transaction using those UUID values.
3. Put resulting ClickHouse changes in the PostgreSQL outbox in that same
   transaction.
4. Resolve response objects with one bulk ClickHouse read; never query
   ClickHouse once per PostgreSQL row.

This applies to workflow launch payloads. Pydantic service DTOs replace ORM
relationships across the database boundary.

### MOS flow

- Pair sampling and membership validation run in ClickHouse against `datasets`,
  `dataset_audio_files`, and `audio_files`.
- `audio_files.score` is a trained-model inference value. `score_a`, `score_b`,
  `preferred_audio_id`, and the previous-score fields in `mos_comparisons` are
  human-label state. MOS writes must never update `audio_files.score`.
- A create inserts one row. A correction inserts the same comparison ID with a
  later `updated_at`. Undo explicitly deletes the comparison. The original
  `created_at` remains stable across replacements.
- Corrections or deletion may also change the previous-score fields of the next
  comparison involving either audio item. A single ordered MOS command worker
  per dataset calculates the affected current rows and writes all replacement
  rows in one ClickHouse insert block. Commands and assigned timestamps are
  durable and idempotent so retries insert the same rows.
- Canonical MOS reads collapse replacements with `argMax` or narrowly filtered
  `FINAL` queries. History pages then fetch referenced audio rows in one bulk
  ClickHouse query.
- The application validates that both audio IDs belong to the dataset, differ,
  and contain `preferred_audio_id`; ClickHouse does not replace these former
  PostgreSQL constraints.

Audio and dataset deletion becomes an idempotent service-level saga: execute the
required ClickHouse delete mutations, apply the former PostgreSQL
cascade/restrict/set-null actions, and record completion. Deletion is allowed to
be slow because it is rare. A reconciliation job reports dangling UUID
references and retries incomplete deletions.

## Speaker persistence removal

Remove these PostgreSQL tables in one Alembic migration after deleting their
model and CRUD dependencies:

- `speaker_cluster_audits`
- `speaker_cluster_summaries`
- `speaker_clustering_artifacts`
- `speaker_clustering_runs`
- `speaker_embedding_runs`
- `speaker_embedding_shards`
- `workflow_reviews`

Also remove the associated run-registration, sealing, reconciliation, and audit
CRUD paths from `src/shared/db/speakers`. Runner nodes that currently persist
those records must either pass their results as normal workflow artifacts or be
removed; do not recreate the six tables in ClickHouse. Existing clustering and
embedding objects referenced through `extra_files` are backfilled into
ClickHouse `assets` before the PostgreSQL manifests are dropped.

Remove the reviews backend routes, schemas, CRUD module, frontend review UI, and
the job-list review-count join. Workflow continuation must no longer depend on a
persisted review row.

The Speakers screen is unaffected conceptually: speakers are still derived by
grouping the current `audio_segments.speaker_id` values. Rename and clear
operations remain segment updates and do not require a `speakers` table.

## ClickHouse data model

Small current-state tables follow `default.ch.hcl` and use
`ReplacingMergeTree(updated_at)`. Audio files and segments instead keep one
physical current row in plain `MergeTree`; their updates and rare deletions use
explicit ClickHouse mutations. Replacement-table writers allocate a strictly
increasing `updated_at` per logical key and retries reuse it. Readers use `FINAL`
or `argMax` only for replacement tables. Background merges are an optimization
rather than a correctness boundary.

Engine decision:

| Table | Engine | Read rule |
| --- | --- | --- |
| `audio_files` | `MergeTree` | One physical current row per ID; reads never use `FINAL` |
| `dataset_audio_files` | `ReplacingMergeTree` | Filter by dataset and collapse membership replacements |
| `datasets` | `ReplacingMergeTree` | Small table; `FINAL` is acceptable |
| `assets` | `ReplacingMergeTree` | Filter by ID or kind before `FINAL` |
| `audio_waveforms` | `ReplacingMergeTree` | Filter by audio IDs before `FINAL` |
| `statistics_entries` | `ReplacingMergeTree` | Small table; `FINAL` is acceptable |
| `configs` | `ReplacingMergeTree` | Small table; filter by type and use `FINAL` |
| `mos_comparisons` | `ReplacingMergeTree` | Filter by dataset, collapse replacements, then page by creation order |
| `audio_segments` | `MergeTree` | One physical current row per stable segment ID; reads never use `FINAL` |
| `bucket_files` | `MergeTree` | Immutable completed packs; no `FINAL` |

Ten million audio rows make broad catalog and training reads more important than
fast metadata updates. Audio rename, score, language, prompt, and storage-location
changes use explicit mutations so catalog queries never pay a `FINAL` cost.
Membership replacement churn and merge backlog still require monitoring.

Segments have logical identity `(audio_file_id, id)`, where `id` is the stable
string identifier exposed by the segment API and stored as `source_id` in
PostgreSQL. Their physical order is `(updated_at, audio_file_id, id)`.
`position` is mutable UI ordering data and never participates in identity.
Updating audio metadata never writes segments. New segments are bulk inserted;
edits use explicit ClickHouse update mutations; removed segments use explicit
delete mutations. A complete replacement deletes the audio item's old segments
and bulk-inserts the replacement. All writes for one audio ID are serialized.
Readers query one table without `FINAL` or an audio revision join and order the
result by `position`.

This deliberately accepts a brief empty or mixed-state window during a complete
transcript replacement in exchange for simple reads, one physical row per
segment, and no duplicated transcript snapshots. Workflows that require a stable
transcript must finish replacement before publishing the audio ID downstream.

Cross-store writes use an outbox in PostgreSQL. The transaction appends an
idempotent catalog command; a relay writes ClickHouse in large blocks and marks
commands delivered. Row identity and its assigned `updated_at` make retries
deterministic.

Packs have no published open state. A writer builds a complete pack in a local
temporary file, assigns every byte offset locally, uploads the completed object,
inserts its immutable `id`/`kind`/`path`/`size` row, and only then updates the
audio location or publishes the replacement waveform location. A pack whose location publication fails
is unreferenced and safe to retry. Compaction repeats this process with a new
pack ID and publishes new location rows; it never edits an existing pack.

Old pack rows remain as immutable history. A garbage collector finds packs that
are unreferenced by current audio/waveform locations, waits a grace
period covering readers and outbox retries, then deletes only the object bytes.
No `used_bytes`, `sealed`, or mutable pack timestamp is required: every published
pack is complete and its used bytes equal `size`.

## Query implications

- Audio lookup, catalog paging, language/name/metadata search, training scans,
  duration bins, segment previews, annotations, and speaker summaries move into
  ClickHouse-backed shared CRUD modules, including waveform location reads.
- Fetch audio and waveform storage locations from ClickHouse when bytes are
  required.
- Dataset definitions, membership reads, and counts come from ClickHouse. The
  service layer must enforce unique active dataset names before inserting a new
  row because ClickHouse has no unique constraint.
- Add projections only for access paths proven slow by production-shaped
  benchmarks. Do not duplicate the base schema preemptively.
- Remove name ordering from the audio catalog API. The supported catalog orders
  are `updated_at` with `id` as its cursor tie-breaker, and `duration` with `id`
  as its cursor tie-breaker.
- Current `%substring%` search over name and serialized JSON needs a measured
  skipping-index or token-index design. Do not encode an unverified index in the
  initial schema. Preserve the API, benchmark representative data, then use the
  pinned server's text index or a query-specific projection.

## Delivery plan

### 1. Measure and freeze contracts

Record row counts, compressed/uncompressed bytes, write rates, query latency,
and the largest metadata/alignment payloads for every candidate table. Capture
the outputs of catalog, dataset training, annotation, speaker, and waveform
CRUD calls as parity fixtures. Define acceptable outbox lag and backfill/retry
windows before choosing partitions or TTLs.

### 2. Introduce database boundaries

Add a ClickHouse connection and typed row models under `src/shared/db`, plus a
PostgreSQL outbox CRUD facade. Keep public CRUD signatures stable. Route bulk
operations through services that build complete packs, upload bytes, and emit
the immutable pack plus replacement location rows as one retryable command.

### 3. Deploy schema and dual-write

Generate an Atlas migration from `db/schema/default.ch.hcl` with
`nix develop -c atlas-migrate-diff`, and apply it locally. Start the outbox
relay, dual-write all catalog mutations, expose lag/failure metrics, and keep
reads on PostgreSQL. Exercise writes through real workflow graphs.

### 4. Backfill

Backfill in large UUID-keyset blocks with a fixed high-water mark. Export audio,
datasets, memberships, waveforms, assets, statistics, configs, and MOS
comparisons with a stable migration `updated_at` without changing model-inferred
audio scores. Export only the current segment rows and both pack families as
immutable rows. Persist progress checkpoints, checksums, counts, null counts,
duration sums, and per-dataset/per-speaker aggregates. Re-run changed rows from
the outbox change stream after the bulk copy.

### 5. Shadow and cut over reads

Run ClickHouse queries in shadow mode and compare ordered IDs, payload hashes,
counts, and aggregates. Cut over endpoint families independently: configs,
assets, statistics, waveform reads, catalog pages, segment reads, dataset scans,
MOS, then speaker aggregates. Keep all feature access behind the existing
shared CRUD facades.

### 6. Cut over writes and retire PostgreSQL bulk rows

After parity and lag objectives hold through representative workflows, make
ClickHouse authoritative for catalog rows. Stop PostgreSQL dual-writes, take a
recoverable snapshot, and remove bulk columns/tables in a later PostgreSQL
migration. Retain the outbox. Rollback before
retirement is a read-routing switch plus outbox replay; after retirement it is
restore-and-replay from the snapshot.

## Schema review items before implementation

- Validate `ReplacingMergeTree(updated_at)`, delete-mutation behavior, and
  insert-deduplication settings against the production ClickHouse version.
- Partition `dataset_audio_files` by immutable `toYYYYMM(created_at)` to bound
  part growth by creation month. Dataset reads prune by the sorting key rather
  than partition. Replacements must preserve `created_at` because ClickHouse
  cannot collapse versions across partitions.
- Leave `audio_files` and `audio_segments` unpartitioned. Their dominant key is
  a uniformly distributed audio UUID; hash partitioning would create parts in
  every bucket for each bulk insert without improving query pruning. The sparse
  primary index already serves audio-ID reads.
- Leave `bucket_files`, `audio_waveforms`, `assets`, `statistics_entries`, and
  `configs` unpartitioned. The current database has about 53,000 audio packs,
  186 waveform packs, 178 waveform descriptors, 91 stored assets, seven
  statistics entries, and nine configs. Partition `mos_comparisons` by immutable
  `toYYYYMM(created_at)` as well.
- Decide which JSON metadata keys deserve typed materialized columns only after
  measuring real filters and scan cost.
- Enforce uniqueness of segment `id` within each audio item at the write API.
  Reordering changes only `position`; it must not change segment identity.
- Define retention for outbox records only after backup and replay requirements
  are agreed.
