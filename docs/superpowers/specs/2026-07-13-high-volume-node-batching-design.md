# High-Volume Node Batching Design

## Goal

Eliminate per-item database, object-storage, and model calls from high-volume runner node batches. A node that receives many audio-related inputs should collect the batch, invoke a collection-shaped dependency once per bounded batch, and restore ordered outputs with unchanged lineage and fan-out behavior.

The immediate performance failure is `DeleteAudioRecords`: it accepts up to 256 inputs but calls single-record deletion for every input. Each call separately resolves storage configuration, deletes and commits waveform state, deletes and commits the audio record, and scans or prunes packs. The nominal bulk delete facade also deletes waveforms with one commit per record and loads packed records individually, so changing only the node call would not fix the root cause.

## Scope

Batch high-cardinality workflow paths:

- audio record metadata and packed bytes;
- audio segments and waveforms;
- statistics entries produced by workflows;
- voice lookup, creation, and assignment for audio batches;
- membership changes involving many audio IDs and one or a few datasets;
- audio loading, extraction, persistence, updates, and deletion;
- ASR, alignment, diarization, enhancement, TTS, and synthesis when a node receives multiple compatible inputs;
- high-volume per-audio artifacts;
- runner state and log flush paths, which must remain set-based.

Keep naturally low-cardinality administrative CRUD single-record unless an audited caller supplies a collection:

- dataset entities;
- settings and integration configuration;
- workflows;
- checkpoints and training-result publication;
- other administrative entities whose cardinality is bounded and small.

Dataset membership is distinct from dataset entity CRUD. Adding or removing hundreds of audio IDs from one dataset remains a high-volume audio operation and must use a bulk association-table statement.

## Batch Contract

High-volume adapters accept typed sequences or ID-keyed payload mappings and return equally sized ordered sequences or explicit ID-keyed results. Nodes follow four steps:

1. Validate and collect every input in the scheduler batch.
2. Deduplicate shared record IDs or references for dependency work.
3. Invoke each expensive database, storage, network, decode, or model adapter at the batch boundary.
4. Map results back to every input in original order while preserving output cardinality and lineage.

Cheap in-memory output shaping may use loops. The prohibited pattern is an expensive dependency call or transaction inside a per-input loop when the dependency can accept the collection.

Single-record CRUD functions remain available for genuinely singular API operations, but high-volume families implement the bulk operation as the source of truth. A single-record wrapper delegates to the bulk implementation when doing so preserves clear return and error semantics.

## CRUD Design

### Audio and packs

Audio CRUD gains or completes set-based get, read, create, update, and delete operations. Bulk functions must:

- load requested rows with `WHERE id IN (...)` rather than repeated `one(...)` calls;
- report all missing IDs before mutation;
- preserve caller order where returning sequences;
- resolve storage configuration and construct the object-store client once;
- group packed reads by bucket object so each pack is downloaded once;
- aggregate `used_bytes` changes by pack;
- commit once per bounded batch;
- run fragmented-pack pruning once after the batch mutation.

Deleting audio loads all relevant audio rows and waveforms set-wise, aggregates audio and waveform pack byte reductions, deletes waveform and audio rows in bulk, commits once, and then performs one audio-pack prune pass. Existing database cascades remain responsible for dependent relational records.

### Segments and waveforms

Segment collection reads and replacements remain collection-shaped and validate missing audio IDs consistently. Waveforms gain bulk deletion used by audio updates and deletes. Bulk waveform replacement deletes prior waveform rows set-wise, updates affected pack usage in aggregate, writes replacement packs through one writer, and commits once.

### Voices and statistics

Voice operations used by batch nodes load names and IDs once, create missing voices as one collection, and return a lookup map. Assigning a voice loads audio rows in one query and performs a bulk metadata/segment update.

Statistics persistence accepts a sequence of typed create payloads, inserts all entries in one transaction, and returns results in payload order.

### Dataset membership

Bulk add already uses a set-based association-table insert. Bulk remove uses one association-table delete scoped by dataset ID and audio IDs. Dataset create, rename, list, and delete remain unchanged.

### Assets and other CRUD

High-volume per-audio artifacts use a bulk facade that reuses one object-store client, stages uploads, inserts metadata together, and returns ordered records. The remaining CRUD modules are audited for collection-shaped callers. No unused bulk API is introduced solely for symmetry.

Job coordination keeps its existing bulk flush approach. Any repeated lookup or mutation discovered in its audited collection paths is converted to a set-based statement without changing lease semantics.

## Node-Family Design

### Database and storage nodes

`LoadAudio`, audio record writeback, segment writeback, split extraction and persistence, dataset membership, voice assignment, deletion, statistics writeback, and artifact writeback collect the full node batch before opening their service boundary. They call the corresponding bulk facade once and rebuild outputs in input order.

Split persistence groups newly created audio records, segment payloads, membership changes, and completed source replacements. Source IDs repeated by multiple split outputs are deduplicated. A source is removed or deleted only after its final group is represented in the same successful batch, preserving the current replace-all contract.

### ASR and alignment

ASR adapters accept all audio paths, durations, and compatible settings together. Parakeet and Canary already expose underlying multi-path APIs; their nodes must stop wrapping one path at a time. Whisper receives a batch adapter at the model boundary so shared preprocessing and batched decoding are performed together.

Alignment materializes the audio inputs together and sends compatible transcript/audio requests through a batch alignment adapter. Results remain grouped by source audio before segment reconstruction.

### Diarization and enhancement

Sortformer already has a multi-audio model call and remains the reference pattern. Preprocessing and voice creation surrounding it become batch-shaped. Enhancement adapters accept audio collections, batch decode or tensor preparation, invoke the underlying model on the batch when supported, and encode ordered outputs together.

### TTS and synthesis

TTS runtimes gain typed synthesis requests and `synthesize_batch`. Nodes expand voice fan-out into requests, group compatible requests by checkpoint, engine, language, and settings, invoke the runtime once per compatible group, and associate results with their originating inputs.

StyleTTS follows the same request-batch shape. Shared references and checkpoint components are resolved once, compatible reference audio is loaded in bulk, and runtime inference processes the request group without reloading resources.

### Remote and training paths

Remote services use one provider bulk request when supported. If a provider only supports singular requests, bounded concurrency is isolated behind a collection-shaped adapter so the node still has one batch boundary and cancellation/progress remain centralized.

Training nodes continue to rely on framework dataloaders for sample batching. Their setup, manifest, and database preparation paths use bulk reads and writes when supplied with audio collections. Independent whole training jobs are not combined into one model operation.

## Transactions and External Storage

Database mutations are atomic per scheduler batch. All referenced IDs are validated before mutation; missing or invalid items fail the batch with actionable identifiers. No per-item commit is permitted inside a high-volume bulk operation.

Object storage cannot participate in the PostgreSQL transaction. Bulk creates and replacements therefore stage object writes, persist metadata in one transaction, and delete newly staged objects if database persistence fails. Packed audio and waveform writers flush once per batch. Deletes update database truth first; physical pack reclamation stays behind the existing prune helpers and runs once after the batch.

The collection size remains bounded by node batch policies and existing source page sizes. Operations that can exceed database parameter or memory limits process explicit bounded chunks, but never regress to a commit or prune per record.

## Cancellation, Progress, and Errors

Nodes check cancellation while collecting inputs, before each bounded dependency call, and while reconstructing large output groups. Model adapters propagate cancellation callbacks where the library supports them. An indivisible native batch call may finish before cancellation is observed.

Progress reports use completed item counts. Batch-level failures identify the operation and relevant record IDs. Output count mismatches from model or storage adapters are invariant violations and fail explicitly.

## Verification

Repository rules prohibit committed tests unless requested, so verification uses temporary scripts and real smoke graphs that are removed before completion.

Representative graphs cover:

- deleting a multi-item audio batch with and without waveforms;
- loading packed audio with multiple records sharing packs;
- bulk audio byte and metadata updates;
- segment load and replacement;
- waveform replacement and deletion;
- dataset membership removal for many audio IDs;
- voice assignment and missing-voice creation;
- statistics and artifact persistence;
- split extraction and replace-all persistence;
- ASR, alignment, diarization, enhancement, and representative synthesis batches.

Nodes are submitted through `POST /graphs/runs` and inspected with the CLI through `nix develop --command ...`. Before-and-after diagnostics record SQL statement counts, commit counts, object downloads/uploads, and pack-prune invocations. A high-volume node passes only when expensive-call counts scale by batch or unique pack/group count rather than input count. Existing smoke workflows and focused static checks run through Nix after implementation.

## Completion Criteria

- No audited high-volume node performs CRUD, object-store, network, decode, or model calls once per input when a collection boundary is available.
- High-volume CRUD uses set-based loading/mutation and one commit per bounded batch.
- Audio deletion performs one waveform/audio mutation transaction and one prune pass per node batch.
- Output order, cardinality, lineage, fan-out, cancellation, and actionable error behavior remain correct.
- Low-cardinality administrative CRUD is not expanded with unused bulk APIs.
- Real graph verification demonstrates reduced external-call and transaction counts.
