# Object Store and Audio Read Cleanup

## Goal

Make object storage the single owner of remote byte access. Audio CRUD resolves
database metadata once and submits typed object ranges directly to storage.

## Storage boundary

`shared.storage` owns one `ObjectStore` interface and these request types:

- `ObjectRange(path, byte_offset, byte_length)`
- `ObjectWrite(path, data)` only if multiple existing upload loops justify it

The interface contains only operations used by current callers:

- upload bytes
- upload a local path
- download a complete object
- read one range
- read multiple ranges
- delete an object
- test connectivity

`S3ObjectStore` explicitly implements this interface. It owns concurrent range
requests, preserves input ordering, validates returned byte counts, and lets
storage errors propagate.

## Audio reads

Audio CRUD performs one query for all requested audio IDs. The query returns the
object path, byte offset, byte length, and storage kind. Packed rows become
`ObjectRange` requests and are passed to `ObjectStore.read_ranges` once.

Single-file and partial reads use the same location resolver without routing
through bulk wrappers. Missing rows, external records, invalid subranges, and
short object responses fail immediately.

Segment reads use the same location query and one storage batch read, then slice
the returned WAV bytes into requested time ranges. Storage does not import audio
models or SQLAlchemy.

## Removed structure

- audio-local and waveform-local `ObjectStore` protocols
- `DeletableObjectStore`
- `AudioFileCache`
- `BulkWavReader`
- unused `BoundedObjectReader` and `read_wav_ranges`
- packed audio read wrappers and their private thread pool
- unused bulk partial-read API
- caller-controlled audio fetch-worker settings

Pack modules retain pack allocation, buffered writes, database accounting, and
pruning. They use the canonical storage interface.

## Local caching

Persistent local caching remains limited to checkpoints, models, configs, and
extra files. Audio data is read from object storage by range and is not cached
on local disk.

## Verification

- import every first-party Python module without writing bytecode
- verify S3 range batching preserves order and rejects short responses
- verify single, bulk, partial, and segment audio reads against packed fixtures
- run representative audio and Beetle graphs when services and required assets
  are available
- verify the runner registry, backend routes, frontend typecheck, and diff
  whitespace checks
