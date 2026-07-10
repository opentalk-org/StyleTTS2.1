# StyleTTS DataLoader Bulk Prefetch Design

## Goal

Replace the StyleTTS training loader's custom bucket stream, speaker ordering,
prefetch thread, and disk cache with PyTorch DataLoader prefetch. Each prefetched
training batch reads its audio through one shared audio CRUD bulk operation.

## Data flow

The training manifest continues to identify database-backed training audio by
UUID-based `.wav` paths. It no longer creates or stores a bucket stream plan.
Validation audio remains materialized because the validation loader is small and
already uses ordinary file reads.

The map-style training dataset implements `__getitems__(indices)`. For each list
of indices supplied by a DataLoader worker, it selects one same-speaker reference
for every primary row, deduplicates the combined primary and reference audio IDs,
and calls `audio_crud.bulk_read_audio_files` once inside a shared database
session. Audio is decoded directly from the returned bytes without temporary
files. The dataset then applies the existing waveform, text, mel, and reference
preparation to produce the normal sample tuples.

The DataLoader uses its standard worker queue and prefetch behavior with
shuffling enabled for training. There is no application-owned prefetch thread,
cursor, resident set, cache budget, bucket order, or local audio cache.

## Batch ordering

The custom ordering of buckets and samples is removed. DataLoader shuffling
determines which samples enter a training batch. The existing mel-length sort in
`Collater` remains unchanged so tensors within an already-selected batch retain
the ordering expected by the training implementation.

## Configuration and cleanup

The obsolete stream-plan path, cache directory, and bucket-cache budget are
removed from StyleTTS training configuration and node settings. The manifest's
database-backed training mode remains the switch that avoids materializing the
training split. The bucket stream implementation and stream-plan module are
removed once no callers remain.

## Failures and cancellation

Missing audio IDs and storage failures propagate from the shared CRUD facade and
fail the DataLoader worker rather than being hidden or retried by a second cache
layer. The main training loop retains its cancellation checks. DataLoader worker
shutdown provides the lifecycle boundary for prefetched work.

## Verification

Temporary tests will verify that a fetched training batch makes one CRUD bulk
read containing deduplicated primary and reference IDs, returns every requested
sample, and retains the collater's mel-length ordering. A DataLoader integration
test will exercise worker prefetch without the custom stream cache. The focused
tests and relevant static checks will run through `nix develop --command`, and
temporary test files will be removed before completion as required by the
repository policy.
