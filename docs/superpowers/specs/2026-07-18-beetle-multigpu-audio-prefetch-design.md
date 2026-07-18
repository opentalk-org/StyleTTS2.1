# Beetle Multi-GPU and Audio Prefetch Design

## Scope

This change has two outcomes:

1. Beetle training runs through Hugging Face Accelerate with normal data-parallel
   semantics and a configured batch size per GPU.
2. Training audio is fetched and prefetched without the measured hundreds of
   sequential object-store reads that reduced Stage 1 throughput by about 3x.

Model architecture, losses, stage ordering, validation behavior, and training
objectives do not change.

## Data-parallel semantics

Every process builds the same deterministic dataset index. The distributed batch
planner draws one global batch containing `batch_size * world_size` target
samples, then gives each rank a disjoint `batch_size` slice. For 5,000 samples and
two ranks, each rank consumes 2,500 distinct samples before the shuffled dataset
is repeated. Training remains continuous and exposes no epoch configuration,
epoch metrics, or epoch termination condition.

Voice and style groups are sampling units rather than flat examples. The planner
draws `voices_per_batch * world_size` complete voice groups and
`recordings_per_batch * world_size` complete style groups, then assigns whole
groups to ranks. All utterances for one same-voice or same-recording group remain
on one rank. The configured group counts and `batch_size` are per GPU.

All ranks advance the same global planner state. A checkpoint therefore stores
one committed planner position while rank-local stochastic state is stored per
rank. Resume requires the same world size so exact continuation is well-defined.

## Accelerate runtime

The CLI creates one `Accelerator` and uses its process index, process count,
device, mixed precision, gradient synchronization, and collective operations.
Trainable modules and optimizers are prepared before the continuous loop.
Backward and gradient clipping go through Accelerate; accumulation suppresses
unnecessary synchronization until the optimizer boundary.

Loss metrics are reduced across ranks before reporting. Only the main process
writes MLflow data, validation artifacts, and shared checkpoint manifests. All
ranks participate in checkpoint barriers and contribute rank-local RNG state.
Cancellation and failures are propagated so one failed rank terminates the whole
distributed run instead of leaving peers blocked in collectives.

Single-process launch uses the same code path with `world_size == 1`.

The documented multi-GPU command uses the project interpreter:

```bash
nix develop --command python -m accelerate.commands.launch \
  --num_processes 2 -m runner.nodes.training.beetle.scripts.train_stage1 ...
```

## Bulk audio reader

The existing ranged WAV reader opens a remote WAV through a seekable facade.
Python's WAV parser turns header seeks and reads into separate S3 range requests;
one measured 64-file batch issued 620 range requests and two whole-pack
downloads.

The replacement resolves each distinct audio file once, reads its complete WAV
byte slice with one object-store range request, and extracts every requested clip
from the in-memory WAV. Requests for different audio files execute concurrently
through one persistent S3 client. Results retain request order and duplicate clip
requests reuse the same fetched WAV bytes.

Small pack files are not downloaded in full merely because they fall below a
size threshold. The reader fetches only the selected audio-file slices. This
keeps cold-cache transferred bytes proportional to the selected audio rather
than to pack size.

## Host-shared audio cache

Each runner host has a bounded WAV cache under
`$XDG_CACHE_HOME/runflow/audio`. Cache entries contain the complete stored WAV
for one audio-file location and are keyed by bucket path, byte offset, and byte
length. Those immutable storage coordinates prevent stale content from being
accepted after an audio update.

Cache population uses a per-entry cross-process file lock, a temporary file, and
an atomic rename. Accelerate ranks on the same host can request an entry
concurrently, but only one downloads it. Other hosts maintain independent
caches. A global cache lock protects budget accounting and least-recently-used
eviction; locked or currently populated entries are not eviction candidates.

The Beetle prefetch configuration explicitly provides:

- local audio-cache byte budget;
- concurrent cold-cache fetch worker count;
- number of prepared batches retained in memory;
- maximum decoded bytes retained in memory.

## Dataset prefetch pipeline

Each rank owns its planned-batch queue and consumes only its distributed shard.
One batch-preparation worker preserves deterministic order and performs segment
metadata resolution, cache-backed audio fetch, decoding, mel construction, and
collation. Cold audio-file misses are fetched concurrently inside the bulk
reader. The worker continues preparing bounded future batches while GPU training
uses the current batch.

Using one preparation worker avoids the measured Python-thread contention that
inflated CUDA event time when two complete collators ran beside the training
thread. Look-ahead comes from the bounded prepared-batch queue and concurrent
audio I/O rather than multiple simultaneous mel/collate jobs.

Prefetched batches do not advance committed sampler state. A batch becomes
committed only after the training loop marks it consumed. After resume,
unconsumed prefetched work is deterministically regenerated.

## Failure behavior

Missing database rows, missing packed bytes, corrupt WAV data, cache write
failures, and worker failures stop the run with the affected audio ID or storage
location in the error. Cache corruption removes the invalid entry and retries
one cold read; a second failure is reported. Cancellation is checked between
planning, bulk fetch, collation, and queue waits.

Distributed startup rejects incompatible world size on exact resume and rejects
insufficient voice or recording groups for the configured per-rank group counts.

## Verification

Temporary tests outside the repository cover:

- one remote range read per distinct cold audio file and stable request ordering;
- one cache population under concurrent processes and bounded eviction;
- disjoint target samples across ranks with per-rank batch size preserved;
- complete voice/style groups on one rank;
- committed sampler state ignoring unconsumed prefetched batches;
- single-rank and two-rank exact-resume behavior.

A real Stage 1 benchmark uses the existing database dataset, batch size 64 per
GPU, 9,600-sample adversarial crops, BF16, and compiled production path. After
queue warmup, real-prefetch step time must be within 20% of the cached-CPU-batch
baseline. The cold-cache probe must reduce the measured 620 range requests to no
more than one request per distinct audio file and must perform no whole-pack
downloads.
