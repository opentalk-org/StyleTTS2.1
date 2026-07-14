# Scalable Speaker Clustering Pipeline Design

## Goal

Add a node-based pipeline that embeds, clusters, audits, and assigns 5–10 million
speech segments without materializing the corpus in runner memory. The default
scope is one selected dataset and one versioned clustering run. Uncertain segments
remain unassigned so a weak match cannot silently contaminate a speaker identity.

## Chosen approach

Use ECAPA-TDNN embeddings, a sharded FAISS candidate index, conservative sparse
graph bootstrap, and prototype-based incremental assignment.

Two simpler approaches are rejected as the primary design:

- Centroid-only assignment is efficient after a trustworthy catalog exists but
  cannot discover an unknown set of speakers.
- MiniBatch K-means scales computationally but requires a speaker count, forces
  every item into a cluster, and does not model rejection. It may be used only as
  an internal coarse partitioning aid.

The clustering path prioritizes pair precision over recall. Splitting one speaker
into provisional clusters is recoverable; merging unrelated speakers poisons all
later centroid assignment.

## Workflow

The example workflow is:

`SpeakerSegmentSource -> ECAPASpeakerEmbed -> CollectSpeakerEmbeddings ->`
`ClusterSpeakerEmbeddings -> AuditSpeakerClusters -> ApplySpeakerClusters`

`SpeakerSegmentSource` keyset-pages audio records in dataset scope, expands their
stored segment JSON in bounded pages, and bulk-reads only the required audio.
Each output is a mono segment clip with stable audio-file and segment identity.
It never constructs a list of all segment IDs.

`ECAPASpeakerEmbed` duration-sorts each micro-batch, converts the whole batch to
16 kHz mono, pads it once, supplies relative lengths, and calls SpeechBrain's
ECAPA encoder once. The model remains loaded through node lifecycle. Embeddings
are cast to float32 for validation and explicitly L2-normalized for cosine
similarity. Empty, non-finite, and too-short inputs fail or receive an explicit
quality rejection according to settings; no invalid vector enters clustering.

The embedding node writes one columnar shard per bounded group instead of passing
millions of raw vectors through downstream queues. A `SpeakerEmbeddingShardRef`
contains the artifact ID, row count, model identity, preprocessing identity,
dimension, and run ID. The shard stores stable segment identity, source audio ID,
duration, quality fields, and float16 vectors. Float32 is used for inference,
centroids, and exact candidate reranking.

`CollectSpeakerEmbeddings` registers shards idempotently through speaker CRUD.
When the durable row count reaches the source's expected count, it atomically
seals the embedding run and emits one `SpeakerEmbeddingSetRef`. Completion is a
database invariant, not an in-memory counter, so bounded batches and retries are
safe.

`ClusterSpeakerEmbeddings` trains a FAISS IVF candidate index on a deterministic
representative sample, adds shards in batches, and queries bounded blocks. For
5–10 million 192-dimensional vectors, the starting profile is an IVF index with
65,536 lists; IVFPQ is used when memory is constrained, while exact float32
cosine reranking decides every accepted edge. Index type and search parameters
remain settings and are recorded in the run.

Bootstrap clustering forms a sparse graph from exact-reranked candidates. Edges
must pass the calibrated cosine threshold and reciprocal-neighbor requirement.
Components are initially conservative microclusters. Prototype consolidation
requires reciprocal prototype matches plus multiple cross-cluster supporting
pairs; a single similar segment cannot join two large clusters. Large or
high-dispersion components are rejected for audit rather than force-merged.

After bootstrap, routine assignment searches cluster prototypes instead of all
segments. It records best and second-best scores and uses three outcomes:

- accepted when the best score and best-minus-second margin both pass;
- provisional new speaker when no candidate reaches the new-speaker threshold;
- ambiguous otherwise.

Only high-confidence members update float32 prototype sums. Ambiguous and
low-quality segments never update prototypes. Multiple exemplars or sub-centroids
are retained for speakers spanning channels or recording conditions.

`AuditSpeakerClusters` scans persisted assignments and candidate scores to publish
a generic workflow review with quantitative metrics and bounded audio samples.
Approval starts a linked `ApplySpeakerClusters` continuation exactly once. Apply
creates voices in bulk and rewrites each affected audio record's complete segment
JSON exactly once, preserving every unrelated segment field. Ambiguous/rejected
segments keep `voice_id=None` and an actionable assignment reason in the run
artifact.

## Durable data

Add a focused `shared.db.speakers` feature with typed schemas, models, and CRUD:

- embedding runs: scope, model revision, preprocessing version, expected and
  stored counts, state, shard manifest, and failure details;
- clustering runs: embedding run, index parameters, thresholds, algorithm
  version, counts by outcome, state, and audit artifact IDs;
- cluster summaries: stable cluster ID, generated voice ID, member count,
  duration, centroid artifact location, dispersion, and status;
- registered embedding, candidate, assignment, and exemplar shards.

Bulk vector and assignment payloads live in Parquet artifacts managed through
asset CRUD. PostgreSQL remains the source of truth for run state, shard keys,
counts, and cluster summaries. Vectors are not placed in `AudioFile.segments` or
statistics JSONB. Applied `voice_id` remains in the existing segment schema.

Migrations add only speaker-run metadata and summaries. All database access goes
through the shared CRUD facade. Run and shard keys have uniqueness constraints so
retries cannot duplicate data.

## Node contracts and runtime policies

Add concrete `SpeakerEmbeddingShardRefPort`, `SpeakerEmbeddingSetRefPort`,
`SpeakerClusterRunRefPort`, and `SpeakerAuditRefPort` datatypes. No union ports or
speaker behavior enter `runflow`.

The source and collector lease generic I/O resources. ECAPA leases one accelerator
and declares VRAM honestly, uses micro-batches bounded by both item count and total
audio seconds, and keeps the model loaded. Clustering leases accelerator and I/O
resources and processes shard/query chunks with cancellation checks. Audit and
apply use bounded I/O batches. Long stages report segments, bytes, shards,
candidates, and assignments rather than vague percentages.

Every artifact and run records the exact model repository/revision, model file
hash, preprocessing version, index factory string, FAISS parameters, random seed,
threshold version, and input scope. Rerunning with the same identifiers is
idempotent; changed configuration creates a new run.

## Calibration and merge protection

No universal VoxCeleb cosine threshold is embedded as a silent default. The
production thresholds come from an in-domain labeled calibration artifact, or
must be set explicitly for an exploratory run. Calibration trials are stratified
by duration, language, channel, SNR, and overlap where those labels exist.

The operating point targets a configured false-accept rate because false merges
are the destructive error. Calibration uses a realistically sized search index,
not only balanced same/different pairs, so nearest impostor behavior is measured.
Short and low-quality segments receive stricter policies or rejection.

Every decision persists best score, second score, margin, candidate cluster IDs,
threshold version, quality flags, and reason. This makes assignments reproducible
and inspectable rather than only writing a final voice UUID.

## Verification outputs

Without ground-truth labels, audit reports:

- member-to-centroid and sampled within-cluster score distributions;
- nearest cross-cluster and ordinary random-impostor distributions;
- cluster dispersion, low-tail scores, low-margin assignments, and cluster sizes;
- suspicious bimodal or oversized clusters;
- worst within-cluster pairs, closest cross-cluster pairs, boundary assignments,
  and proposed merges as deterministic listening samples;
- ANN recall@k and assignment-decision agreement against exact search for a
  deterministic 1,000–10,000-query sample.

With known speaker labels, audit additionally reports same-predicted-cluster pair
precision/recall, weighted purity, adjusted Rand index, adjusted mutual
information, per-speaker fragmentation, and distinct true speakers per predicted
cluster. Pair precision and the worst merged clusters are the primary checks for
the user's random-speaker failure mode. Sampled silhouette is secondary only.

The example workflow includes a small synthetic verification mode built from
several clips per known speaker plus deliberately unrelated speakers. It must
demonstrate high same-speaker similarity, rejected unrelated pairs, stable output
under shuffled input order, and reproducible assignments for a fixed seed.

## Validation

Development follows temporary test-first coverage and removes tests before handoff
per repository policy. Coverage establishes batched ECAPA preprocessing and one
encoder call per micro-batch, explicit L2 normalization, shard idempotency,
bounded-memory iteration, reciprocal edge filtering, rejection and margin logic,
multi-support consolidation, metrics, and whole-record writeback preservation.

Final verification runs commands through Nix and executes nodes only through real
graphs submitted to `POST /graphs/runs`. A labeled multi-speaker smoke corpus
validates audit metrics and listening artifacts. A generated large metadata/vector
fixture validates streaming, shard sizing, cancellation, and peak-memory behavior
without committing the fixture.
