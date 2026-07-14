# Generic Workflow Reviews Design

## Goal

Replace speaker-audit HTML, JSON, ZIP, and listening-manifest artifacts with a
single reusable review capability rendered inside the existing Jobs UI. A
workflow may publish bounded metrics and media samples, and an approved review
may launch a typed continuation graph.

## Scope

- Add one generic persisted review model and CRUD feature under `src/shared/db`.
- Add generic review reads and decisions to the backend.
- Add a reusable review drawer to the existing Jobs screen.
- Make speaker clustering the first review producer.
- Remove report files and artifact-based review plumbing completely.
- Keep `runflow` domain-agnostic and free of review-specific scheduler behavior.

There is no new navigation destination, speaker-specific backend router, or
speaker-specific frontend feature.

## Typed review contract

`shared.db.reviews.schemas` owns immutable Pydantic models for the serialized
review payload:

- `ReviewMetric`: stable key, human label, formatted value, optional numeric
  value, and neutral/success/warning/danger tone.
- `ReviewField`: stable key, human label, and formatted value.
- `AudioSegmentReviewMedia`: discriminated media reference containing an audio
  file ID, segment ID, start/end seconds, duration, and display name.
- `ReviewItem`: stable item key, title, optional subtitle, fields, and zero or
  more media references.
- `ReviewGroup`: stable key, title, explanation, tone, and bounded items.
- `ReviewPayload`: headline metrics, warning messages, and review groups.
- `ReviewContinuation`: a validated `InlineGraphRunRequest` without a run ID.

The database stores Pydantic-serialized JSONB, never unvalidated dictionaries at
feature boundaries. The payload is bounded by its producer; speaker audits keep
at most the configured `category_limit` entries per group.

## Persistence

Add `workflow_reviews` with:

- UUID primary key;
- unique `(kind, source_key)` identity for retry safety;
- `producer_run_id` foreign key to `jobs.run_id` with cascade deletion;
- kind, title, pending/approved/rejected state;
- typed payload JSONB;
- optional typed continuation graph JSONB;
- optional continuation run ID;
- created, decided, and updated timestamps.

The shared CRUD facade provides paginated/list-by-run reads, one detail read,
idempotent creation, and row-locked decision transitions. Reviews are immutable
after creation except for their decision and continuation-run fields.

`speaker_cluster_audits` remains the domain record for audit identity, metrics,
and resumable apply progress. It gains a required `review_id` when completed.
The report and listening artifact columns and their foreign keys are removed.
Existing review files are not migrated or retained; this project does not carry
legacy compatibility paths.

## Speaker audit producer

`AuditSpeakerClusters` continues to scan assignment Parquet in bounded batches.
It computes the existing quantitative metrics and deterministic category
selection, but returns typed in-memory results instead of writing files.

For selected rows it bulk-loads the referenced audio segment metadata through
the public audio CRUD facade and creates `AudioSegmentReviewMedia` references.
It publishes these review groups:

- weakest accepted members;
- closest cross-cluster candidates;
- lowest-margin boundaries;
- suspicious labeled merges.

The audit also publishes outcome coverage and labeled quality metrics. Its
continuation graph contains `SpeakerAuditSource -> ApplySpeakerClusters`, with
both nodes pinned to the durable audit ID. The source only emits completed
audits whose linked generic review is approved. Apply repeats that invariant
check before any writeback.

The audit completes in one transaction that stores its metrics, generic review,
and review link. Retry returns the same completed audit and review.

Delete the assignment-audit renderer, ZIP creation, report upload, listening
manifest upload, and artifact identity validation from apply.

## Backend API

Add one generic router with three operations:

- `GET /reviews?run_id=<run_id>` returns the bounded reviews for one job.
- `GET /reviews/{review_id}` returns one complete typed review.
- `POST /reviews/{review_id}/decision` accepts `approved` or `rejected`.

Approval row-locks the pending review, assigns a deterministic continuation run
ID, records the decision, and submits the stored graph through the existing
`BackendManager`. Repeated approval returns the same continuation run rather
than creating another. Rejection never submits a graph. A review without a
continuation may still be approved as a recorded decision.

Job summaries expose `review_count` using the existing paginated jobs query so
the frontend does not issue one request per row.

## Frontend behavior

Jobs with reviews show a `Review` action in their existing row. It opens a
reusable drawer/modal rather than navigating to a new screen.

The drawer contains:

- review title and pending/approved/rejected badge;
- responsive metric cards generated from `ReviewMetric` values;
- warning banners supplied by the producer;
- one tab per `ReviewGroup`;
- virtualized review items;
- existing `WaveformPlayer` instances for audio segment media;
- Approve and Reject actions with confirmation.

Extend the shared waveform player with optional clip start/end bounds and reuse
the existing ranged audio-file and waveform endpoints. No review-specific media
endpoint is added.

After approval, the drawer shows and links the continuation job state. Query
invalidation refreshes both reviews and jobs. Backend errors appear through the
existing toast system and do not optimistically change the decision.

The frontend knows only generic review metrics, fields, groups, tones, media
discriminators, and decisions. It contains no speaker thresholds or cluster
logic.

## Failure and concurrency behavior

- Creating the same producer identity twice returns the original review.
- Only pending reviews may transition; repeating the same decision is
  idempotent and conflicting decisions fail clearly.
- The continuation run ID is deterministic from the review ID, so retries cannot
  launch duplicate assignment jobs.
- Apply requires both a completed speaker audit and an approved linked review.
- Missing audio referenced by a review item produces an unavailable-player state
  without hiding the rest of the review.
- Cancellation during audit scanning leaves no completed audit or review.
- Cancellation during apply retains the existing checkpoint and is resumable by
  retrying the same approved continuation.

## Verification

Temporary tests, removed before handoff, cover:

- typed review payload validation and idempotent CRUD;
- decision conflicts and deterministic continuation IDs;
- backend list/detail/decision contracts;
- generic frontend rendering, media bounds, and decision actions;
- speaker audit persistence without report/listening artifacts;
- source/apply rejection before approval.

Final validation runs a real graph through
`SpeakerEmbeddingSetSource -> ClusterSpeakerEmbeddings -> AuditSpeakerClusters`,
opens its review through the backend, approves it, observes the automatically
submitted `SpeakerAuditSource -> ApplySpeakerClusters` graph, and verifies that
only accepted segments receive voices. It also runs lint, type checking,
frontend build, compilation, Alembic checks, cancellation, and retry checks
through the Nix development shell.
