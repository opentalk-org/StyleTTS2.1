# Audio Annotation Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize voice prompts, style prompts, scores, and genuine transcription accuracy values for every audio row without changing metadata, transcripts, or audio data.

**Architecture:** Add a small shared CRUD facade that projects only annotation columns and performs typed partial updates. A temporary dataset-aware backfill program will derive proposals, dry-run them, apply them in batches, and verify immutable-column fingerprints before being removed.

**Tech Stack:** Python 3.12, Pydantic, SQLAlchemy, PostgreSQL JSONB, Nix development shell.

## Global Constraints

- Never read audio bytes.
- Never modify `metadata`, `segments`, alignment, storage, dataset membership, language, or speaker identity.
- Decode opaque values only with dataset-specific evidence.
- Prefer per-audio MOS over system or condition aggregates.
- Populate accuracy only from transcription accuracy/confidence evidence.
- Run every Python and test command through `nix develop --command python ...`.
- Keep temporary validation tests and the backfill program out of the finished tree.

---

### Task 1: Partial annotation CRUD facade

**Files:**
- Create: `src/shared/db/audio/annotations/schemas.py`
- Create: `src/shared/db/audio/annotations/crud.py`
- Test temporarily: `.tmp_test_annotation_backfill.py`

**Interfaces:**
- Produces: `AudioAnnotationRow`, `AudioAnnotationUpdate`, `iter_audio_annotations(session, batch_size)`, and `bulk_update_audio_annotations(session, updates)`.
- The row projection includes ID, dataset names, prompts, score, accuracy, metadata, and immutable metadata/segment hashes; the update type contains only style prompt, voice prompt, score, and accuracy.

- [ ] Write a temporary test asserting the update schema has exactly four writable fields and the generated SQL update changes no immutable column.
- [ ] Run `nix develop --command python .tmp_test_annotation_backfill.py` and verify it fails because the facade does not exist.
- [ ] Implement keyset-paginated projection and `executemany` updates. Reject empty update sets and non-finite numbers.
- [ ] Re-run the temporary test and verify it passes.

### Task 2: Dataset-aware derivation and dry run

**Files:**
- Create temporarily: `.tmp_audio_annotation_backfill.py`
- Read: `imports/order.md`
- Read: relevant `imports/stage1/**/prepare.py` files.

**Interfaces:**
- Consumes: `AudioAnnotationRow`.
- Produces: immutable `AnnotationProposal` values and a JSON dry-run report.

- [ ] Implement readable keyword composition with stable ordering and duplicate removal.
- [ ] Implement explicit dataset mappings for opaque emotion, intensity, age, gender, MOS, and transcription-confidence fields found in the full database inventory.
- [ ] Preserve existing prompts unless a proposal is readable and materially richer; reject numeric-only results.
- [ ] Run the dry run against every row and inspect counts, conflicts, and before/after examples.
- [ ] Correct false-positive mappings, especially identifiers, system-level MOS, transcript text, political affiliation, and non-audible metadata.

### Task 3: Apply and verify

**Files:**
- Use temporarily: `.tmp_audio_annotation_backfill.py`
- Remove: `.tmp_audio_annotation_backfill.py`
- Remove: `.tmp_test_annotation_backfill.py`

**Interfaces:**
- Consumes: validated `AnnotationProposal` values.
- Produces: committed partial annotation updates and a post-run audit.

- [ ] Capture aggregate metadata and segment fingerprints before applying.
- [ ] Apply proposals in bounded transactions through `bulk_update_audio_annotations`.
- [ ] Re-query every row and assert metadata/segment fingerprints and row counts are unchanged.
- [ ] Assert no numeric-only prompt remains among changed rows and all written scores/accuracy values are finite.
- [ ] Report per-field and per-dataset update totals plus unresolved cases.
- [ ] Remove temporary files and run `git status --short` to confirm only the reusable facade and approved documentation remain.
