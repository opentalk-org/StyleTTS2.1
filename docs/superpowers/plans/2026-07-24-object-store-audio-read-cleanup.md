# Object Store Audio Read Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make object storage the sole owner of remote byte access and collapse audio reads to one metadata query plus one storage call.

**Architecture:** `shared.storage.ObjectStore` defines the actual storage operations and `S3ObjectStore` implements them. Audio CRUD converts database rows to ordered `ObjectRange` values; storage performs concurrent range reads and returns ordered bytes.

**Tech Stack:** Python, boto3, SQLAlchemy, Pydantic

## Global Constraints

- Storage remains domain-agnostic and must not import audio or SQLAlchemy models.
- Audio has no persistent local cache.
- Missing rows, invalid ranges, external records, and short responses fail.
- Project commands run through `./nix/run-venv.sh`.
- Temporary verification scripts are not committed.

---

### Task 1: Canonical storage contract

**Files:**
- Modify: `src/shared/storage/object_store.py`
- Modify: `src/shared/storage/__init__.py`

**Interfaces:**
- Produces: `ObjectRange(path: str, byte_offset: int, byte_length: int)`
- Produces: `ObjectStore` with `upload`, `upload_path`, `download`, `read_range`, `read_ranges`, `delete`, and `test_connection`
- Produces: `S3ObjectStore(ObjectStore)`

- [ ] Add validated `ObjectRange` and the canonical abstract storage interface.
- [ ] Make `S3ObjectStore` inherit it.
- [ ] Implement ordered concurrent `read_ranges(requests)` inside S3 storage and reject short responses.
- [ ] Verify ordering and short-response failure with an inline fake-client script.

### Task 2: Collapse audio byte reads

**Files:**
- Modify: `src/shared/db/audio/files.py`
- Modify: `src/shared/db/audio/packed.py`
- Modify: `src/shared/db/audio/pack_store.py`
- Modify: `src/shared/db/audio/maintenance.py`
- Modify: `src/shared/db/audio/crud.py`

**Interfaces:**
- Consumes: `ObjectStore`, `ObjectRange`
- Produces: direct `read_audio_file`, `read_audio_part`, and `bulk_read_audio_files`

- [ ] Add one audio-location query returning rows in requested-ID order.
- [ ] Build exact `ObjectRange` values in `files.py` and call storage directly.
- [ ] Delete packed read wrappers, `PackedRangeRead`, and their thread pool.
- [ ] Delete the unused bulk partial-read API and update the CRUD export surface.
- [ ] Replace feature-local storage protocol imports with the canonical interface.
- [ ] Verify whole, partial, missing, external, and short reads with temporary database/storage fixtures.

### Task 3: Collapse segment reads and remove audio caching

**Files:**
- Modify: `src/shared/db/audio/ranges/reader.py`
- Modify: `src/shared/db/audio/ranges/wav.py`
- Modify: `src/shared/db/audio/ranges/__init__.py`
- Delete: `src/shared/db/audio/ranges/cache.py`
- Modify: `src/runner/nodes/training/beetle/data/loader.py`
- Modify: `src/runner/nodes/speaker_clustering/source.py`

**Interfaces:**
- Consumes: ordered audio locations and `ObjectStore.read_ranges`
- Produces: `bulk_read_wav_segments(session, requests)`

- [ ] Replace `BulkWavReader` with one function that queries locations once and calls `read_ranges` once.
- [ ] Preserve request ordering while grouping time ranges per audio file.
- [ ] Delete `AudioFileCache`, `BoundedObjectReader`, and unused `read_wav_ranges`.
- [ ] Remove Beetle cache/fetch-worker construction and lifecycle.
- [ ] Remove caller-controlled speaker audio-fetch workers.
- [ ] Verify repeated audio IDs and multiple time ranges return clips in request order.

### Task 4: Unify all storage consumers

**Files:**
- Modify: `src/shared/db/waveforms/pack_store.py`
- Modify: `src/shared/db/waveforms/crud.py`
- Modify: `src/shared/db/staged_objects.py`
- Modify: `src/shared/db/assets/crud.py`
- Modify: `src/shared/db/assets/file_store.py`

**Interfaces:**
- Consumes: canonical `ObjectStore`
- Removes: waveform-local `ObjectStore` and `DeletableObjectStore`

- [ ] Replace concrete `S3ObjectStore` annotations and local protocols with `ObjectStore`.
- [ ] Keep checkpoint/model cache behavior unchanged.
- [ ] Parse every first-party Python file without bytecode writes.
- [ ] Import backend routes and runner registry.
- [ ] Run frontend TypeScript checking and `git diff --check`.
- [ ] Run representative graph smoke tests when local services and assets are available.
