# Pack Sharding and Repacking Design

## Goal

Use 256 MiB target packs for audio and waveforms. Store pack objects in folders
that normally contain about 256 files and continue to behave well under repeated
delete/add workloads. After validating the write path, repack all existing audio
packs and verify database metadata and stored bytes before removing old objects.

## Pack layout

Audio objects use `audio-packs/<folder-id>/<pack-id>.bin`. Waveform objects use
`waveform-packs/<folder-id>/<pack-id>.bin`. Folder and pack identifiers are UUIDs.
Existing object paths remain readable throughout migration.

A shared pack-folder allocator examines registered bucket-file paths for the
requested prefix. It chooses an existing folder with fewer than 256 registered
objects or creates a new UUID folder. Deleted packs therefore free capacity that
later writes can reuse. Concurrent allocation may exceed 256 by a small amount;
the limit is a distribution target, not a correctness invariant.

## Write behavior

Audio and waveform pack configurations default to 256 MiB. Each writer obtains a
folder through the allocator and creates packs inside it. CRUD transaction and
staged-object cleanup behavior remain authoritative: PostgreSQL owns paths and
metadata, while the Storage Box owns object bytes behind the storage facade.

Oversized individual payloads remain valid single-object packs. Normal pack
creation must remain batch-aware, cancellable where a caller provides a context,
and safe under concurrent writers.

## Existing audio migration

Migration runs in tmux through a dedicated command using shared CRUD and storage
facades. It processes bounded groups so it never materializes the full corpus.
For each group it:

1. Reads live audio bytes from existing packs through audio CRUD.
2. Writes replacement 256 MiB packs under the sharded layout.
3. Atomically updates audio pack references, offsets, lengths, and replacement
   pack metadata in PostgreSQL.
4. Commits before old packs become eligible for deletion.
5. Deletes old objects through pack cleanup only after no audio rows reference
   them.

A failed group leaves committed audio readable from either the old or replacement
pack and staged-object cleanup removes uncommitted uploads. The operation can be
rerun safely by selecting packs outside the new layout.

Existing waveform packs are not migrated. Waveform writes adopt the new target
and folder layout after validation.

## Validation

Before migration, real graph/API smoke tests create, read, update, and delete audio
and waveform data through their public CRUD paths. Tests cover concurrent writers,
folder reuse after deletion, minor folder overshoot, oversized payloads, and staged
upload cleanup.

Migration proceeds only after those tests pass. Final verification requires:

- every audio row still belongs to the same datasets and retains metadata,
  segments, duration, language, and byte length;
- audio bytes read through CRUD match pre-migration hashes;
- no audio row references an absent pack and every registered pack object exists
  with the recorded size;
- old unreferenced audio objects are removed;
- new audio packs target 256 MiB and use folders averaging roughly 256 objects;
- representative workflows and R3 readbacks pass after migration.

The migration reports pack counts, folder populations, total/used bytes, orphan
objects, and size mismatches before and after repacking.
