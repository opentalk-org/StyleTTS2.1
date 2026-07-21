# Stage 1 Dataset Imports Design

## Scope

Prepare every dataset listed in `imports/stage1.md` under `imports/stage1/<slug>/` with source code, normalized WAV files, complete import metadata, and an empty temporary directory. A dataset is terminal only when its requested duration passes the audit or access/source evidence proves the request impossible.

## Structure

Each dataset owns its download and preparation scripts because label taxonomies, speaker identities, transcripts, scores, licenses, and provenance differ. Mechanical audio decoding, resampling, channel reduction, and PCM-24 writing may use a small shared helper when doing so removes real duplication without interpreting metadata.

Every adapter retains the complete publisher row under `metadata`, promotes known fields into the import schema, uses dataset-prefixed speaker identifiers, and leaves genuinely absent values null. It must not infer speakers, labels, language, ratings, or transcripts from unsupported evidence.

## Data Flow

1. Download official artifacts or a byte-identical public mirror into `tmp/` from a resumable tmux command bounded to 45 minutes per attempt.
2. Inspect archive schemas before conversion and validate filename or table invariants across the full source inventory.
3. Select real source clips up to the requested duration without duplication, padding, or synthetic silence.
4. Normalize in parallel to 24 kHz, mono, PCM-24 WAV and create one complete record per output.
5. Write `data.json` only after all workers succeed, then remove temporary artifacts.
6. Run `imports/stage1/audit.py` against file headers, durations, counts, segments, and metadata.

## Failures and Access

Downloads resume from partial files where the host supports ranges. Rate limits receive bounded delayed retries. Dead official URLs are checked against publisher mirrors and public repository artifacts. Gated sources are tested with the configured token and documented as impossible only when the host requires account-side acceptance or approval that cannot be performed programmatically.

If the complete official release is shorter than the requested table duration, the importer preserves all authentic audio and records the measured discrepancy; it never fabricates duration. The status evidence names the authoritative release, measured result, and unsuccessful alternatives.

## Verification

The final audit covers all 46 named datasets. Passing evidence requires the requested rounded duration, matching JSON/WAV counts, required record and segment fields, nonempty provenance metadata, valid 24 kHz mono PCM-24 headers, no unreferenced WAVs, and empty `tmp/`. Impossible entries require a `STATUS.md` with reproducible access or source evidence and no unsupported success claim.
