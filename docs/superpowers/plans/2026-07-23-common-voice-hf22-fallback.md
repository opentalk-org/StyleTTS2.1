# Common Voice HF22 Fallback Implementation Plan

> **For agentic workers:** Execute inline. Subagents and git worktrees are prohibited for this task.

**Goal:** Collect the 42 Common Voice locales still missing from the CV26 terms list from the CV22 Hugging Face mirror, producing backend-ready local Stage 1 manifests and normalized WAVs without importing them.

**Architecture:** Three concurrent workers own Parts 1, 2, and 3, so no two processes write the same manifest. Each locale is a resumable unit with a 45-minute deadline: download all five transcript TSVs, stream selected validated audio shards through bounded parallel conversion, merge records atomically into the part manifest, update durable status, and remove locale temporary files before continuing.

**Tech Stack:** Python 3.12 through `nix develop`, requests, tarfile, ffmpeg, soundfile, Pydantic Stage 1 schemas, tmux.

## Global Constraints

- Do not use subagents, git worktrees, branches, commits, or backend import commands.
- Store output under the existing `imports/stage1/common_voice_partN/` dataset folders.
- Normalize every retained clip to 24 kHz, mono, PCM_24 WAV.
- Preserve every selected publisher field in metadata and download metadata for all five CV22 splits.
- Use only validated `train`, `dev`, and `test` audio; never select `other` or `invalidated`.
- Cap each language at 50 hours and at its waterfill target or CV22 validated availability, whichever is lower.
- Enforce 45 minutes per locale, then checkpoint and continue.
- Keep at least 48 GB free and delete each locale's tar/MP3 temporary data before advancing.
- Process the three parts concurrently and prefer the largest target locales first.

---

### Task 1: Build the exact target catalog

**Files:**
- Create: `imports/stage1/common/hf22_catalog.py`

**Interfaces:**
- Produces: immutable `LocaleSpec` records with canonical language, HF locale, part, target hours, split shard counts, and expected metadata counts.

- [ ] Parse remaining locales from `imports/mozilla-common-voice-26-terms-links.md`.
- [ ] Parse CV26 target allocations from `imports/waterfill-75h-by-language-dataset.md`.
- [ ] Apply the seven verified HF locale aliases.
- [ ] Cap targets by 50 hours and CV22 `validHrs`.
- [ ] Assert exactly 42 unique locales distributed 21/9/12 across Parts 1/2/3.

### Task 2: Implement resumable metadata and shard transfer

**Files:**
- Create: `imports/stage1/common/hf22_download.py`

**Interfaces:**
- Consumes: `LocaleSpec`, HF token, locale temporary directory.
- Produces: five verified TSV paths and resumable tar shard paths.

- [ ] Load `HF_TOKEN` from `.env` without logging it.
- [ ] Download with HTTP Range resume, retries, throughput reporting, and a bounded timeout.
- [ ] Download all five TSVs before audio work and record their row counts.
- [ ] Generate an evenly distributed train/dev/test shard order.
- [ ] Check the locale deadline and disk reserve before each network operation.

### Task 3: Implement bounded conversion and manifest records

**Files:**
- Create: `imports/stage1/common/hf22_prepare.py`

**Interfaces:**
- Consumes: one tar shard, parsed publisher rows, current speaker totals, target duration.
- Produces: `AudioRecord` objects and normalized WAVs.

- [ ] Stream tar members without extracting the archive wholesale.
- [ ] Balance accepted clips with a per-speaker duration/count cap derived from the locale target.
- [ ] Convert with a bounded thread pool so MP3 bytes cannot accumulate without limit.
- [ ] Clean enclosing or majority-span ASCII quotes using the established Common Voice rule.
- [ ] Retain the full publisher row, split, release, HF locale, and original sentence when cleaning changed it.
- [ ] Validate each WAV and `AudioRecord` before checkpointing.

### Task 4: Implement the per-part orchestrator

**Files:**
- Create: `imports/stage1/common/hf22_worker.py`
- Create: `imports/stage1/common_voice_part1/src/hf22.py`
- Create: `imports/stage1/common_voice_part2/src/hf22.py`
- Create: `imports/stage1/common_voice_part3/src/hf22.py`

**Interfaces:**
- Command: `nix develop --command python -m imports.stage1.common_voice_partN.src.hf22`
- Produces: merged `data.json`, `HF22_STATUS.json`, `HF22_STATUS.md`, normalized WAVs, and empty locale-specific temporary directories.

- [ ] Load existing records so interrupted locales resume without reconversion.
- [ ] Sort pending locales by descending target hours.
- [ ] Atomically merge each locale's records and actual collected hours into `data.json`.
- [ ] Mark `COMPLETE_LOCAL`, `TIME_LIMIT`, `DISK_LIMIT`, or `FAILED` with counts, duration, bytes, elapsed time, and error.
- [ ] Always remove the locale-specific temporary directory in `finally`.

### Task 5: Test and benchmark

**Files:**
- Temporary test: `imports/stage1/test_hf22_pipeline.py`

- [ ] Write failing tests for catalog membership, shard ordering, quote cleanup, metadata preservation, and manifest merge isolation.
- [ ] Run with `nix develop --command python -m unittest imports.stage1.test_hf22_pipeline -v` and confirm the intended failures.
- [ ] Implement until the tests pass, then remove the temporary test.
- [ ] Benchmark a small locale and tune conversion workers if elapsed time exceeds four minutes.

### Task 6: Run and verify all parts

**Runtime artifacts:**
- `imports/stage1/common_voice_partN/hf22.log`
- `imports/stage1/common_voice_partN/HF22_STATUS.json`
- `imports/stage1/common_voice_partN/HF22_STATUS.md`

- [ ] Start three named tmux sessions, one per part.
- [ ] Monitor sessions, disk free space, throughput, and terminal locale states.
- [ ] Retry recoverable network failures while respecting the per-locale deadline.
- [ ] Validate all manifests with `DatasetManifest`.
- [ ] Inspect every retained WAV header and sum duration/bytes by locale.
- [ ] Assert no locale-specific HF temporary directory remains.
- [ ] Mark Groups 5–7 `COMPLETE_LOCAL` only when every assigned locale is locally complete; otherwise record exact terminal failures.

