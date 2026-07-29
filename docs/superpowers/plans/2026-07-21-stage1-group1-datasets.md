# Stage 1 Group 1 Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents and git worktrees are prohibited for this work.

**Goal:** Prepare all 148 logical sources in download Group 1 as audited Stage 1 datasets, processing sources from greatest requested duration to least.

**Architecture:** Each logical source owns a resumable downloader and a metadata-aware preparation adapter under `imports/stage1/<slug>/`. Shared code performs only mechanical selection, audio normalization, schema validation, and auditing; source adapters retain publisher metadata and map it into the common import contract. Work proceeds in disk-bounded batches, with each dataset reaching a verified or evidence-backed terminal state before moving on.

**Tech Stack:** Python 3.12, Pydantic, soundfile/ffmpeg, Hugging Face Hub where applicable, tqdm, Nix development shell, tmux.

## Global Constraints

- Work in the current checkout; do not create subagents, worktrees, or branches.
- Start with Group 1 because it contains the most logical sources (148), then process its sources in descending `Hours to get` order.
- Limit each dataset attempt to 45 minutes and preserve resumable partial downloads.
- Keep disk use below 512 GB; Stage 1 should remain near 250 GB and `tmp/` must be empty for terminal datasets.
- Select by duration while maximizing distinct and balanced speakers and preserving available demographic, accent, style, and recording-condition diversity.
- Output WAV audio at 24 kHz, mono, PCM-24 without padding, duplication, or synthetic duration.
- Retain complete source metadata, transcripts, word alignments, speaker IDs, language, score/MOS, accuracy, style prompt, voice prompt, and timed vocal events when present.
- Do not upload Group 1 to the backend until the whole group passes audit; then verify import fidelity before pruning staged files.
- Run project commands through `nix develop --command ...`; run long acquisition/preparation jobs in tmux.
- Do not commit temporary tests, archives, caches, audio, or generated `data.json` files.
- Mark every terminal dataset in `STATUS.md` with exactly `COMPLETE`, `ACCESS_DENIED`, or `FAIL`.

---

### Task 1: Restore the Stage 1 validation foundation

**Files:**
- Create: `imports/stage1/common/schema.py`
- Create: `imports/stage1/common/audio.py`
- Create: `imports/stage1/common/selection.py`
- Create: `imports/stage1/audit.py`

**Interfaces:**
- Produces `DatasetManifest`, `AudioRecord`, `SegmentRecord`, and `AlignmentWord` Pydantic models for `data.json`.
- Produces parallel `normalize_audio(...)` and speaker-balanced `select_duration(...)` helpers.
- Produces `audit.py <slug...>` with header, duration, count, reference, metadata, and empty-`tmp` checks.

- [ ] Write a throwaway test that constructs valid and invalid manifests, a stereo 48 kHz fixture, and a speaker-skewed selection inventory.
- [ ] Run it with `nix develop --command python /tmp/stage1_foundation_test.py` and confirm missing implementations fail.
- [ ] Implement the four focused modules with no dataset-specific assumptions.
- [ ] Re-run the throwaway test and confirm schema rejection, 24 kHz mono PCM-24 output, and speaker-balanced duration selection.
- [ ] Run `nix develop --command python -m compileall -q imports/stage1` and remove the throwaway fixture.

### Task 2: Prepare VCTK (44 h)

**Files:**
- Create: `imports/stage1/vctk/src/download.py`
- Create: `imports/stage1/vctk/src/prepare.py`
- Create: `imports/stage1/vctk/src/run.sh`
- Generate locally: `imports/stage1/vctk/data.json`
- Generate locally: `imports/stage1/vctk/wavs/*.wav`

**Interfaces:**
- Consumes the official VCTK 0.92 artifact and the shared Stage 1 foundation.
- Produces records for up to 44 h across all available speakers, with `vctk_<speaker_id>`, transcript, age, gender, accent, region, source file identity, and license provenance.

- [ ] Probe the official Edinburgh artifact, record its size/checksum, and verify resumable range support.
- [ ] Write a throwaway adapter test using two speakers and source metadata fixtures; confirm it fails before implementation.
- [ ] Implement resumable download, full inventory validation, speaker-balanced selection, parallel normalization, and atomic manifest writing.
- [ ] Run `timeout 45m nix develop --command bash imports/stage1/vctk/src/run.sh` inside tmux and retain resumable state if the limit expires.
- [ ] Run `nix develop --command python imports/stage1/audit.py vctk`; require 44 rounded hours or document the measured official shortfall.
- [ ] Remove VCTK temporary artifacts and the throwaway test after audit passes.

### Task 3: Prepare the remaining Group 1 sources

**Files:**
- Create per source: `imports/stage1/<slug>/src/download.py` or `download.sh`
- Create per source: `imports/stage1/<slug>/src/prepare.py`
- Create per source: `imports/stage1/<slug>/src/run.sh`
- Generate locally per source: `imports/stage1/<slug>/data.json`, `wavs/`, and empty `tmp/`

**Interfaces:**
- Consumes rows and caveats from `imports/waterfill-75h-by-language-dataset.md` and the Group 1 ordering from `imports/dataset-download-groups-1000h.md`.
- Produces one independently auditable Stage 1 folder per logical source.

- [ ] Process the remaining sources in descending target order, beginning MWA Western Armenian, StarRail, multilingual LibriSpeech, AISHELL-3, YODAS, FCBH Shan, ftspeech, vocal-burst-db, and THAI-SER.
- [ ] Before each source, inspect its complete schema and write a small failing fixture for every source-specific metadata mapping.
- [ ] Implement only that source's downloader and adapter, including all table-row language/configuration allocations and caveats.
- [ ] Run the source job in tmux with a 45-minute timeout; continue resumable work or record reproducible `STATUS.md` evidence when access is impossible.
- [ ] Audit the source and empty `tmp/` before moving to the next source.
- [ ] Continue through the final 0.173 h MESD source without skipping failed or gated sources silently.

### Task 4: Verify, import, and prune Group 1

**Files:**
- Modify if schema coverage requires it: `imports/stage1_backend.py`
- Generate locally: Group 1 audit and import logs outside git.

**Interfaces:**
- Consumes all 148 terminal Group 1 source folders.
- Produces backend datasets whose audio counts and stored annotations match every `data.json` record.

- [ ] Run the complete Stage 1 audit across all Group 1 slugs and reconcile every requested allocation against the waterfill table.
- [ ] Use a throwaway import-contract test to prove every manifest field is retained by `stage1_backend.py`; patch only demonstrated gaps.
- [ ] Start the shared development stack and run the import in tmux through `nix develop --command python imports/stage1_backend.py import ...` with bounded batch size and workers.
- [ ] Run `nix develop --command python imports/stage1_backend.py verify ...` and independently compare dataset counts, durations, segments, alignments, speakers, prompts, annotations, and metadata.
- [ ] Prune only verified Group 1 staged WAVs and manifests, retain source adapters and terminal evidence, and report reclaimed disk space.

## Self-review

- The plan covers source ordering, disk bounds, timeout/resume behavior, source-specific metadata, audio format, selection diversity, audit, deferred upload, fidelity verification, and safe pruning.
- No compatibility or migration path is introduced; this is a greenfield Stage 1 contract.
- VCTK is the first executable source because it is the largest Group 1 allocation.
