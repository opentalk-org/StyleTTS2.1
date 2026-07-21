# Stage 1 Dataset Imports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents and worktrees are prohibited by the repository instructions. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce audited Stage 1 WAV and metadata imports for all 46 datasets in `imports/stage1.md`, or reproducibly document why a dataset is impossible.

**Architecture:** Each corpus owns a downloader and metadata-aware preparation adapter. The audit is the completion authority and checks real output files rather than trusting logs or status prose.

**Tech Stack:** Nix dev shell, Python 3.12, soundfile, scipy, datasets/huggingface_hub, curl, tmux, JSON.

## Global Constraints

- Run every import attempt in tmux through `nix develop --command ...`, with a 45-minute limit per dataset attempt.
- Emit only 24,000 Hz mono PCM-24 WAV files.
- Preserve every available publisher metadata field and never invent speaker identities, labels, transcripts, or scores.
- Never duplicate, pad, or synthesize audio to reach a duration target.
- Keep `tmp/` empty for every terminal dataset and keep total disk use below 512 GB.
- Do not retain repository tests; temporary regression checks must be removed before completion.

---

### Task 1: Completion audit

**Files:**
- Modify: `imports/stage1/audit.py`

**Interfaces:**
- Consumes: all `imports/stage1/<slug>/data.json`, WAV files, and `STATUS.md` files.
- Produces: one PASS, FAIL, MISSING, or IMPOSSIBLE result per dataset and a failing exit status unless all 46 are terminal.

- [ ] Verify the audit rejects wrong duration, count, sample rate, channels, subtype, missing fields, missing metadata, extra WAV files, and nonempty `tmp/`.
- [ ] Run `nix develop --command python imports/stage1/audit.py`; expect a nonzero exit until all datasets are terminal.
- [ ] Keep target values synchronized exactly with the 46 rows in `imports/stage1.md`.

### Task 2: Reconcile prepared and active datasets

**Files:**
- Modify as failures require: `imports/stage1/{pstn_speech_quality_corpus,urgent_2024_human_mos,asvp_esd_v2,emogator,fsd50k,vocalsound,singmos_pro,nonspeech7k,esc50}/src/*`

**Interfaces:**
- Consumes: current tmux jobs, partial downloads, archives, and generated WAVs.
- Produces: audited terminal outputs for the nine named datasets.

- [ ] Inspect each active session and its log, distinguishing waiting, downloading, converting, failed, and completed states.
- [ ] For every parser failure, write a temporary minimal regression check and observe the expected failure.
- [ ] Make the smallest metadata-aware correction, rerun the temporary check, then remove it.
- [ ] Resume conversions in tmux and run the audit after each completed dataset.

### Task 3: Remaining vocal-event corpora

**Files:**
- Create or modify downloader and preparation code under:
  `imports/stage1/{podcastfillers,audioset,vggsound,fsdkaggle2019,coughvid,coswara,icbhi_2017}/src/`
- Produce: the corresponding `wavs/`, `tmp/`, and `data.json` artifacts.

**Interfaces:**
- Consumes: official releases, official URL inventories, or byte-identical public mirrors.
- Produces: event-tagged segments and complete source provenance up to each requested duration.

- [ ] Work in descending target duration and benchmark any preparation still running after four minutes.
- [ ] Preserve timestamps for PodcastFillers and clip-level multilabel taxonomies for general sound datasets.
- [ ] Preserve per-file licenses for Freesound-derived corpora and availability evidence for URL-derived corpora.
- [ ] Run each importer in tmux, clear `tmp/`, and require its audit row to become PASS.

### Task 4: Emotion and expression corpora

**Files:**
- Create downloader and preparation code under:
  `imports/stage1/{msp_podcast,beat,mead,esd,emov_db,subesco,emozionalmente,crema_d,shemo,ased,jl_corpus,enterface_05,emodb,mesd}/src/`
- Complete existing code under `imports/stage1/{emns_imz,cafe,aesdd}/src/`.

**Interfaces:**
- Consumes: official archives and their speaker/emotion/intensity tables.
- Produces: dataset-prefixed authentic speaker IDs, style prompts backed by released labels, transcripts when released, and complete raw metadata.

- [ ] Validate speaker, label, intensity, language, and transcript mappings across each complete source inventory before bulk conversion.
- [ ] Keep intended and perceived emotion ratings separate wherever both exist.
- [ ] Normalize in parallel, retain the requested amount of authentic audio, clear `tmp/`, and require every audit row to pass.

### Task 5: Remaining MOS and quality corpora

**Files:**
- Complete existing code under `imports/stage1/{nisqa,somos,tcd_voip,blizzard_challenge_2019,chime_7_udase}/src/`.
- Create downloader and preparation code under:
  `imports/stage1/{tencent_speech_quality,tmhint_qi,bvcc_voicemos_2022,ttsds2}/src/`.

**Interfaces:**
- Consumes: audio, utterance/system score tables, raw listener judgments, degradation conditions, and split metadata.
- Produces: mean score fields plus uncropped raw rating/protocol metadata.

- [ ] Preserve listener IDs, individual ratings, question/protocol, system IDs, degradation, language, and source grouping fields.
- [ ] Never merge MOS, CMOS, SMOS, intelligibility, or multidimensional quality scales into one unlabeled score.
- [ ] Normalize, clear `tmp/`, and require every audit row to pass.

### Task 6: Exhaust gated and discrepant sources

**Files:**
- Modify: `imports/stage1/{nvspeech,synparaspeech,mnv_17,ehehe_corpus}/STATUS.md`
- Create `STATUS.md` only for any additional dataset proven impossible.

**Interfaces:**
- Consumes: configured credentials, host/API responses, publisher mirrors, release manifests, and measured official audio duration.
- Produces: reproducible impossibility evidence only after automated alternatives are exhausted.

- [ ] Retry authenticated official downloads and verify whether account-side gate acceptance is required.
- [ ] Search publisher-controlled mirrors and public release artifacts without substituting a different corpus.
- [ ] For duration discrepancies, measure the complete authentic release and document why padding or duplication would falsify the corpus.
- [ ] Ensure each impossible status names attempted URLs, date, exact response, and the external action or unavailable bytes preventing completion.

### Task 7: Full completion verification

**Files:**
- Modify: `imports/stage1/audit.py` only if final evidence exposes a missing invariant.

**Interfaces:**
- Consumes: all 46 dataset directories.
- Produces: authoritative all-terminal audit output.

- [ ] Run `nix develop --command python imports/stage1/audit.py` and require all 46 rows to be PASS or IMPOSSIBLE with exit code zero.
- [ ] Independently count dataset rows, `data.json` files, status files, WAV files, and nonempty temporary directories.
- [ ] Run `git diff --check` and verify no temporary tests, downloaded archives, caches, or generated WAV/data files are staged for commit.
- [ ] Check `df -h /workspace` and confirm disk use stayed below 512 GB.
