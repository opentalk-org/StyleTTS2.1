# TTS Corpus Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove repeated clone conditioning and make corpus generation deterministically shardable across GPU runners.

**Architecture:** Clone runtimes cache only their active voice conditioning, while corpus planning selects deterministic non-overlapping shards before durable resume filtering. A real graph benchmark gates replacement of the active campaign.

**Tech Stack:** Python, PyTorch, Runflow nodes, FastAPI graph submission, PostgreSQL audio metadata

## Global Constraints

- Preserve high-tier model settings.
- Preserve `tts_<engine>` dataset lineage and transcript segments.
- Never delete existing audio or manufacture duplicates.
- Run project commands through `nix develop --command`.
- Do not commit temporary tests or generated audio.

---

### Task 1: Chatterbox Active-Voice Conditioning Cache

**Files:**
- Modify: `src/runner/nodes/tts/engines/chatterbox.py`
- Temporary test: `/tmp/test_chatterbox_conditioning_cache.py`

**Interfaces:**
- Consumes: `Voice.clone` and `ChatterboxMultilingualTTS.prepare_conditionals`
- Produces: `ChatterboxRuntime.synthesize` with identical output contract and one conditioning operation per consecutive clone voice

- [ ] **Step 1: Write the failing test**

Create a temporary test with a fake model that records `prepare_conditionals` and `generate` calls. Invoke `synthesize` twice with the same clone and assert one preparation call and two generation calls without `audio_prompt_path`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
nix develop --command python /tmp/test_chatterbox_conditioning_cache.py
```

Expected: FAIL because the runtime passes an audio prompt on both calls.

- [ ] **Step 3: Implement minimal active-voice caching**

Compute a stable digest from the clone WAV bytes, retain the active digest, call `prepare_conditionals` only when it changes, and call `generate` without `audio_prompt_path`.

- [ ] **Step 4: Run test to verify it passes**

Run the temporary test again. Expected: PASS.

- [ ] **Step 5: Run a real graph benchmark**

Submit at least 100 consecutive jobs through `OtherTtsCorpusSynthesis`, then inspect:

```bash
nix develop --command python -m cli perf <run_id>
nix develop --command python -m cli logs <run_id>
```

Expected: success, correct stored records, and sustained rate materially above the current 0.18 items/second.

### Task 2: Deterministic Corpus Shards

**Files:**
- Modify: `src/runner/nodes/tts/corpus/other.py`
- Modify: `imports/run_other_tts_corpus_campaign.py`
- Temporary test: `/tmp/test_other_corpus_shards.py`

**Interfaces:**
- Consumes: ordered `OtherCorpusJob` plan
- Produces: `shard_index: int`, `shard_count: int` settings selecting plan positions where `position % shard_count == shard_index`

- [ ] **Step 1: Write the failing test**

Create a temporary test for a pure shard selector. Assert that four shards are disjoint and their union equals the complete ordered plan.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
nix develop --command python /tmp/test_other_corpus_shards.py
```

Expected: FAIL because the selector and settings do not exist.

- [ ] **Step 3: Implement deterministic selection**

Add validated shard settings, select plan positions before querying/filtering completed source keys, and expose launcher arguments for shard index and count.

- [ ] **Step 4: Run test to verify it passes**

Run the temporary test again. Expected: PASS.

- [ ] **Step 5: Validate two small real graph shards**

Submit two non-overlapping shard graphs with bounded jobs. Expected: both succeed and no duplicate `source_key` is stored.

### Task 3: Campaign Cutover and Monitoring

**Files:**
- Modify: `imports/run_other_tts_corpus_campaign.py`

**Interfaces:**
- Consumes: online runner list, remaining dataset counts, benchmark throughput
- Produces: one campaign shard per explicitly targeted online runner and a required-rate comparison

- [ ] **Step 1: Record the active run and dataset counts**

Capture the current graph ID and counts without stopping it.

- [ ] **Step 2: Calculate benchmark projection**

Require measured throughput to exceed the prior implementation and report the number of equivalent GPUs needed for the five-hour deadline.

- [ ] **Step 3: Cut over safely**

Stop the prior graph only after the benchmark passes, then launch resumable production shards against available runners.

- [ ] **Step 4: Verify production progress**

Inspect graph status, GPU utilization, stored dataset count growth, dataset flags, and transcript segments.

- [ ] **Step 5: Remove temporary tests and commit**

Delete temporary files, run repository checks relevant to the modified modules, and commit only the implementation files.
