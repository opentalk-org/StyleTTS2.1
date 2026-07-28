# FineWiki TTS Text Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ignored `imports/tts_text_data` workspace with a deterministic FineWiki selector that emits 741 per-voice text files containing exactly 101,250 balanced source-text lines.

**Architecture:** A small local package owns the immutable corpus plan, normalization and prior-sample matching, phoneme-bin allocation, Parquet selection, atomic output, and validation. It reads `/workspace/lang-pl-bert` only as an external data source and publishes generated files beneath `imports/tts_text_data/output`.

**Tech Stack:** Python 3.12, PyArrow, standard-library dataclasses/hashlib/difflib/tempfile, project Nix development shell.

## Global Constraints

- Do not modify any file under `/workspace/lang-pl-bert`.
- Registered voices receive 450 lines; Piper voices receive 90 lines.
- Generate exactly 101,250 lines for the 15 approved languages.
- Exclude source rows whose normalized phonemes match old Lang-PL-BERT TXT phonemes at a score of at least 0.70.
- Write normalized source text, never phoneme strings.
- Prefer unique rows and never use one source row more than twice corpus-wide.
- Run all Python commands through `nix develop --command python`.
- Remove temporary tests before completion.

---

### Task 1: Clean workspace and define the corpus plan

**Files:**
- Delete: `imports/tts_text_data/`
- Create: `imports/tts_text_data/tts_text/__init__.py`
- Create: `imports/tts_text_data/tts_text/config.py`
- Test temporarily: `imports/tts_text_data/tests/test_config.py`

**Interfaces:**
- Produces: `Bin(lower: int, upper: int)`, `LanguagePlan(code: str, lines: int, registered: int)`, `BINS`, `LANGUAGES`, `voice_plans()`.

- [ ] **Step 1: Remove the explicitly superseded ignored workspace**

Resolve the exact target with `realpath imports/tts_text_data`, verify that it is `/workspace/styletts_studio_v2/imports/tts_text_data`, then remove only that directory and recreate the package/test directories.

- [ ] **Step 2: Write failing plan-contract tests**

Test that the configuration produces 96 registered and 645 Piper voices, that their line counts sum to 101,250, and that each language total matches the approved table.

- [ ] **Step 3: Run the focused test and confirm the intended failure**

Run:
`nix develop --command python -m unittest discover -s imports/tts_text_data/tests -p 'test_config.py' -v`

Expected: import failure because `tts_text.config` does not exist.

- [ ] **Step 4: Implement immutable configuration**

Define 32 bins `(1, 16), (16, 32), …, (496, 513)` and the exact language table. Derive Piper count using:

```python
piper = (plan.lines - plan.registered * 450) // 90
```

Assert exact divisibility and generate stable identities
`registered-<lang>-NNN` and `piper-<lang>-NNN`.

- [ ] **Step 5: Run the focused test**

Expected: all configuration tests pass.

### Task 2: Normalize and identify previously sampled rows

**Files:**
- Create: `imports/tts_text_data/tts_text/normalize.py`
- Create: `imports/tts_text_data/tts_text/exclusion.py`
- Test temporarily: `imports/tts_text_data/tests/test_exclusion.py`

**Interfaces:**
- Produces: `normalize_text(value: str) -> str`, `normalize_phonemes(value: str) -> str`, `ExclusionIndex.from_lines(lines)`, and `ExclusionIndex.score(value: str) -> float`.

- [ ] **Step 1: Write failing normalization and threshold tests**

Cover NFC normalization, `<m/>` removal, whitespace collapse, exact matches,
scores below and above 0.70, length-incompatible strings, and short strings.

- [ ] **Step 2: Run the focused tests and confirm missing-interface failures**

Run the two temporary test modules through Nix.

- [ ] **Step 3: Implement normalization and exhaustive matching**

Use batched `rapidfuzz.fuzz.ratio` matrices to compare every source value with
every normalized prior line in its language. Report every prior-line index
meeting the threshold, including duplicate prior lines. Return the maximum
score, or 0.0 when no prior line meets the threshold.

- [ ] **Step 4: Run tests and confirm the 0.70 boundary**

Expected: all normalization/exclusion tests pass.

### Task 3: Allocate balanced bins per voice

**Files:**
- Create: `imports/tts_text_data/tts_text/balance.py`
- Test temporarily: `imports/tts_text_data/tests/test_balance.py`

**Interfaces:**
- Consumes: `BINS` and voice plans.
- Produces: `length_bin(length: int) -> int | None` and `bin_quotas(line_count: int, rotation: int) -> tuple[int, ...]`.

- [ ] **Step 1: Write failing quota tests**

Assert that 450 produces 14 or 15 rows per bin and totals 450; 90 produces two
or three rows per bin and totals 90; rotations move remainder bins; lengths 1
and 512 are accepted while 0 and 513 are rejected.

- [ ] **Step 2: Verify red**

Run `test_balance.py`; expected failure is the missing balance module.

- [ ] **Step 3: Implement rotating even quotas**

Compute `base, remainder = divmod(line_count, len(BINS))`, initialize every bin
with `base`, and increment `remainder` bins beginning at
`rotation % len(BINS)`.

- [ ] **Step 4: Verify green**

Expected: all balance tests pass.

### Task 4: Select FineWiki rows with bounded reuse

**Files:**
- Create: `imports/tts_text_data/tts_text/models.py`
- Create: `imports/tts_text_data/tts_text/select.py`
- Test temporarily: `imports/tts_text_data/tests/test_select.py`

**Interfaces:**
- Produces: `SourceRow(language, row_index, text, phonemes, bin_index)`,
  `VoiceSelection(identity, kind, language, rows, bin_counts)`, and
  `select_language(parquet_path, old_txt_path, plan, seed)`.

- [ ] **Step 1: Write failing selection tests with tiny Parquets**

Create temporary PyArrow Parquets containing known bins. Assert excluded rows
are absent, output values equal `text`, rows are unique on the first pass,
second-pass reuse never exceeds two, deterministic seeds reproduce output, and
a genuine shortage raises an error naming the deficient voice and bins.

- [ ] **Step 2: Verify red**

Run `test_select.py`; expected failure is the missing selector.

- [ ] **Step 3: Implement batched Parquet collection**

Validate exact columns `lang`, `text`, and `phonemes`. Iterate batches, enforce
the filename language, normalize both strings, assign phoneme bin, apply the
0.70 exclusion index, and retain candidates by bin in deterministic
Blake2-priority order. Bound each bin with a priority heap sized to its
aggregate voice quota, count all eligible rows, and retain exclusion statistics.

- [ ] **Step 4: Implement voice assignment**

Assign unique candidates across voices first. For remaining deficits, admit a
second occurrence from the same bin while rejecting any third occurrence and
any duplicate within one voice. Raise an actionable `ValueError` if deficits
remain.

- [ ] **Step 5: Verify green**

Expected: all selector tests pass.

### Task 5: Publish atomic per-voice files and manifest

**Files:**
- Create: `imports/tts_text_data/tts_text/output.py`
- Create: `imports/tts_text_data/generate.py`
- Test temporarily: `imports/tts_text_data/tests/test_output.py`

**Interfaces:**
- Consumes: all language selections.
- Produces: `write_corpus(selections, output_dir, metadata) -> Path` and the CLI
  options `--parquet-dir`, `--prior-txt-dir`, `--output-dir`, and `--seed`.

- [ ] **Step 1: Write failing atomic-output tests**

Assert one line per selected text, paths of the form
`voices/<lang>/<kind>/<block>/<identity>.txt`, no block above 15 files, manifest
counts matching files, and preservation of an old output when validation
fails.

- [ ] **Step 2: Verify red**

Run `test_output.py`; expected failure is the missing writer.

- [ ] **Step 3: Implement atomic writing**

Write into a sibling `.<name>-<random>` directory. Normalize newlines in each
text, validate all selections, write blocks of at most 15 files, and serialize
a manifest with plan, per-voice bin counts, source row indices, exclusions,
reuse counts, and input paths. Replace only the generated output directory
after all validation succeeds, using an atomic directory exchange when an old
output is present and retaining failed staging output for diagnostics.

- [ ] **Step 4: Implement the CLI**

Default inputs to `/workspace/lang-pl-bert/data/parquet` and
`/workspace/lang-pl-bert/data/language_phonemes`; default output to
`imports/tts_text_data/output`; print a compact JSON completion summary.

- [ ] **Step 5: Verify green**

Expected: all temporary tests pass.

### Task 6: Generate and audit the complete corpus

**Files:**
- Generate: `imports/tts_text_data/output/`
- Delete: `imports/tts_text_data/tests/`

**Interfaces:**
- Produces the final 741 TXT files and `manifest.json`.

- [ ] **Step 1: Run the complete generator**

Run:

```bash
nix develop --command python imports/tts_text_data/generate.py
```

Expected summary: 15 languages, 741 voices, 101250 lines.

- [ ] **Step 2: Run an independent audit**

Through Nix Python, read `manifest.json` and every TXT file. Assert exact
language totals, 96/645 voice totals, 450/90 per-file counts, nonempty
single-line text, at-most-two source-row use, and manifest/file agreement.

- [ ] **Step 3: Re-run temporary tests before removing them**

Run the entire temporary suite and confirm zero failures.

- [ ] **Step 4: Remove temporary tests**

Delete only `imports/tts_text_data/tests`, as required by repository policy.

- [ ] **Step 5: Run final production verification**

Run `compileall` on the generator package, rerun the independent corpus audit,
run `git diff --check`, and verify `/workspace/lang-pl-bert` has no changes.
