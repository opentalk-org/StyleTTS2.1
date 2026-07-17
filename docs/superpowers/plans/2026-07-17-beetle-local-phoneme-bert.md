# Beetle Local Phoneme BERT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Beetle's ALBERT-specific phoneme encoder with a custom BERT loaded from a local directory and consolidate the phoneme/aligner vocabulary into one 178-token architecture setting.

**Architecture:** `ArchitectureConfig.phoneme_token_count` is the sole phoneme vocabulary contract. `PhonemeEncoder` wraps a supplied `BertModel`; runtime loads that model and its tokenizer from `PhonemeConfig.model_path` with local-only Transformers APIs and performs no explicit checkpoint metadata validation.

**Tech Stack:** Python 3.12, PyTorch, Transformers BERT, Pydantic v2, PyYAML, Nix.

## Global Constraints

- Work only in the current checkout and do not create a branch or worktree.
- Do not use subagents.
- Keep production files below 300 lines and folders below 16 files.
- Keep temporary tests under `/tmp`; do not commit them.
- Run every Python, Ruff, and pytest command through `nix develop --command ...`.
- Do not import implementation modules from another runner node family.
- Do not explicitly validate the local BERT checkpoint's vocabulary or hidden width.
- Preserve the 100M–150M inference parameter report and report if the loaded BERT makes it exceed 150M.

---

### Task 1: One phoneme token-count configuration

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/architecture.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/main.md`
- Modify temporarily: `/tmp/test_beetle_config.py`

**Interfaces:**
- Produces: `ArchitectureConfig.phoneme_token_count: int`, `PhonemeConfig.model_path: str`.
- Removes: `PhonemeConfig.vocabulary_size`, `PhonemeConfig.albert_hidden_channels`, and `AlignerConfig.vocabulary_size`.

- [x] Add a failing temporary config test that asserts the default has `phoneme_token_count == 178`, uses a non-empty local `phoneme.model_path`, and no longer exposes either duplicated vocabulary field or `albert_hidden_channels`.
- [x] Run `nix develop --command uv run --no-sync --with pytest python -m pytest -q /tmp/test_beetle_config.py`; expect the new assertions to fail against the ALBERT/2,048-token configuration.
- [x] Add `phoneme_token_count: int = Field(gt=1)` to `ArchitectureConfig`; replace `PhonemeConfig.pretrained_model` and ALBERT-specific fields with `model_path: str = Field(min_length=1)`; remove `AlignerConfig.vocabulary_size`.
- [x] Set `architecture.phoneme_token_count: 178` and `architecture.phoneme.model_path: src/runner/nodes/training/beetle/files/phoneme_bert` in `default.yaml`; update `main.md` to state the local BERT and sole token-count contract without claiming checkpoint validation.
- [x] Re-run the focused config tests and expect PASS.
- [x] Commit the production changes with `git commit -m 'refactor: configure local beetle phoneme bert'`.

### Task 2: BERT-backed phoneme model composition

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/modules/text.py`
- Modify: `src/runner/nodes/training/beetle/models/stage2.py`
- Modify temporarily: `/tmp/test_beetle_stage2.py`
- Modify temporarily: `/tmp/test_beetle_stage2_bundle.py`

**Interfaces:**
- Produces: `PhonemeEncoder(bert: BertModel, output_channels: int)` and `Stage2Dependencies.phoneme_bert: BertModel`.
- Consumes: the BERT's own `config.hidden_size` for the projection width and performs no separate vocabulary/width comparison.

- [x] Replace temporary ALBERT fixtures with tiny `BertConfig`/`BertModel` fixtures and assert gradients reach `PhonemeEncoder.bert`, masked tokens remain zero, and pooled output geometry is unchanged.
- [x] Add a bundle test whose BERT config deliberately differs from any removed expected-width field and assert `build_stage2_models()` accepts it without explicit metadata validation.
- [x] Run the focused Stage 2 tests and expect failure because the implementation still requires `AlbertModel`, `.albert`, and `Stage2Dependencies.albert`.
- [x] Change imports and typed fields to `BertModel`, rename stable responsibility fields to `.bert` and `.phoneme_bert`, remove ALBERT vocabulary/hidden-width checks, and continue deriving the projection input from `bert.config.hidden_size`.
- [x] Run the focused Stage 2, bundle, Stage 2 runtime, and structure tests; expect PASS.
- [x] Commit with `git commit -m 'refactor: use bert for beetle phonemes'`.

### Task 3: Local-only phoneme resource loading

**Files:**
- Create: `src/runner/nodes/training/beetle/training/runtime.py`
- Modify: `src/runner/nodes/training/beetle/training/__init__.py`
- Modify temporarily: `/tmp/test_beetle_runtime.py`

**Interfaces:**
- Produces: `PhonemeResources`, `load_phoneme_resources(model_path: Path) -> PhonemeResources`.
- `PhonemeResources` contains `model: BertModel` and `tokenizer: BertTokenizerFast`.
- Consumes: a local Transformers directory; both loaders receive `local_files_only=True`.

- [x] Add a failing temporary test that monkeypatches `BertModel.from_pretrained` and `BertTokenizerFast.from_pretrained`, calls `load_phoneme_resources(Path('/models/phoneme-bert'))`, and asserts both receive that path plus `local_files_only=True` and that no vocabulary comparison occurs.
- [x] Run `nix develop --command uv run --no-sync --with pytest python -m pytest -q /tmp/test_beetle_runtime.py`; expect import failure because `training/runtime.py` does not exist.
- [x] Implement the frozen `PhonemeResources` dataclass and local-only loader without path probing, fallback downloads, vocabulary checks, or hidden-width checks; export both through `training/__init__.py`.
- [x] Run the runtime test and all `/tmp/test_beetle_*.py` tests; expect PASS.
- [x] Run Ruff and compileall over `src/runner/nodes/training/beetle`; expect exit zero.
- [x] Run the parameter report with the available synthetic BERT fixture; record the count and explicitly report any result above 150M.
- [x] Commit with `git commit -m 'feat: load local beetle phoneme bert'`.

### Task 4: Resume the standalone training runtime

**Files:**
- Modify: `docs/superpowers/plans/2026-07-17-beetle-training-runtime.md`

**Interfaces:**
- Consumes: `load_phoneme_resources()` during model allocation after database/checkpoint preflight.
- Produces: an updated Task 8 assembly order that loads the local phoneme BERT only after eligibility and resume validation.

- [x] Update the existing runtime plan so `prepare_run()` remains data/checkpoint-only and model allocation uses `PhonemeResources` afterward.
- [x] Verify the runtime plan contains no ALBERT/2,048-token references and retains the Python-script/future-callback-node seam.
- [x] Commit with `git commit -m 'docs: update beetle runtime for local bert'`.
