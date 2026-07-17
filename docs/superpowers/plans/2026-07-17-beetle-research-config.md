# Beetle Research and Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish pinned primary references, verified duration/latent-flow mathematics, and strict configuration contracts before model implementation.

**Architecture:** Reference repositories and paper extracts are local evidence, while concise Markdown derivations lock the equations used later. Split Pydantic modules validate fixed model geometry, data selection, loss schedules, runtime limits, and step-based checkpointing without importing training code.

**Tech Stack:** Git shallow clones, arXiv PDFs, `pdftotext`, Python 3.12, Pydantic v2, PyYAML, hashlib, Nix.

## Global Constraints

- Store clones only under `src/runner/nodes/training/beetle/external/` and paper material only under `beetle/papers/`.
- Pin resolved commits and source URLs in Markdown; never copy equations from memory.
- Keep Beetle below 20 GB and do not download checkpoints or datasets.
- Configuration is strict and step-based; it contains no epoch field.
- Use temporary tests under `/tmp` and run them only through Nix.

---

### Task 1: Pin repositories and primary papers

**Files:**
- Create: `src/runner/nodes/training/beetle/papers/SOURCES.md`
- Create: `src/runner/nodes/training/beetle/papers/<paper>/paper.pdf`
- Create: `src/runner/nodes/training/beetle/papers/<paper>/paper.txt`
- Create: `src/runner/nodes/training/beetle/external/StyleTTS2/`
- Create: `src/runner/nodes/training/beetle/external/piper/`

**Interfaces:**
- Produces: pinned StyleTTS2 and Piper source plus locally searchable paper text.
- Consumes: URLs named in the approved spec.

- [ ] Clone `https://github.com/yl4579/StyleTTS2.git` and `https://github.com/rhasspy/piper.git` with `--depth 1`; record `git rev-parse HEAD`, remote URL, license, and retrieval date in `SOURCES.md`.
- [ ] Download arXiv `2306.07691`, `2106.06103`, `2210.02747`, `2407.01392`, `2410.12557`, `2308.07117`, and `2010.05646` into named paper subfolders.
- [ ] Extract every PDF with `nix develop --command pdftotext -layout paper.pdf paper.txt`; verify each text contains its title and equation section headings.
- [ ] Run `du -sh src/runner/nodes/training/beetle` and fail if it reaches 20 GB.
- [ ] Commit only source manifests and intended reference files: `git commit -m 'docs: add beetle research references'`.

### Task 2: Verify and document generative mathematics

**Files:**
- Create: `src/runner/nodes/training/beetle/papers/duration-flow.md`
- Create: `src/runner/nodes/training/beetle/papers/latent-flow.md`
- Create temporarily: `/tmp/test_beetle_research.py`

**Interfaces:**
- Produces: exact conventions later consumed by `losses/duration.py`, `losses/flow.py`, `models/duration.py`, and `models/latent_flow.py`.
- Consumes: pinned VITS/Piper, Flow Matching, Diffusion Forcing, and Shortcut sources.

- [ ] Write a temporary source-audit test that asserts both notes name tensor shapes, masks, forward/reverse direction, log-determinant sign, probability path, velocity target, per-token `t`, dyadic `d`, EMA target, stop-gradient boundary, and sampling update.
- [ ] Run `nix develop --command pytest -q /tmp/test_beetle_research.py`; expect failure because the notes do not exist.
- [ ] Trace Piper/VITS duration-flow code line by line and document its actual augmented variables, transforms, base density, masking, log determinant, NLL reduction, and reverse path with source file/line citations.
- [ ] Trace the three latent-generation papers and document one internally consistent merge: conditional path definition, analytic velocity, independently sampled valid-token noise levels, shortcut base case, two-half-step EMA bootstrap, loss weighting, and one/multi-step inference.
- [ ] Re-run the audit; expect PASS, then manually compare every equation symbol against cited source text.
- [ ] Commit notes: `git commit -m 'docs: derive beetle generative losses'`.

### Task 3: Strict architecture and data configuration

**Files:**
- Create: `src/runner/nodes/training/beetle/config/__init__.py`
- Create: `src/runner/nodes/training/beetle/config/architecture.py`
- Create: `src/runner/nodes/training/beetle/config/data.py`
- Create temporarily: `/tmp/test_beetle_config.py`

**Interfaces:**
- Produces: `ArchitectureConfig`, `ConditioningConfig`, `ConditionDropoutConfig`, `AudioConfig`, `DataConfig`, and `DatabaseSelection`.
- Consumes: no model or trainer imports.

- [ ] Write failing tests that validate hop 300, 1–45 second cuts, `k` range 1–32, 0.01 phoneme dropout, 0.75 boundary-context dropout, UUID dataset selection, positive prefetch limits, and rejection of unknown fields or incompatible channel dimensions.
- [ ] Run `nix develop --command pytest -q /tmp/test_beetle_config.py`; expect import failure.
- [ ] Implement frozen strict Pydantic models with `ConfigDict(extra="forbid", frozen=True)` and cross-field validators for latent/conditioning/decoder geometry.
- [ ] Re-run the configuration tests; expect PASS.
- [ ] Commit: `git commit -m 'feat: add beetle architecture and data config'`.

### Task 4: Training configuration, YAML loading, and fingerprinting

**Files:**
- Create: `src/runner/nodes/training/beetle/config/training.py`
- Create: `src/runner/nodes/training/beetle/config/load.py`
- Create: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify temporarily: `/tmp/test_beetle_config.py`

**Interfaces:**
- Produces: `BeetleConfig`, `StageConfig`, `LossWeights`, `OptimizerConfig`, `RuntimeConfig`, `CheckpointConfig`, `load_config(path)`, and `config_fingerprint(config)`.
- Consumes: config types from Task 3.

- [ ] Add failing tests asserting the default YAML has three stages, no key containing `epoch` or validation configuration, positive checkpoint cadence, separate Stage 1/3 discriminator settings, all named loss weights, and a stable SHA-256 fingerprint independent of YAML key order.
- [ ] Run the focused tests; expect missing training/load modules.
- [ ] Implement strict nested models, canonical `model_dump(mode="json")` serialization, SHA-256 fingerprinting, and YAML loading that reports the exact invalid field path.
- [ ] Run all config tests and `nix develop --command python -c 'from runner.nodes.training.beetle.config import load_config; print(load_config(...))'`; expect PASS and a validated config.
- [ ] Remove `/tmp/test_beetle_config.py` with `apply_patch` after the suite passes.
- [ ] Commit: `git commit -m 'feat: add beetle training config'`.
