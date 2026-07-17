# Beetle Training Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete configurable Beetle model, database data pipeline, three continuously running training scripts, exact checkpoint resume, and stage-specific validation artifacts.

**Architecture:** A strict configuration and research layer feeds a database-native bulk data pipeline, explicit PyTorch models, focused losses, and one callback-driven continuous trainer. The three CLI scripts compose stage-specific model/loss/optimizer sets; a future Runflow node supplies adapters without changing core behavior.

**Tech Stack:** Python 3.12, PyTorch, torchaudio, Transformers, Pydantic v2, SQLAlchemy shared CRUD facades, PostgreSQL, S3-compatible packed audio, YAML, Nix development shell.

## Global Constraints

- Work mainly in `src/runner/nodes/training/beetle/`; shared changes require a demonstrated missing general CRUD capability.
- Keep every project-owned file below 300 lines and every project-owned folder below 16 files.
- Run all Python, pytest, CLI, and formatting commands through `nix develop --command`.
- Keep `src/runflow` domain-agnostic and do not register a Beetle node in this baseline.
- Use explicit model classes with configurable dimensions; do not build architecture graphs dynamically.
- Do not add Wave-U-Net or SLM discriminators; reuse current StyleTTS multi-period and multi-resolution spectrogram discriminators.
- Do not add an epoch concept. Training cycles continuously and all schedules use optimizer steps.
- Keep `src/runner/nodes/training/beetle/` below 20 GB and exclude weights, datasets, generated audio, caches, and run outputs.
- Preserve all pre-existing worktree changes. Use temporary tests under `/tmp` and remove them before completion.
- Do not implement generative losses until the primary-paper/reference research gate passes.

---

## Execution order

The linked plans are sequential. Each ends in a reviewable, testable deliverable and must be completed before the next starts.

1. [Research and configuration](2026-07-17-beetle-research-config.md)
2. [Database data pipeline](2026-07-17-beetle-data-pipeline.md)
3. [Stage 1 audio models](2026-07-17-beetle-stage1-models.md)
4. [Stage 2 conditioning and generative models](2026-07-17-beetle-stage2-models.md)
5. [Continuous training runtime and scripts](2026-07-17-beetle-training-runtime.md)

## Locked file map

```text
src/runner/nodes/training/beetle/
├── __init__.py
├── README.md
├── main.md
├── config/
│   ├── __init__.py
│   ├── architecture.py
│   ├── data.py
│   ├── training.py
│   ├── load.py
│   └── default.yaml
├── data/
│   ├── __init__.py
│   ├── records.py
│   ├── index.py
│   ├── cuts.py
│   ├── sampling.py
│   ├── source.py
│   ├── audio.py
│   ├── collate.py
│   └── pipeline.py
├── models/
│   ├── __init__.py
│   ├── audio_encoder.py
│   ├── features.py
│   ├── decoder.py
│   ├── generator.py
│   ├── phoneme.py
│   ├── context.py
│   ├── embeddings.py
│   ├── duration.py
│   ├── latent_flow.py
│   ├── aligner.py
│   ├── text_prompt.py
│   ├── discriminators.py
│   ├── bundle.py
│   └── modules/
│       ├── __init__.py
│       ├── convolution.py
│       ├── conditioning.py
│       ├── flows.py
│       ├── pooling.py
│       ├── source.py
│       └── istft.py
├── losses/
│   ├── __init__.py
│   ├── acoustic.py
│   ├── adversarial.py
│   ├── duration.py
│   ├── flow.py
│   ├── alignment.py
│   ├── embeddings.py
│   └── composition.py
├── training/
│   ├── __init__.py
│   ├── callbacks.py
│   ├── state.py
│   ├── checkpoint.py
│   ├── optimizer.py
│   ├── loop.py
│   ├── stage1.py
│   ├── stage2.py
│   ├── stage3.py
│   ├── validation.py
│   └── runtime.py
└── scripts/
    ├── __init__.py
    ├── common.py
    ├── train_stage1.py
    ├── train_stage2.py
    └── train_stage3.py
```

`external/`, `files/`, and `papers/` contain reference material described by the research plan. Upstream clone contents are not modified.

## Completion gate

- [ ] Every linked plan is complete and reviewed.
- [ ] Temporary tests pass through Nix and are removed.
- [ ] A synthetic uninterrupted run and save/resume run produce equivalent next-step state for all three stages.
- [ ] Stage 3 updates both current StyleTTS discriminator families.
- [ ] Parameter reporting places inference-time Beetle modules within 100M–120M, excluding TextEncoder and training-only models.
- [ ] All three CLIs validate config and report empty-dataset eligibility without loading model weights.
- [ ] `find src/runner/nodes/training/beetle -type f -printf '%s\n' | awk '{s+=$1} END {print s}'` reports less than 20 GB.
- [ ] `git diff --check` passes and unrelated dirty files remain untouched.
