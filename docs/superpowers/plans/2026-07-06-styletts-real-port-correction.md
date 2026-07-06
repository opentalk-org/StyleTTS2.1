# StyleTTS Real Port Correction Plan

> **For agentic workers:** use `superpowers:subagent-driven-development`. Treat the merged node branch as graph/schema scaffolding plus a few real ports, not as a completed source-code port.

## Goal

Port source StyleTTS Studio behavior into v2 runner/backend nodes without rewriting v2 persistence or segment semantics. V2 segment storage remains JSON entries on `AudioFile.segments`; multiple entries may share a time range, and each segment entry has one text/phoneme pair.

## Current Reality

- Real or mostly real: segment CRUD/writeback, split grouping/extraction shape, normalize, some statistics, asset/checkpoint refs.
- Partial: manifest generation, StyleTTS config generation, DeepFilterNet dependency bridge.
- Scaffold/bridge: StyleTTS training, F0 training, ASR training, StyleTTS synthesis, catalog downloads, phonemizer.
- Must not remain hidden: node execution must call runner-owned Python code directly.

## Correct Implementation Order

### Task 1: Training Manifest Compatibility

- [x] Make `BuildTrainingManifestNode` write source-compatible `wav|joined_phonemes|speaker` rows.
- [x] Materialize source WAV files through `shared.db.audio.crud.read_audio_file`.
- [x] Materialize files in the same directory referenced by manifest rows, including configured `root_path`.
- [x] Preserve stacked segment semantics by sorting and joining all usable segment entries for the same source audio.
- [x] Preserve per-segment text/phoneme pairs in the JSONL sidecar.
- [x] Make F0/ASR training manifest input required.

### Task 2: StyleTTS Config And Layout

- [x] Add runner-side StyleTTS config templates from source `data/base.yaml`, `data/asr.yml`, and `data/plbert.yml`.
- [x] Add library layout helpers for `config.yml` and latest `.pth` resolution.
- [x] Make `BuildStyleTtsFinetuneConfigNode` write real `config.yaml`, not a generic JSON wrapper.
- [x] Use typed checkpoint/assets ports on real preparation nodes instead of scaffold JSON unions.
- [x] Count phoneme symbols by symbol list length, not serialized string character length.
- [x] Require generated StyleTTS config input for `StyleTtsFinetuneNode`.
- [ ] Add strict slot-layout resolution for ASR bundle config/weights, F0 inner weights, and PL-BERT `.t7` plus `config.yml`.
- [ ] Preserve symbol-count resizing metadata for ASR, PL-BERT, and base checkpoint mismatches.

### Task 3: Symbols And Phonemization

- [x] Port source StyleTTS symbol table and text cleaner.
- [ ] Replace `_placeholder_phonemes` in production text nodes.
- [ ] Keep alternate phonemizations as multiple segment entries at the same range.

### Task 4: StyleTTS Finetune Runtime

- [x] Move source `training/*` modules under the `StyleTtsFinetune` node package.
- [x] Keep training core free of `runflow`, `Node`, and port imports.
- [ ] Keep model/training internals out of `src/runflow`.
- [ ] Replace `StyleTtsFinetuneNode` external-command bridge with direct runner training invocation.
- [ ] Publish epoch checkpoints through `shared.db.assets.crud.create_checkpoint`.
- [x] Emit metrics/artifacts through runner-owned training code and v2 asset CRUD, not source processing-job state.

### Task 5: F0 Training Runtime

- [ ] Port F0 dataset, optimizer, trainer, and checkpoint finalizer.
- [ ] Add missing settings: `lambda_f0`, `num_workers`, optimizer scheduler controls, and checkpoint interval naming.
- [ ] Validate and publish `final.pth` as a v2 checkpoint folder.

### Task 6: ASR Training Runtime

- [ ] Port ASR reference config loading, dataset, trainer, and checkpoint publisher.
- [x] Add ASR config node helper for symbols, CTC blank, `n_token`, and effective YAML.
- [ ] Publish ASR bundle with `final.pth`, `asr_train_config.yaml`, and symbols metadata.

### Task 7: StyleTTS Synthesis Runtime

- [ ] Port checkpoint loading, style reference resolution, text cleaning/symbol resolution, synthesis, and sweep behavior.
- [x] Replace synthesis command bridge.
- [ ] Save generated audio through v2 audio CRUD/writeback nodes.

### Task 8: Catalog Downloads

- [ ] Port `styletts2_catalog` HTTP/GitHub/download logic.
- [ ] Store downloaded checkpoint/extra-file artifacts through shared asset CRUD.
- [ ] Remove catalog scaffold metadata responses.

## Verification

- Run `python -m compileall src/backend src/shared src/runner src/runflow`.
- Run schema export smoke and confirm training nodes expose required manifest/config ports.
- Do not add tests; repository instructions explicitly say there are no tests and not to add them.
- Frontend build remains blocked until `src/frontend` dependencies are installed.
