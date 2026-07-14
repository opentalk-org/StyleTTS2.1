# Batched Smart Turn Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a checkpoint-driven, genuinely batched `SmartTurnPredict` node that preserves each audio input and emits a typed completion decision plus probability using Pipecat Smart Turn v3.2 CPU ONNX inference.

**Architecture:** A `turn_models` catalog entry stores the official quantized ONNX file through shared checkpoint CRUD. A focused `runner.nodes.smart_turn` package owns bulk byte loading, upstream-compatible 16 kHz/8-second waveform preparation, batched Whisper feature extraction, the persistent ONNX session, and node lifecycle. Only a reusable boolean runner port is added outside the feature package.

**Tech Stack:** Python 3.11+, runflow, Pydantic, NumPy, librosa, Transformers, ONNX Runtime, Hugging Face Hub, PostgreSQL/shared CRUD, Nix/uv.

## Global Constraints

- Keep Smart Turn and audio-specific behavior out of `src/runflow`.
- Use typed `AudioPort`, `CheckpointRefPort`, `BoolPort`, and `FloatPort`; do not encode scalar results as JSON.
- Execute one ONNX call for the whole runtime batch and return exactly one output per input.
- Fetch absent bytes once with `audio_crud.bulk_read_audio_files` and store weights through existing checkpoint helpers.
- Run Python, uv, backend, runner, and CLI commands through `nix develop --command ...`.
- Never call the node's `execute()` directly; the execution test must use `POST /graphs/runs`.
- Temporary tests are required for TDD but must be removed before commits.
- Preserve unrelated dirty-worktree changes; keep files below 300 lines and folders below 16 files.

## File Map

- Create `src/runner/nodes/smart_turn/{__init__,audio,inference,node}.py`.
- Modify `src/runner/nodes/datatypes.py`, `src/runner/nodes/registry.py`.
- Modify `src/runner/nodes/assets/catalog.py` and `catalog_runtime/tasks.py`.
- Modify `pyproject.toml`, regenerate `uv.lock`.
- Create `workflows/smart_turn_predict.json`; update `workflows/README.md`.
- Create/remove `tests/test_smart_turn_temporary.py`; never commit it.

---

### Task 1: Boolean Port and Model Catalog

**Files:** temporary test; modify `datatypes.py`, `assets/catalog.py`, `assets/catalog_runtime/tasks.py`.

**Produces:** `BoolPort(TYPE_NAME="BOOL", python_type=bool)`; `CatalogKey.TURN_MODELS`; `bootstrap_turn_model(...)`; checkpoint kind `smart_turn`.

- [ ] Write failing tests that assert:

```python
registry = register_runner_types(TypeRegistry())
assert registry.get("BOOL") is BoolPort
assert BoolPort.python_type is bool
assert CatalogKey.TURN_MODELS.value == "turn_models"
```

Patch `ensure_model_checkpoint` and `download_hf_snapshot`, invoke `bootstrap_turn_model("pipecat-ai/smart-turn-v3")`, invoke its captured download callback, and assert exact calls:

```python
assert ensure.call_args.args[:2] == ("smart_turn", "pipecat-ai/smart-turn-v3")
assert captured == {
    "model_id": "pipecat-ai/smart-turn-v3",
    "allow_patterns": ["smart-turn-v3.2-cpu.onnx"],
}
assert result["model_checkpoint"]["kind"] == "smart_turn"
```

- [ ] Run RED: `nix develop --command pytest -q tests/test_smart_turn_temporary.py`; expect missing port/catalog symbols.
- [ ] Add this generalized port beside the existing scalar ports and include it in `ALL_PORT_TYPES`:

```python
@dataclass(frozen=True)
class BoolPort(Port):
    TYPE_NAME = "BOOL"
    python_type = bool
    color = "#0F766E"
    description = "Boolean"
```

- [ ] Add constants `_SMART_TURN_MODEL = "pipecat-ai/smart-turn-v3"` and `_SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"`. Implement `bootstrap_turn_model` to reject every other item, call:

```python
ref = ensure_model_checkpoint(
    "smart_turn",
    model_id,
    lambda folder: download_hf_snapshot(model_id, folder, allow_patterns=[_SMART_TURN_FILE]),
)
```

Return `{"model_checkpoint": {"kind": "smart_turn", "model_id": model_id, "checkpoint_id": str(ref.checkpoint_id), "name": ref.name}}`; register `turn_models` in `CATALOG_DOWNLOAD_TASKS` and `CatalogKey`.
- [ ] Run GREEN; expect all Task 1 tests pass. Delete the temporary test with `apply_patch`.
- [ ] Commit only these three source files: `git commit -m "feat: add Smart Turn model catalog"`.

---

### Task 2: Audio Preparation and One-Call Batch Inference

**Files:** temporary test; create `smart_turn/audio.py`, `smart_turn/inference.py`.

**Produces:** `TARGET_SAMPLE_RATE=16_000`, `WINDOW_SAMPLES=128_000`, `load_audio_bytes`, `prepare_waveform(s)`, `SmartTurnBundle`, `load_smart_turn_bundle`, `predict_probabilities`, `is_turn_complete`.

The official model was inspected before planning: input `input_features` is `[dynamic_batch, 80, 800]`; output `logits` is `[dynamic_batch, 1]`.

- [ ] Write failing preprocessing tests using in-memory PCM WAVs:

```python
prepared = prepare_waveform(audio, one_second_mono_16khz)
assert prepared.shape == (128_000,)
assert prepared.dtype == np.float32
assert np.count_nonzero(prepared[:-16_000]) == 0
assert np.allclose(prepared[-16_000:], 0.25, atol=1e-4)

prepared = prepare_waveform(audio, ten_seconds_with_distinct_first_two_seconds)
assert np.allclose(prepared, expected_last_eight_seconds, atol=1e-4)

prepared = prepare_waveform(audio, one_second_stereo_8khz)
assert np.mean(prepared[-16_000:]) == pytest.approx(0.3, abs=0.01)
```

Assert empty audio with `id="audio_empty"` raises `ValueError("SmartTurnPredict requires non-empty audio: audio_empty")`. Patch `database_session` and `bulk_read_audio_files`; assert two missing items trigger one bulk call and original order is preserved.
- [ ] Write a fake extractor returning `(2, 80, 800)` and fake session returning `[[0.25], [0.75]]`; assert `predict_probabilities` calls the session once and returns both values. Assert NaN, `1.2`, and wrong output count raise `smart_turn_non_finite_probability`, `smart_turn_probability_out_of_range`, and `smart_turn_output_count_mismatch`. Assert `is_turn_complete(0.5)` is false and `is_turn_complete(0.500001)` is true.
- [ ] Run RED; expect `runner.nodes.smart_turn` import failure.
- [ ] Implement `audio.py` with module-top imports. `load_audio_bytes(audios)` must collect missing UUIDs, make one bulk CRUD call, and use direct dictionary indexing. `prepare_waveform` must use `librosa.load(BytesIO(data), sr=16_000, mono=True)`, convert to one-dimensional float32, reject empty samples, retain `samples[-128_000:]`, and left-pad only when short. `prepare_waveforms` uses strict ordered zip.
- [ ] Implement `SmartTurnBundle(feature_extractor, session)` as a frozen dataclass. `load_smart_turn_bundle(checkpoint_dir)` finds the single `.onnx`, creates upstream session options (`ORT_SEQUENTIAL`, one inter-op thread, all graph optimizations), forces `CPUExecutionProvider`, and creates `WhisperFeatureExtractor(chunk_length=8)`.
- [ ] Implement `predict_probabilities` with one extractor call using `sampling_rate=16_000`, NumPy tensors, max-length padding/truncation at 128,000, and normalization. Convert features to float32; call `session.run(None, {"input_features": features})` once; flatten output; enforce count, finite values, and `[0,1]`; return Python floats. `is_turn_complete` is exactly `probability > 0.5`.
- [ ] Run GREEN; expect all preparation/batch/validation tests pass. Delete the temporary test with `apply_patch`.
- [ ] Commit only `audio.py` and `inference.py`: `git commit -m "feat: add batched Smart Turn inference"`.

---

### Task 3: Node Lifecycle and Registration

**Files:** temporary test; create `smart_turn/node.py`, `smart_turn/__init__.py`; modify `runner/nodes/registry.py`.

**Produces:** registered `SmartTurnPredictNode` with persistent checkpoint-scoped bundle and typed outputs.

- [ ] Write a failing registry-only test (never call `execute`):

```python
node = create_node_registry().nodes["SmartTurnPredict"]
assert node.INPUTS["checkpoint"].join_mode is JoinMode.BROADCAST
assert {key: port.TYPE_NAME for key, port in node.OUTPUTS.items()} == {
    "audio": "AUDIO", "turn_complete": "BOOL", "probability": "FLOAT",
}
assert node.BATCH_POLICY == BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=32, max_size=64, sort_by="duration")
assert node.RESOURCE_POLICY.resources == {"cpu_workers": 1}
assert node.RESOURCE_POLICY.keep_loaded is True
assert node.QUEUE_MAX_SIZE == 128
```

- [ ] Run RED; expect registry lookup failure.
- [ ] Define metadata exactly: type `SmartTurnPredict`, category `Audio`, broadcast checkpoint plus audio inputs, unchanged audio plus bool/float outputs, policy asserted above, queue 128, and an explicit description.
- [ ] Constructor fields are `_bundle: SmartTurnBundle | None` and `_loaded_checkpoint_id: UUID | None`; teardown clears both. `_ensure_bundle` requires `checkpoint.metadata["type"] == "smart_turn"`, reuses the matching checkpoint, otherwise loads in `asyncio.to_thread` and records its UUID.
- [ ] `execute` must: resolve `typed_checkpoint(batch[0]["checkpoint"])`; assert all inputs are `Audio`; check cancellation; bulk-load bytes in a thread; prepare ordered chunks of eight in threads with cancellation and item-count progress; check cancellation; call `predict_probabilities` once in a thread; strict-zip outputs as `{"audio": audio, "turn_complete": is_turn_complete(probability), "probability": probability}`.
- [ ] Export the node in `smart_turn/__init__.py`; import and register it next to other audio inference nodes in `registry.py`.
- [ ] Run GREEN plus: `nix develop --command python -c 'from runner.nodes.registry import create_node_registry; assert create_node_registry().nodes["SmartTurnPredict"].OUTPUTS["turn_complete"].TYPE_NAME == "BOOL"'`.
- [ ] Delete the temporary test with `apply_patch`; commit only node package export/node and registry: `git commit -m "feat: register Smart Turn prediction node"`.

---

### Task 4: Dependency and Example Workflow

**Files:** modify `pyproject.toml`, `uv.lock`, `workflows/README.md`; create `workflows/smart_turn_predict.json`.

- [ ] Verify RED with a Nix Python TOML assertion that no direct dependency starts with `onnxruntime`.
- [ ] Add `"onnxruntime>=1.20",` to project dependencies; run `nix develop --command uv lock` then `nix develop --command uv lock --check`.
- [ ] Create a `WorkflowCreate` JSON named `Smart Turn batched completion prediction` with four nodes: selected `AudioSource`, mono/16 kHz `LoadAudio`, `CatalogDownload({"catalog_key":"turn_models","item":"pipecat-ai/smart-turn-v3"})`, and `SmartTurnPredict`. Wire source→load→predict audio and catalog checkpoint→predict checkpoint. Use CPU context resources `{"io":1,"cpu_workers":2}` and the same replaceable selected audio UUID in node params and `launch_source`.
- [ ] Document that `smart_turn_predict.json` preserves inputs, exposes `turn_complete`/`probability`, and requires replacing the example source UUID.
- [ ] Validate: `nix develop --command python -m json.tool workflows/smart_turn_predict.json >/dev/null`; compile imports; verify all touched source files remain below 300 lines and `smart_turn` has four files.
- [ ] Commit only dependency/lock/workflow files: `git commit -m "docs: add Smart Turn example workflow"`.

---

### Task 5: Real-Graph Verification

**Files:** no permanent changes; `/tmp/smart-turn-run.json` and response only.

- [ ] Run `nix develop --command python -m compileall -q src/runner/nodes/smart_turn src/runner/nodes/datatypes.py src/runner/nodes/assets/catalog.py src/runner/nodes/assets/catalog_runtime/tasks.py src/runner/nodes/registry.py` and a registry assertion for output order `(audio, turn_complete, probability)`.
- [ ] Run `nix develop --command runflow-dev-status`; only if inactive, attach/start the single shared stack as the workspace user with `nix develop --command runflow-dev-session`.
- [ ] Select at least two real stored audio UUIDs through existing API/CLI reads. Put the same IDs in both source arrays, extract the workflow's `.data` as `/tmp/smart-turn-run.json`, and submit to `POST http://127.0.0.1:8001/graphs/runs`. Store the response in `/tmp/smart-turn-run-response.json`; expect HTTP 202 and `run_id`.
- [ ] Inspect with `nix develop --command python -m cli runs`, `logs`, and `node-log ... smart_turn` using that returned ID. Require success and evidence that both inputs were prepared in one runtime batch. On failure use `cli failed`, systematic debugging, and a temporary regression test for pure behavior before fixing.
- [ ] Remove both `/tmp` payloads. Run `git diff --check` and `git status --short`; confirm no temporary tests, caches, models, audio, or run output and that unrelated changes remain untouched.
- [ ] Review every port, catalog key/item, checkpoint kind, batch/resource policy, lifecycle field, preprocessing invariant, probability invariant, registration entry, dependency, and real result against the approved design. Confirm no filtering, writeback, metadata mutation, or `src/runflow` domain leak.
