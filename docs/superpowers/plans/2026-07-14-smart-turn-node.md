# Batched Smart Turn Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a checkpoint-driven, genuinely batched `SmartTurnPredict` node that preserves each audio input and emits a typed completion decision plus probability using Pipecat Smart Turn v3.2 CPU ONNX inference.

**Architecture:** A `turn_models` catalog entry stores the official quantized ONNX file through the shared checkpoint CRUD. A focused `runner.nodes.smart_turn` package owns byte loading, upstream-compatible 16 kHz/8-second waveform preparation, batched Whisper feature extraction, the persistent ONNX session, and node lifecycle. The runtime stays domain-agnostic; only a reusable boolean runner port is added outside the feature package.

**Tech Stack:** Python 3.11+, runflow nodes/ports/policies, Pydantic, NumPy, librosa, Transformers `WhisperFeatureExtractor`, ONNX Runtime, Hugging Face Hub, shared PostgreSQL/audio/assets CRUD, Nix/uv.

## Global Constraints

- Keep Smart Turn and audio-specific behavior out of `src/runflow`.
- Use `AudioPort`, `CheckpointRefPort`, `BoolPort`, and `FloatPort`; do not use union ports or JSON for typed scalar results.
- Process the whole incoming node batch with one ONNX call and preserve one output per input.
- Fetch absent audio bytes with `shared.db.audio.crud.bulk_read_audio_files`.
- Store model files through the existing catalog/checkpoint helpers; inference must not download weights.
- Run all Python, uv, backend, runner, and CLI commands through `nix develop --command ...`.
- Do not commit temporary tests, generated audio, model files, caches, or run output.
- Keep every modified or created source file under 300 lines and every folder under 16 files.

---

## File Map

- Create `src/runner/nodes/smart_turn/__init__.py`: public node-family export.
- Create `src/runner/nodes/smart_turn/audio.py`: bulk byte resolution and exact waveform preparation.
- Create `src/runner/nodes/smart_turn/inference.py`: model bundle loading, batched feature extraction, ONNX call, and probability validation.
- Create `src/runner/nodes/smart_turn/node.py`: node metadata, lifecycle, cancellation, progress, and output assembly.
- Modify `src/runner/nodes/datatypes.py`: reusable `BoolPort` and type registration.
- Modify `src/runner/nodes/assets/catalog.py`: expose the `turn_models` catalog key.
- Modify `src/runner/nodes/assets/catalog_runtime/tasks.py`: download only `smart-turn-v3.2-cpu.onnx` as a `smart_turn` checkpoint.
- Modify `src/runner/nodes/registry.py`: register `SmartTurnPredictNode` for runner and UI schema discovery.
- Modify `pyproject.toml`: declare the directly imported `onnxruntime` dependency.
- Modify `uv.lock`: Nix-wrapped uv lock update.
- Create `workflows/smart_turn_predict.json`: runnable example graph using `CatalogDownload`, `AudioSource`, `LoadAudio`, and `SmartTurnPredict`.
- Modify `workflows/README.md`: document the example and its environment-specific source selection.
- Create and remove `tests/test_smart_turn_temporary.py`: test-first development only; never commit it.

---

### Task 1: Typed Boolean Port and Smart Turn Catalog

**Files:**
- Create temporarily: `tests/test_smart_turn_temporary.py`
- Modify: `src/runner/nodes/datatypes.py:18-118`
- Modify: `src/runner/nodes/assets/catalog.py:18-26`
- Modify: `src/runner/nodes/assets/catalog_runtime/tasks.py:12-249`

**Interfaces:**
- Consumes: `download_hf_snapshot(model_id: str, folder: Path, *, allow_patterns: list[str] | None = None, ignore_patterns: list[str] | None = None)` and `ensure_model_checkpoint(kind: str, model_id: str, download: Callable[[Path], None]) -> CheckpointRef`.
- Produces: `BoolPort` with `TYPE_NAME = "BOOL"`; `CatalogKey.TURN_MODELS`; `bootstrap_turn_model(item: str = "", *, logger: logging.Logger | None = None) -> dict[str, Any]`; catalog item `pipecat-ai/smart-turn-v3` resolving checkpoint kind `smart_turn`.

- [ ] **Step 1: Write failing port and catalog tests**

```python
from pathlib import Path
from unittest.mock import patch

from runflow.registry.type_registry import TypeRegistry
from runner.nodes.assets.catalog import CatalogKey
from runner.nodes.assets.catalog_runtime import tasks
from runner.nodes.datatypes import BoolPort, register_runner_types


def test_bool_port_is_registered() -> None:
    registry = register_runner_types(TypeRegistry())
    assert registry.get("BOOL") is BoolPort
    assert BoolPort.python_type is bool


def test_turn_model_catalog_downloads_only_v32_cpu() -> None:
    captured: dict[str, object] = {}

    def fake_download(model_id: str, folder: Path, **kwargs: object) -> None:
        captured.update(model_id=model_id, folder=folder, kwargs=kwargs)

    with (
        patch.object(tasks, "download_hf_snapshot", side_effect=fake_download),
        patch.object(tasks, "ensure_model_checkpoint") as ensure,
    ):
        ensure.return_value.checkpoint_id = "checkpoint-id"
        ensure.return_value.name = "smart-turn"
        result = tasks.bootstrap_turn_model("pipecat-ai/smart-turn-v3")
        download = ensure.call_args.args[2]
        download(Path("/tmp/checkpoint"))

    assert CatalogKey.TURN_MODELS.value == "turn_models"
    assert ensure.call_args.args[:2] == ("smart_turn", "pipecat-ai/smart-turn-v3")
    assert captured["model_id"] == "pipecat-ai/smart-turn-v3"
    assert captured["kwargs"] == {"allow_patterns": ["smart-turn-v3.2-cpu.onnx"]}
    assert result["model_checkpoint"]["kind"] == "smart_turn"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `nix develop --command pytest -q tests/test_smart_turn_temporary.py`

Expected: collection fails because `BoolPort`, `CatalogKey.TURN_MODELS`, and `bootstrap_turn_model` do not exist.

- [ ] **Step 3: Implement the reusable port and catalog task**

Add beside the existing scalar ports in `datatypes.py`:

```python
@dataclass(frozen=True)
class BoolPort(Port):
    TYPE_NAME = "BOOL"
    python_type = bool
    color = "#0F766E"
    description = "Boolean"
```

Add `BoolPort` to `ALL_PORT_TYPES`. Add `TURN_MODELS = "turn_models"` to `CatalogKey`. In `catalog_runtime/tasks.py`, add:

```python
_SMART_TURN_MODEL = "pipecat-ai/smart-turn-v3"
_SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"


def bootstrap_turn_model(item: str = "", *, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or _LOGGER
    model_id = item.strip()
    if model_id != _SMART_TURN_MODEL:
        raise ValueError(f"catalog_item_unknown:{item}")
    log.info("Smart Turn model download starting model=%s", model_id)
    ref = ensure_model_checkpoint(
        "smart_turn",
        model_id,
        lambda folder: download_hf_snapshot(model_id, folder, allow_patterns=[_SMART_TURN_FILE]),
    )
    log.info("Smart Turn model download resolved model=%s checkpoint=%s", model_id, ref.checkpoint_id)
    return {
        "model_checkpoint": {
            "kind": "smart_turn",
            "model_id": model_id,
            "checkpoint_id": str(ref.checkpoint_id),
            "name": ref.name,
        }
    }
```

Register it in `CATALOG_DOWNLOAD_TASKS` as `"turn_models": CatalogTask(key="turn_models", run=bootstrap_turn_model)`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `nix develop --command pytest -q tests/test_smart_turn_temporary.py`

Expected: `2 passed`.

- [ ] **Step 5: Remove the temporary test and commit only implementation files**

Run: `rm tests/test_smart_turn_temporary.py`

Run: `git add src/runner/nodes/datatypes.py src/runner/nodes/assets/catalog.py src/runner/nodes/assets/catalog_runtime/tasks.py && git commit -m "feat: add Smart Turn model catalog"`

Expected: the commit contains no test, cache, model, or unrelated workspace files.

---

### Task 2: Audio Preparation and Batched ONNX Inference

**Files:**
- Create temporarily: `tests/test_smart_turn_temporary.py`
- Create: `src/runner/nodes/smart_turn/audio.py`
- Create: `src/runner/nodes/smart_turn/inference.py`

**Interfaces:**
- Produces: `TARGET_SAMPLE_RATE = 16_000`, `WINDOW_SAMPLES = 128_000`, `load_audio_bytes(audios: list[Audio]) -> list[bytes]`, `prepare_waveform(audio: Audio, data: bytes) -> np.ndarray`, `SmartTurnBundle(feature_extractor: Any, session: Any)`, `load_smart_turn_bundle(checkpoint_dir: Path) -> SmartTurnBundle`, and `predict_probabilities(bundle: SmartTurnBundle, waveforms: list[np.ndarray]) -> list[float]`.
- Guarantees: every prepared waveform is one-dimensional `float32` with 128,000 samples; every prediction is finite and in `[0.0, 1.0]`; one ONNX call returns one probability per waveform.

- [ ] **Step 1: Write failing preprocessing and batch-inference tests**

```python
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4
import wave

import numpy as np
import pytest

from runner.nodes.models import Audio
from runner.nodes.smart_turn.audio import WINDOW_SAMPLES, prepare_waveform
from runner.nodes.smart_turn.inference import SmartTurnBundle, predict_probabilities


def audio_item(data: bytes, duration: float) -> Audio:
    return Audio(uuid4(), "sample.wav", data, 16_000, 1, 0.0, duration, 1.0, "audio", "lineage")


def wav_bytes(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((samples * 32767.0).astype("<i2").tobytes())
    return output.getvalue()


def test_prepare_waveform_left_pads_short_audio() -> None:
    samples = np.ones(16_000, dtype=np.float32) * 0.25
    prepared = prepare_waveform(audio_item(wav_bytes(samples), 1.0), wav_bytes(samples))
    assert prepared.dtype == np.float32
    assert prepared.shape == (WINDOW_SAMPLES,)
    assert np.count_nonzero(prepared[:-16_000]) == 0
    assert np.allclose(prepared[-16_000:], 0.25, atol=1e-4)


def test_prepare_waveform_keeps_last_eight_seconds() -> None:
    samples = np.concatenate((np.zeros(32_000, dtype=np.float32), np.ones(WINDOW_SAMPLES, dtype=np.float32) * 0.5))
    prepared = prepare_waveform(audio_item(wav_bytes(samples), 10.0), wav_bytes(samples))
    assert np.allclose(prepared, 0.5, atol=1e-4)


def test_prepare_waveform_rejects_empty_audio() -> None:
    with pytest.raises(ValueError, match="SmartTurnPredict requires non-empty audio: audio"):
        prepare_waveform(audio_item(wav_bytes(np.array([], dtype=np.float32)), 0.0), wav_bytes(np.array([], dtype=np.float32)))


def test_predict_probabilities_uses_one_batched_onnx_call() -> None:
    class Extractor:
        def __call__(self, waveforms: list[np.ndarray], **kwargs: object) -> SimpleNamespace:
            assert len(waveforms) == 2
            return SimpleNamespace(input_features=np.zeros((2, 80, 800), dtype=np.float32))

    class Session:
        calls = 0

        def run(self, outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            self.calls += 1
            assert inputs["input_features"].shape == (2, 80, 800)
            return [np.array([[0.25], [0.75]], dtype=np.float32)]

    session = Session()
    probabilities = predict_probabilities(SmartTurnBundle(Extractor(), session), [np.zeros(WINDOW_SAMPLES), np.zeros(WINDOW_SAMPLES)])
    assert session.calls == 1
    assert probabilities == pytest.approx([0.25, 0.75])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `nix develop --command pytest -q tests/test_smart_turn_temporary.py`

Expected: collection fails because the `runner.nodes.smart_turn` package does not exist.

- [ ] **Step 3: Implement exact waveform preparation and bulk byte loading**

In `audio.py`, import all dependencies at module top. `load_audio_bytes` must gather missing UUIDs, call `audio_crud.bulk_read_audio_files` once inside `database_session`, and preserve input order. `prepare_waveform` must decode with `librosa.load(BytesIO(data), sr=TARGET_SAMPLE_RATE, mono=True)`, convert to a one-dimensional `np.float32` array, reject zero samples with the audio id in the exception, keep `samples[-WINDOW_SAMPLES:]`, and use `np.pad(samples, (WINDOW_SAMPLES - samples.size, 0))` only when short.

- [ ] **Step 4: Implement the persistent model bundle and true batch inference**

In `inference.py`, define:

```python
@dataclass(frozen=True)
class SmartTurnBundle:
    feature_extractor: Any
    session: Any


def load_smart_turn_bundle(checkpoint_dir: Path) -> SmartTurnBundle:
    model_path = single_checkpoint_file(checkpoint_dir, (".onnx",))
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return SmartTurnBundle(
        feature_extractor=WhisperFeatureExtractor(chunk_length=8),
        session=ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"]),
    )
```

`predict_probabilities` must call the extractor once with the waveform list, `sampling_rate=16_000`, `return_tensors="np"`, `padding="max_length"`, `max_length=128_000`, `truncation=True`, and `do_normalize=True`; convert `input_features` to `float32`; call `session.run(None, {"input_features": features})` once; flatten the first output; require exactly one value per waveform; and reject non-finite or out-of-range values with explicit `RuntimeError` messages.

- [ ] **Step 5: Run the focused tests and add invalid-output cases**

Append tests that return `np.array([[np.nan]])`, `np.array([[1.2]])`, and two outputs for one waveform, asserting `RuntimeError` messages `smart_turn_non_finite_probability`, `smart_turn_probability_out_of_range`, and `smart_turn_output_count_mismatch` respectively.

Run: `nix develop --command pytest -q tests/test_smart_turn_temporary.py`

Expected: all preprocessing, one-call batching, and validation tests pass.

- [ ] **Step 6: Remove the temporary test and commit implementation files**

Run: `rm tests/test_smart_turn_temporary.py`

Run: `git add src/runner/nodes/smart_turn/audio.py src/runner/nodes/smart_turn/inference.py && git commit -m "feat: add batched Smart Turn inference"`

Expected: only the two feature implementation files are committed.

---

### Task 3: Node Lifecycle, Outputs, and Registry Discovery

**Files:**
- Create temporarily: `tests/test_smart_turn_temporary.py`
- Create: `src/runner/nodes/smart_turn/node.py`
- Create: `src/runner/nodes/smart_turn/__init__.py`
- Modify: `src/runner/nodes/registry.py:5-135`

**Interfaces:**
- Consumes: the Task 1 ports and Task 2 audio/inference functions.
- Produces: `SmartTurnPredictNode` with node type `SmartTurnPredict`, broadcast checkpoint input, audio input, unchanged audio output, boolean `turn_complete`, float `probability`, micro-batching, `cpu_workers` resource declaration, and persistent model lifecycle.

- [ ] **Step 1: Write a failing registry/schema test without calling `execute` directly**

```python
from runflow.core.ports import JoinMode
from runflow.policies import BatchMode
from runner.nodes.registry import create_node_registry


def test_smart_turn_node_contract_is_registered() -> None:
    node = create_node_registry().get("SmartTurnPredict")
    assert node.INPUTS["checkpoint"].join_mode is JoinMode.BROADCAST
    assert node.INPUTS["audio"].TYPE_NAME == "AUDIO"
    assert node.OUTPUTS["audio"].TYPE_NAME == "AUDIO"
    assert node.OUTPUTS["turn_complete"].TYPE_NAME == "BOOL"
    assert node.OUTPUTS["probability"].TYPE_NAME == "FLOAT"
    assert node.BATCH_POLICY.mode is BatchMode.MICRO_BATCH
    assert node.RESOURCE_POLICY.keep_loaded is True
    assert node.RESOURCE_POLICY.resources == {"cpu_workers": 1}
```

- [ ] **Step 2: Run the registry test and verify RED**

Run: `nix develop --command pytest -q tests/test_smart_turn_temporary.py`

Expected: `SmartTurnPredict` is absent from the runner registry.

- [ ] **Step 3: Implement the node lifecycle and output contract**

`node.py` must define `SmartTurnPredictNode` with:

```python
NODE_TYPE = "SmartTurnPredict"
CATEGORY = "Audio"
INPUTS = {
    "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
    "audio": AudioPort(),
}
OUTPUTS = {
    "audio": AudioPort(),
    "turn_complete": BoolPort(),
    "probability": FloatPort(),
}
BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=32, max_size=64, sort_by="duration")
RESOURCE_POLICY = ResourcePolicy(resources={"cpu_workers": 1}, keep_loaded=True)
QUEUE_MAX_SIZE = 128
```

The constructor stores `_bundle: SmartTurnBundle | None` and `_loaded_checkpoint_id: UUID | None`. `_ensure_bundle` requires `checkpoint.metadata["type"] == "smart_turn"`, uses `asyncio.to_thread(load_smart_turn_bundle, checkpoint.path)`, and reuses the bundle for the same checkpoint. `teardown` clears both fields.

`execute` must not be unit-tested directly. It must:

1. Resolve the broadcast checkpoint with `typed_checkpoint` and ensure the bundle.
2. Assert every input audio is `Audio` and bulk-resolve bytes once.
3. Check cancellation before preparation and before inference.
4. Prepare waveforms in order, checking cancellation and reporting item-count progress every eight items and at the end.
5. Call `predict_probabilities` once through `asyncio.to_thread`.
6. Return one dictionary per strict audio/probability pair: `{"audio": audio, "turn_complete": probability > 0.5, "probability": probability}`.

Export the class from `smart_turn/__init__.py`, import it at the top of `runner/nodes/registry.py`, and add it adjacent to other audio inference nodes in `register_runner_nodes`.

- [ ] **Step 4: Run schema and import checks and verify GREEN**

Run: `nix develop --command pytest -q tests/test_smart_turn_temporary.py`

Run: `nix develop --command python -c 'from runner.nodes.registry import create_node_registry; assert create_node_registry().get("SmartTurnPredict").OUTPUTS["turn_complete"].TYPE_NAME == "BOOL"'`

Expected: the temporary test passes and the import command exits zero.

- [ ] **Step 5: Remove the temporary test and commit node registration**

Run: `rm tests/test_smart_turn_temporary.py`

Run: `git add src/runner/nodes/smart_turn/__init__.py src/runner/nodes/smart_turn/node.py src/runner/nodes/registry.py && git commit -m "feat: register Smart Turn prediction node"`

Expected: node files and the localized registry change only are committed.

---

### Task 4: Dependency Lock and Example Workflow

**Files:**
- Modify: `pyproject.toml:10-52`
- Modify: `uv.lock`
- Create: `workflows/smart_turn_predict.json`
- Modify: `workflows/README.md`

**Interfaces:**
- Produces: direct `onnxruntime>=1.20` dependency and a UI-visible example graph wired to the new catalog/node contract.

- [ ] **Step 1: Confirm the direct dependency is absent**

Run: `nix develop --command python -c 'import tomllib, pathlib; deps=tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["dependencies"]; assert not any(item.startswith("onnxruntime") for item in deps)'`

Expected: command exits zero, proving the direct dependency has not already been declared.

- [ ] **Step 2: Add ONNX Runtime and update the lock through Nix**

Add `"onnxruntime>=1.20",` beside the other inference dependencies in `pyproject.toml`.

Run: `nix develop --command uv lock`

Expected: lock succeeds; the root package dependency list now includes `onnxruntime`, reusing the resolved compatible package already present transitively where possible.

- [ ] **Step 3: Add the example workflow**

Create `workflows/smart_turn_predict.json` as a `WorkflowCreate` payload with:

- `AudioSource` configured for one replaceable selected `audio_file_id`;
- `LoadAudio` configured for mono 16 kHz;
- `CatalogDownload` configured with `catalog_key: "turn_models"` and `item: "pipecat-ai/smart-turn-v3"`;
- `SmartTurnPredict` with empty params/runtime;
- edges `AudioSource.audio -> LoadAudio.audio`, `LoadAudio.audio -> SmartTurnPredict.audio`, and `CatalogDownload.checkpoint -> SmartTurnPredict.checkpoint`;
- a CPU context containing at least `io: 1` and `cpu_workers: 2`;
- matching `launch_source` selected audio metadata.

Add a short `smart_turn_predict.json` section to `workflows/README.md` explaining that the graph preserves each input and exposes `turn_complete`/`probability`, and that the example audio UUID must be replaced locally.

- [ ] **Step 4: Validate dependency declaration, lock consistency, workflow JSON, and file limits**

Run: `nix develop --command uv lock --check`

Run: `nix develop --command python -m json.tool workflows/smart_turn_predict.json >/dev/null`

Run: `find src/runner/nodes/smart_turn -maxdepth 1 -type f | wc -l && wc -l src/runner/nodes/smart_turn/*.py src/runner/nodes/datatypes.py src/runner/nodes/registry.py src/runner/nodes/assets/catalog.py src/runner/nodes/assets/catalog_runtime/tasks.py`

Expected: lock check and JSON validation exit zero; the Smart Turn folder contains four source files; every listed file is below 300 lines.

- [ ] **Step 5: Commit dependency and workflow files**

Run: `git add pyproject.toml uv.lock workflows/smart_turn_predict.json workflows/README.md && git commit -m "docs: add Smart Turn example workflow"`

Expected: only dependency/lock and workflow documentation files are committed.

---

### Task 5: End-to-End Verification Through a Real Graph

**Files:**
- No permanent file changes.
- Temporary graph payload may be copied under `/tmp`; do not add it to the repository.

**Interfaces:**
- Verifies the same backend endpoint and NATS-backed runner path used by the frontend, including registry export, checkpoint acquisition, lifecycle, routing, and output serialization.

- [ ] **Step 1: Run static and registry verification**

Run: `nix develop --command python -m compileall -q src/runner/nodes/smart_turn src/runner/nodes/datatypes.py src/runner/nodes/assets/catalog.py src/runner/nodes/assets/catalog_runtime/tasks.py src/runner/nodes/registry.py`

Run: `nix develop --command python -c 'from runner.nodes.registry import create_node_registry; r=create_node_registry(); n=r.get("SmartTurnPredict"); assert tuple(n.OUTPUTS) == ("audio", "turn_complete", "probability")'`

Expected: both commands exit zero with no traceback.

- [ ] **Step 2: Check or start the single shared development stack**

Run: `nix develop --command runflow-dev-status`

If it is not active, run as the workspace user: `nix develop --command runflow-dev-session`.

Expected: exactly one `runflow-dev` Zellij session owns NATS, backend, and runner.

- [ ] **Step 3: Select a real audio record and submit the graph through the backend**

Use the existing backend/API or CLI read path to identify one stored audio UUID. Replace both occurrences of the placeholder UUID from `workflows/smart_turn_predict.json`, extract its `.data` object into an `InlineGraphRunRequest`, and submit it:

Run: `curl -fsS -X POST http://127.0.0.1:8001/graphs/runs -H 'Content-Type: application/json' --data-binary @/tmp/smart-turn-run.json`

Expected: HTTP 202 JSON containing a new `run_id`. This is the only valid node execution test; do not call `SmartTurnPredictNode.execute()` by hand.

- [ ] **Step 4: Inspect completion and node logs**

Run: `nix develop --command python -m cli runs`

Run: `nix develop --command python -m cli logs <run_id>`

Run: `nix develop --command python -m cli node-log <run_id> smart_turn`

Expected: the run completes, the node reports model loading/preparation, and no failed node appears. If it fails, run `nix develop --command python -m cli failed <run_id>`, diagnose systematically, add a failing temporary regression test for pure behavior where possible, then fix and repeat.

- [ ] **Step 5: Verify clean scoped diff and remove temporary artifacts**

Run: `rm -f /tmp/smart-turn-run.json`

Run: `git diff --check && git status --short`

Expected: no temporary tests, payloads, caches, model files, or run outputs appear in the repository; pre-existing unrelated user changes remain untouched.

- [ ] **Step 6: Review implementation against the approved spec**

Check every node port, catalog identifier, checkpoint type, batch policy, resource policy, lifecycle field, preprocessing constant, probability invariant, registration entry, dependency, and real-graph result against `docs/superpowers/specs/2026-07-14-smart-turn-node-design.md`.

Expected: no spec gaps, no unrequested filtering/writeback/metadata mutation, and no audio-specific changes under `src/runflow`.
