# Beetle Language Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit ordered language vocabulary whose learned vectors condition both duration prediction and latent flow alongside the same style, voice, phoneme, and context sources.

**Architecture:** Carry the database language string through indexing and collation as a stable configured ID. Stage 2 selects one shared learned language vector, builds equivalent raw condition sets at phoneme and latent rates, applies one set of per-source dropout decisions, sends concatenated raw conditions through the duration predictor's existing linear input projection, and sends independently projected conditions to latent-flow AdaLN and concat layers.

**Tech Stack:** Python 3.12, PyTorch, Pydantic, SQLAlchemy, pytest, Ruff, Nix development shell.

## Global Constraints

- Work in `/workspace/styletts_studio_v2`; do not create a branch, worktree, or subagent.
- Run Python, pytest, Ruff, and compile checks through `nix develop --command`.
- Keep temporary tests under `/tmp`; do not delete existing temporary tests.
- Keep files below 300 lines and folders below 16 files.
- Missing and unconfigured languages are rejected from every training stage.
- Configured language order is the checkpoint-stable ID order.
- One learned language vector is shared by duration prediction and latent flow.
- Full inference must remain at or below 150M parameters and latent-to-audio below 15 GFLOPs/s; report violations.

---

### Task 1: Stable Vocabulary and Database Indexing

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/architecture.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/shared/db/audio/segment_references_crud.py`
- Modify: `src/runner/nodes/training/beetle/data/records.py`
- Modify: `src/runner/nodes/training/beetle/data/index.py`
- Modify: `src/runner/nodes/training/beetle/training/runtime.py`
- Test: `/tmp/test_beetle_language_conditioning.py`

**Interfaces:**
- Produces: `LanguageConfig.values: tuple[str, ...]` and `embedding_channels: int`.
- Produces: `SegmentReference.language: str | None` and `IndexedSegment.language: str`.
- Produces: index builders accepting `languages: tuple[str, ...]`.

- [ ] **Step 1: Write failing config and indexing tests**

```python
def test_language_order_is_preserved(config_dict):
    config_dict["architecture"]["language"] = {
        "values": ["de", "en", "ja"], "embedding_channels": 64,
    }
    assert BeetleConfig.model_validate(config_dict).architecture.language.values == (
        "de", "en", "ja",
    )


def test_index_rejects_missing_and_unknown_languages(reference_factory):
    references = [reference_factory(language="en"), reference_factory(language=None),
                  reference_factory(language="fr")]
    index = DatabaseSegmentIndex.from_references(DATASET_ID, (), ("en",), references)
    assert tuple(item.language for item in index.records.values()) == ("en",)
    assert index.report.excluded_language == 2
```

- [ ] **Step 2: Run tests and verify failure**

Run: `nix develop --command pytest -q /tmp/test_beetle_language_conditioning.py -k 'language_order or index_rejects'`

Expected: failure because language configuration and index arguments are absent.

- [ ] **Step 3: Implement the contract**

```python
class LanguageConfig(StrictConfigModel):
    values: tuple[str, ...] = Field(min_length=1)
    embedding_channels: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_values(self) -> "LanguageConfig":
        if any(not value.strip() for value in self.values):
            raise ValueError("language values must not be empty")
        if len(set(self.values)) != len(self.values):
            raise ValueError("language values must be unique")
        return self
```

Add language to the shared SQL projection and dataclass. Filter missing and
unconfigured values before constructing records, add `excluded_language`, include
language in the index fingerprint, and pass configured values from runtime.

- [ ] **Step 4: Run focused tests**

Run: `nix develop --command pytest -q /tmp/test_beetle_language_conditioning.py -k 'language_order or index_rejects'`

Expected: selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/shared/db/audio/segment_references_crud.py src/runner/nodes/training/beetle/config/architecture.py src/runner/nodes/training/beetle/config/default.yaml src/runner/nodes/training/beetle/data/records.py src/runner/nodes/training/beetle/data/index.py src/runner/nodes/training/beetle/training/runtime.py
git commit -m "feat: index configured beetle languages"
```

### Task 2: Mixed-Language Batch Propagation

**Files:**
- Modify: `src/runner/nodes/training/beetle/data/source.py`
- Modify: `src/runner/nodes/training/beetle/data/collate.py`
- Modify: `src/runner/nodes/training/beetle/data/pipeline.py`
- Modify: `src/runner/nodes/training/beetle/data/records.py`
- Test: `/tmp/test_beetle_language_conditioning.py`

**Interfaces:**
- Consumes: configured values and `IndexedSegment.language`.
- Produces: `FetchedExample.language: str` and `BeetleBatch.language_ids: Tensor[B]`.

- [ ] **Step 1: Write the failing collation test**

```python
def test_collator_uses_configured_language_order(collator_factory, fetched_factory):
    batch = collator_factory(("de", "en", "ja")).collate(
        fetched_factory(("ja", "de", "en"))
    )
    assert batch.language_ids.dtype == torch.long
    assert batch.language_ids.tolist() == [2, 0, 1]
```

- [ ] **Step 2: Run test and verify failure**

Run: `nix develop --command pytest -q /tmp/test_beetle_language_conditioning.py -k collator_uses`

Expected: failure because language does not reach the batch.

- [ ] **Step 3: Propagate and resolve IDs**

```python
self.language_ids = {name: index for index, name in enumerate(languages)}
language_ids = torch.tensor(
    [self.language_ids[item.source.language] for item in prepared],
    dtype=torch.long,
)
```

Add the source and batch fields, pass configured values from `build_data_pipeline`,
add synthetic IDs, and require shape `[B]` with dtype `torch.long`.

- [ ] **Step 4: Run focused test**

Run: `nix develop --command pytest -q /tmp/test_beetle_language_conditioning.py -k collator_uses`

Expected: selected test passes.

- [ ] **Step 5: Commit**

```bash
git add src/runner/nodes/training/beetle/data/source.py src/runner/nodes/training/beetle/data/collate.py src/runner/nodes/training/beetle/data/pipeline.py src/runner/nodes/training/beetle/data/records.py
git commit -m "feat: collate beetle language ids"
```

### Task 3: Rate-Aware Conditioning Primitives

**Files:**
- Modify: `src/runner/nodes/training/beetle/config/architecture.py`
- Modify: `src/runner/nodes/training/beetle/config/default.yaml`
- Modify: `src/runner/nodes/training/beetle/models/modules/conditioning.py`
- Modify: `src/runner/nodes/training/beetle/models/modules/embeddings.py`
- Modify: `src/runner/nodes/training/beetle/models/modules/latent_flow/model.py`
- Test: `/tmp/test_beetle_conditioning.py`
- Test: `/tmp/test_beetle_latent_flow.py`
- Test: `/tmp/test_beetle_language_conditioning.py`

**Interfaces:**
- Produces: `LanguageEmbedding.forward(Tensor[B]) -> Tensor[B,C]`.
- Produces: nine-source `ConditionInputs`, `ConditionKeep`, and `ProjectedConditions`.
- Produces: `CONDITION_SOURCE_NAMES: tuple[str, ...]` in dataclass field order.
- Produces: `ConditionInputs.dropped_concatenated(keep)` and shared keep sampling.

- [ ] **Step 1: Write failing vector and dropout tests**

```python
def test_one_language_vector_expands_at_both_rates(condition_vectors):
    duration = condition_vectors.at_rate(torch.ones(2, 6, 5), 5)
    latent = condition_vectors.at_rate(torch.ones(2, 6, 9), 9)
    assert torch.equal(duration.language[:, :, 0], condition_vectors.language)
    assert torch.equal(latent.language[:, :, 0], condition_vectors.language)


def test_language_dropout_uses_one_keep_decision(condition_bank, inputs_by_rate, config):
    keep = condition_bank.sample_keep(2, torch.device("cpu"), config, generator)
    duration = inputs_by_rate.duration.dropped_concatenated(keep)
    latent = condition_bank(inputs_by_rate.latent, keep)
    language_width = inputs_by_rate.duration.language.shape[1]
    expected = inputs_by_rate.duration.language * keep.language
    assert torch.equal(duration[:, -language_width:], expected)
    projected_present = latent.language.abs().sum(dim=(1, 2)) > 0
    assert torch.equal(projected_present, keep.language[:, 0, 0])
```

- [ ] **Step 2: Run tests and verify failures**

Run: `nix develop --command pytest -q /tmp/test_beetle_conditioning.py /tmp/test_beetle_latent_flow.py /tmp/test_beetle_language_conditioning.py -k language`

Expected: failures for the absent ninth condition and embedding.

- [ ] **Step 3: Implement nine-source primitives**

```python
@dataclass(frozen=True)
class ConditionVectors:
    style: Tensor
    voice: Tensor
    pooled_phoneme: Tensor
    pre_text: Tensor
    post_text: Tensor
    pre_audio: Tensor
    post_audio: Tensor
    language: Tensor

    def at_rate(self, phoneme: Tensor, token_count: int) -> ConditionInputs:
        return ConditionInputs(phoneme, *(
            value.unsqueeze(2).expand(-1, -1, token_count)
            for value in (self.style, self.voice, self.pooled_phoneme,
                          self.pre_text, self.post_text, self.pre_audio,
                          self.post_audio, self.language)
        ))
```

Add language to all condition records, add `dropout.language: 0.01`, sample one
`[B,1,1]` keep tensor per source, concatenate zeroed raw conditions for duration,
project zeroed latent inputs, and change latent-flow concat width from eight to nine.
Define `CONDITION_SOURCE_NAMES` from the stable field order shown in the design.

- [ ] **Step 4: Run conditioning tests**

Run: `nix develop --command pytest -q /tmp/test_beetle_conditioning.py /tmp/test_beetle_latent_flow.py /tmp/test_beetle_language_conditioning.py`

Expected: all selected files pass.

- [ ] **Step 5: Commit**

```bash
git add src/runner/nodes/training/beetle/config/architecture.py src/runner/nodes/training/beetle/config/default.yaml src/runner/nodes/training/beetle/models/modules/conditioning.py src/runner/nodes/training/beetle/models/modules/embeddings.py src/runner/nodes/training/beetle/models/modules/latent_flow/model.py
git commit -m "feat: add beetle language conditioning"
```

### Task 4: Give Both Models the Same Conditioning Sources

**Files:**
- Modify: `src/runner/nodes/training/beetle/models/stage2.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2_inputs.py`
- Modify: `src/runner/nodes/training/beetle/training/stage2_setup.py`
- Create: `src/runner/nodes/training/beetle/training/stage2_conditions.py`
- Test: `/tmp/test_beetle_stage2.py`
- Test: `/tmp/test_beetle_stage2_objective.py`
- Test: `/tmp/test_beetle_language_conditioning.py`

**Interfaces:**
- Consumes: batch language IDs, condition vectors, and shared keep decisions.
- Produces: duration condition `[B,sum(source widths),P]` and projected latent conditions `[B,common,L]`.
- Produces: inference-counted, optimizer-owned `Stage2Models.language_embedding`.

- [ ] **Step 1: Write failing complete-source tests**

```python
def test_duration_and_latent_flow_receive_all_nine_sources(stage2_fixture):
    result = stage2_fixture.build_inputs(language_ids=torch.tensor([1, 0]))
    assert result.duration_sources == CONDITION_SOURCE_NAMES
    assert result.latent_sources == CONDITION_SOURCE_NAMES
    assert result.duration_condition.shape[-1] == result.phoneme_count
    assert result.conditions.language.shape[-1] == result.latent_count
```

- [ ] **Step 2: Run tests and verify failure**

Run: `nix develop --command pytest -q /tmp/test_beetle_stage2.py /tmp/test_beetle_stage2_objective.py /tmp/test_beetle_language_conditioning.py -k 'nine_sources or duration_condition'`

Expected: failure because duration currently receives only phoneme features.

- [ ] **Step 3: Build both rates from one vector set**

```python
vectors = ConditionVectors(
    target_style, target_voice, phoneme.pooled, pre_text, post_text,
    pre_audio, post_audio, models.language_embedding(values.language_ids),
)
keep = models.condition_bank.sample_keep(
    values.waveform.shape[0], self.device,
    self.config.architecture.conditioning.dropout,
    self._generator(loop, "condition-dropout"),
)
duration_inputs = vectors.at_rate(duration_tokens, duration_tokens.shape[2])
latent_inputs = vectors.at_rate(aligned_tokens, posterior.latent.shape[2])
duration_nll = models.duration_predictor.log_prob(
    durations, duration_inputs.dropped_concatenated(keep), phoneme.mask, generator,
)
conditions = models.condition_bank(latent_inputs, keep)
```

Move context-vector extraction into `stage2_conditions.py` to preserve file limits.
Add the embedding to construction, inference accounting, and optimizer ownership.
Validate duration input width as the exact sum of all nine raw source widths.

- [ ] **Step 4: Run all stage-2 tests**

Run: `nix develop --command pytest -q /tmp/test_beetle_stage2.py /tmp/test_beetle_stage2_objective.py /tmp/test_beetle_stage2_bundle.py /tmp/test_beetle_stage2_runtime.py /tmp/test_beetle_language_conditioning.py`

Expected: all selected files pass.

- [ ] **Step 5: Commit**

```bash
git add src/runner/nodes/training/beetle/models/stage2.py src/runner/nodes/training/beetle/training/stage2_inputs.py src/runner/nodes/training/beetle/training/stage2_setup.py src/runner/nodes/training/beetle/training/stage2_conditions.py
git commit -m "feat: share beetle conditions across prediction rates"
```

### Task 5: Documentation and Full Verification

**Files:**
- Modify: `src/runner/nodes/training/beetle/main.md`
- Modify: `docs/superpowers/plans/2026-07-17-beetle-language-conditioning.md`

**Interfaces:**
- Produces: documented vocabulary, rejection, dropout, rate, and accounting behavior.

- [ ] **Step 1: Update `main.md` with every approved behavior**

Document the ordered vocabulary, rejection policy, one shared vector, all nine
sources, phoneme-rate duration path, latent-rate flow path, independent dropout,
and parameter accounting.

- [ ] **Step 2: Run the preserved temporary suite**

Run: `nix develop --command pytest -q /tmp/test_beetle_*.py`

Expected: every test passes and no validation lifecycle is introduced.

- [ ] **Step 3: Run static checks**

Run: `nix develop --command ruff check src/runner/nodes/training/beetle src/shared/db/audio/segment_references_crud.py`

Run: `nix develop --command python -m compileall -q src/runner/nodes/training/beetle src/shared/db/audio/segment_references_crud.py`

Expected: both commands exit zero.

- [ ] **Step 4: Measure model limits**

Run the existing Beetle parameter and latent-audio complexity paths through Nix.
Record exact inference parameters and GFLOPs/s; report rather than hide any result
over 150M parameters or 15 GFLOPs/s.

- [ ] **Step 5: Verify file and diff integrity**

Run: `find src/runner/nodes/training/beetle -type f -name '*.py' -print0 | xargs -0 wc -l | sort -n | tail`

Run: `git diff --check`

Expected: edited Python files stay below 300 lines and no whitespace errors exist.

- [ ] **Step 6: Commit documentation**

```bash
git add src/runner/nodes/training/beetle/main.md docs/superpowers/plans/2026-07-17-beetle-language-conditioning.md
git commit -m "docs: describe beetle language conditioning"
```
