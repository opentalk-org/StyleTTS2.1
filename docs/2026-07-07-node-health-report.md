# Runner Node & Port Health Report

Date: 2026-07-07 (updated after cleanup pass)
Scope: `src/runner/nodes/**` (node layer) + the `src/frontend` training UI that builds graphs.
Vendored StyleTTS2 model code under `training/styletts/finetune/training/modules/**` is out of scope.

Everything below the "Cleanup applied" section was verified: the runner registry imports and builds its
schema, affected nodes instantiate from their schema defaults, the OOD-from-bundle path was exercised
end-to-end, every frontend graph-template edge resolves against the schema, and the frontend passes
`tsc --noEmit`.

---

## Cleanup applied in this pass

### Port fragmentation collapsed (+ 2 more nodes removed)
The fragmented port names were unified to one canonical name per datatype, which turned the rename-only
`Prefetch*` nodes into pure identity pass-throughs, so they were removed:
- **`CHECKPOINT_REF`** ports `checkpoint_ref` / `base_checkpoint` / `pretrained_checkpoint` → all **`checkpoint`**
  (SelectCheckpoint/ResolveCheckpoint outputs; BuildTrainingManifest / StyleTtsFinetune / F0 / ASR inputs).
- **`ASSET_BUNDLE`** ports `asset_refs` / `pretrained_assets` → all **`assets`**.
- statistics `features` (output) → **`feature_records`**, matching the AggregateDatasetStatistics input.
- Removed **`PrefetchCheckpoint`** and **`PrefetchTrainingAssets`** (their `prefetch_*` helpers were no-op
  isinstance guards; the real bucket fetch happens in `resolve_checkpoint_ref` /
  `resolve_training_asset_bundle`, so no fetching was lost). Frontend training + testing graph templates
  were rewired to connect Select→consumer directly with the unified names.

### Nodes removed (2)
- **`CalculateAudioStats`** — duplicate of `AnalyzeAudioFeatures` (shared its settings/policy and called
  the same `analyze_audio_features`). Dropped from `audio_processing`, the package `__init__`, and the registry.
- **`BuildStyleTtsFinetuneConfig`** — its `training_config` output was consumed by nothing;
  `StyleTtsFinetune` already rebuilds the config inline. Removed the node; folded `config_output_dir`
  into `StyleTtsFinetuneSettings`; deleted the now-dead `training_config_output_dir` helper.

### OOD text sets are now plain assets (special-casing removed)
OOD text sets are extra-file assets, and `ResolveTrainingAssets` + `node_config._ood_text_path` already
knew how to source them from the asset bundle (role `ood_text_set`). The parallel JSON path was deleted:
- Removed **`SelectOodTextSets`** and **`PrefetchOodTextSets`** nodes, `SelectOodTextSetsSettings`,
  `OodTextSet`, and the `_resolve_ood_text_sets` / `_prefetch_ood_text_sets` helpers.
- Removed the `ood_text_sets` input port from `StyleTtsFinetune` and the `ood_text_set_refs` / `ood_text_sets`
  JSON ports entirely.
- `SelectTrainingAssets` gained an `ood_text_set_file_ids` setting and now bundles OOD like every other asset.
  `node_config._training_asset_paths` collects **all** `ood_text_set` role paths and concatenates multiple
  sets into one file (single set passes through untouched).
- **Fetch semantics preserved:** OOD files are still fetched from the bucket exactly once via
  `asset_crud.get_extra_file_path` inside `resolve_training_asset_bundle`; the config reads the cached
  local paths. No extra bucket round-trips.
- Frontend updated to match: `training/logic.ts` (dropped the two nodes + their edges + the `oodSets` id),
  `OodEditor.tsx` (now edits `ood_text_set_file_ids` on the assets node), `StyleTtsForm.tsx`, `QueueCard.tsx`.

### Dead ports removed
- `segment_records` input on `AggregateDatasetStatistics` — no producer existed; segments still flow
  embedded inside `feature_records` via `_flatten_segments`.
- `training_config` output and `stats` output — died with the two removed nodes.
- `ood_text_set_refs` / `ood_text_sets` — died with the OOD nodes.

### Dead code / dead settings / dead imports removed
- ASR: the entire `_transcribe_segment_paths` method family (base + 3 overrides), the `segment_batch_size`
  settings, `write_segment_wavs`, `wav_duration`, and `transcribe_wav_to_text` (all only reachable through it).
- `audio_segments/writeback.py`: dead `_save_group_segments`.
- `text/runtime/phonemize.py`: singular `phonemize_text` (+ its `__init__` re-exports).
- `text/runtime/symbols.py`: unused `symbols` alias.
- `statistics/audio_features.py`: `StatisticsFeatureRecord` (emitted an unread `lineage_id`) collapsed to a
  plain dict; dead `_statistics_lineage_id`.
- No-op settings deleted: `CutAudioSettings.fade_ms` (whole class gone), `DeepFilterNetSettings.strength`,
  `NormalizeSettings.target_lufs`, `AudioFeatureSettings.histogram_bins`,
  `StyleTtsRequestSettings.output_name` + `weights_file` (and the unreachable `resolve_weights_path` branch).
- Unused imports: `extract.py` `Field`, `ds_v1_parquet.py` `UUID`, `denoise.py` `Field`, plus the imports
  orphaned by the removals above.

### Duplicated helpers consolidated
- `_typed_checkpoint` (**6 copies**) and `_typed_assets` (**2 copies**) → single `typed_checkpoint` /
  `typed_assets` in `runner/nodes/models.py`.
- `_maybe_cuda_half` (parakeet + canary) → `maybe_cuda_half` in `accelerator_memory.py`.
- `_prompt_text` (synthesis) → one copy in `actions.py`, reused by `styletts.py`.
- Removed the redundant `load_synthesis_runtime` wrapper in `actions.py`; callers use the runtime module directly.

> Note: `StrictSettings` was deliberately left `extra="forbid"` (still catches typo'd params). Removing the
> settings above will make older UI-authored graphs that persisted those keys fail at compile — accepted per
> the cleanup decision to allow breaking old graphs.

---

## Remaining findings (not addressed — structural or lower value)

### Duplicate nodes still present
These are safe subset-duplicates but removing them means committing to one of two parallel designs; left as-is
because the frontend training template uses the `Select*` path while `Resolve*` stays registered:
- `SelectCheckpoint` ⊂ `ResolveCheckpoint` (same `resolve_checkpoint_ref`, Select just adds a `run` port).
- `SelectTrainingAssets` ⊂ `ResolveTrainingAssets` (same resolver; now both carry OOD).
- `TestingRunInputNode` ≡ `TrainingRunInputNode` — byte-identical seed nodes in different families; could
  share one `RunInputNode` base.

### Mergeable groups (parameterize / shared base)
- 3 audio-source nodes (`AudioSource` + the two `HetznerDsV*Parquet`) — the two hetzner variants still share
  ~120 duplicated lines (`_download_sftp_file`, `_iter_parquet_rows`, …) that want a common base.
- 4 `dataset_writeback` CRUD nodes → one `op=add|remove|delete` node.
- 3 training nodes (`Asr`/`F0`/`StyleTtsFinetune`) → a `BaseTrainingNode` (teardown + accelerator policy +
  execute loop) with F0/ASR sharing a `_run_seq_training` template.
- `DeepFilterNetDenoise` + `NormalizeLoudness` — keep separate (resource asymmetry) but extract the shared
  optional-dependency loader.
- `_without_large_fields` still exists in two slightly divergent copies (styletts strips `{data, wav_base64}`,
  actions strips `{wav_base64}`) — left because the behavior differs.

### Datatype-level smells
- **`JSON` still carries ~11 distinct logical schemas** (`run`, `dataset_ref`, `audio_file_ids`,
  `phoneme_alphabet`, `prompt_text`, `style_reference`, `feature_records`, `statistics`,
  `statistics_entry`, `writeback_result`, `catalog_item`) — typed ports give no safety across them. Promote
  the stable-shaped ones to real datatypes or use `UnionDataType` (which no port uses today).
- `TEXT`, `INT`, `FLOAT` are still registered but wired to zero ports.
- `SAVE_RESULT`, `TRAINING_RESULT`, `SYNTHESIS_RESULT` are sink-only (never an input) — confirm the runtime/UI
  surfaces them.

### Other smells (unchanged)
- `assert` used for runtime validation in many nodes (stripped under `python -O`).
- Voice lookups do O(n) "list all + Python filter" instead of a `voice_crud.get_or_create_by_name` facade.
- `PlanSegmentGroups → Extract → PersistSplit` pass an untyped `metadata` dict as their cross-node contract.
- Input nodes with DB side effects (`AudioSource` reads; `HetznerDsV2Parquet` upserts voices).
- CATEGORY taxonomy is ad hoc (siblings split across `Inputs`/`Training`/`Assets`).
