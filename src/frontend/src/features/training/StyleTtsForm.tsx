import type { SchemaValues } from "@/shared/schema-form/types";
import { Field } from "@/shared/ui/form/Field";
import { Toggle } from "@/shared/ui/form/Toggle";
import { useFileAssetsQuery } from "../assets/query";
import { useCheckpointsQuery } from "../checkpoints/query";
import { useDatasetsQuery } from "../datasets/query";
import type { WorkflowGraph, WorkflowSchema } from "../workflows/types";

import { AlphabetEditor } from "./AlphabetEditor";
import { AssetSlot } from "./AssetSlot";
import { FormSection } from "./FormSection";
import { FormSelect } from "./FormSelect";
import { checkpointOptions, checkpointSymbolCount, datasetOptions, fileAssetOptions, oodSetValues, trainingNode, type TrainingWorkflowSpec, updateNodeParams } from "./logic";
import { OodEditor } from "./OodEditor";
import { SettingField, SettingNumberInput, settingLabel } from "./SettingsField";

const TOGGLE_ROWS = [
  { key: "multispeaker", sub: "Per-speaker style encoder" },
  { key: "checkpoint_each_stage", sub: "Save base / diffusion / joint separately" },
  { key: "mixed_precision", sub: "Faster, slightly less stable" },
];

/** The full StyleTTS finetune configuration form. */
export function StyleTtsForm({
  schema,
  graph,
  spec,
  onChange,
}: {
  schema: WorkflowSchema;
  graph: WorkflowGraph;
  spec: TrainingWorkflowSpec;
  onChange: (graph: WorkflowGraph) => void;
}) {
  const datasets = useDatasetsQuery();
  const checkpoints = useCheckpointsQuery();
  const f0Assets = useFileAssetsQuery("f0");
  const asrAssets = useFileAssetsQuery("asr");
  const plbertAssets = useFileAssetsQuery("plbert");
  const oodAssets = useFileAssetsQuery("ood_text");
  const training = trainingNode(graph, spec.ids.training);
  const dataset = trainingNode(graph, spec.ids.dataset);
  const checkpoint = trainingNode(graph, spec.ids.checkpoint);
  const assets = trainingNode(graph, spec.ids.assets as string);
  const alphabet = trainingNode(graph, spec.ids.alphabet as string);
  const oodSets = trainingNode(graph, spec.ids.oodSets as string);
  const values = training.params;
  const trainingInfo = schema.nodes[training.type];
  if (!trainingInfo) throw new Error(`Training node is not registered: ${training.type}`);
  const settingsSchema = trainingInfo.settings;
  const seqSeconds = Number(values.max_sequence_seconds);
  const frames = Math.round((seqSeconds * 24000) / 300);
  const updateParams = (nodeId: string, params: SchemaValues) => onChange(updateNodeParams(graph, nodeId, params));
  const selectedCheckpoint = (checkpoints.data ?? []).find((item) => item.id === String(checkpoint.params.checkpoint_id));

  return (
    <>
      <FormSection title="Identity & data" tag="Run">
        <div className="grid grid-cols-2 gap-3.5">
          <SettingField schema={settingsSchema} values={values} name="display_name" onChange={(params) => updateParams(training.id, params)} />
          <Field label="Training dataset">
            <FormSelect
              defaultValue=""
              value={String(dataset.params.dataset_id)}
              onChange={(dataset_id) => updateParams(dataset.id, { ...dataset.params, dataset_id })}
              options={datasetOptions(datasets.data ?? [])}
            />
          </Field>
        </div>
        <div className="h-3.5" />
        <div className="grid grid-cols-2 gap-3.5">
          <SettingField schema={settingsSchema} values={values} name="validation_samples" onChange={(params) => updateParams(training.id, params)} />
          <Field label="Base checkpoint" hint="Required — finetuning resumes from here.">
            <FormSelect
              defaultValue=""
              value={String(checkpoint.params.checkpoint_id)}
              onChange={(checkpoint_id) => updateParams(checkpoint.id, { ...checkpoint.params, checkpoint_id })}
              options={checkpointOptions(checkpoints.data ?? [], "styletts2", "— select base checkpoint —")}
            />
          </Field>
        </div>
      </FormSection>

      <FormSection title="Asset slots" tag="Optional pretrained">
        <div className="grid grid-cols-3 gap-3.5">
          <AssetSlot
            label="F0 model"
            value={String(assets.params.f0_model)}
            onChange={(f0_model) => updateParams(assets.id, { ...assets.params, f0_model })}
            options={fileAssetOptions(f0Assets.data ?? [], "— select F0 file —")}
          />
          <AssetSlot
            label="ASR model"
            value={String(assets.params.asr_model)}
            onChange={(asr_model) => updateParams(assets.id, { ...assets.params, asr_model })}
            options={fileAssetOptions(asrAssets.data ?? [], "— select ASR file —")}
          />
          <AssetSlot
            label="PL-BERT"
            value={String(assets.params.plbert_model)}
            onChange={(plbert_model) => updateParams(assets.id, { ...assets.params, plbert_model })}
            options={fileAssetOptions(plbertAssets.data ?? [], "— select PL-BERT file —")}
          />
        </div>
      </FormSection>

      <AlphabetEditor
        values={alphabet.params}
        baseSymbolCount={checkpointSymbolCount(selectedCheckpoint)}
        onChange={(params) => updateParams(alphabet.id, params)}
      />

      <FormSection title="Optimization" tag="Optimizer">
        <div className="grid grid-cols-3 gap-3.5">
          <SettingField schema={settingsSchema} values={values} name="batch_size" onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="learning_rate" onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="numeric_precision" onChange={(params) => updateParams(training.id, params)} />
        </div>
        <div className="mt-4 mb-2.5 text-xs font-semibold text-txt-dim">
          Gradient clipping (max norm)
        </div>
        <div className="grid grid-cols-3 gap-3.5">
          <SettingField schema={settingsSchema} values={values} name="clip_total" onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="clip_diffusion" onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="clip_slm" onChange={(params) => updateParams(training.id, params)} />
        </div>
      </FormSection>

      <FormSection title="Schedule" tag="Epochs & sequence">
        <div className="grid grid-cols-3 gap-3.5">
          <SettingField schema={settingsSchema} values={values} name="epochs_base" onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="epochs_diffusion" onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="epochs_joint" onChange={(params) => updateParams(training.id, params)} />
        </div>
        <div className="h-3.5" />
        <div className="grid grid-cols-3 gap-3.5">
          <SettingNumberInput schema={settingsSchema} values={values} name="max_sequence_seconds" hint={`≈ ${frames} frames @ 300 hop`} step={0.5} onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="save_interval_epochs" onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="decoder" onChange={(params) => updateParams(training.id, params)} />
        </div>
      </FormSection>

      <FormSection title="SLM adversarial" tag="Discriminator">
        <div className="grid grid-cols-3 gap-3.5">
          <SettingField schema={settingsSchema} values={values} name="slm_weight" onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="diffusion_samples" onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="slm_scale" onChange={(params) => updateParams(training.id, params)} />
        </div>
        <div className="mt-4 flex flex-col gap-3">
          {TOGGLE_ROWS.map((r) => (
            <label key={r.key} className="flex items-center gap-3 cursor-pointer">
              <span className="flex-1">
                <div className="text-[13px] font-semibold text-txt">{settingLabel(settingsSchema, r.key)}</div>
                <div className="text-xs text-txt-mute">{r.sub}</div>
              </span>
              <Toggle
                checked={Boolean(values[r.key])}
                onChange={(value) => updateParams(training.id, { ...values, [r.key]: value })}
              />
            </label>
          ))}
        </div>
      </FormSection>

      <OodEditor
        values={oodSets.params}
        availableSets={oodSetValues(oodAssets.data ?? [])}
        onChange={(params) => updateParams(oodSets.id, params)}
      />
    </>
  );
}
