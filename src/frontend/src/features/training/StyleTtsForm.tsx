import type { SchemaValues } from "@/shared/schema-form/types";
import { Field } from "@/shared/ui/form/Field";
import { Toggle } from "@/shared/ui/form/Toggle";
import { useCreateTextFileAssetMutation, useFileAssetsQuery } from "../assets/query";
import { useCheckpointsQuery } from "../checkpoints/query";
import { useDatasetsQuery } from "../datasets/query";
import type { WorkflowGraph, WorkflowSchema } from "../workflows/types";

import { AlphabetEditor } from "./AlphabetEditor";
import { AssetSlot } from "./AssetSlot";
import { FormSection } from "./FormSection";
import { FormSelect } from "./FormSelect";
import { checkpointOptions, checkpointSymbolCount, datasetOptions, oodSetValues, pretrainedAssetOptions, styleTtsParamsForBaseCheckpoint, trainingNode, type TrainingWorkflowSpec, updateNodeParams, updateTrainingParams } from "./logic";
import { OodEditor } from "./OodEditor";
import { useCreateTrainingConfigMutation, useTrainingConfigsQuery } from "./query";
import { SettingField, SettingNumberInput, settingLabel } from "./SettingsField";

const TOGGLE_ROWS = [
  { key: "multispeaker", sub: "Per-speaker style encoder" },
  { key: "checkpoint_each_stage", sub: "Save base / diffusion / joint separately" },
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
  const oodAssets = useFileAssetsQuery("OOD_TEXT_SET");
  const alphabets = useTrainingConfigsQuery("phoneme_alphabet");
  const createOod = useCreateTextFileAssetMutation("OOD_TEXT_SET");
  const createAlphabet = useCreateTrainingConfigMutation("phoneme_alphabet");
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
  const updateTraining = (params: SchemaValues) => onChange(updateTrainingParams(graph, spec, params));
  const selectedCheckpoint = (checkpoints.data ?? []).find((item) => item.id === String(checkpoint.params.checkpoint_id));
  const selectBaseCheckpoint = (checkpoint_id: string) => {
    const nextCheckpoint = (checkpoints.data ?? []).find((item) => item.id === checkpoint_id);
    let next = updateNodeParams(graph, checkpoint.id, { ...checkpoint.params, checkpoint_id });
    next = updateTrainingParams(next, spec, styleTtsParamsForBaseCheckpoint(nextCheckpoint, values));
    onChange(next);
  };

  return (
    <>
      <FormSection title="Identity & data" tag="Run">
        <div className="grid grid-cols-2 gap-3.5">
          <SettingField schema={settingsSchema} values={values} name="display_name" onChange={updateTraining} />
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
          <SettingField schema={settingsSchema} values={values} name="validation_samples" onChange={updateTraining} />
          <Field label="Base checkpoint" hint="Required — finetuning resumes from here.">
            <FormSelect
              defaultValue=""
              value={String(checkpoint.params.checkpoint_id)}
              onChange={selectBaseCheckpoint}
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
            options={pretrainedAssetOptions(f0Assets.data ?? [], checkpoints.data ?? [], "f0", "— select F0 asset —")}
          />
          <AssetSlot
            label="ASR model"
            value={String(assets.params.asr_model)}
            onChange={(asr_model) => updateParams(assets.id, { ...assets.params, asr_model })}
            options={pretrainedAssetOptions(asrAssets.data ?? [], checkpoints.data ?? [], "asr", "— select ASR asset —")}
          />
          <AssetSlot
            label="PL-BERT"
            value={String(assets.params.plbert_model)}
            onChange={(plbert_model) => updateParams(assets.id, { ...assets.params, plbert_model })}
            options={pretrainedAssetOptions(plbertAssets.data ?? [], checkpoints.data ?? [], "plbert", "— select PL-BERT asset —")}
          />
        </div>
      </FormSection>

      <AlphabetEditor
        values={alphabet.params}
        presets={alphabets.data ?? []}
        baseSymbolCount={checkpointSymbolCount(selectedCheckpoint)}
        onChange={(params) => updateParams(alphabet.id, params)}
        onSave={(name, symbols) =>
          createAlphabet.mutate({
            name,
            type_: "phoneme_alphabet",
            metadata: { preset: "custom", symbols: symbols.trim().split(/\s+/).filter(Boolean) },
          })
        }
      />

      <FormSection title="Optimization" tag="Optimizer">
        <div className="grid grid-cols-3 gap-3.5">
          <SettingField schema={settingsSchema} values={values} name="batch_size" onChange={updateTraining} />
          <SettingField schema={settingsSchema} values={values} name="learning_rate" onChange={updateTraining} />
          <SettingField schema={settingsSchema} values={values} name="numeric_precision" onChange={updateTraining} />
        </div>
      </FormSection>

      <FormSection title="Schedule" tag="Epochs & sequence">
        <div className="grid grid-cols-3 gap-3.5">
          <SettingField schema={settingsSchema} values={values} name="epochs_base" onChange={updateTraining} />
          <SettingField schema={settingsSchema} values={values} name="epochs_diffusion" onChange={updateTraining} />
          <SettingField schema={settingsSchema} values={values} name="epochs_joint" onChange={updateTraining} />
        </div>
        <div className="h-3.5" />
        <div className="grid grid-cols-3 gap-3.5">
          <SettingNumberInput schema={settingsSchema} values={values} name="max_sequence_seconds" hint={`≈ ${frames} frames @ 300 hop`} step={0.5} onChange={updateTraining} />
          <SettingField schema={settingsSchema} values={values} name="save_interval_epochs" onChange={updateTraining} />
          <SettingField schema={settingsSchema} values={values} name="decoder" onChange={updateTraining} />
        </div>
      </FormSection>

      <FormSection title="SLM adversarial" tag="Discriminator">
        <div className="grid grid-cols-3 gap-3.5">
          <SettingField schema={settingsSchema} values={values} name="slmadv_min_len" onChange={updateTraining} />
          <SettingField schema={settingsSchema} values={values} name="slmadv_max_len" onChange={updateTraining} />
          <SettingField schema={settingsSchema} values={values} name="slm_scale" onChange={updateTraining} />
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
                onChange={(value) => updateTraining({ ...values, [r.key]: value })}
              />
            </label>
          ))}
        </div>
      </FormSection>

      <OodEditor
        values={oodSets.params}
        availableSets={oodSetValues(oodAssets.data ?? [])}
        onChange={(params) => updateParams(oodSets.id, params)}
        onCreate={async ({ name, content }) => {
          const item = await createOod.mutateAsync({
            name,
            type_: "OOD_TEXT_SET",
            content,
            metadata: { source: "training_tab" },
          });
          const created = oodSetValues([item])[0];
          if (!created) throw new Error("OOD text set was not created");
          return created;
        }}
      />
    </>
  );
}
