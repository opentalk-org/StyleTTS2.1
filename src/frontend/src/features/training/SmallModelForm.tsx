import type { SchemaValues } from "@/shared/schema-form/types";
import { Field } from "@/shared/ui/form/Field";
import { useCheckpointsQuery } from "../checkpoints/query";
import { useDatasetsQuery } from "../datasets/query";
import type { WorkflowGraph, WorkflowSchema } from "../workflows/types";

import { AlphabetEditor } from "./AlphabetEditor";
import { FormSection } from "./FormSection";
import { FormSelect } from "./FormSelect";
import { checkpointOptions, datasetOptions, trainingNode, type TrainingWorkflowSpec, updateNodeParams } from "./logic";
import { SettingField } from "./SettingsField";

/** Compact finetune form for the F0 pitch extractor and ASR aligner models. */
export function SmallModelForm({
  variant,
  schema,
  graph,
  spec,
  onChange,
}: {
  variant: "f0" | "asr";
  schema: WorkflowSchema;
  graph: WorkflowGraph;
  spec: TrainingWorkflowSpec;
  onChange: (graph: WorkflowGraph) => void;
}) {
  const datasets = useDatasetsQuery();
  const checkpoints = useCheckpointsQuery();
  const isAsr = variant === "asr";
  const training = trainingNode(graph, spec.ids.training);
  const dataset = trainingNode(graph, spec.ids.dataset);
  const checkpoint = trainingNode(graph, spec.ids.checkpoint);
  const alphabet = isAsr && spec.ids.alphabet ? trainingNode(graph, spec.ids.alphabet) : null;
  const trainingInfo = schema.nodes[training.type];
  if (!trainingInfo) throw new Error(`Training node is not registered: ${training.type}`);
  const settingsSchema = trainingInfo.settings;
  const values = training.params;
  const updateParams = (nodeId: string, params: SchemaValues) => onChange(updateNodeParams(graph, nodeId, params));
  const checkpointType = isAsr ? "asr" : "f0";

  return (
    <>
      <FormSection
        title={isAsr ? "ASR model" : "F0 model"}
        tag={isAsr ? "Aligner" : "Pitch extractor"}
      >
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
          <Field label="Pretrained (optional)">
            <FormSelect
              defaultValue=""
              value={String(checkpoint.params.checkpoint_id)}
              onChange={(checkpoint_id) => updateParams(checkpoint.id, { ...checkpoint.params, checkpoint_id })}
              options={checkpointOptions(checkpoints.data ?? [], checkpointType, "— train from scratch —")}
            />
          </Field>
          <SettingField schema={settingsSchema} values={values} name="validation_samples" onChange={(params) => updateParams(training.id, params)} />
        </div>
      </FormSection>

      <FormSection title="Optimization" tag="Optimizer">
        <div className={isAsr ? "grid grid-cols-4 gap-3.5" : "grid grid-cols-3 gap-3.5"}>
          <SettingField schema={settingsSchema} values={values} name="batch_size" onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="learning_rate" onChange={(params) => updateParams(training.id, params)} />
          <SettingField schema={settingsSchema} values={values} name="epochs" onChange={(params) => updateParams(training.id, params)} />
          {isAsr ? <SettingField schema={settingsSchema} values={values} name="dataloader_workers" onChange={(params) => updateParams(training.id, params)} /> : null}
        </div>
        <div className="h-3.5" />
        <div className="grid grid-cols-2 gap-3.5">
          <SettingField schema={settingsSchema} values={values} name="save_interval_epochs" onChange={(params) => updateParams(training.id, params)} />
          <div />
        </div>
      </FormSection>

      {alphabet ? <AlphabetEditor values={alphabet.params} onChange={(params) => updateParams(alphabet.id, params)} /> : null}
    </>
  );
}
