import { Field } from "@/shared/ui/form/Field";
import { useCheckpointsQuery } from "../checkpoints/query";
import { useDatasetsQuery } from "../datasets/query";
import type { WorkflowGraph, WorkflowSchema } from "../workflows/types";
import { FormSection } from "./FormSection";
import { FormSelect } from "./FormSelect";
import { checkpointOptions, datasetOptions, trainingNode, type TrainingWorkflowSpec, updateNodeParams } from "./logic";
import { SettingField } from "./SettingsField";

export function MosModelForm({
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
  const training = trainingNode(graph, spec.ids.training);
  const dataset = trainingNode(graph, spec.ids.dataset);
  const checkpoint = trainingNode(graph, spec.ids.checkpoint);
  const manifest = trainingNode(graph, spec.ids.manifest);
  const trainingInfo = schema.nodes[training.type];
  const manifestInfo = schema.nodes[manifest.type];
  if (!trainingInfo || !manifestInfo) throw new Error("MOS training nodes are not registered");
  const update = (nodeId: string, params: Record<string, unknown>) => onChange(updateNodeParams(graph, nodeId, params));

  return (
    <>
      <FormSection title="MOS model" tag="Wav2Vec2 XLS-R">
        <div className="grid grid-cols-2 gap-3.5">
          <SettingField schema={trainingInfo.settings} values={training.params} name="display_name" onChange={(params) => update(training.id, params)} />
          <Field label="Rating dataset">
            <FormSelect
              defaultValue=""
              value={String(dataset.params.dataset_id)}
              onChange={(dataset_id) => update(dataset.id, { ...dataset.params, dataset_id })}
              options={datasetOptions(datasets.data ?? [])}
            />
          </Field>
        </div>
        <div className="h-3.5" />
        <div className="grid grid-cols-2 gap-3.5">
          <Field label="Wav2Vec2 base checkpoint">
            <FormSelect
              defaultValue=""
              value={String(checkpoint.params.checkpoint_id)}
              onChange={(checkpoint_id) => update(checkpoint.id, { ...checkpoint.params, checkpoint_id })}
              options={checkpointOptions(checkpoints.data ?? [], "mos_base", "— select MOS base —")}
            />
          </Field>
          <SettingField schema={manifestInfo.settings} values={manifest.params} name="validation_comparisons" onChange={(params) => update(manifest.id, params)} />
        </div>
      </FormSection>

      <FormSection title="Optimization" tag="MOS + comparison loss">
        <div className="grid grid-cols-4 gap-3.5">
          <SettingField schema={trainingInfo.settings} values={training.params} name="batch_size" onChange={(params) => update(training.id, params)} />
          <SettingField schema={trainingInfo.settings} values={training.params} name="learning_rate" onChange={(params) => update(training.id, params)} />
          <SettingField schema={trainingInfo.settings} values={training.params} name="epochs" onChange={(params) => update(training.id, params)} />
          <SettingField schema={trainingInfo.settings} values={training.params} name="dataloader_workers" onChange={(params) => update(training.id, params)} />
        </div>
        <div className="h-3.5" />
        <div className="grid grid-cols-3 gap-3.5">
          <SettingField schema={trainingInfo.settings} values={training.params} name="comparison_weight" onChange={(params) => update(training.id, params)} />
          <SettingField schema={trainingInfo.settings} values={training.params} name="weight_decay" onChange={(params) => update(training.id, params)} />
          <SettingField schema={trainingInfo.settings} values={training.params} name="save_interval_epochs" onChange={(params) => update(training.id, params)} />
        </div>
      </FormSection>
    </>
  );
}
