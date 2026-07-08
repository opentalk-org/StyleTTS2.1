import type { SchemaValues } from "@/shared/schema-form/types";
import { Card } from "@/shared/ui/Card";
import { Field } from "@/shared/ui/form/Field";
import { NumberInput } from "@/shared/ui/form/NumberInput";
import { Slider } from "@/shared/ui/form/Slider";
import { Input } from "@/shared/ui/Input";
import { Select } from "@/shared/ui/Select";
import { Textarea } from "@/shared/ui/Textarea";
import { useCheckpointsQuery } from "../checkpoints/query";
import { AudioSinglePickerField } from "../workflows/components/WorkflowFieldPickers";
import type { WorkflowGraph, WorkflowSchema } from "../workflows/types";
import { DatasetField } from "./DatasetField";
import { checkpointOptions, numericSetting, testingNode, type TestingWorkflowSpec, updateNodeParams } from "./logic";

function GroupTitle({ children }: { children: string }) {
  return (
    <div className="text-[11px] font-bold uppercase tracking-[0.06em] text-blue-500">
      {children}
    </div>
  );
}

/** Auto-fit grid of form fields; each cell can shrink below content width. */
function FieldGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-x-5 gap-y-4 [&>*]:min-w-0">
      {children}
    </div>
  );
}

/** Single-synthesis config: Text & inference / Model / Style reference. */
export function ConfigCard({
  schema,
  graph,
  spec,
  onChange,
}: {
  schema: WorkflowSchema;
  graph: WorkflowGraph;
  spec: TestingWorkflowSpec;
  onChange: (graph: WorkflowGraph) => void;
}) {
  const checkpoints = useCheckpointsQuery();
  if (!spec.ids.styleRef || !spec.ids.synthesis) {
    throw new Error("Single testing workflow ids are incomplete");
  }
  const prompt = testingNode(graph, spec.ids.prompt);
  const checkpoint = testingNode(graph, spec.ids.checkpoint);
  const styleRef = testingNode(graph, spec.ids.styleRef);
  const synthesis = testingNode(graph, spec.ids.synthesis);
  const steps = numericSetting(schema, synthesis, "diffusion_steps");
  const emb = numericSetting(schema, synthesis, "embedding_scale");
  const alpha = numericSetting(schema, styleRef, "alpha");
  const beta = numericSetting(schema, styleRef, "beta");
  const updateParams = (nodeId: string, params: SchemaValues) => onChange(updateNodeParams(graph, nodeId, params));

  return (
    <Card className="flex flex-col gap-5 rounded-xl px-6 py-[22px]">
      <div className="flex flex-col gap-2.5">
        <GroupTitle>Text &amp; inference</GroupTitle>
        <Field label="Text">
          <Textarea
            value={String(prompt.params.text)}
            onChange={(e) => updateParams(prompt.id, { ...prompt.params, text: e.target.value })}
          />
        </Field>
        <FieldGrid>
          <Field label="Language">
            <Input
              value={String(prompt.params.language)}
              onChange={(event) => updateParams(prompt.id, { ...prompt.params, language: event.target.value })}
              filled
            />
          </Field>
          <Field label="Diffusion steps">
            <Slider
              value={Number(synthesis.params.diffusion_steps)}
              onChange={(diffusion_steps) => updateParams(synthesis.id, { ...synthesis.params, diffusion_steps })}
              min={steps.min}
              max={steps.max}
              step={1}
            />
          </Field>
          <Field label="Embedding scale">
            <Slider
              value={Number(synthesis.params.embedding_scale)}
              onChange={(embedding_scale) => updateParams(synthesis.id, { ...synthesis.params, embedding_scale })}
              min={emb.min}
              max={emb.max}
              step={0.1}
              format={(v) => v.toFixed(1)}
            />
          </Field>
        </FieldGrid>
      </div>

      <div className="h-px bg-line" />

      <div className="flex flex-col gap-2.5">
        <GroupTitle>Model</GroupTitle>
        <FieldGrid>
          <Field label="Checkpoint">
            <Select
              value={String(checkpoint.params.checkpoint_id)}
              onChange={(checkpoint_id) => updateParams(checkpoint.id, { ...checkpoint.params, checkpoint_id })}
              options={checkpointOptions(checkpoints.data ?? [])}
            />
          </Field>
          <DatasetField graph={graph} datasetNodeId={spec.ids.dataset} onChange={onChange} />
        </FieldGrid>
      </div>

      <div className="h-px bg-line" />

      <div className="flex flex-col gap-2.5">
        <GroupTitle>Style reference</GroupTitle>
        <FieldGrid>
          <AudioSinglePickerField
            label="Reference"
            value={styleRef.params.reference_id}
            onChange={(reference_id) => updateParams(styleRef.id, { ...styleRef.params, reference_id })}
          />
          <Field label="Alpha" hint="Timbre adherence to reference.">
            <NumberInput
              value={Number(styleRef.params.alpha)}
              onChange={(value) => updateParams(styleRef.id, { ...styleRef.params, alpha: value })}
              min={alpha.min}
              max={alpha.max}
              step={0.05}
            />
          </Field>
          <Field label="Beta" hint="Rhythm & intonation from reference.">
            <NumberInput
              value={Number(styleRef.params.beta)}
              onChange={(value) => updateParams(styleRef.id, { ...styleRef.params, beta: value })}
              min={beta.min}
              max={beta.max}
              step={0.05}
            />
          </Field>
        </FieldGrid>
      </div>
    </Card>
  );
}
