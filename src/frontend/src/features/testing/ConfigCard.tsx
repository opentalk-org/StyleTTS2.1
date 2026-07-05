import { useCheckpoints } from "@/features/checkpoints/store";
import { seedStyleRefs } from "@/mock/data";
import type { SchemaValues } from "@/shared/schema-form/types";
import { showToast } from "@/shared/feedback/Toast";
import { Card } from "@/shared/ui/Card";
import { Field } from "@/shared/ui/form/Field";
import { Slider } from "@/shared/ui/form/Slider";
import { Select, type Option } from "@/shared/ui/Select";
import { Textarea } from "@/shared/ui/Textarea";
import type { WorkflowGraph, WorkflowSchema } from "../workflows/types";
import { checkpointOptions, enumOptions, numericSetting, testingNode, type TestingWorkflowSpec, updateNodeParams } from "./logic";

const STYLE_REFS: Option[] = [
  ...seedStyleRefs().map((r) => ({ value: r.id, label: `${r.name}  (${r.voice})` })),
  { value: "upload", label: "⤴ Upload reference…" },
];

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
  const checkpoints = useCheckpoints((s) => s.checkpoints);
  if (!spec.ids.styleRef || !spec.ids.synthesis) {
    throw new Error("Single testing workflow ids are incomplete");
  }
  const prompt = testingNode(graph, spec.ids.prompt);
  const alphabet = testingNode(graph, spec.ids.alphabet);
  const checkpoint = testingNode(graph, spec.ids.checkpoint);
  const styleRef = testingNode(graph, spec.ids.styleRef);
  const synthesis = testingNode(graph, spec.ids.synthesis);
  const steps = numericSetting(schema, synthesis, "diffusion_steps");
  const emb = numericSetting(schema, synthesis, "embedding_scale");
  const styleMix = numericSetting(schema, styleRef, "style_mix");
  const prosodyMix = numericSetting(schema, styleRef, "prosody_mix");
  const updateParams = (nodeId: string, params: SchemaValues) => onChange(updateNodeParams(graph, nodeId, params));

  const onStyleRef = (v: string) => {
    if (v === "upload") {
      showToast("Choose a .wav style reference");
      return;
    }
    updateParams(styleRef.id, { ...styleRef.params, reference_id: v });
  };

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
            <Select
              value={String(prompt.params.language)}
              onChange={(language) => updateParams(prompt.id, { ...prompt.params, language })}
              options={enumOptions(schema, prompt, "language")}
            />
          </Field>
          <Field label="Phoneme alphabet">
            <Select
              value={String(alphabet.params.preset)}
              onChange={(preset) => updateParams(alphabet.id, { ...alphabet.params, preset })}
              options={enumOptions(schema, alphabet, "preset")}
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
        <Field label="Alphabet symbols">
          <Textarea
            value={String(alphabet.params.symbols)}
            onChange={(e) => updateParams(alphabet.id, { ...alphabet.params, symbols: e.target.value })}
            spellCheck={false}
            className="min-h-[58px] text-xs font-mono leading-relaxed"
          />
        </Field>
      </div>

      <div className="h-px bg-line" />

      <div className="flex flex-col gap-2.5">
        <GroupTitle>Model</GroupTitle>
        <FieldGrid>
          <Field label="Checkpoint">
            <Select
              value={String(checkpoint.params.checkpoint_id)}
              onChange={(checkpoint_id) => updateParams(checkpoint.id, { ...checkpoint.params, checkpoint_id })}
              options={checkpointOptions(checkpoints)}
            />
          </Field>
          <Field label="Weights file">
            <Select
              value={String(synthesis.params.weights_file)}
              onChange={(weights_file) => updateParams(synthesis.id, { ...synthesis.params, weights_file })}
              options={enumOptions(schema, synthesis, "weights_file")}
            />
          </Field>
        </FieldGrid>
      </div>

      <div className="h-px bg-line" />

      <div className="flex flex-col gap-2.5">
        <GroupTitle>Style reference</GroupTitle>
        <FieldGrid>
          <Field label="Reference">
            <Select value={String(styleRef.params.reference_id)} onChange={onStyleRef} options={STYLE_REFS} />
          </Field>
          <Field label="Style mix" hint="Timbre adherence to reference.">
            <Slider
              value={Number(styleRef.params.style_mix)}
              onChange={(value) => updateParams(styleRef.id, { ...styleRef.params, style_mix: value })}
              min={styleMix.min}
              max={styleMix.max}
              step={0.05}
              format={(v) => v.toFixed(2)}
            />
          </Field>
          <Field label="Prosody mix" hint="Rhythm & intonation from reference.">
            <Slider
              value={Number(styleRef.params.prosody_mix)}
              onChange={(value) => updateParams(styleRef.id, { ...styleRef.params, prosody_mix: value })}
              min={prosodyMix.min}
              max={prosodyMix.max}
              step={0.05}
              format={(v) => v.toFixed(2)}
            />
          </Field>
        </FieldGrid>
      </div>
    </Card>
  );
}
