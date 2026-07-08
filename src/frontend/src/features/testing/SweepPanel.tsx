import { WaveformPlayer } from "@/shared/media/WaveformPlayer";
import { Icon } from "@/shared/icons";
import type { SchemaValues } from "@/shared/schema-form/types";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { Field } from "@/shared/ui/form/Field";
import { NumberInput } from "@/shared/ui/form/NumberInput";
import { Slider } from "@/shared/ui/form/Slider";
import { Input } from "@/shared/ui/Input";
import { Select } from "@/shared/ui/Select";
import { Textarea } from "@/shared/ui/Textarea";
import { cn } from "@/shared/ui/cn";
import { backendResourceUrl } from "@/app/backend";
import { showToast } from "@/shared/feedback/Toast";
import { useCheckpointsQuery } from "../checkpoints/query";
import { useVoicesQuery } from "../voices/query";
import { defaultWorkflowContext, graphPayload, runtimeConfigForGraph } from "../workflows/logic";
import type { WorkflowGraph, WorkflowSchema } from "../workflows/types";
import { DatasetField } from "./DatasetField";
import { checkpointOptions, numericSetting, sweepConfigFromGraph, testingNode, type TestingWorkflowSpec, updateNodeParams } from "./logic";
import { useTesting, type SweepDisplay } from "./store";

const VOICE_QUERY = { query: "", limit: 200, offset: 0 };

function VoiceChip({ name, on, onClick }: { name: string; on: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex h-[34px] items-center gap-1.5 rounded-full border-[1.5px] px-3.5 text-[12.5px] font-semibold cursor-pointer transition-colors",
        on
          ? "border-blue-500 bg-blue-50 text-blue-700"
          : "border-line-2 bg-panel text-txt-dim",
      )}
    >
      {on ? <Icon name="check" size={13} strokeWidth={3} className="text-blue-600" /> : null}
      {name}
    </button>
  );
}

/** Sweep config form + a grid of per-voice result cards. */
export function SweepPanel({
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
  const voicesQuery = useVoicesQuery(VOICE_QUERY);
  const results = useTesting((s) => s.sweepResults);
  const runSweep = useTesting((s) => s.runSweep);
  const pending = results.some((r) => r.state === "running");
  if (!spec.ids.styleSweep) throw new Error("Sweep testing workflow ids are incomplete");
  const prompt = testingNode(graph, spec.ids.prompt);
  const checkpoint = testingNode(graph, spec.ids.checkpoint);
  const styleSweep = testingNode(graph, spec.ids.styleSweep);
  const samples = numericSetting(schema, styleSweep, "samples_per_voice");
  const alpha = numericSetting(schema, styleSweep, "alpha");
  const beta = numericSetting(schema, styleSweep, "beta");
  const voices = voicesQuery.data?.rows ?? [];
  const sweep = sweepConfigFromGraph(graph, spec, voices);
  const selectedIds = selectedVoiceIds(styleSweep.params.voices);
  const updateParams = (nodeId: string, params: SchemaValues) => onChange(updateNodeParams(graph, nodeId, params));
  const toggleVoice = (voiceId: string) => {
    const nextVoices = selectedIds.includes(voiceId)
      ? selectedIds.filter((item) => item !== voiceId)
      : [...selectedIds, voiceId];
    updateParams(styleSweep.id, { ...styleSweep.params, voices: nextVoices });
  };

  return (
    <div>
      <Card className="mb-5 flex flex-col gap-3.5 p-5">
        <Field label="Text">
          <Textarea
            value={String(prompt.params.text)}
            onChange={(e) => updateParams(prompt.id, { ...prompt.params, text: e.target.value })}
            className="min-h-[60px]"
          />
        </Field>
        <div className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-x-5 gap-y-4 [&>*]:min-w-0">
          <Field label="Checkpoint">
            <Select
              value={String(checkpoint.params.checkpoint_id)}
              onChange={(checkpoint_id) => updateParams(checkpoint.id, { ...checkpoint.params, checkpoint_id })}
              options={checkpointOptions(checkpoints.data ?? [])}
            />
          </Field>
          <Field label="Language">
            <Input
              value={String(prompt.params.language)}
              onChange={(event) => updateParams(prompt.id, { ...prompt.params, language: event.target.value })}
              filled
            />
          </Field>
          <DatasetField graph={graph} datasetNodeId={spec.ids.dataset} onChange={onChange} />
        </div>
        <Field label="Voices">
          <div className="flex flex-wrap gap-2">
            {voices.length ? (
              voices.map((voice) => (
                <VoiceChip key={voice.id} name={voice.name} on={selectedIds.includes(voice.id)} onClick={() => toggleVoice(voice.id)} />
              ))
            ) : (
              <span className="text-xs text-txt-mute">No voices available.</span>
            )}
          </div>
        </Field>
        <div className="flex flex-wrap items-end gap-4">
          <div className="w-[200px]">
            <Field label="Samples per voice">
              <Slider
                value={sweep.n}
                onChange={(samples_per_voice) => updateParams(styleSweep.id, { ...styleSweep.params, samples_per_voice })}
                min={samples.min}
                max={samples.max}
                step={1}
              />
            </Field>
          </div>
          <div className="w-[120px]">
            <Field label="Alpha">
              <NumberInput
                value={sweep.alpha}
                onChange={(value) => updateParams(styleSweep.id, { ...styleSweep.params, alpha: value })}
                min={alpha.min}
                max={alpha.max}
                step={0.05}
              />
            </Field>
          </div>
          <div className="w-[120px]">
            <Field label="Beta">
              <NumberInput
                value={sweep.beta}
                onChange={(value) => updateParams(styleSweep.id, { ...styleSweep.params, beta: value })}
                min={beta.min}
                max={beta.max}
                step={0.05}
              />
            </Field>
          </div>
          <div className="flex-1" />
          <span className="text-xs text-txt-mute">
            {sweep.voices.length} voices × {sweep.n} = {sweep.voices.length * sweep.n} samples
          </span>
          <Button
            variant="primary"
            size="lg"
            icon="sparkles"
            disabled={pending}
            onClick={() => {
              const config = sweepConfigFromGraph(graph, spec, voices);
              if (!config.ckpt) {
                showToast("Select a checkpoint first", undefined, "error");
                return;
              }
              if (!config.voices.length) {
                showToast("Select at least one voice", undefined, "error");
                return;
              }
              if (!config.dataset) {
                showToast("Select a dataset to save to", undefined, "error");
                return;
              }
              const display: SweepDisplay[] = config.voices.flatMap((voice) =>
                Array.from({ length: config.n }, (_unused, sample) => ({ voice: voice.name, sample: sample + 1 })),
              );
              const runtimeConfig = runtimeConfigForGraph(schema, graph, schema.runtime_config_defaults);
              const payload = graphPayload(graph, null, defaultWorkflowContext(runtimeConfig));
              void runSweep(payload, display);
            }}
          >
            {pending ? "Synthesizing…" : "Generate sweep"}
          </Button>
        </div>
      </Card>

      <div className="mb-3 text-xs font-bold uppercase tracking-[0.05em] text-txt-mute">
        Results ({results.length})
      </div>
      {results.length ? (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-3.5">
          {results.map((r) => (
            <Card key={r.id} className="p-4">
              <div className="mb-3 flex items-center gap-2">
                <div className="flex h-[30px] w-[30px] items-center justify-center rounded-full bg-emerald-50">
                  <Icon name="mic" size={15} strokeWidth={2.2} className="text-emerald-600" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-bold text-txt">{r.voice}</div>
                  <div className="text-[11px] text-txt-mute">sample {r.sample}</div>
                </div>
              </div>
              {r.state === "succeeded" && r.audioFileId ? (
                <WaveformPlayer
                  seed={r.id.length}
                  duration={r.duration ?? 0}
                  fileName={r.name ?? "synthesis.wav"}
                  src={backendResourceUrl(`/audio-files/${encodeURIComponent(r.audioFileId)}/content`)}
                />
              ) : r.state === "failed" ? (
                <div className="text-xs font-semibold text-red-600">{r.error ?? "Failed"}</div>
              ) : (
                <div className="flex items-center gap-2 text-xs text-txt-mute">
                  <Icon name="loader" size={13} className="animate-spin text-blue-500" /> Running…
                </div>
              )}
            </Card>
          ))}
        </div>
      ) : (
        <Card className="rounded-[10px] border-dashed border-line-2 p-12 text-center text-sm text-txt-mute">
          Select voices and generate to compare the same line across speakers.
        </Card>
      )}
    </div>
  );
}

function selectedVoiceIds(value: unknown): string[] {
  if (!Array.isArray(value)) throw new Error("Testing sweep voices must be an array");
  return value.map((item) => String(item));
}
