import { Icon } from "@/shared/icons";
import { Card } from "@/shared/ui/Card";
import { defaultWorkflowContext, graphPayload, runtimeConfigForGraph } from "../workflows/logic";
import { useStartGraphMutation } from "../workflows/query";
import type { WorkflowGraph, WorkflowNode, WorkflowSchema } from "../workflows/types";

const BASE_SYMBOLS = 178;

type ValidationRow = { ok: boolean; label: string };

export function QueueCard({
  schema,
  graph,
}: {
  schema: WorkflowSchema | null;
  graph: WorkflowGraph | null;
}) {
  const start = useStartGraphMutation();
  const training = graph?.nodes.find((node) => node.type.endsWith("Training") || node.type === "StyleTtsFinetune");
  const dataset = graph?.nodes.find((node) => node.type === "SelectTrainingDataset");
  const checkpoint = graph?.nodes.find((node) => node.type === "SelectCheckpoint");
  const alphabet = graph?.nodes.find((node) => node.type === "PhonemeAlphabet");
  const assets = graph?.nodes.find((node) => node.type === "SelectTrainingAssets");
  const alphabetSymbols = String(alphabet?.params.symbols ?? "");
  const alphabetValid = !alphabet || alphabetSymbols.trim().split(/\s+/).filter(Boolean).length === BASE_SYMBOLS;

  const rows: ValidationRow[] = [
    { ok: Boolean(training?.params.display_name), label: "Display name set" },
    { ok: Boolean(dataset?.params.dataset_id), label: "Training dataset selected" },
    { ok: Boolean(checkpoint?.params.checkpoint_id), label: "Base checkpoint required" },
    { ok: alphabetValid, label: "Phoneme alphabet valid" },
    { ok: hasOodSet(assets), label: "OOD reference set added" },
  ];

  const queue = () => {
    if (!schema || !graph) return;
    const config = runtimeConfigForGraph(schema, graph, schema.runtime_config_defaults);
    start.mutate(graphPayload(graph, null, defaultWorkflowContext(config)));
  };

  return (
    <Card className="p-[18px]">
      <div className="mb-3.5 flex items-center justify-between">
        <div className="text-sm font-bold text-txt">Queue training</div>
        <span className="flex items-center gap-1.5 text-[11px] font-semibold text-emerald-700">
          <Icon name="check" size={13} strokeWidth={2.5} className="text-emerald-600" />
          Draft saved
        </span>
      </div>

      <div className="mb-4 flex flex-col gap-2">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-2 text-[12.5px]">
            <Icon
              name={r.ok ? "check-circle" : "alert"}
              size={16}
              strokeWidth={2.2}
              className={r.ok ? "text-emerald-600" : "text-amber-600"}
            />
            <span
              className={
                r.ok ? "font-medium text-txt-dim" : "font-semibold text-amber-700"
              }
            >
              {r.label}
            </span>
          </div>
        ))}
      </div>

      <button
        onClick={queue}
        disabled={!schema || !graph || start.isPending}
        className="flex h-11 w-full items-center justify-center gap-2 rounded-lg border-0 bg-blue-500 text-sm font-semibold text-white cursor-pointer transition-colors hover:bg-blue-600 disabled:cursor-default disabled:opacity-50"
      >
        <Icon name="bolt" size={17} strokeWidth={2.2} />
        Queue training job
      </button>

      <p className="mt-2.5 text-[11px] leading-relaxed text-txt-mute">
        Drafts autosave to this browser. Queuing returns a job id.
      </p>
    </Card>
  );
}

function hasOodSet(node: WorkflowNode | undefined): boolean {
  const ids = node?.params.ood_text_set_file_ids;
  return Array.isArray(ids) && ids.length > 0;
}
