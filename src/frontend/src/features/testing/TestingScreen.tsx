import { useEffect } from "react";

import { showToast } from "@/shared/feedback/Toast";
import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { Tabs } from "@/shared/ui/Tabs";
import { defaultWorkflowContext, graphPayload, runtimeConfigForGraph } from "../workflows/logic";
import { useWorkflowSchemaQuery } from "../workflows/query";
import type { WorkflowGraph, WorkflowSchema } from "../workflows/types";
import { ConfigCard } from "./ConfigCard";
import { createTestingGraph, singleConfigFromGraph, TESTING_OPTIONS, TESTING_WORKFLOWS, type TestingMode, type TestingWorkflowSpec } from "./logic";
import { ResultCard } from "./ResultCard";
import { SweepPanel } from "./SweepPanel";
import { useTesting } from "./store";

function SingleMode({
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
  const results = useTesting((s) => s.testResults);
  const runSingle = useTesting((s) => s.runSingle);
  const pending = results.some((r) => r.state === "running");

  const generate = () => {
    const config = singleConfigFromGraph(graph, spec);
    if (!config.ckpt) {
      showToast("Select a checkpoint first", undefined, "error");
      return;
    }
    if (!config.styleRef) {
      showToast("Select a reference audio first", undefined, "error");
      return;
    }
    if (!config.dataset) {
      showToast("Select a dataset to save to", undefined, "error");
      return;
    }
    const runtimeConfig = runtimeConfigForGraph(schema, graph, schema.runtime_config_defaults);
    const payload = graphPayload(graph, null, defaultWorkflowContext(runtimeConfig));
    void runSingle(payload, { text: config.text, steps: config.steps, emb: config.emb });
  };

  return (
    <>
      <ConfigCard schema={schema} graph={graph} spec={spec} onChange={onChange} />
      <Button
        variant="primary"
        size="lg"
        icon="sparkles"
        disabled={pending}
        onClick={generate}
        className="mt-4 h-12 w-full"
      >
        {pending ? "Synthesizing…" : "Generate speech"}
      </Button>

      <div className="mt-[26px]">
        <div className="mb-3 text-xs font-bold uppercase tracking-[0.05em] text-txt-mute">
          Results ({results.length})
        </div>
        {results.length ? (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(380px,1fr))] gap-3.5">
            {results.map((r) => (
              <ResultCard key={r.id} result={r} />
            ))}
          </div>
        ) : (
          <Card className="border-dashed border-line-2 p-10 text-center text-txt-mute">
            <div className="mb-2.5 flex justify-center">
              <Icon name="play-circle" size={30} strokeWidth={1.8} className="text-txt-mute" />
            </div>
            <div className="text-sm font-semibold text-txt-dim">No results yet</div>
            <div className="mt-0.5 text-[12.5px]">
              Configure the model and generate to audition synthesis.
            </div>
          </Card>
        )}
      </div>
    </>
  );
}

export function TestingScreen() {
  const mode = useTesting((s) => s.testMode);
  const setMode = useTesting((s) => s.setMode);
  const graph = useTesting((s) => s.graphsByMode[mode]);
  const ensureGraph = useTesting((s) => s.ensureGraph);
  const setGraph = useTesting((s) => s.setGraph);
  const schemaQuery = useWorkflowSchemaQuery();
  const spec = TESTING_WORKFLOWS[mode];

  useEffect(() => {
    if (!schemaQuery.data) return;
    ensureGraph(mode, createTestingGraph(schemaQuery.data, spec));
  }, [schemaQuery.data, mode, spec, ensureGraph]);

  const updateGraph = (next: WorkflowGraph) => setGraph(mode, next);

  return (
    <div className="mx-auto max-w-[1100px] px-7 pb-[70px] pt-[22px]">
      <Tabs value={mode} onChange={(v) => setMode(v as TestingMode)} options={TESTING_OPTIONS} className="mb-[22px]" />
      {schemaQuery.isLoading ? <Card className="p-6 text-sm text-txt-mute">Loading node schema...</Card> : null}
      {schemaQuery.error ? <Card className="p-6 text-sm font-semibold text-amber-700">Could not load workflow schema.</Card> : null}
      {schemaQuery.data && graph ? (
        mode === "single" ? (
          <SingleMode schema={schemaQuery.data} graph={graph} spec={spec} onChange={updateGraph} />
        ) : (
          <SweepPanel schema={schemaQuery.data} graph={graph} spec={spec} onChange={updateGraph} />
        )
      ) : null}
    </div>
  );
}
