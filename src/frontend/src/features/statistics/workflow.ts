import { fetchRun, startGraph } from "../workflows/api";
import { defaultWorkflowContext, graphPayload, runtimeConfigForGraph } from "../workflows/execution";
import type { WorkflowEdge, WorkflowGraph, WorkflowSchema } from "../workflows/types";

export type StatisticsMode = "database" | "acoustic";

type NodeTemplate = { id: string; type: string; x: number; y: number };

const COMMON_NODES: NodeTemplate[] = [
  { id: "source", type: "AudioSource", x: 0, y: 200 },
  { id: "segments", type: "LoadAudioSegments", x: 480, y: 200 },
  { id: "aggregate", type: "AggregateDatasetStatistics", x: 960, y: 200 },
  { id: "save", type: "SaveStatisticsEntry", x: 1200, y: 200 },
];

const FINAL_EDGES: WorkflowEdge[] = [
  { source_node: "features", source_port: "feature_records", target_node: "aggregate", target_port: "feature_records" },
  { source_node: "aggregate", source_port: "statistics", target_node: "save", target_port: "statistics" },
];

function buildStatisticsGraph(schema: WorkflowSchema, datasetId: string, name: string, mode: StatisticsMode, sampleCount: number | null): WorkflowGraph {
  const modeNodes: NodeTemplate[] = mode === "acoustic"
    ? [{ id: "load", type: "LoadAudio", x: 240, y: 200 }, { id: "features", type: "AnalyzeAudioFeatures", x: 720, y: 200 }]
    : [{ id: "features", type: "DatabaseStatisticsFeatures", x: 720, y: 200 }];
  const nodes = [...COMMON_NODES, ...modeNodes].map((node) => {
    const info = schema.nodes[node.type];
    if (!info) throw new Error(`Statistics node is not registered: ${node.type}`);
    const params = structuredClone(info.settings_defaults);
    if (node.id === "source") {
      params.source = "dataset";
      params.dataset_id = datasetId;
      params.include_virtual = mode === "database";
      params.selection = sampleCount === null ? "all" : "random";
      if (sampleCount !== null) params.count = sampleCount;
    }
    if (node.id === "save") {
      params.name = name;
      params.dataset_id = datasetId;
    }
    return { id: node.id, type: node.type, x: node.x, y: node.y, params, runtime: structuredClone(info.runtime_defaults) };
  });
  const inputEdges: WorkflowEdge[] = mode === "acoustic"
    ? [
        { source_node: "source", source_port: "audio", target_node: "load", target_port: "audio" },
        { source_node: "load", source_port: "audio", target_node: "segments", target_port: "audio" },
        { source_node: "segments", source_port: "audio", target_node: "features", target_port: "audio" },
      ]
    : [
        { source_node: "source", source_port: "audio", target_node: "segments", target_port: "audio" },
        { source_node: "segments", source_port: "audio", target_node: "features", target_port: "audio" },
      ];
  return { nodes, edges: [...inputEdges, ...FINAL_EDGES] };
}

const TERMINAL = new Set(["succeeded", "failed", "stopped"]);

async function pollRun(runId: string): Promise<void> {
  const deadline = Date.now() + 30 * 60 * 1000;
  while (Date.now() < deadline) {
    const run = await fetchRun(runId);
    if (TERMINAL.has(run.state)) {
      if (run.state !== "succeeded") throw new Error(run.error || `Statistics run ${run.state}`);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error("Statistics run timed out");
}

export async function computeDatasetStatistics(
  schema: WorkflowSchema,
  datasetId: string,
  name: string,
  mode: StatisticsMode,
  sampleCount: number | null,
): Promise<void> {
  const graph = buildStatisticsGraph(schema, datasetId, name, mode, sampleCount);
  const runtimeConfig = runtimeConfigForGraph(schema, graph, schema.runtime_config_defaults);
  const payload = graphPayload(graph, null, defaultWorkflowContext(runtimeConfig));
  const run = await startGraph(payload);
  await pollRun(run.run_id);
}
