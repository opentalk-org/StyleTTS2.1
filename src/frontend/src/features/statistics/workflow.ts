import { fetchRun, startGraph } from "../workflows/api";
import { defaultWorkflowContext, graphPayload, runtimeConfigForGraph } from "../workflows/logic";
import type { WorkflowEdge, WorkflowGraph, WorkflowSchema } from "../workflows/types";

// The canonical dataset-statistics pipeline: pull a dataset's audio, hydrate its stored
// segments (carrying word-level alignment), compute per-file audio features, aggregate the
// whole dataset, and persist a statistics entry.
const NODES: { id: string; type: string; x: number; y: number }[] = [
  { id: "source", type: "AudioSource", x: 0, y: 200 },
  { id: "load", type: "LoadAudio", x: 240, y: 200 },
  { id: "segments", type: "LoadAudioSegments", x: 480, y: 200 },
  { id: "features", type: "AnalyzeAudioFeatures", x: 720, y: 200 },
  { id: "aggregate", type: "AggregateDatasetStatistics", x: 960, y: 200 },
  { id: "save", type: "SaveStatisticsEntry", x: 1200, y: 200 },
];

const EDGES: WorkflowEdge[] = [
  { source_node: "source", source_port: "audio", target_node: "load", target_port: "audio" },
  { source_node: "load", source_port: "audio", target_node: "segments", target_port: "audio" },
  { source_node: "segments", source_port: "audio", target_node: "features", target_port: "audio" },
  { source_node: "features", source_port: "feature_records", target_node: "aggregate", target_port: "feature_records" },
  { source_node: "aggregate", source_port: "statistics", target_node: "save", target_port: "statistics" },
];

function buildStatisticsGraph(schema: WorkflowSchema, datasetId: string, name: string): WorkflowGraph {
  const nodes = NODES.map((node) => {
    const info = schema.nodes[node.type];
    if (!info) throw new Error(`Statistics node is not registered: ${node.type}`);
    const params = structuredClone(info.settings_defaults);
    if (node.id === "source") {
      params.source = "dataset";
      params.dataset_id = datasetId;
    }
    if (node.id === "save") {
      params.name = name;
      params.dataset_id = datasetId;
    }
    return { id: node.id, type: node.type, x: node.x, y: node.y, params, runtime: structuredClone(info.runtime_defaults) };
  });
  return { nodes, edges: EDGES };
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

export async function computeDatasetStatistics(schema: WorkflowSchema, datasetId: string, name: string): Promise<void> {
  const graph = buildStatisticsGraph(schema, datasetId, name);
  const runtimeConfig = runtimeConfigForGraph(schema, graph, schema.runtime_config_defaults);
  const payload = graphPayload(graph, null, defaultWorkflowContext(runtimeConfig));
  const run = await startGraph(payload);
  await pollRun(run.run_id);
}
