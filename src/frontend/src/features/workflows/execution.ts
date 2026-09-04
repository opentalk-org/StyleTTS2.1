import type { SchemaValues } from "@/shared/schema-form/types";
import type { WorkflowDefinition, WorkflowGraph, WorkflowNode, WorkflowPayload, WorkflowRunContext, WorkflowSchema } from "./types";

export function graphPayload(graph: WorkflowGraph, runId: string | null, context: WorkflowRunContext, runnerId: string | null = null): WorkflowPayload {
  return {
    run_id: runId,
    runner_id: runnerId,
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      type: node.type,
      x: node.x,
      y: node.y,
      params: node.params,
      runtime: node.runtime,
    })),
    edges: graph.edges,
    context,
  };
}

export function defaultWorkflowContext(config: SchemaValues): WorkflowRunContext {
  return { work_dir: "work", cache_dir: "cache", output_dir: "outputs", device: "cuda", config, input_items: [] };
}

export function workflowDefinition(graph: WorkflowGraph, config: SchemaValues): WorkflowDefinition {
  const payload = graphPayload(graph, null, defaultWorkflowContext(config));
  return { nodes: payload.nodes, edges: payload.edges, panels: graph.panels ?? [], context: payload.context, launch_source: null };
}

export function runtimeConfigForGraph(schema: WorkflowSchema, graph: WorkflowGraph, config: SchemaValues): SchemaValues {
  return { ...config, resources: resourceLimitsForGraph(schema, graph, numberRecord(config["resources"], "runtime resources")) };
}

export function resourceLimitsForGraph(schema: WorkflowSchema, graph: WorkflowGraph, currentLimits: Record<string, number>): Record<string, number> {
  const requirements = resourceRequirementsForGraph(schema, graph);
  return Object.fromEntries(
    Object.entries(requirements)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, amount]) => [key, currentLimits[key] ?? amount]),
  );
}

export function resourceRequirementsForGraph(schema: WorkflowSchema, graph: WorkflowGraph): Record<string, number> {
  const requirements: Record<string, number> = {};
  for (const node of graph.nodes) {
    const resources = nodeResourceRequirements(schema, node);
    for (const [key, amount] of Object.entries(resources)) requirements[key] = Math.max(requirements[key] ?? 0, amount);
  }
  return requirements;
}

export function nodeResourceRequirements(schema: WorkflowSchema, node: WorkflowNode): Record<string, number> {
  const info = schema.nodes[node.type];
  if (!info) throw new Error(`Unknown node type: ${node.type}`);
  const runtime = mergeRuntimeDefaults(info.runtime_defaults, node.runtime);
  const policy = recordValue(runtime["resource_policy"], `${node.id} resource policy`);
  return numberRecord(policy["resources"], `${node.id} resource requirements`);
}

function mergeRuntimeDefaults(defaults: SchemaValues, runtime: SchemaValues): SchemaValues {
  const merged = { ...defaults };
  for (const [key, value] of Object.entries(runtime)) {
    const current = merged[key];
    merged[key] = isRecord(current) && isRecord(value) ? { ...current, ...value } : value;
  }
  return merged;
}

export function numberRecord(value: unknown, label: string): Record<string, number> {
  const record = recordValue(value, label);
  for (const [key, item] of Object.entries(record)) {
    if (typeof item !== "number" || Number.isNaN(item)) throw new Error(`${label} contains a non-number value: ${key}`);
  }
  return record as Record<string, number>;
}

function recordValue(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as Record<string, unknown>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
