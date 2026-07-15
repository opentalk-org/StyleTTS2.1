import type { JsonSchema, SchemaValues } from "@/shared/schema-form/types";

export type WorkflowPort = {
  name: string;
  type: string;
  mode: string;
  join_mode: "item" | "broadcast";
  optional: boolean;
  default: unknown;
  description: string;
};

export type WorkflowNodeSchema = {
  type: string;
  category: string;
  description: string;
  is_input: boolean;
  inputs: Record<string, WorkflowPort>;
  outputs: Record<string, WorkflowPort>;
  settings: JsonSchema;
  settings_defaults: SchemaValues;
  runtime: JsonSchema;
  runtime_defaults: SchemaValues;
};

export type WorkflowTypeSchema = {
  name: string;
  description: string;
  color: string;
};

export type WorkflowSchema = {
  types: Record<string, WorkflowTypeSchema>;
  nodes: Record<string, WorkflowNodeSchema>;
  runtime_config: JsonSchema;
  runtime_config_defaults: SchemaValues;
};

export type WorkflowNode = {
  id: string;
  type: string;
  x: number;
  y: number;
  params: SchemaValues;
  runtime: SchemaValues;
};

export type WorkflowEdge = {
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;
};

export type ControlTarget = {
  node_id: string;
  key: string;
};

export type WorkflowControl = {
  id: string;
  label: string;
  description?: string;
  targets: ControlTarget[];
};

export type ControlPanel = {
  id: string;
  x: number;
  y: number;
  title: string;
  controls: WorkflowControl[];
};

export type WorkflowGraph = {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  panels?: ControlPanel[];
};

export type WorkflowRunContext = {
  work_dir: string;
  cache_dir: string;
  output_dir: string;
  device: string;
  config: SchemaValues;
  input_items: unknown[];
};

export type WorkflowLaunchSource =
  | { kind: "selected_audio"; audio_file_ids: string[] }
  | { kind: "dataset_audio"; dataset_id: string; include_virtual: boolean }
  | { kind: "all_audio"; include_virtual: boolean };

export type WorkflowPayload = WorkflowGraph & {
  run_id: string | null;
  runner_id: string | null;
  context: WorkflowRunContext;
};

export type WorkflowDefinition = WorkflowGraph & {
  context: WorkflowRunContext;
  launch_source: WorkflowLaunchSource | null;
};

export type SavedWorkflow = {
  id: string;
  name: string;
  data: WorkflowDefinition;
  hidden: boolean;
};

export type SaveWorkflowPayload = {
  name: string;
  data: WorkflowDefinition;
  hidden: boolean;
};

export type Viewport = {
  x: number;
  y: number;
  zoom: number;
};

export type WireDraft = {
  node: string;
  port: string;
  kind: "input" | "output";
  x: number;
  y: number;
} | null;

export type PortAnchorKey = `${string}:${string}:${"input" | "output"}`;

export type PortAnchor = {
  x: number;
  y: number;
};

export type RunStatus = {
  run_id: string;
  state: string;
  workflow_path: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  event_count: number;
};

export type RunErrorEvent = {
  kind: string;
  created_at: string;
  message: string;
  node_id: string | null;
  detail: Record<string, unknown>;
};

export type RunnerStatus = {
  total_runs: number;
  active_runs: number;
  runs: RunStatus[];
};

export type NodeRunSnapshot = {
  node_id: string;
  status: string;
  loaded: boolean;
  queue_size: number;
  remaining_items: number | null;
  running_batches: number;
  processing_items: number;
  latest_batch_index: number | null;
  latest_message: string;
  error: string | null;
  counters: Record<string, number>;
  performance: NodePerformanceSnapshot;
};

export type BatchPerformanceSnapshot = {
  batch_index: number;
  input_items: number;
  output_items: number;
  queue_wait_ms: number;
  resource_wait_ms: number;
  load_ms: number;
  execute_ms: number;
  unload_ms: number;
  route_ms: number;
  total_ms: number;
};

export type NodePerformanceSnapshot = {
  batches: number;
  arrived_items: number;
  departed_items: number;
  arrival_rate: number;
  departure_rate: number;
  queue_size: number;
  queue_capacity: number;
  queue_fill_ratio: number;
  queue_growth_rate: number;
  busy_ratio: number;
  resource_wait_ratio: number;
  resource_wait_ms: number | null;
  downstream_blocked_ms: number;
  batch_p50_ms: number;
  batch_p95_ms: number;
  service_capacity: number;
  current_batch_started_at: string | null;
  recent_batches: BatchPerformanceSnapshot[];
};

export type RatePoint = {
  timestamp: string;
  count: number;
  rate: number;
};

export type GraphPerformanceSnapshot = {
  started_items: number;
  completed_items: number;
  inflight_items: number;
  abandoned_items: number;
  rolling_throughput: number;
  average_throughput: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  history: RatePoint[];
};

export type EdgePerformanceSnapshot = {
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;
  delivered_items: number;
  rolling_rate: number;
  enqueue_blocked_ms: number;
  join_waiting_items: number;
};

export type RunSnapshot = {
  run_id: string;
  total_event_count: number;
  error_count: number;
  event_counts: Record<string, number>;
  performance: GraphPerformanceSnapshot;
  nodes: NodeRunSnapshot[];
  edges: EdgePerformanceSnapshot[];
};

export type WorkflowSocketMessage =
  | { type: "runner_status"; status: RunnerStatus }
  | { type: "run_status"; status: RunStatus }
  | { type: "run_snapshot"; run_id: string; status: RunStatus; snapshot: RunSnapshot };
