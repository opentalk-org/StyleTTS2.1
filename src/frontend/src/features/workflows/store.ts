import { create } from "zustand";

import type { SchemaValues } from "@/shared/schema-form/types";
import { connect, deleteNodes, moveNodes, renameNode, zoomViewport } from "./logic";
import type { RunSnapshot, RunStatus, Viewport, WireDraft, WorkflowEdge, WorkflowGraph, WorkflowSchema } from "./types";

type WorkflowStore = {
  schema: WorkflowSchema | null;
  graph: WorkflowGraph;
  selectedNodeIds: string[];
  viewport: Viewport;
  wireDraft: WireDraft;
  runtimeConfig: SchemaValues;
  activeRunId: string | null;
  runs: RunStatus[];
  snapshots: Record<string, RunSnapshot>;
  setSchema: (schema: WorkflowSchema) => void;
  setGraph: (graph: WorkflowGraph) => void;
  selectNode: (nodeId: string | null, additive?: boolean) => void;
  selectNodes: (nodeIds: string[]) => void;
  deleteSelection: () => void;
  moveSelection: (dx: number, dy: number) => void;
  renameSelectedNode: (nextId: string) => void;
  addEdge: (edge: WorkflowEdge) => void;
  setViewport: (viewport: Viewport) => void;
  panViewport: (dx: number, dy: number) => void;
  zoomAt: (nextZoom: number, anchorX: number, anchorY: number) => void;
  setWireDraft: (wireDraft: WireDraft) => void;
  setRuntimeConfig: (runtimeConfig: SchemaValues) => void;
  setActiveRunId: (activeRunId: string | null) => void;
  applyRunnerStatus: (runs: RunStatus[]) => void;
  applyRunStatus: (run: RunStatus) => void;
  applyRunSnapshot: (runId: string, snapshot: RunSnapshot) => void;
};

export const useWorkflowStore = create<WorkflowStore>((set) => ({
  schema: null,
  graph: { nodes: [], edges: [] },
  selectedNodeIds: [],
  viewport: { x: 0, y: 0, zoom: 1 },
  wireDraft: null,
  runtimeConfig: {},
  activeRunId: null,
  runs: [],
  snapshots: {},
  setSchema: (schema) => set({ schema, runtimeConfig: structuredClone(schema.runtime_config_defaults) as SchemaValues }),
  setGraph: (graph) => set({ graph }),
  selectNode: (nodeId, additive = false) => set((state) => {
    if (nodeId === null) return { selectedNodeIds: [] };
    if (!additive) return { selectedNodeIds: [nodeId] };
    const selected = new Set(state.selectedNodeIds);
    if (selected.has(nodeId)) selected.delete(nodeId);
    else selected.add(nodeId);
    return { selectedNodeIds: [...selected] };
  }),
  selectNodes: (selectedNodeIds) => set({ selectedNodeIds }),
  deleteSelection: () => set((state) => ({ graph: deleteNodes(state.graph, state.selectedNodeIds), selectedNodeIds: [] })),
  moveSelection: (dx, dy) => set((state) => ({ graph: moveNodes(state.graph, state.selectedNodeIds, dx, dy) })),
  renameSelectedNode: (nextId) => set((state) => {
    const previous = state.selectedNodeIds[0];
    if (!previous) return {};
    return { graph: renameNode(state.graph, previous, nextId), selectedNodeIds: [nextId] };
  }),
  addEdge: (edge) => set((state) => {
    if (!state.schema) return {};
    return { graph: connect(state.schema, state.graph, edge), wireDraft: null };
  }),
  setViewport: (viewport) => set({ viewport }),
  panViewport: (dx, dy) => set((state) => ({ viewport: { ...state.viewport, x: state.viewport.x + dx, y: state.viewport.y + dy } })),
  zoomAt: (nextZoom, anchorX, anchorY) => set((state) => ({ viewport: zoomViewport(state.viewport, nextZoom, anchorX, anchorY) })),
  setWireDraft: (wireDraft) => set({ wireDraft }),
  setRuntimeConfig: (runtimeConfig) => set({ runtimeConfig }),
  setActiveRunId: (activeRunId) => set({ activeRunId }),
  applyRunnerStatus: (runs) => set({ runs }),
  applyRunStatus: (run) => set((state) => {
    const others = state.runs.filter((item) => item.run_id !== run.run_id);
    return { runs: [run, ...others] };
  }),
  applyRunSnapshot: (runId, snapshot) => set((state) => ({ snapshots: { ...state.snapshots, [runId]: snapshot } })),
}));
