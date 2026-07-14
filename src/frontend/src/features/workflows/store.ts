import { create } from "zustand";

import type { SchemaValues } from "@/shared/schema-form/types";
import { autoLayoutGraph, connect, deleteNodes, moveNodes, renameNode, runtimeConfigForGraph, zoomViewport } from "./logic";
import type { ControlTarget, PortAnchor, PortAnchorKey, RunSnapshot, RunStatus, Viewport, WireDraft, WorkflowControl, WorkflowEdge, WorkflowGraph, WorkflowNode, WorkflowSchema } from "./types";

function shortId(prefix: string): string {
  return `${prefix}_${crypto.randomUUID().slice(0, 8)}`;
}

type WorkflowStore = {
  schema: WorkflowSchema | null;
  graph: WorkflowGraph;
  selectedNodeIds: string[];
  viewport: Viewport;
  wireDraft: WireDraft;
  runtimeConfig: SchemaValues;
  activeRunId: string | null;
  inspectorOpen: boolean;
  inspectorTab: "settings" | "runtime" | "logs";
  runs: RunStatus[];
  snapshots: Record<string, RunSnapshot>;
  portAnchors: Record<PortAnchorKey, PortAnchor>;
  setSchema: (schema: WorkflowSchema) => void;
  setGraph: (graph: WorkflowGraph) => void;
  patchNode: (nodeId: string, patch: Partial<WorkflowNode>) => void;
  selectNode: (nodeId: string | null, additive?: boolean) => void;
  selectNodes: (nodeIds: string[]) => void;
  deleteSelection: () => void;
  autoLayout: () => void;
  moveSelection: (dx: number, dy: number) => void;
  renameSelectedNode: (nextId: string) => void;
  addEdge: (edge: WorkflowEdge) => void;
  startWire: (nodeId: string, portName: string, kind: "input" | "output", x: number, y: number) => void;
  finishWire: (nodeId: string | null, portName: string | null, kind: "input" | "output" | null) => void;
  setViewport: (viewport: Viewport) => void;
  panViewport: (dx: number, dy: number) => void;
  zoomAt: (nextZoom: number, anchorX: number, anchorY: number) => void;
  setWireDraft: (wireDraft: WireDraft) => void;
  setRuntimeConfig: (runtimeConfig: SchemaValues) => void;
  setActiveRunId: (activeRunId: string | null) => void;
  openInspector: (tab?: "settings" | "runtime" | "logs") => void;
  closeInspector: () => void;
  setInspectorTab: (tab: "settings" | "runtime" | "logs") => void;
  applyRunnerStatus: (runs: RunStatus[]) => void;
  applyRunStatus: (run: RunStatus) => void;
  applyRunSnapshot: (runId: string, snapshot: RunSnapshot) => void;
  setPortAnchor: (key: PortAnchorKey, anchor: PortAnchor) => void;
  addPanel: (x: number, y: number) => void;
  movePanel: (panelId: string, dx: number, dy: number) => void;
  deletePanel: (panelId: string) => void;
  renamePanel: (panelId: string, title: string) => void;
  addControl: (panelId: string, target: ControlTarget, label: string) => void;
  updateControl: (panelId: string, controlId: string, patch: Partial<Pick<WorkflowControl, "label" | "description">>) => void;
  removeControl: (panelId: string, controlId: string) => void;
  exposeSetting: (panelId: string | null, target: ControlTarget, label: string, position: { x: number; y: number }) => void;
};

function mapPanels(graph: WorkflowGraph, fn: (panels: NonNullable<WorkflowGraph["panels"]>) => NonNullable<WorkflowGraph["panels"]>): WorkflowGraph {
  return { ...graph, panels: fn(graph.panels ?? []) };
}

export const useWorkflowStore = create<WorkflowStore>((set) => ({
  schema: null,
  graph: { nodes: [], edges: [] },
  selectedNodeIds: [],
  viewport: { x: 0, y: 0, zoom: 1 },
  wireDraft: null,
  runtimeConfig: {},
  activeRunId: null,
  inspectorOpen: false,
  inspectorTab: "settings",
  runs: [],
  snapshots: {},
  portAnchors: {},
  setSchema: (schema) => set((state) => {
    const defaults = structuredClone(schema.runtime_config_defaults) as SchemaValues;
    return { schema, runtimeConfig: runtimeConfigForGraph(schema, state.graph, defaults) };
  }),
  setGraph: (graph) => set((state) => ({
    graph,
    selectedNodeIds: [],
    wireDraft: null,
    portAnchors: {},
    runtimeConfig: state.schema ? runtimeConfigForGraph(state.schema, graph, state.runtimeConfig) : state.runtimeConfig,
  })),
  patchNode: (nodeId, patch) => set((state) => ({
    graph: { ...state.graph, nodes: state.graph.nodes.map((node) => (node.id === nodeId ? { ...node, ...patch } : node)) },
  })),
  selectNode: (nodeId, additive = false) => set((state) => {
    if (nodeId === null) return { selectedNodeIds: [] };
    if (!additive) return { selectedNodeIds: [nodeId] };
    const selected = new Set(state.selectedNodeIds);
    if (selected.has(nodeId)) selected.delete(nodeId);
    else selected.add(nodeId);
    return { selectedNodeIds: [...selected] };
  }),
  selectNodes: (selectedNodeIds) => set({ selectedNodeIds }),
  deleteSelection: () => set((state) => {
    const graph = deleteNodes(state.graph, state.selectedNodeIds);
    return {
      graph,
      selectedNodeIds: [],
      runtimeConfig: state.schema ? runtimeConfigForGraph(state.schema, graph, state.runtimeConfig) : state.runtimeConfig,
    };
  }),
  autoLayout: () => set((state) => {
    if (!state.schema) return {};
    const nodeWidths = new Map<string, number>();
    for (const element of document.querySelectorAll<HTMLElement>("[data-workflow-node]")) {
      const nodeId = element.dataset.workflowNode;
      if (!nodeId) throw new Error("Rendered workflow node is missing its identifier");
      nodeWidths.set(nodeId, element.offsetWidth);
    }
    return {
      graph: autoLayoutGraph(state.schema, state.graph, nodeWidths),
      portAnchors: {},
      viewport: { x: 0, y: 0, zoom: 1 },
      wireDraft: null,
    };
  }),
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
  startWire: (nodeId, portName, kind, x, y) => set((state) => {
    if (kind === "input") {
      const edge = state.graph.edges.find((item) => item.target_node === nodeId && item.target_port === portName);
      if (edge) {
        return {
          graph: {
            ...state.graph,
            edges: state.graph.edges.filter((item) => item !== edge),
          },
          wireDraft: { node: edge.source_node, port: edge.source_port, kind: "output", x, y },
        };
      }
    }
    return { wireDraft: { node: nodeId, port: portName, kind, x, y } };
  }),
  finishWire: (nodeId, portName, kind) => set((state) => {
    const wire = state.wireDraft;
    if (!wire || !state.schema || !nodeId || !portName || !kind) return { wireDraft: null };
    if (wire.kind === "output" && kind === "input") {
      const graph = connect(state.schema, state.graph, { source_node: wire.node, source_port: wire.port, target_node: nodeId, target_port: portName });
      return { graph, wireDraft: null };
    }
    if (wire.kind === "input" && kind === "output") {
      const graph = connect(state.schema, state.graph, { source_node: nodeId, source_port: portName, target_node: wire.node, target_port: wire.port });
      return { graph, wireDraft: null };
    }
    return { wireDraft: null };
  }),
  setViewport: (viewport) => set({ viewport }),
  panViewport: (dx, dy) => set((state) => ({ viewport: { ...state.viewport, x: state.viewport.x + dx, y: state.viewport.y + dy } })),
  zoomAt: (nextZoom, anchorX, anchorY) => set((state) => ({ viewport: zoomViewport(state.viewport, nextZoom, anchorX, anchorY) })),
  setWireDraft: (wireDraft) => set({ wireDraft }),
  setRuntimeConfig: (runtimeConfig) => set({ runtimeConfig }),
  setActiveRunId: (activeRunId) => set({ activeRunId }),
  openInspector: (inspectorTab = "settings") => set({ inspectorOpen: true, inspectorTab }),
  closeInspector: () => set({ inspectorOpen: false }),
  setInspectorTab: (inspectorTab) => set({ inspectorTab }),
  applyRunnerStatus: (runs) => set({ runs }),
  applyRunStatus: (run) => set((state) => {
    const others = state.runs.filter((item) => item.run_id !== run.run_id);
    return { runs: [run, ...others] };
  }),
  applyRunSnapshot: (runId, snapshot) => set((state) => ({ snapshots: { ...state.snapshots, [runId]: snapshot } })),
  setPortAnchor: (key, anchor) => set((state) => {
    const current = state.portAnchors[key];
    if (current && Math.abs(current.x - anchor.x) < 0.5 && Math.abs(current.y - anchor.y) < 0.5) return {};
    return { portAnchors: { ...state.portAnchors, [key]: anchor } };
  }),
  addPanel: (x, y) => set((state) => ({
    graph: mapPanels(state.graph, (panels) => [...panels, { id: shortId("panel"), x, y, title: "Controls", controls: [] }]),
  })),
  movePanel: (panelId, dx, dy) => set((state) => ({
    graph: mapPanels(state.graph, (panels) => panels.map((panel) => (panel.id === panelId ? { ...panel, x: panel.x + dx, y: panel.y + dy } : panel))),
  })),
  deletePanel: (panelId) => set((state) => ({
    graph: mapPanels(state.graph, (panels) => panels.filter((panel) => panel.id !== panelId)),
  })),
  renamePanel: (panelId, title) => set((state) => ({
    graph: mapPanels(state.graph, (panels) => panels.map((panel) => (panel.id === panelId ? { ...panel, title } : panel))),
  })),
  addControl: (panelId, target, label) => set((state) => ({
    graph: mapPanels(state.graph, (panels) =>
      panels.map((panel) => {
        if (panel.id !== panelId) return panel;
        if (panel.controls.some((control) => control.targets.some((item) => item.node_id === target.node_id && item.key === target.key))) return panel;
        return { ...panel, controls: [...panel.controls, { id: shortId("control"), label, description: "", targets: [target] }] };
      }),
    ),
  })),
  updateControl: (panelId, controlId, patch) => set((state) => ({
    graph: mapPanels(state.graph, (panels) =>
      panels.map((panel) => (
        panel.id === panelId
          ? { ...panel, controls: panel.controls.map((control) => (control.id === controlId ? { ...control, ...patch } : control)) }
          : panel
      )),
    ),
  })),
  removeControl: (panelId, controlId) => set((state) => ({
    graph: mapPanels(state.graph, (panels) =>
      panels.map((panel) => (panel.id === panelId ? { ...panel, controls: panel.controls.filter((control) => control.id !== controlId) } : panel)),
    ),
  })),
  exposeSetting: (panelId, target, label, position) => set((state) => {
    const control = { id: shortId("control"), label, description: "", targets: [target] };
    const bound = (panel: { controls: { targets: ControlTarget[] }[] }) =>
      panel.controls.some((item) => item.targets.some((entry) => entry.node_id === target.node_id && entry.key === target.key));
    if (panelId) {
      return {
        graph: mapPanels(state.graph, (panels) => panels.map((panel) => (panel.id === panelId && !bound(panel) ? { ...panel, controls: [...panel.controls, control] } : panel))),
      };
    }
    return { graph: mapPanels(state.graph, (panels) => [...panels, { id: shortId("panel"), x: position.x, y: position.y, title: "Controls", controls: [control] }]) };
  }),
}));
