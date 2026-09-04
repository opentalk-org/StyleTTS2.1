import {
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import { Maximize2, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import "@xyflow/react/dist/style.css";

import type { ModelComponent, Run } from "@/shared/types";
import { IconButton } from "@/shared/ui";
import { MonitorInspector } from "./MonitorInspector";
import { useArrayMetricNames, useModelGraph } from "./query";

export function ModelMonitor({ run }: { run: Run }) {
  const graphQuery = useModelGraph(run.id, run.status === "running");
  const namesQuery = useArrayMetricNames(run.id, run.status === "running");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const components = graphQuery.data ?? [];

  useEffect(() => {
    if (components.length === 0 || expanded.size > 0) return;
    setExpanded(new Set(components.filter((item) => item.parent_id === null).map((item) => item.id)));
  }, [components, expanded.size]);

  const visible = useMemo(
    () => visibleComponents(components, expanded, search),
    [components, expanded, search],
  );
  const nodes = useMemo(() => graphNodes(visible, selectedId), [visible, selectedId]);
  const edges = useMemo(() => graphEdges(visible), [visible]);
  const selected = components.find((item) => item.id === selectedId);

  return (
    <section className="relative m-3 flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-line bg-inset">
      <div className="z-10 flex h-12 flex-none items-center justify-between gap-3 border-b border-line bg-elevated px-3">
        <label className="flex h-8 items-center gap-2 rounded-md border border-line bg-inset px-2.5">
          <Search size={14} className="text-fg-muted" />
          <input
            className="w-52 border-0 bg-transparent text-xs text-fg outline-none"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="module"
          />
        </label>
        <IconButton label="Fullscreen" onClick={() => document.documentElement.requestFullscreen()}>
          <Maximize2 size={15} />
        </IconButton>
      </div>
      <div className="relative min-h-0 flex-1">
        <div className="absolute inset-0">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            colorMode="dark"
            proOptions={{ hideAttribution: true }}
            minZoom={0.08}
            maxZoom={2}
            onNodeClick={(_, node) => setSelectedId(node.id)}
            onNodeDoubleClick={(_, node) => setExpanded(toggle(expanded, node.id))}
          >
            <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color="#363942" />
            <Controls position="bottom-left" />
            <MiniMap
              position="bottom-right"
              nodeColor={(node) => node.id === selectedId ? "#2563eb" : "#d4d4d8"}
              maskColor="rgba(5,6,8,.72)"
            />
          </ReactFlow>
        </div>
      </div>
      {selected === undefined ? null : (
        <MonitorInspector
          runId={run.id}
          component={selected}
          names={namesQuery.data ?? []}
          running={run.status === "running"}
          onClose={() => setSelectedId(null)}
        />
      )}
    </section>
  );
}

function visibleComponents(
  components: ModelComponent[],
  expanded: Set<string>,
  search: string,
) {
  if (search.length > 0) {
    const needle = search.toLowerCase();
    return components.filter((item) =>
      item.id.toLowerCase().includes(needle) || item.module_type.toLowerCase().includes(needle),
    );
  }
  const byId = new Map(components.map((item) => [item.id, item]));
  return components.filter((item) => ancestorsExpanded(item, byId, expanded));
}

function ancestorsExpanded(
  component: ModelComponent,
  byId: Map<string, ModelComponent>,
  expanded: Set<string>,
): boolean {
  if (component.parent_id === null) return true;
  const parent = byId.get(component.parent_id);
  return parent !== undefined && expanded.has(parent.id) && ancestorsExpanded(parent, byId, expanded);
}

function graphNodes(components: ModelComponent[], selectedId: string | null): Node[] {
  const rows = new Map<number, number>();
  return components.map((component) => {
    const depth = component.id.split(".").length - 1;
    const row = rows.get(depth) ?? 0;
    rows.set(depth, row + 1);
    return {
      id: component.id,
      position: { x: 340 * depth + 80, y: 125 * row + 80 },
      data: {
        label: <div className="flex flex-col gap-1 text-left">
          <span className="font-mono text-xs opacity-60">{component.id}</span>
          <span className="text-base font-semibold text-white">{component.module_type}</span>
        </div>,
      },
      selected: component.id === selectedId,
      style: {
        width: 270,
        padding: "14px 18px",
        color: "#fff",
        background: component.id === selectedId ? "#1357dc" : "#111318",
        border: `1px solid ${component.id === selectedId ? "#3b82f6" : "#d4d4d8"}`,
        borderRadius: 9,
        boxShadow: component.id === selectedId ? "0 0 0 1px #2563eb" : "none",
      },
    };
  });
}

function graphEdges(components: ModelComponent[]): Edge[] {
  const ids = new Set(components.map((item) => item.id));
  return components.flatMap((component) =>
    component.parent_id !== null && ids.has(component.parent_id) ? [{
      id: `${component.parent_id}-${component.id}`,
      source: component.parent_id,
      target: component.id,
      type: "smoothstep",
      markerEnd: { type: MarkerType.ArrowClosed, color: "#b8bbc4" },
      style: { stroke: "#b8bbc4", strokeWidth: 1.5 },
    }] : [],
  );
}

function toggle(values: Set<string>, value: string) {
  const next = new Set(values);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}
