import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  useReactFlow,
  ReactFlowProvider,
  type NodeProps,
} from "@xyflow/react";
import { ChevronsDownUp, ChevronsUpDown, Maximize2, Scan } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import "@xyflow/react/dist/style.css";

import { useViewerStore } from "@/features/viewer/store";
import type { ChartTheme } from "@/shared/chart";
import type { Run } from "@/shared/types";
import { cn, IconButton, SearchInput, Toolbar } from "@/shared/ui";

import {
  formatParameterCount,
  graphEdges,
  graphNodes,
  NODE_HEIGHT,
  NODE_WIDTH,
  toggleSet,
  visibleComponents,
  type GraphNodeData,
} from "./graph";
import { MonitorInspector } from "./MonitorInspector";
import { useArrayMetricNames, useModelGraph } from "./query";

const NODE_TYPES = { module: ModuleNode };

export function ModelMonitor({ run, chart }: { run: Run; chart: ChartTheme }) {
  return (
    <ReactFlowProvider>
      <ModelMonitorInner run={run} chart={chart} />
    </ReactFlowProvider>
  );
}

function ModelMonitorInner({ run, chart }: { run: Run; chart: ChartTheme }) {
  const running = run.status === "running";
  const graphQuery = useModelGraph(run.id, running);
  const namesQuery = useArrayMetricNames(run.id, running);
  const { fitView } = useReactFlow();
  const theme = useViewerStore((state) => state.theme);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const components = graphQuery.data ?? [];

  useEffect(() => {
    if (components.length === 0 || expanded.size > 0) return;
    setExpanded(new Set(components.filter((item) => item.parent_id === null).map((item) => item.id)));
  }, [components, expanded.size]);

  const visible = useMemo(() => visibleComponents(components, expanded, search), [components, expanded, search]);
  const nodes = useMemo(() => graphNodes(visible, components, expanded, selectedId), [visible, components, expanded, selectedId]);
  const edges = useMemo(() => graphEdges(visible), [visible]);
  const selected = components.find((item) => item.id === selectedId);

  return (
    <div className="flex min-h-0 flex-1 flex-row">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <Toolbar
          start={
            <SearchInput label="Search modules" value={search} onValue={setSearch} placeholder="Search modules by name or type" className="w-72" />
          }
          end={
            <>
              <span className="font-mono text-xs tabular-nums text-fg-muted">{visible.length} of {components.length} modules</span>
              <IconButton label="Expand all" onClick={() => setExpanded(new Set(components.map((item) => item.id)))}>
                <ChevronsUpDown size={14} />
              </IconButton>
              <IconButton label="Collapse all" onClick={() => setExpanded(new Set(components.filter((item) => item.parent_id === null).map((item) => item.id)))}>
                <ChevronsDownUp size={14} />
              </IconButton>
              <IconButton label="Fit to view" onClick={() => void fitView({ duration: 200 })}>
                <Scan size={14} />
              </IconButton>
              <IconButton label="Fullscreen" onClick={() => void document.documentElement.requestFullscreen()}>
                <Maximize2 size={14} />
              </IconButton>
            </>
          }
        />
        <div className="relative min-h-0 flex-1 bg-canvas">
          <div className="absolute inset-0">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={NODE_TYPES}
              fitView
              colorMode={theme}
              proOptions={{ hideAttribution: true }}
              minZoom={0.08}
              maxZoom={2}
              onNodeClick={(_, node) => setSelectedId(node.id)}
              onNodeDoubleClick={(_, node) => setExpanded(toggleSet(expanded, node.id))}
            >
              <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color={chart.grid} />
              <Controls position="bottom-left" showInteractive={false} />
              <MiniMap
                position="bottom-right"
                pannable
                zoomable
                nodeColor={(node) => (node.id === selectedId ? chart.series[0] : chart.hoverBorder)}
                maskColor={theme === "dark" ? "rgb(0 0 0 / 0.5)" : "rgb(255 255 255 / 0.6)"}
                style={{ background: chart.hoverBg, border: `1px solid ${chart.grid}`, borderRadius: 6 }}
              />
            </ReactFlow>
          </div>
        </div>
      </div>
      {selected === undefined ? null : (
        <MonitorInspector runId={run.id} component={selected} names={namesQuery.data ?? []} running={running} chart={chart} onClose={() => setSelectedId(null)} />
      )}
    </div>
  );
}

function ModuleNode({ data, selected }: NodeProps & { data: GraphNodeData }) {
  return (
    <div
      style={{ width: NODE_WIDTH, height: NODE_HEIGHT }}
      className={cn(
        "flex flex-col justify-center rounded-md border bg-surface px-3 text-left shadow-none",
        selected ? "border-accent outline-2 outline-accent/40" : "border-strong",
      )}
    >
      <Handle type="target" position={Position.Left} className="!size-1.5 !border-0 !bg-strong" />
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-[13px] font-semibold text-fg">{data.moduleType}</span>
        <span className="shrink-0 font-mono text-[11px] text-fg-muted">{formatParameterCount(data.parameterCount)}</span>
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="truncate font-mono text-[11px] text-fg-muted">{data.id}</span>
        {data.hasChildren && !data.expanded ? (
          <span className="shrink-0 text-[11px] text-fg-muted">double-click to expand</span>
        ) : null}
      </div>
      <Handle type="source" position={Position.Right} className="!size-1.5 !border-0 !bg-strong" />
    </div>
  );
}
