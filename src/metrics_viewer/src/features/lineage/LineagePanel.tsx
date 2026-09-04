import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { GitBranch, Scan } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import "@xyflow/react/dist/style.css";

import { useViewerStore } from "@/features/viewer/store";
import type { ChartTheme } from "@/shared/chart";
import type { Lineage, Run } from "@/shared/types";
import { Checkbox, EmptyState, IconButton, SearchInput, Skeleton, Toolbar, cn } from "@/shared/ui";

import { CheckpointInspector } from "./CheckpointInspector";
import { LineageEdge, type LineageEdgeData } from "./LineageEdge";
import {
  ancestorChain,
  layoutLineage,
  matchesSearch,
  NODE_HEIGHT,
  NODE_WIDTH,
  type LineageLayout,
  type PlacedCheckpoint,
  type RunBox,
} from "./logic";
import { useLineageQuery } from "./query";

interface LineagePanelProps {
  allRuns: Run[];
  runColors: Record<string, string>;
  chart: ChartTheme;
}

export function LineagePanel(props: LineagePanelProps) {
  return (
    <ReactFlowProvider>
      <LineageGraph {...props} />
    </ReactFlowProvider>
  );
}

interface CheckpointNodeData extends Record<string, unknown> {
  checkpoint: PlacedCheckpoint;
  color: string;
  active: boolean;
}

interface RunBoxNodeData extends Record<string, unknown> {
  box: RunBox;
  color: string;
  active: boolean;
  onOpen: (runId: string, additive: boolean) => void;
}

const NODE_TYPES = { checkpoint: CheckpointNode, runBox: RunBoxNode };
const EDGE_TYPES = { lineage: LineageEdge };
/** Run boxes sit under the lines, the lines under the checkpoints. */
const BOX_Z = 0;
const EDGE_Z = 1;
const NODE_Z = 2;

function LineageGraph({ allRuns, runColors, chart }: LineagePanelProps) {
  const projectId = useViewerStore((state) => state.projectId);
  const selectedRunIds = useViewerStore((state) => state.selectedRunIds);
  const focusRun = useViewerStore((state) => state.focusRun);
  const theme = useViewerStore((state) => state.theme);
  const { fitView } = useReactFlow();
  const [search, setSearch] = useState("");
  const [onlySelected, setOnlySelected] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const lineageQuery = useLineageQuery(projectId, true);
  const lineage: Lineage | undefined = lineageQuery.data;

  const runNames = useMemo(() => {
    const names = new Map(allRuns.map((run) => [run.id, run.name]));
    for (const run of lineage?.runs ?? []) if (!names.has(run.id)) names.set(run.id, run.name);
    return names;
  }, [allRuns, lineage]);

  const layout: LineageLayout | null = useMemo(
    () => (lineage === undefined ? null : layoutLineage(lineage, runNames)),
    [lineage, runNames],
  );

  const selection = useMemo(() => new Set(selectedRunIds), [selectedRunIds]);
  const colorOf = useMemo(() => {
    const palette = chart.series;
    return (box: RunBox) => runColors[box.runId] ?? palette[box.lane % palette.length];
  }, [chart.series, runColors]);

  const visible = useMemo(() => {
    if (layout === null) return null;
    const boxes = layout.boxes.filter(
      (box) => matchesSearch(box, search) && (!onlySelected || selection.has(box.runId)),
    );
    const runIds = new Set(boxes.map((box) => box.runId));
    const nodes = layout.nodes.filter((node) => runIds.has(node.runId));
    const ids = new Set(nodes.map((node) => node.id));
    const links = layout.links.filter((link) => ids.has(link.source) && ids.has(link.target));
    return { boxes, nodes, links };
  }, [layout, onlySelected, search, selection]);

  const flowNodes = useMemo<Node[]>(() => {
    if (visible === null) return [];
    const boxNodes: Node<RunBoxNodeData>[] = visible.boxes.map((box) => ({
      id: `run-${box.runId}`,
      type: "runBox",
      draggable: false,
      selectable: false,
      zIndex: BOX_Z,
      width: box.width,
      height: box.height,
      position: { x: box.x, y: box.y },
      data: { box, color: colorOf(box), active: selection.has(box.runId), onOpen: focusRun },
    }));
    const byRun = new Map(visible.boxes.map((box) => [box.runId, box]));
    const checkpointNodes: Node<CheckpointNodeData>[] = visible.nodes.map((checkpoint) => {
      const box = byRun.get(checkpoint.runId);
      return {
        id: checkpoint.id,
        type: "checkpoint",
        draggable: false,
        zIndex: NODE_Z,
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
        position: { x: checkpoint.x, y: checkpoint.y },
        selected: checkpoint.id === selectedId,
        data: {
          checkpoint,
          color: box === undefined ? chart.series[0] : colorOf(box),
          active: selection.has(checkpoint.runId),
        },
      };
    });
    return [...boxNodes, ...checkpointNodes];
  }, [chart.series, colorOf, focusRun, selectedId, selection, visible]);

  // Selecting a checkpoint traces its resume chain: those links come forward, the rest fade.
  const chainEdgeIds = useMemo(() => {
    if (visible === null || selectedId === null) return null;
    const chain = ancestorChain(selectedId, visible.nodes);
    return new Set(chain.slice(1).map((node, index) => `${chain[index].id}-${node.id}`));
  }, [selectedId, visible]);

  const flowEdges = useMemo<Edge[]>(() => {
    if (visible === null) return [];
    const byRun = new Map(visible.boxes.map((box) => [box.runId, box]));
    return visible.links.map((link) => {
      const box = byRun.get(link.runId);
      const color = box === undefined ? chart.grid : colorOf(box);
      const highlighted = chainEdgeIds?.has(link.id) ?? false;
      const data: LineageEdgeData = {
        channelX: link.channelX,
        fork: link.fork,
        color,
        dimmed: chainEdgeIds !== null && !highlighted,
        highlighted,
      };
      return {
        id: link.id,
        source: link.source,
        target: link.target,
        type: "lineage",
        animated: false,
        zIndex: EDGE_Z,
        selectable: false,
        focusable: false,
        data,
      };
    });
  }, [chainEdgeIds, chart.grid, colorOf, visible]);

  // Re-frame when the filters change the visible part of the graph.
  useEffect(() => {
    if (flowNodes.length === 0) return;
    const timer = window.setTimeout(() => void fitView({ duration: 200, padding: 0.08 }), 60);
    return () => window.clearTimeout(timer);
  }, [fitView, flowNodes.length]);

  const selected = visible?.nodes.find((node) => node.id === selectedId);
  const checkpointCount = visible?.nodes.length ?? 0;
  const runCount = visible?.boxes.length ?? 0;

  return (
    <div className="flex min-h-0 flex-1 flex-row">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <Toolbar
          start={
            <>
              <SearchInput
                label="Search checkpoints"
                value={search}
                onValue={setSearch}
                placeholder="Search runs or checkpoints"
                className="w-72"
              />
              <Checkbox
                checked={onlySelected}
                onChange={(event) => setOnlySelected(event.currentTarget.checked)}
              >
                Only selected runs
              </Checkbox>
            </>
          }
          end={
            <>
              <span className="font-mono text-xs tabular-nums text-fg-muted">
                {checkpointCount} checkpoints · {runCount} runs
              </span>
              <IconButton label="Fit to view" onClick={() => void fitView({ duration: 200, padding: 0.08 })}>
                <Scan size={14} />
              </IconButton>
            </>
          }
        />
        {lineageQuery.isPending ? (
          <div className="flex flex-col gap-3 p-4">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
        ) : checkpointCount === 0 ? (
          <EmptyState
            icon={<GitBranch />}
            title="No checkpoint lineage"
            description={
              lineageQuery.error === null
                ? "No checkpoint of this project records the run it belongs to and the checkpoint it was resumed from."
                : String(lineageQuery.error)
            }
          />
        ) : (
          <div className="relative min-h-0 flex-1 bg-canvas">
            <div className="absolute inset-0">
              <ReactFlow
                nodes={flowNodes}
                edges={flowEdges}
                nodeTypes={NODE_TYPES}
                edgeTypes={EDGE_TYPES}
                fitView
                colorMode={theme}
                proOptions={{ hideAttribution: true }}
                minZoom={0.1}
                maxZoom={2}
                nodesConnectable={false}
                onNodeClick={(_, node) => setSelectedId(node.type === "checkpoint" ? node.id : null)}
                onPaneClick={() => setSelectedId(null)}
              >
                <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color={chart.grid} />
                <Controls position="bottom-left" showInteractive={false} />
                <MiniMap
                  position="bottom-right"
                  pannable
                  zoomable
                  nodeColor={(node) => (node.type === "runBox" ? "transparent" : chart.hoverBorder)}
                  maskColor={theme === "dark" ? "rgb(0 0 0 / 0.5)" : "rgb(255 255 255 / 0.6)"}
                  style={{ background: chart.hoverBg, border: `1px solid ${chart.grid}`, borderRadius: 6 }}
                />
              </ReactFlow>
            </div>
          </div>
        )}
      </div>
      {selected === undefined || visible === null ? null : (
        <CheckpointInspector
          checkpoint={selected}
          nodes={visible.nodes}
          runNames={runNames}
          run={allRuns.find((run) => run.id === selected.runId)}
          onOpenRun={(runId, additive) => focusRun(runId, additive)}
          onSelect={setSelectedId}
          onClose={() => setSelectedId(null)}
        />
      )}
    </div>
  );
}

function CheckpointNode({ data, selected }: NodeProps & { data: CheckpointNodeData }) {
  const { checkpoint, color, active } = data;
  return (
    <div
      style={{ width: NODE_WIDTH, height: NODE_HEIGHT, borderColor: selected === true ? color : undefined }}
      className={cn(
        "flex flex-col justify-center rounded-md border bg-surface px-2 text-left",
        selected === true ? "outline-2 outline-accent/40" : "border-line",
        active ? "" : "opacity-70",
      )}
    >
      <Handle type="target" position={Position.Left} className="!size-1.5 !border-0 !bg-strong" />
      <div className="flex items-center gap-1.5">
        <span aria-hidden className="size-2 shrink-0 rounded-full" style={{ background: color }} />
        <span className="truncate font-mono text-[12px] font-semibold text-fg">
          step {checkpoint.step.toLocaleString()}
        </span>
      </div>
      <span className="truncate pl-3.5 font-mono text-[10px] text-fg-muted">
        {checkpoint.lineageStep.toLocaleString()} total steps
      </span>
      <span className="truncate pl-3.5 font-mono text-[10px] text-fg-muted">
        {new Date(checkpoint.createdAt).toLocaleDateString()}
      </span>
      <Handle type="source" position={Position.Right} className="!size-1.5 !border-0 !bg-strong" />
    </div>
  );
}

function RunBoxNode({ data }: NodeProps & { data: RunBoxNodeData }) {
  const { box, color, active, onOpen } = data;
  return (
    <div
      style={{ width: box.width, height: box.height, borderColor: color }}
      className={cn("rounded-lg border border-dashed", active ? "bg-selected/40" : "bg-transparent")}
    >
      <button
        type="button"
        onClick={(event) => onOpen(box.runId, event.metaKey || event.ctrlKey)}
        title={`${box.name} — click to show this run, ⌘/Ctrl-click to add it`}
        className="pointer-events-auto -mt-2.5 ml-2 flex max-w-[calc(100%-1rem)] items-center gap-1.5 rounded-sm bg-canvas px-1.5 text-[11px] font-medium text-fg-secondary hover:text-fg"
      >
        <span aria-hidden className="size-1.5 shrink-0 rounded-full" style={{ background: color }} />
        <span className="truncate">{box.name}</span>
      </button>
    </div>
  );
}
