import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "@xyflow/react/dist/style.css";

import { useViewerStore } from "@/features/viewer/store";
import type { ChartTheme } from "@/shared/chart";
import type { Run } from "@/shared/types";
import { EmptyState, Skeleton } from "@/shared/ui";

import { buildGraph } from "./graph";
import { GraphToolbar } from "./GraphToolbar";
import { buildTree } from "./hierarchy";
import { containersToDepth, groupKey, levelView, matchingIds, revealPaths, type LevelNode } from "./level";
import { ContainerNode, GraphActions, ModuleNode } from "./ModuleNodes";
import { MonitorInspector } from "./MonitorInspector";
import { hueIndex } from "./palette";
import { useArrayMetricNames, useModelGraph } from "./query";
import { findRepeats } from "./repeats";
import { RoutedEdge } from "./RoutedEdge";

const NODE_TYPES = { module: ModuleNode, container: ContainerNode };
const EDGE_TYPES = { routed: RoutedEdge };
const DEFAULT_DETAIL = 2;
/** Fit never shrinks past legibility: a crowded level is scrolled, not squinted at. */
const FIT = { duration: 200, minZoom: 0.4, maxZoom: 1.1 };

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
  const { fitView, getZoom, getViewport, setViewport } = useReactFlow();
  const theme = useViewerStore((state) => state.theme);
  const [detail, setDetail] = useState(DEFAULT_DETAIL);
  const [overrides, setOverrides] = useState(new Map<string, boolean>());
  const [search, setSearch] = useState("");
  const [showShapes, setShowShapes] = useState(false);
  const [parametersOnly, setParametersOnly] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const anchor = useRef<{ id: string; left: number; top: number } | null>(null);

  const tree = useMemo(() => buildTree(graphQuery.data ?? []), [graphQuery.data]);
  const repeats = useMemo(() => findRepeats(tree), [tree]);
  const hues = useMemo(() => hueIndex(tree), [tree]);
  const matches = useMemo(() => matchingIds(tree, search), [tree, search]);
  const expanded = useMemo(() => {
    const open = containersToDepth(tree, detail);
    for (const [id, wanted] of overrides) {
      if (wanted) open.add(id);
      else open.delete(id);
    }
    for (const id of revealPaths(tree, repeats, matches)) open.add(id);
    return open;
  }, [tree, repeats, detail, overrides, matches]);
  const view = useMemo(
    () => levelView(tree, repeats, { expanded, matches, parametersOnly }),
    [tree, repeats, expanded, matches, parametersOnly],
  );
  const graph = useMemo(
    () => buildGraph(view, { hues, selectedId, searching: matches.size > 0, showShapes }),
    [view, hues, selectedId, matches.size, showShapes],
  );

  /**
   * Opening a box relays out everything around it. The box that was clicked is pinned to where it
   * was on screen and the viewport moves instead, so the canvas grows from that point rather than
   * sliding out from under the pointer.
   */
  const remember = useCallback((id: string) => {
    const rect = document.querySelector(`.react-flow__node[data-id="${CSS.escape(id)}"]`)?.getBoundingClientRect();
    anchor.current = rect === undefined ? null : { id, left: rect.left, top: rect.top };
  }, []);
  const setOpen = useCallback(
    (id: string, open: boolean) => {
      remember(id);
      setOverrides((current) => new Map(current).set(id, open));
    },
    [remember],
  );
  const setUnfolded = useCallback(
    (id: string, unfolded: boolean) => {
      remember(id);
      setOverrides((current) => new Map(current).set(groupKey(id), unfolded));
    },
    [remember],
  );
  const actions = useMemo(() => ({ setOpen, setUnfolded }), [setOpen, setUnfolded]);

  useEffect(() => {
    const held = anchor.current;
    anchor.current = null;
    if (held === null) return;
    const frame = requestAnimationFrame(() => {
      const rect = document
        .querySelector(`.react-flow__node[data-id="${CSS.escape(held.id)}"]`)
        ?.getBoundingClientRect();
      if (rect === undefined) return;
      const viewport = getViewport();
      setViewport({ ...viewport, x: viewport.x + held.left - rect.left, y: viewport.y + held.top - rect.top });
    });
    return () => cancelAnimationFrame(frame);
  }, [graph, getViewport, setViewport]);

  /** The depth control redraws the canvas wholesale, so the viewport is re-seated with it. */
  useEffect(() => {
    setOverrides(new Map());
    const frame = requestAnimationFrame(() => void fitView(FIT));
    return () => cancelAnimationFrame(frame);
  }, [detail, parametersOnly, tree, fitView]);

  /** Arrow keys walk the dataflow from the selected box, bringing each step into view. */
  const step = useCallback(
    (event: KeyboardEvent) => {
      if (selectedId === null || (event.key !== "ArrowDown" && event.key !== "ArrowUp")) return;
      const downstream = event.key === "ArrowDown";
      const next = view.edges.find((edge) => (downstream ? edge.source : edge.target) === selectedId);
      if (next === undefined) return;
      event.preventDefault();
      const id = downstream ? next.target : next.source;
      setSelectedId(id);
      void fitView({ nodes: [{ id }], duration: 200, maxZoom: getZoom() });
    },
    [selectedId, view, fitView, getZoom],
  );

  useEffect(() => {
    window.addEventListener("keydown", step);
    return () => window.removeEventListener("keydown", step);
  }, [step]);

  if (graphQuery.isPending) return <Skeleton className="m-3 flex-1" />;
  if (graphQuery.isError) {
    return (
      <EmptyState
        icon={<span />}
        title="No model graph"
        description="This run has not written monitor/model_graph.json yet."
      />
    );
  }

  // The panel is for modules that hold weights of their own; a group answers a click by opening.
  const found = selectedId === null ? undefined : findBox(view.roots, selectedId);
  const selected = found !== undefined && found.node.childIds.length === 0 && found.parameterCount > 0 ? found : undefined;
  return (
    <div className="flex min-h-0 flex-1 flex-row">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <GraphToolbar
          search={search}
          onSearch={setSearch}
          detail={detail}
          maxDetail={tree.maxDepth + 1}
          onDetail={setDetail}
          showShapes={showShapes}
          onShowShapes={setShowShapes}
          parametersOnly={parametersOnly}
          onParametersOnly={setParametersOnly}
          boxCount={view.boxCount}
          moduleCount={tree.nodes.size}
          onFit={() => void fitView(FIT)}
        />
        <div className="relative min-h-0 flex-1 bg-canvas">
          <div className="absolute inset-0">
            <GraphActions.Provider value={actions}>
              <ReactFlow
                nodes={graph.nodes}
                edges={graph.edges}
                nodeTypes={NODE_TYPES}
                edgeTypes={EDGE_TYPES}
                fitView
                fitViewOptions={FIT}
                colorMode={theme}
                proOptions={{ hideAttribution: true }}
                minZoom={0.05}
                maxZoom={2.5}
                nodesDraggable={false}
                onNodeClick={(_, node) =>
                  node.data.container === true ? setOpen(node.id, true) : setSelectedId(node.id)
                }
                onPaneClick={() => setSelectedId(null)}
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
            </GraphActions.Provider>
          </div>
        </div>
      </div>
      {selected === undefined ? null : (
        <MonitorInspector
          runId={run.id}
          box={selected}
          names={namesQuery.data ?? []}
          loading={namesQuery.isPending}
          running={running}
          chart={chart}
          onClose={() => setSelectedId(null)}
        />
      )}
    </div>
  );
}

function findBox(boxes: LevelNode[], id: string): LevelNode | undefined {
  for (const box of boxes) {
    if (box.id === id) return box;
    const found = findBox(box.children, id);
    if (found !== undefined) return found;
  }
  return undefined;
}
