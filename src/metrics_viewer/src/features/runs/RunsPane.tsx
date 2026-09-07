import { useVirtualizer } from "@tanstack/react-virtual";
import { Check, Minus } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";

import type { ViewerLayout } from "@/features/viewer/layout";
import { useViewerStore } from "@/features/viewer/store";
import type { ChartTheme } from "@/shared/chart";
import { moveBefore } from "@/shared/order";
import type { ProjectColumns, Run, RunStatus } from "@/shared/types";
import { Button, cn, EmptyState } from "@/shared/ui";

import { columnMinWidth, columnWidth, filterRuns, type RunSort } from "./logic";
import { ROW_HEIGHT, RunRow } from "./RunRow";
import { RunInspector } from "./RunInspector";
import { ColumnHeader, SkeletonRows } from "./RunsTableParts";
import { RunsToolbar } from "./RunsToolbar";

const CHECK_WIDTH = 28;
const SWATCH_WIDTH = 24;

interface RunsPaneProps {
  runs: Run[];
  projectColumns: ProjectColumns;
  loading: boolean;
  runColors: Record<string, string>;
  palette: string[];
  chart: ChartTheme;
  layout: ViewerLayout;
  revealRunId: string | null;
  onCollapse: () => void;
  onStack: () => void;
  onResetLayout: () => void;
}

export function RunsPane({ runs, projectColumns, loading, runColors, palette, chart, layout, revealRunId, onCollapse, onStack, onResetLayout }: RunsPaneProps) {
  const viewer = useViewerStore();
  const [query, setQuery] = useState("");
  const [statuses, setStatuses] = useState<RunStatus[]>([]);
  const [sort, setSort] = useState<RunSort | null>(null);
  const [focusIndex, setFocusIndex] = useState(0);
  const [inspectedRun, setInspectedRun] = useState<Run | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const anchorRef = useRef<string | null>(null);
  const focusPending = useRef(false);

  const filtered = useMemo(
    () => filterRuns(runs, { query, statuses }, sort, viewer.starredRunIds),
    [runs, query, statuses, sort, viewer.starredRunIds],
  );

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 12,
  });

  useEffect(() => {
    if (revealRunId === null) return;
    setQuery("");
    setStatuses([]);
    const visibleRuns = filterRuns(runs, { query: "", statuses: [] }, sort, viewer.starredRunIds);
    const index = visibleRuns.findIndex((run) => run.id === revealRunId);
    if (index === -1) return;
    setFocusIndex(index);
    requestAnimationFrame(() => virtualizer.scrollToIndex(index, { align: "center" }));
  }, [revealRunId, runs, sort, viewer.starredRunIds]);

  useEffect(() => {
    if (!focusPending.current) return;
    focusPending.current = false;
    const focusRow = () => {
      const node = document.querySelector<HTMLElement>(`[data-row="${focusIndex}"]`);
      node?.focus({ preventScroll: true });
      return node !== null;
    };
    if (!focusRow()) requestAnimationFrame(focusRow);
  }, [focusIndex, filtered.length]);

  const { columns } = viewer;
  const gridTemplateColumns = `${CHECK_WIDTH}px ${SWATCH_WIDTH}px ${columns.map(columnWidth).join(" ")}`;
  const minWidth = CHECK_WIDTH + SWATCH_WIDTH + columns.reduce((total, column) => total + columnMinWidth(column), 0);
  const selectedCount = viewer.selectedRunIds.length;
  const allFilteredSelected =
    filtered.length > 0 && filtered.every((run) => viewer.selectedRunIds.includes(run.id));

  function toggleRun(id: string, extend: boolean) {
    const anchor = anchorRef.current;
    if (extend && anchor !== null && anchor !== id) {
      const from = filtered.findIndex((run) => run.id === anchor);
      const to = filtered.findIndex((run) => run.id === id);
      if (from !== -1 && to !== -1) {
        const span = filtered.slice(Math.min(from, to), Math.max(from, to) + 1).map((run) => run.id);
        viewer.selectRuns([...new Set([...viewer.selectedRunIds, ...span])]);
        return;
      }
    }
    anchorRef.current = id;
    viewer.toggleRun(id);
  }

  function moveFocus(index: number) {
    const next = Math.min(filtered.length - 1, Math.max(0, index));
    focusPending.current = true;
    setFocusIndex(next);
    virtualizer.scrollToIndex(next, { align: "auto" });
  }

  function reorderColumns(from: string, to: string) {
    const next = moveBefore(columns, from, to);
    // The run name stays the first column so rows keep a stable anchor.
    viewer.setColumns(["name", ...next.filter((column) => column !== "name")]);
  }

  function cycleSort(column: string) {
    setSort((current) => {
      if (current === null || current.column !== column) return { column, direction: "asc" };
      if (current.direction === "asc") return { column, direction: "desc" };
      return null;
    });
  }

  function onListKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "a" && !event.metaKey && !event.ctrlKey) {
      event.preventDefault();
      viewer.selectRuns(filtered.map((run) => run.id));
    } else if (event.key === "Escape") {
      viewer.selectRuns([]);
    }
  }

  return (
    <section aria-label="Runs" className="flex min-h-0 min-w-0 flex-1 flex-col bg-surface">
      <RunsToolbar
        runs={runs}
        projectColumns={projectColumns}
        query={query}
        onQuery={setQuery}
        statuses={statuses}
        onStatuses={setStatuses}
        columns={columns}
        onColumns={viewer.setColumns}
        layout={layout}
        onCollapse={onCollapse}
        onStack={onStack}
        onResetLayout={onResetLayout}
      />

      <div ref={headerRef} className="flex-none overflow-hidden border-b border-line bg-surface">
        <div role="row" className="grid h-thead items-center" style={{ gridTemplateColumns, minWidth }}>
          <span role="columnheader" className="flex h-full items-center justify-center">
            <button
              type="button"
              aria-label={allFilteredSelected ? "Deselect all runs" : "Select all matching runs"}
              aria-pressed={allFilteredSelected}
              disabled={filtered.length === 0}
              onClick={() => viewer.selectRuns(allFilteredSelected ? [] : filtered.map((run) => run.id))}
              className={cn(
                "grid size-3.5 place-items-center rounded-sm border disabled:opacity-40",
                selectedCount > 0 ? "border-accent bg-accent text-accent-fg" : "border-strong bg-inset text-transparent hover:border-accent",
              )}
            >
              {allFilteredSelected ? <Check size={10} strokeWidth={3} /> : <Minus size={10} strokeWidth={3} />}
            </button>
          </span>
          <span role="columnheader" />
          {columns.map((column) => (
            <ColumnHeader key={column} column={column} sort={sort} onSort={() => cycleSort(column)} onReorder={reorderColumns} />
          ))}
        </div>
      </div>

      <div
        ref={scrollRef}
        onScroll={(event) => {
          if (headerRef.current !== null) headerRef.current.scrollLeft = event.currentTarget.scrollLeft;
        }}
        onKeyDown={onListKeyDown}
        className="min-h-0 flex-1 overflow-auto"
      >
        <div role="grid" aria-rowcount={filtered.length} className="min-w-full" style={{ minWidth }}>
          {loading ? <SkeletonRows /> : null}
          {!loading && runs.length === 0 ? (
            <EmptyState compact icon={<span />} title="No runs yet" description="Runs appear here as soon as the project logs one." />
          ) : null}
          {!loading && runs.length > 0 && filtered.length === 0 ? (
            <EmptyState compact icon={<span />} title="No run matches" description="Try a different search or clear the status filter.">
              <Button size="sm" onClick={() => { setQuery(""); setStatuses([]); }}>Clear filters</Button>
            </EmptyState>
          ) : null}
          <div className="relative" style={{ height: virtualizer.getTotalSize() }}>
            {virtualizer.getVirtualItems().map((row) => {
              const run = filtered[row.index];
              return (
                <RunRow
                  key={run.id}
                  run={run}
                  rowIndex={row.index}
                  lastIndex={filtered.length - 1}
                  tabbable={row.index === Math.min(focusIndex, filtered.length - 1)}
                  selected={viewer.selectedRunIds.includes(run.id)}
                  starred={viewer.starredRunIds.includes(run.id)}
                  color={runColors[run.id]}
                  hasCustomColor={viewer.runColorOverrides[run.id] !== undefined}
                  palette={palette}
                  columns={columns}
                  gridTemplateColumns={gridTemplateColumns}
                  offset={row.start}
                  onToggle={(extend) => toggleRun(run.id, extend)}
                  onFocusRun={(additive) => {
                    anchorRef.current = run.id;
                    viewer.focusRun(run.id, additive);
                  }}
                  onInspect={() => setInspectedRun(run)}
                  onStar={() => viewer.toggleStar(run.id)}
                  onColor={(color) => viewer.setRunColor(run.id, color)}
                  onFocusRow={() => setFocusIndex(row.index)}
                  onMoveFocus={moveFocus}
                />
              );
            })}
          </div>
        </div>
      </div>

      <footer className="flex h-8 flex-none items-center justify-between gap-2 border-t border-line bg-surface px-3">
        <span className="flex items-center gap-2 text-xs text-fg-muted">
          <span className="font-mono tabular-nums">
            <span className="text-fg">{selectedCount}</span> of {filtered.length} selected
          </span>
          {selectedCount > 0 ? (
            <Button size="sm" variant="ghost" onClick={() => viewer.selectRuns([])}>
              Clear
            </Button>
          ) : null}
        </span>
        <Button
          size="sm"
          variant="ghost"
          disabled={filtered.length === 0 || allFilteredSelected}
          onClick={() => viewer.selectRuns(filtered.map((run) => run.id))}
        >
          Select all
        </Button>
      </footer>
      {inspectedRun === null ? null : (
        <RunInspector
          run={runs.find((run) => run.id === inspectedRun.id) ?? inspectedRun}
          color={runColors[inspectedRun.id]}
          chart={chart}
          onClose={() => setInspectedRun(null)}
        />
      )}
    </section>
  );
}
