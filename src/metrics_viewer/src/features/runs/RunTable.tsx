import { useVirtualizer, useWindowVirtualizer } from "@tanstack/react-virtual";
import { ArrowDown, ArrowUp, Check, Columns3, Star, X } from "lucide-react";
import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
} from "react";

import { RUN_COLOR_PALETTE, runColor } from "@/shared/chart";
import { runColumnLabel, runColumnOptions } from "@/shared/metrics";
import type { Run, Scalar } from "@/shared/types";
import {
  cn,
  ColorPicker,
  GroupLabel,
  IconButton,
  Popover,
  SearchInput,
  SearchOptionList,
  StatusBadge,
} from "@/shared/ui";

const ROW_HEIGHT = 40;
const HEAD_HEIGHT = 32;
const CHECK_WIDTH = 32;

const ROW_PADDING = 16;

type SortDirection = "asc" | "desc";
interface Sort {
  column: string;
  direction: SortDirection;
}

interface RunTableProps {
  runs: Run[];
  selected: string[];
  columns: string[];
  runColors: Record<string, string>;
  starred: string[];
  loading?: boolean;

  scroll?: "self" | "page";
  onToggle: (id: string) => void;
  onSelect: (ids: string[]) => void;
  onColumns: (items: string[]) => void;
  onRunColor: (runId: string, color: string | null) => void;
  onStar: (runId: string) => void;
  className?: string;
}

export function RunTable({
  runs,
  selected,
  columns,
  runColors,
  starred,
  loading = false,
  scroll = "self",
  onToggle,
  onSelect,
  onColumns,
  onRunColor,
  onStar,
  className,
}: RunTableProps) {
  const [query, setQuery] = useState("");
  const [showColumns, setShowColumns] = useState(false);
  const [sort, setSort] = useState<Sort | null>(null);
  const [focusIndex, setFocusIndex] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);

  const anchorRef = useRef<string | null>(null);

  const focusPending = useRef(false);

  const filtered = useMemo(() => {
    const normalized = query.toLowerCase();
    const matches = runs.filter((run) =>
      `${run.name} ${run.status} ${Object.values(run.params).join(" ")}`
        .toLowerCase()
        .includes(normalized),
    );
    const ordered =
      sort === null
        ? matches
        : [...matches].sort(
            (a, b) =>
              (sort.direction === "asc" ? 1 : -1) *
              compare(sortValue(a, sort.column), sortValue(b, sort.column)),
          );

    return [
      ...ordered.filter((run) => starred.includes(run.id)),
      ...ordered.filter((run) => !starred.includes(run.id)),
    ];
  }, [query, runs, sort, starred]);

  const paged = scroll === "page";


  const [listTop, setListTop] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (!paged) return;
    function measure() {
      const node = listRef.current;
      if (node === null) return;
      setListTop(node.getBoundingClientRect().top + window.scrollY);
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [paged, columns.length, query]);


  const containerVirtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 10,
  });
  const windowVirtualizer = useWindowVirtualizer({
    count: filtered.length,
    estimateSize: () => ROW_HEIGHT,
    overscan: 10,
    scrollMargin: listTop,
  });
  const virtualizer = paged ? windowVirtualizer : containerVirtualizer;

  const scrollMargin = paged ? listTop : 0;

  useEffect(() => {
    if (!focusPending.current) return;
    focusPending.current = false;
    const focusRow = () => {
      const node = document.querySelector<HTMLElement>(`[data-row="${focusIndex}"]`);
      node?.focus({ preventScroll: true });
      return node !== null && node !== undefined;
    };

    if (!focusRow()) requestAnimationFrame(focusRow);
  }, [focusIndex, filtered.length]);

  const gridTemplateColumns = `${CHECK_WIDTH}px ${columns.map(columnWidth).join(" ")}`;


  const minWidth =
    CHECK_WIDTH + ROW_PADDING + columns.reduce((total, column) => total + columnMinWidth(column), 0);
  const allSelected = filtered.length > 0 && selected.length === filtered.length;





  function pickRun(id: string, extend: boolean) {
    const anchor = anchorRef.current;
    if (extend && anchor !== null && anchor !== id) {
      const from = filtered.findIndex((run) => run.id === anchor);
      const to = filtered.findIndex((run) => run.id === id);
      if (from !== -1 && to !== -1) {
        const span = filtered.slice(Math.min(from, to), Math.max(from, to) + 1).map((run) => run.id);
        onSelect([...new Set([...selected, ...span])]);
        return;
      }
    }
    anchorRef.current = id;
    onToggle(id);
  }


  function moveFocus(index: number) {
    const next = Math.min(filtered.length - 1, Math.max(0, index));
    focusPending.current = true;
    setFocusIndex(next);
    virtualizer.scrollToIndex(next, { align: "auto" });
  }

  function cycleSort(column: string) {
    setSort((current) => {
      if (current === null || current.column !== column) return { column, direction: "asc" };
      if (current.direction === "asc") return { column, direction: "desc" };
      return null;
    });
  }

  return (
    <section
      aria-label="Runs"
      className={cn("flex min-h-0 min-w-0 flex-1 flex-col bg-elevated", className)}
    >
      <div className="flex h-13 flex-none items-center gap-2 border-b border-line px-3 py-2.5">
        <SearchInput
          label="Filter runs"
          value={query}
          onValue={setQuery}
          placeholder="Filter runs or params"
          className="flex-1"
        />
        <Popover
          open={showColumns}
          onClose={() => setShowColumns(false)}
          panelClassName="w-72 overflow-hidden p-0"
          trigger={
            <IconButton
              label="Choose columns"
              active={showColumns}
              onClick={() => setShowColumns(!showColumns)}
            >
              <Columns3 size={15} />
            </IconButton>
          }
        >
          <SearchOptionList
            multiple
            options={runColumnOptions()}
            selected={columns}
            placeholder="Search columns"
            emptyMessage="No column matches"
            onSelect={(id) =>
              onColumns(columns.includes(id) ? columns.filter((item) => item !== id) : [...columns, id])
            }
          />
        </Popover>
      </div>

      <div className="flex h-9 flex-none items-center justify-between border-b border-line px-3">
        <div className="flex items-center gap-3">
          <button
            type="button"
            disabled={filtered.length === 0 || allSelected}
            onClick={() => onSelect(filtered.map((run) => run.id))}
            className="rounded-md text-xs font-medium text-fg-secondary transition-colors duration-150 hover:text-accent-bright disabled:opacity-40 disabled:hover:text-fg-secondary"
          >
            Select all
          </button>
          {selected.length > 0 ? (
            <button
              type="button"
              onClick={() => onSelect([])}
              className="flex items-center gap-1 rounded-md text-xs font-medium text-fg-muted transition-colors duration-150 hover:text-negative"
            >
              <X size={11} />
              Clear {selected.length}
            </button>
          ) : null}
        </div>
        <span
          className="font-mono text-xs tabular-nums text-fg-muted"
          title="Click a row to toggle it, shift-click to select a range"
        >
          {selected.length} selected · {filtered.length} runs
        </span>
      </div>



      <div
        ref={headerRef}
        className={cn(
          "flex-none overflow-hidden border-b border-line bg-inset",
          paged ? "sticky top-14 z-20" : "",
        )}
      >
        <div
          role="row"
          className="grid items-center px-2"
          style={{ gridTemplateColumns, height: HEAD_HEIGHT, minWidth }}
        >
          <span role="columnheader" />
          {columns.map((column) => (
            <ColumnHeader key={column} column={column} sort={sort} onSort={() => cycleSort(column)} />
          ))}
        </div>
      </div>

      <div
        ref={scrollRef}
        onScroll={(event) => {
          if (headerRef.current !== null) headerRef.current.scrollLeft = event.currentTarget.scrollLeft;
        }}
        className={cn(paged ? "overflow-x-auto" : "min-h-0 flex-1 overflow-auto")}
      >
        <div role="grid" aria-rowcount={filtered.length} className="min-w-full" style={{ minWidth }}>
          {loading ? (
            <p className="m-0 px-4 py-10 text-center text-xs text-fg-muted">Loading runs…</p>
          ) : null}
          {!loading && runs.length === 0 ? (
            <p className="m-0 px-4 py-10 text-center text-xs text-fg-muted">
              This project has no runs yet.
            </p>
          ) : null}
          {!loading && runs.length > 0 && filtered.length === 0 ? (
            <p className="m-0 px-4 py-10 text-center text-xs text-fg-muted">
              No run matches “{query}”.
            </p>
          ) : null}

          <div
            ref={listRef}
            className="relative"
            style={{ height: Math.max(0, virtualizer.getTotalSize() - scrollMargin) }}
          >
            {virtualizer.getVirtualItems().map((row) => {
              const run = filtered[row.index];
              const active = selected.includes(run.id);
              return (
                <RunRow
                  key={run.id}
                  run={run}
                  rowIndex={row.index}
                  tabbable={row.index === Math.min(focusIndex, filtered.length - 1)}
                  active={active}
                  columns={columns}
                  color={runColor(run.id, row.index, runColors)}
                  starred={starred.includes(run.id)}
                  onStar={() => onStar(run.id)}
                  hasCustomColor={runColors[run.id] !== undefined}
                  gridTemplateColumns={gridTemplateColumns}
                  offset={row.start - scrollMargin}
                  onPick={(extend) => pickRun(run.id, extend)}
                  onFocusRow={() => setFocusIndex(row.index)}
                  onMoveFocus={moveFocus}
                  lastIndex={filtered.length - 1}
                  onColor={(color) => onRunColor(run.id, color)}
                />
              );
            })}
          </div>
        </div>
      </div>

    </section>
  );
}

function ColumnHeader({
  column,
  sort,
  onSort,
}: {
  column: string;
  sort: Sort | null;
  onSort: () => void;
}) {
  const active = sort?.column === column;
  return (
    <button
      type="button"
      role="columnheader"
      aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}
      title={`Sort by ${runColumnLabel(column)}`}
      onClick={onSort}
      className={cn(
        "group flex h-full min-w-0 items-center gap-1 px-2 text-left transition-colors duration-150",
        active ? "text-accent-bright" : "text-fg-muted hover:text-fg-secondary",
      )}
    >
      <GroupLabel className="truncate text-inherit">{runColumnLabel(column)}</GroupLabel>
      {active ? (
        sort.direction === "asc" ? (
          <ArrowUp size={11} className="shrink-0" />
        ) : (
          <ArrowDown size={11} className="shrink-0" />
        )
      ) : (
        <ArrowUp size={11} className="shrink-0 opacity-0 transition-opacity group-hover:opacity-60" />
      )}
    </button>
  );
}

interface RunRowProps {
  run: Run;
  rowIndex: number;
  tabbable: boolean;
  active: boolean;
  columns: string[];
  color: string;
  starred: boolean;
  onStar: () => void;
  hasCustomColor: boolean;
  gridTemplateColumns: string;
  offset: number;
  lastIndex: number;
  onPick: (extend: boolean) => void;
  onFocusRow: () => void;
  onMoveFocus: (index: number) => void;
  onColor: (color: string | null) => void;
}

function RunRow({
  run,
  rowIndex,
  tabbable,
  active,
  columns,
  color,
  starred,
  onStar,
  hasCustomColor,
  gridTemplateColumns,
  offset,
  lastIndex,
  onPick,
  onFocusRow,
  onMoveFocus,
  onColor,
}: RunRowProps) {
  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.target !== event.currentTarget) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onPick(event.shiftKey);
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      onMoveFocus(rowIndex + 1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      onMoveFocus(rowIndex - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      onMoveFocus(0);
    } else if (event.key === "End") {
      event.preventDefault();
      onMoveFocus(lastIndex);
    }
  }

  return (


    <div
      role="row"
      data-row={rowIndex}
      aria-selected={active}
      aria-rowindex={rowIndex + 1}

      tabIndex={tabbable ? 0 : -1}
      onFocus={onFocusRow}

      onMouseDown={(event: MouseEvent<HTMLDivElement>) => {
        if (event.shiftKey) event.preventDefault();
      }}
      onClick={(event: MouseEvent<HTMLDivElement>) => onPick(event.shiftKey)}
      onKeyDown={onKeyDown}
      style={{ gridTemplateColumns, transform: `translateY(${offset}px)`, height: ROW_HEIGHT }}
      className={cn(
        "absolute inset-x-0 grid cursor-pointer items-center border-b border-line/60 px-2 text-left text-xs select-none",
        "transition-colors duration-150 ease-out",
        active ? "bg-accent-surface shadow-[inset_2px_0_0_0_var(--color-accent)]" : "hover:bg-surface",
      )}
    >
      <span role="gridcell" className="flex items-center">
        <span
          aria-hidden
          className={cn(
            "ml-0.5 grid size-4 place-items-center rounded-[4px] border transition-colors duration-150",
            active ? "border-accent bg-accent text-white" : "border-line-hover bg-inset text-transparent",
          )}
        >
          <Check size={11} />
        </span>
      </span>
      {columns.map((column) => (
        <span
          key={column}
          role="gridcell"
          className={cn(
            "flex min-w-0 items-center gap-2 px-2",
            column === "name"
              ? cn("font-medium", active ? "text-accent-bright" : "text-fg")
              : "font-mono tabular-nums text-fg-secondary",
          )}
        >
          {column === "name" ? (
            <button
              type="button"
              aria-pressed={starred}
              title={starred ? "Unstar this run" : "Star this run, pinning it to the top"}
              aria-label={starred ? `Unstar ${run.name}` : `Star ${run.name}`}
              onClick={(event) => {
                event.stopPropagation();
                onStar();
              }}
              className={cn(
                "grid size-5 shrink-0 place-items-center rounded-md transition-colors duration-150",
                starred
                  ? "text-notice hover:text-notice/80"
                  : "text-fg-muted/50 hover:bg-surface hover:text-fg-secondary",
              )}
            >
              <Star size={12} fill={starred ? "currentColor" : "none"} />
            </button>
          ) : null}
          {column === "name" ? (
            <ColorPicker
              label={`Plot color for ${run.name}`}
              value={color}
              colors={RUN_COLOR_PALETTE}
              onValue={onColor}
              onReset={hasCustomColor ? () => onColor(null) : undefined}
            />
          ) : null}
          <span className="min-w-0 truncate">{cellValue(run, column)}</span>
        </span>
      ))}
    </div>
  );
}

function columnWidth(column: string): string {
  if (column === "name") return "minmax(160px,1.5fr)";
  if (column === "startedAt") return "116px";
  return "minmax(96px,1fr)";
}

function columnMinWidth(column: string): number {
  if (column === "name") return 160;
  if (column === "startedAt") return 116;
  return 96;
}

function durationMs(run: Run): number {
  if (run.endedAt === 0) return Number.POSITIVE_INFINITY;
  return run.endedAt - run.startedAt;
}


function sortValue(run: Run, column: string): Scalar {
  if (column === "name") return run.name;
  if (column === "status") return run.status;
  if (column === "startedAt") return run.startedAt;
  if (column === "duration") return durationMs(run);
  if (column.startsWith("param:")) return run.params[column.slice(6)] ?? "";
  if (column.startsWith("metric:")) return run.summary[column.slice(7)] ?? Number.NaN;
  return "";
}

function compare(a: Scalar, b: Scalar): number {
  if (typeof a === "number" && typeof b === "number") {
    if (Number.isNaN(a)) return 1;
    if (Number.isNaN(b)) return -1;
    return a - b;
  }
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

function cellValue(run: Run, column: string): ReactNode {
  if (column === "name") return run.name;
  if (column === "status") return <StatusBadge status={run.status} />;
  if (column === "startedAt")
    return new Date(run.startedAt).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  if (column === "duration")
    return run.endedAt === 0
      ? "—"
      : `${Math.round(durationMs(run) / 60000)}m`;
  if (column.startsWith("param:")) return String(run.params[column.slice(6)] ?? "—");
  if (column.startsWith("metric:")) return run.summary[column.slice(7)]?.toFixed(4) ?? "—";

  return "—";
}
