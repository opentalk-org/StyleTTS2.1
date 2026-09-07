import { ArrowDown, ArrowUp, Columns3 } from "lucide-react";
import { useMemo, useState } from "react";

import { runColumnLabel } from "@/shared/metrics";
import type { Run } from "@/shared/types";
import { Caption, cn, EmptyState, Tooltip } from "@/shared/ui";

import { cellText, compareValue, isIdentical, isNumeric, rawValue } from "./logic";

const MAX_CHARS = 25;

interface Sort {
  column: string;
  direction: "asc" | "desc";
}

interface CompareTableProps {
  runs: Run[];
  columns: string[];
  runColors: Record<string, string>;
  onlyDifferences: boolean;
}

/** Runs as rows, chosen parameters and final metrics as columns. */
export function CompareTable({ runs, columns, runColors, onlyDifferences }: CompareTableProps) {
  const [sort, setSort] = useState<Sort | null>(null);
  const visibleColumns = useMemo(
    () => (onlyDifferences && runs.length > 1 ? columns.filter((column) => !isIdentical(runs, column)) : columns),
    [columns, runs, onlyDifferences],
  );
  const numeric = useMemo(() => new Set(visibleColumns.filter((column) => isNumeric(runs, column))), [visibleColumns, runs]);
  const ordered = useMemo(() => {
    if (sort === null) return runs;
    return [...runs].sort((a, b) => compareValue(rawValue(a, sort.column), rawValue(b, sort.column), sort.direction));
  }, [runs, sort]);

  function cycleSort(column: string) {
    setSort((current) => {
      if (current === null || current.column !== column) return { column, direction: "asc" };
      if (current.direction === "asc") return { column, direction: "desc" };
      return null;
    });
  }

  if (runs.length === 0) {
    return <EmptyState compact icon={<Columns3 />} title="No runs selected" description="Tick runs in the list on the left to fill the table." />;
  }
  if (visibleColumns.length === 0) {
    return (
      <EmptyState
        compact
        icon={<Columns3 />}
        title={columns.length === 0 ? "No columns chosen" : "No differences"}
        description={columns.length === 0 ? "Pick parameters and final metrics with the Columns button." : "Every chosen column is identical across these runs."}
      />
    );
  }

  const template = `240px ${visibleColumns.map((column) => numeric.has(column) ? "120px" : "168px").join(" ")}`;

  return (
    <div className="w-max min-w-full">
      <div className="sticky top-0 z-10 grid border-b border-line bg-surface" style={{ gridTemplateColumns: template }}>
        <Caption className="sticky left-0 z-10 flex h-thead items-center border-r border-line bg-surface px-2">Run</Caption>
        {visibleColumns.map((column) => {
          const active = sort?.column === column;
          return (
            <button
              key={column}
              type="button"
              onClick={() => cycleSort(column)}
              title={column}
              className={cn(
                "group flex h-thead min-w-0 items-center gap-1 px-2 text-left",
                numeric.has(column) ? "flex-row-reverse" : "",
                active ? "text-fg" : "text-fg-muted hover:text-fg-secondary",
              )}
            >
              <Caption className="truncate text-inherit">{runColumnLabel(column)}</Caption>
              {active ? (
                sort.direction === "asc" ? <ArrowUp size={11} className="shrink-0" /> : <ArrowDown size={11} className="shrink-0" />
              ) : (
                <ArrowUp size={11} className="shrink-0 opacity-0 group-hover:opacity-60" />
              )}
            </button>
          );
        })}
      </div>
      {ordered.map((run) => (
        <div key={run.id} className="grid border-b border-line hover:bg-hover" style={{ gridTemplateColumns: template }}>
          <span className="sticky left-0 z-10 flex min-w-0 items-center gap-2 border-r border-line bg-surface px-2 py-1.5">
            <span className="size-2.5 shrink-0 rounded-full" style={{ background: runColors[run.id] }} />
            <Clipped value={run.name} className="text-xs font-medium text-fg" />
          </span>
          {visibleColumns.map((column) => (
            <Clipped
              key={column}
              value={cellText(run, column)}
              className={cn("px-2 py-1.5 font-mono text-xs tabular-nums text-fg-secondary", numeric.has(column) ? "text-right" : "")}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

/** Text longer than 25 characters is clipped; the full text shows in a tooltip on hover. */
function Clipped({ value, className }: { value: string; className: string }) {
  if (value.length <= MAX_CHARS) return <span className={cn("block min-w-0 truncate", className)}>{value}</span>;
  return (
    <Tooltip content={<span className="max-w-96 font-mono whitespace-pre-wrap break-all">{value}</span>} className="min-w-0">
      <span className={cn("block min-w-0 truncate", className)}>{`${value.slice(0, MAX_CHARS)}…`}</span>
    </Tooltip>
  );
}
