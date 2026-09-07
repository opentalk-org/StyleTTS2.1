import { ArrowDown, ArrowUp } from "lucide-react";
import { useState, type DragEvent } from "react";

import { isNumericColumn, runColumnLabel } from "@/shared/metrics";
import { Caption, Skeleton, cn } from "@/shared/ui";

import type { RunSort } from "./logic";

const COLUMN_DRAG_TYPE = "application/x-metrics-column";

interface ColumnHeaderProps {
  column: string;
  sort: RunSort | null;
  onSort: () => void;
  onReorder: (from: string, to: string) => void;
}

/** Sortable header cell; drag one onto another to reorder the columns. */
export function ColumnHeader({ column, sort, onSort, onReorder }: ColumnHeaderProps) {
  const active = sort?.column === column;
  const [dropTarget, setDropTarget] = useState(false);

  function onDragOver(event: DragEvent<HTMLButtonElement>) {
    if (!event.dataTransfer.types.includes(COLUMN_DRAG_TYPE)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    setDropTarget(true);
  }

  return (
    <button
      type="button"
      role="columnheader"
      aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}
      title={`Sort by ${runColumnLabel(column)}. Drag to reorder`}
      draggable
      onDragStart={(event) => {
        event.dataTransfer.setData(COLUMN_DRAG_TYPE, column);
        event.dataTransfer.effectAllowed = "move";
      }}
      onDragOver={onDragOver}
      onDragLeave={() => setDropTarget(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDropTarget(false);
        onReorder(event.dataTransfer.getData(COLUMN_DRAG_TYPE), column);
      }}
      onClick={onSort}
      className={cn(
        "group flex h-full min-w-0 cursor-grab items-center gap-1 px-2 text-left active:cursor-grabbing",
        isNumericColumn(column) ? "flex-row-reverse" : "",
        active ? "text-fg" : "text-fg-muted hover:text-fg-secondary",
        dropTarget ? "shadow-[inset_2px_0_0_0_var(--color-accent)]" : "",
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
}

export function SkeletonRows() {
  return (
    <div className="flex flex-col gap-px p-2">
      {Array.from({ length: 8 }, (_, index) => (
        <div key={index} className="flex h-7 items-center gap-3">
          <Skeleton className="size-3.5" />
          <Skeleton className="size-3 rounded-full" />
          <Skeleton className="h-3 w-40" />
          <Skeleton className="h-3 w-16" />
          <Skeleton className="ml-auto h-3 w-12" />
        </div>
      ))}
    </div>
  );
}
