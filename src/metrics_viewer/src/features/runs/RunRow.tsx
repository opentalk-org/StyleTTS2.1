import { Check, PanelRightOpen, Star } from "lucide-react";
import { useState, type KeyboardEvent, type MouseEvent } from "react";

import { useCursorStore } from "@/shared/cursor";
import { isNumericColumn } from "@/shared/metrics";
import type { Run } from "@/shared/types";
import { cn, ColorPalette, Popover, StatusBadge, Tooltip } from "@/shared/ui";

import { cellText, formatAbsolute } from "./logic";

export const ROW_HEIGHT = 32;

export interface RunRowProps {
  run: Run;
  rowIndex: number;
  lastIndex: number;
  tabbable: boolean;
  selected: boolean;
  starred: boolean;
  color: string;
  hasCustomColor: boolean;
  palette: string[];
  columns: string[];
  gridTemplateColumns: string;
  offset: number;
  onToggle: (extend: boolean) => void;
  onFocusRun: (additive: boolean) => void;
  onInspect: () => void;
  onStar: () => void;
  onColor: (color: string | null) => void;
  onFocusRow: () => void;
  onMoveFocus: (index: number) => void;
}

export function RunRow({
  run,
  rowIndex,
  lastIndex,
  tabbable,
  selected,
  starred,
  color,
  hasCustomColor,
  palette,
  columns,
  gridTemplateColumns,
  offset,
  onToggle,
  onFocusRun,
  onInspect,
  onStar,
  onColor,
  onFocusRow,
  onMoveFocus,
}: RunRowProps) {
  const [paletteOpen, setPaletteOpen] = useState(false);
  // Hovering a curve marks the run it belongs to, and the rest of its lineage with it.
  const highlighted = useCursorStore((state) => state.runId === run.id);
  const inLineage = useCursorStore((state) => state.runIds?.has(run.id) ?? false);

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.target !== event.currentTarget) return;
    if (event.key === " ") {
      event.preventDefault();
      onToggle(event.shiftKey);
    } else if (event.key === "Enter") {
      event.preventDefault();
      onFocusRun(event.metaKey || event.ctrlKey);
    } else if (event.key === "ArrowDown" || event.key === "j") {
      event.preventDefault();
      onMoveFocus(rowIndex + 1);
    } else if (event.key === "ArrowUp" || event.key === "k") {
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
      aria-selected={selected}
      aria-rowindex={rowIndex + 1}
      tabIndex={tabbable ? 0 : -1}
      onFocus={onFocusRow}
      onMouseDown={(event: MouseEvent<HTMLDivElement>) => {
        if (event.shiftKey) event.preventDefault();
      }}
      onClick={(event: MouseEvent<HTMLDivElement>) => {
        if (event.shiftKey) onToggle(true);
        else onFocusRun(event.metaKey || event.ctrlKey);
      }}
      onKeyDown={onKeyDown}
      style={{ gridTemplateColumns, transform: `translateY(${offset}px)`, height: ROW_HEIGHT }}
      className={cn(
        "group/row absolute inset-x-0 grid cursor-pointer items-center border-b border-line text-xs select-none",
        selected ? "bg-selected" : "hover:bg-hover",
        highlighted ? "shadow-[inset_3px_0_0_0_var(--color-accent)] bg-hover" : "",
        inLineage && !highlighted ? "shadow-[inset_3px_0_0_0_var(--color-strong)]" : "",
      )}
    >
      <span role="gridcell" className="flex h-full items-center justify-center">
        <button
          type="button"
          aria-label={selected ? `Remove ${run.name} from the selection` : `Add ${run.name} to the selection`}
          aria-pressed={selected}
          tabIndex={-1}
          onClick={(event) => {
            event.stopPropagation();
            onToggle(event.shiftKey);
          }}
          className={cn(
            "grid size-3.5 place-items-center rounded-sm border",
            selected ? "border-accent bg-accent text-accent-fg" : "border-strong bg-inset text-transparent",
          )}
        >
          <Check size={10} strokeWidth={3} />
        </button>
      </span>

      <span role="gridcell" className="flex h-full items-center justify-center">
        <Popover
          open={paletteOpen}
          onClose={() => setPaletteOpen(false)}
          align="start"
          width={168}
          trigger={
            <Tooltip content="Chart colour">
              <button
                type="button"
                tabIndex={-1}
                aria-label={`Chart colour for ${run.name}`}
                aria-expanded={paletteOpen}
                onClick={(event) => {
                  event.stopPropagation();
                  setPaletteOpen(!paletteOpen);
                }}
                className="grid size-6 place-items-center rounded-sm hover:bg-hover"
              >
                {/* Filled when charted, a hollow ring otherwise: fading the dot
                    washes every colour out to the same pastel on a light canvas. */}
                <span
                  className="block size-3 rounded-full border-2"
                  style={{ borderColor: color, background: selected ? color : "transparent" }}
                />
              </button>
            </Tooltip>
          }
        >
          <ColorPalette
            value={color}
            colors={palette}
            onValue={(next) => {
              onColor(next);
              setPaletteOpen(false);
            }}
            onCustom={onColor}
            onReset={
              hasCustomColor
                ? () => {
                    onColor(null);
                    setPaletteOpen(false);
                  }
                : undefined
            }
          />
        </Popover>
      </span>

      {columns.map((column) => (
        <span
          key={column}
          role="gridcell"
          className={cn(
            "flex h-full min-w-0 items-center gap-1.5 px-2",
            column === "name" ? "font-medium text-fg" : "text-fg-secondary",
            isNumericColumn(column) ? "justify-end font-mono tabular-nums" : "",
            column.startsWith("param:") ? "font-mono" : "",
          )}
        >
          {column === "name" ? (
            <>
              <button
                type="button"
                tabIndex={-1}
                aria-pressed={starred}
                aria-label={starred ? `Unstar ${run.name}` : `Star ${run.name}`}
                onClick={(event) => {
                  event.stopPropagation();
                  onStar();
                }}
                className={cn(
                  "grid size-5 shrink-0 place-items-center rounded-sm",
                  starred ? "text-queued" : "text-fg-muted opacity-0 group-hover/row:opacity-100 hover:bg-hover",
                )}
              >
                <Star size={12} fill={starred ? "currentColor" : "none"} />
              </button>
              <span className="min-w-0 truncate">{run.name}</span>
              <button
                type="button"
                tabIndex={-1}
                aria-label={`Inspect ${run.name}`}
                onClick={(event) => {
                  event.stopPropagation();
                  onInspect();
                }}
                className="ml-auto grid size-5 shrink-0 place-items-center rounded-sm text-fg-muted opacity-0 hover:bg-hover hover:text-fg group-hover/row:opacity-100 focus:opacity-100"
              >
                <PanelRightOpen size={12} />
              </button>
            </>
          ) : column === "status" ? (
            <StatusBadge status={run.status} />
          ) : column === "startedAt" ? (
            <Tooltip content={formatAbsolute(run.startedAt)}>
              <span className="truncate">{cellText(run, column)}</span>
            </Tooltip>
          ) : (
            <span className="min-w-0 truncate">{cellText(run, column)}</span>
          )}
        </span>
      ))}
    </div>
  );
}
