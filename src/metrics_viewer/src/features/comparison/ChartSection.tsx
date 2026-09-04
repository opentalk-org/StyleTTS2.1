import { Pin } from "lucide-react";
import { lazy, Suspense, useCallback, useMemo } from "react";

import { useViewerStore } from "@/features/viewer/store";
import type { ChartTheme } from "@/shared/chart";
import type { PanelColumns, Run } from "@/shared/types";
import { cn, Collapsible, IconButton } from "@/shared/ui";

import { moveBefore, orderByList } from "@/shared/order";

import type { Plot, Section } from "./logic";

const ChartCard = lazy(() => import("./ChartCard").then((module) => ({ default: module.ChartCard })));

export type Columns = PanelColumns;

export const COLUMN_CLASSES: Record<Columns, string> = {
  "1": "grid-cols-1",
  "2": "grid-cols-2",
  "3": "grid-cols-3",
  auto: "grid-cols-[repeat(auto-fill,minmax(340px,1fr))]",
};

interface ChartSectionProps {
  section: Section<Plot>;
  open: boolean;
  onToggle: () => void;
  columns: Columns;
  runs: Run[];
  runColors: Record<string, string>;
  chart: ChartTheme;
  onExpand: (name: string) => void;
  rangeQueries: boolean;
}

/** One namespace of charts: collapsible, pinnable, with drag-to-reorder cards. */
export function ChartSection({ section, open, onToggle, columns, runs, runColors, chart, onExpand, rangeQueries }: ChartSectionProps) {
  const plotOrder = useViewerStore((state) => state.plotOrder);
  const setPlotOrder = useViewerStore((state) => state.setPlotOrder);
  const togglePinnedSection = useViewerStore((state) => state.togglePinnedSection);
  const ordered = useMemo(() => orderByList(section.items, plotOrder), [section.items, plotOrder]);
  const names = useMemo(() => ordered.map((plot) => plot.name), [ordered]);

  const onMove = useCallback(
    (from: string, to: string) => {
      if (!names.includes(from) || !names.includes(to)) return;
      setPlotOrder(moveBefore(names, from, to));
    },
    [names, setPlotOrder],
  );

  return (
    <Collapsible
      open={open}
      onToggle={onToggle}
      title={section.name}
      count={section.items.length}
      actions={
        <IconButton
          label={section.pinned ? "Unpin section" : "Pin section to the top"}
          size="sm"
          active={section.pinned}
          className={section.pinned ? "" : "opacity-0 group-hover:opacity-100 focus-visible:opacity-100"}
          onClick={() => togglePinnedSection(section.name)}
        >
          <Pin size={12} />
        </IconButton>
      }
    >
      <div className={cn("grid gap-3", COLUMN_CLASSES[columns])}>
        <Suspense fallback={null}>
          {ordered.map((plot) => (
            <ChartCard
              key={plot.name}
              plot={plot}
              runs={runs}
              runColors={runColors}
              chart={chart}
              onExpand={onExpand}
              onMove={onMove}
              rangeQuery={rangeQueries}
            />
          ))}
        </Suspense>
      </div>
    </Collapsible>
  );
}
